# -*- coding: utf-8 -*-
"""入库流水线（离线异步）：
上传保存 -> SHA256 指纹 -> 变更判断(新增/更新/无改动/删除)
-> 解析(标题层级+表格结构化分离+清洗) -> 语义分块(强制拼接父章节标题, 叙事/参数不同粒度)
-> 向量编码只增写 Chroma -> 上线前 QA 评估门禁(不达标自动回滚)
"""
import re
import json
import hashlib
from pathlib import Path

import chromadb

import db
from modelhub import embed_texts

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "storage" / "chroma"

# 分块粒度（按字符近似 token：中文 1 字 ≈ 1 token）
NARR_SIZE, NARR_OVERLAP = 600, 90      # 叙事文本：300~800 token，重叠 10%~20%
PARAM_SIZE, PARAM_OVERLAP = 200, 30    # 参数/列表文本：100~300 token，低重叠

client = chromadb.PersistentClient(path=str(CHROMA_DIR))
try:
    client.delete_collection("doc_chunks")   # 清理旧版 collection
except Exception:
    pass
collection = client.get_or_create_collection(
    name="kb_chunks", metadata={"hnsw:space": "cosine"})


# ================= 1. 指纹与变更判断 =================
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def decide_change(name: str, fingerprint: str) -> dict:
    """返回 action: create / update / unchanged / retry"""
    doc = db.get_doc_by_name(name)
    if doc is None:
        return {"action": "create"}
    if doc["status"] in ("failed", "rolled_back", "archived"):
        # 失败/回滚/已归档的文档重新上传：视为重试，升版本重建
        return {"action": "retry", "old": doc}
    if doc["fingerprint"] == fingerprint:
        return {"action": "unchanged", "old": doc}
    return {"action": "update", "old": doc}


# ================= 2. 解析：标题层级 + 表格分离 + 清洗 =================
HEADING_RE_MD = re.compile(r"^(#{1,4})\s+(.+)$")
LIST_LINE_RE = re.compile(r"^\s*([-*•·]|\d+[.、)]|\([0-9一二三四五六七八九十]+\))\s+")
KV_LINE_RE = re.compile(r"^[^：:]{1,20}[：:]\s*\S")     # “键：值”参数行
NOISE_LINE_RE = re.compile(r"^[\s\d\-—_.·•|/\\]{0,12}$")  # 页码/分隔符等噪声


def is_heading_like(line: str) -> bool:
    """txt/pdf 的启发式标题识别：短行、无结尾标点、非列表"""
    s = line.strip()
    if not s or len(s) > 30:
        return False
    if LIST_LINE_RE.match(s) or NOISE_LINE_RE.match(s):
        return False
    if s.endswith(("。", "，", "；", "！", "？", "：", ".", ",")):
        return False
    return True


def clean_lines(lines):
    out = []
    for ln in lines:
        ln = ln.rstrip()
        if not ln.strip():
            out.append("")
            continue
        if NOISE_LINE_RE.match(ln.strip()):
            continue
        out.append(ln)
    # 折叠连续空行
    cleaned, blank = [], 0
    for ln in out:
        blank = blank + 1 if not ln.strip() else 0
        if blank <= 1:
            cleaned.append(ln)
    return cleaned


def split_markdown_tables(lines):
    """把 markdown 管道表格抽离出来：返回 (剩余行, [tables])
    table = {"name": str, "columns": [...], "rows": [[...]]}
    """
    tables, rest, i = [], [], 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if "|" in line and i + 1 < n and re.match(r"^\s*\|?[\s:|-]+\|[\s:|-]*$", lines[i + 1] or ""):
            # 检测到表格头 + 分隔行
            header = [c.strip() for c in line.strip().strip("|").split("|")]
            i += 2
            rows = []
            while i < n and "|" in lines[i] and lines[i].strip():
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                rows.append(cells + [""] * (len(header) - len(cells)))
                i += 1
            # 表名取表格上方最近的标题行或非空行
            name = "未命名表格"
            for prev in reversed(rest):
                p = prev.strip().lstrip("#").strip()
                if p:
                    name = p[:30]
                    break
            tables.append({"name": name, "columns": header, "rows": rows})
            continue
        rest.append(line)
        i += 1
    return rest, tables


def parse_document(text: str, kind: str):
    """解析为有序 blocks: [{"type": heading/para/table_ref, ...}]
    返回 (blocks, tables)"""
    raw_lines = text.splitlines()
    if kind in ("txt", "md", "pdf"):
        raw_lines, tables = split_markdown_tables(raw_lines)
    else:
        tables = []
    lines = clean_lines(raw_lines)

    blocks, stack = [], []          # stack: 当前标题层级 ["一级","二级"]
    buf = []

    def flush():
        if buf and "".join(buf).strip():
            blocks.append({"type": "para", "section": list(stack), "text": "\n".join(buf).strip()})
        buf.clear()

    for ln in lines:
        m = HEADING_RE_MD.match(ln) if kind == "md" else None
        if m:
            flush()
            level, title = len(m.group(1)), m.group(2).strip()
            stack = stack[:level - 1] + [title]
            blocks.append({"type": "heading", "section": list(stack), "text": title})
        elif is_heading_like(ln):
            # 启发式标题（txt/pdf）：短行、无结尾标点、非列表 -> 一级标题
            flush()
            stack = [ln.strip()]
            blocks.append({"type": "heading", "section": list(stack), "text": ln.strip()})
        elif ln.strip():
            buf.append(ln)
        else:
            flush()
    flush()
    return blocks, tables


def parse_docx(path: Path):
    """docx：按 Heading 样式维护层级，表格直接抽离"""
    from docx import Document as DocxDocument
    doc = DocxDocument(str(path))
    blocks, stack, tables = [], [], []
    body = doc.element.body
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    for child in body.iterchildren():
        if child.tag.endswith("}p"):
            p = Paragraph(child, doc)
            t = p.text.strip()
            if not t:
                continue
            style = (p.style.name or "").lower()
            if style.startswith("heading") or style.startswith("标题"):
                try:
                    level = int(re.sub(r"\D", "", style) or 1)
                except ValueError:
                    level = 1
                stack = stack[:level - 1] + [t]
                blocks.append({"type": "heading", "section": list(stack), "text": t})
            else:
                blocks.append({"type": "para", "section": list(stack), "text": t})
        elif child.tag.endswith("}tbl"):
            table = Table(child, doc)
            cols = [c.text.strip() for c in table.rows[0].cells] if table.rows else []
            rows = [[c.text.strip() for c in r.cells] for r in table.rows[1:]]
            name = stack[-1] if stack else "未命名表格"
            tables.append({"name": name, "columns": cols, "rows": rows})
            blocks.append({"type": "para", "section": list(stack),
                           "text": f"（表格「{name}」已单独结构化存储，共 {len(rows)} 行）"})
    return blocks, tables


# ================= 3. 语义分块 =================
def classify_para(text: str) -> str:
    """参数/列表文本 vs 叙事文本"""
    lines = [l for l in text.splitlines() if l.strip()]
    list_like = sum(1 for l in lines if LIST_LINE_RE.match(l) or KV_LINE_RE.match(l))
    if lines and list_like / len(lines) >= 0.5:
        return "param"
    if text.count("：") + text.count(":") >= 3 and len(text) < 400:
        return "param"
    return "narrative"


def chunk_section(section: list, body: str, ctype: str):
    """把一个 section 的正文切成 chunk，强制拼接父章节标题"""
    size = PARAM_SIZE if ctype == "param" else NARR_SIZE
    overlap = PARAM_OVERLAP if ctype == "param" else NARR_OVERLAP
    prefix = " > ".join(section) if section else "（无标题）"

    if len(body) <= size:
        return [(prefix, body)]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    chunks, buf = [], ""
    for para in paragraphs:
        if len(para) > size:                      # 超长段落按句子滑动窗口
            if buf:
                chunks.append(buf)
                buf = ""
            sentences = [s for s in re.split(r"(?<=[。！？.!?\n])", para) if s.strip()]
            win = ""
            for s in sentences:
                if len(win) + len(s) > size and win:
                    chunks.append(win)
                    win = win[-overlap:] if len(win) > overlap else ""
                win += s
            if win.strip():
                buf = win
            continue
        if buf and len(buf) + len(para) + 2 > size:
            chunks.append(buf)
            if ctype == "narrative" and overlap < len(buf):   # 叙事保留重叠上下文
                buf = buf[-overlap:] + "\n" + para
            else:
                buf = para
        else:
            buf = f"{buf}\n{para}" if buf else para
    if buf.strip():
        chunks.append(buf)
    return [(prefix, c.strip()) for c in chunks if c.strip()]


# ================= 4. 完整入库流程 =================
def ingest_document(path: Path, orig_name: str) -> dict:
    suffix = path.suffix.lower()
    kind = {"md": "md", "txt": "txt", "pdf": "pdf", "docx": "docx"}[suffix.lstrip(".")]

    fingerprint = sha256_file(path)
    decision = decide_change(orig_name, fingerprint)
    if decision["action"] == "unchanged":
        path.unlink(missing_ok=True)
        return {"status": "unchanged", "doc_id": decision["old"]["doc_id"],
                "message": "文档指纹相同，无改动，跳过入库"}

    # 文档元数据
    if decision["action"] == "create":
        doc_id = db.create_doc(orig_name, fingerprint, str(path))
        version = 1
    else:  # retry / update
        old = decision["old"]
        doc_id = old["doc_id"]
        version = old["version"] + 1
        db.update_doc(doc_id, fingerprint=fingerprint, version=version,
                      status="processing", file_path=str(path))
        db.soft_delete_chunks(doc_id)          # 旧版本全部软删除（保留向量用于回滚）

    try:
        # 解析
        if kind == "docx":
            blocks, tables = parse_docx(path)
        else:
            for enc in ("utf-8", "gbk", "utf-16"):
                try:
                    text = path.read_text(encoding=enc)
                    break
                except (UnicodeDecodeError, UnicodeError):
                    text = None
            if text is None:
                text = path.read_text(encoding="utf-8", errors="ignore")
            blocks, tables = parse_document(text, kind)

        # 表格结构化分离（禁止直接丢入向量）
        table_rows = []
        for t in tables:
            table_rows.append((db.new_id(), doc_id, version, t["name"],
                               json.dumps(t["columns"], ensure_ascii=False),
                               json.dumps(t["rows"], ensure_ascii=False)))
        if table_rows:
            db.insert_tables(table_rows)

        # 分块
        chunk_rows, docs2embed, ids, metas = [], [], [], []
        seq = 0
        pending: list = []          # (section, texts) 同类型聚合
        for b in blocks:
            if b["type"] == "heading":
                continue
            section = b["section"] or ["（无标题）"]
            ctype = classify_para(b["text"])
            for prefix, piece in chunk_section(section, b["text"], ctype):
                full_text = f"【{prefix}】\n{piece}"
                chunk_id = db.new_id()
                tags = json.dumps(section + [orig_name], ensure_ascii=False)
                chunk_rows.append((chunk_id, doc_id, version, seq, full_text, piece,
                                   prefix, ctype, tags, db.now()))
                docs2embed.append(full_text)
                ids.append(chunk_id)
                metas.append({"source": orig_name, "doc_id": doc_id, "version": version,
                              "seq": seq, "section_path": prefix, "ctype": ctype})
                seq += 1
                if pending:
                    pending.clear()

        if not chunk_rows:
            db.update_doc(doc_id, status="failed")
            return {"status": "failed", "doc_id": doc_id, "message": "未解析出有效内容"}

        # 向量编码 & 只增写入索引
        B = 64
        for i in range(0, len(docs2embed), B):
            vecs = embed_texts(docs2embed[i:i + B])
            collection.add(ids=ids[i:i + B], documents=docs2embed[i:i + B],
                           metadatas=metas[i:i + B], embeddings=vecs)
        db.insert_chunks(chunk_rows)

        # 上线前评估门禁
        import evaluate
        gate = evaluate.run_eval(gate=True)
        if gate["passed"]:
            db.update_doc(doc_id, status="ready")
            return {"status": "ready", "doc_id": doc_id, "chunks": seq,
                    "tables": len(table_rows), "version": version,
                    "eval": gate["metrics"]}
        else:
            # 评估不达标：回滚到旧版本
            prev = version - 1
            if prev >= 1:
                db.restore_version(doc_id, prev)
                db.update_doc(doc_id, status="rolled_back")
            else:
                db.soft_delete_chunks(doc_id)
                db.update_doc(doc_id, status="failed")
            return {"status": "rolled_back", "doc_id": doc_id, "chunks": seq,
                    "version": version, "eval": gate["metrics"],
                    "message": "评估不达标，已回滚到旧版本索引"}
    except Exception as e:
        db.update_doc(doc_id, status="failed")
        return {"status": "failed", "doc_id": doc_id, "message": f"入库失败: {e}"}


def archive_document(doc_id: str):
    """文档删除：状态 archived + 全部 chunk 软删除"""
    db.soft_delete_chunks(doc_id)
    db.update_doc(doc_id, status="archived")
