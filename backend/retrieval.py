# -*- coding: utf-8 -*-
"""线上实时检索工作流：
Query 预处理(实体抽取/意图分类/别名扩展) -> 路由分发(结构化SQL/稀疏/多路召回)
-> 元数据过滤(valid=true + 实体标签) -> 去重 + CrossEncoder 重排
-> 上下文组装(带来源：文档名/章节/版本/chunk_id) -> 可选 LLM 生成
"""
import re
import json
import math
from collections import Counter

import db
from modelhub import embed_texts, rerank
from ingest import collection

# ---------- Query 预处理 ----------
STRUCT_HINTS = ["多少", "几个", "几秒", "几天", "伤害", "价格", "费用", "花费", "冷却", "cd",
                "上限", "下限", "时长", "天数", "比例", "额度", "标准", "限额", "数值", "参数"]
NOUN_HINTS = ["是什么", "是什么意思", "指的是", "什么意思", "介绍", "定义"]

_cjk_re = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text: str):
    """中文字符 bigram + 英文数字词，用于 BM25 稀疏检索"""
    text = text.lower()
    toks = re.findall(r"[a-z0-9]+", text)
    for run in _cjk_re.findall(text):
        if len(run) == 1:
            toks.append(run)
        else:
            toks.extend(run[i:i + 2] for i in range(len(run) - 1))
    return toks


def preprocess_query(q: str) -> dict:
    """实体抽取 + 意图分类 + 别名扩展"""
    aliases = db.get_aliases()
    entities, expansion = [], []
    for a in aliases:
        for word in {a["entity"], a["alias"]}:
            if word and word in q:
                if a["entity"] not in entities:
                    entities.append(a["entity"])
                if a["alias"] not in expansion and a["alias"] not in q:
                    expansion.append(a["alias"])
    q_lower = q.lower()
    if any(h in q_lower for h in STRUCT_HINTS) or re.search(r"\d+\s*(元|天|次|秒|点|%)", q):
        intent = "structured"
    elif len(q) <= 8 and entities:
        intent = "proper_noun"
    else:
        intent = "open_qa"
    expanded = q + (" " + " ".join(expansion) if expansion else "")
    return {"intent": intent, "entities": entities, "expansion": expansion,
            "expanded_query": expanded}


# ---------- BM25 稀疏检索 ----------
class BM25:
    def __init__(self, corpus_tokens):
        self.docs = corpus_tokens
        self.N = len(corpus_tokens)
        self.avgdl = sum(len(d) for d in corpus_tokens) / max(self.N, 1) or 1
        self.df = Counter()
        for d in corpus_tokens:
            self.df.update(set(d))
        self.k1, self.b = 1.5, 0.75

    def idf(self, term):
        df = self.df.get(term, 0)
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def search(self, query_toks, top_k=30):
        scores = []
        for i, d in enumerate(self.docs):
            if not d:
                scores.append((i, 0.0))
                continue
            tf = Counter(d)
            dl = len(d)
            s = 0.0
            for t in query_toks:
                if t in tf:
                    s += self.idf(t) * tf[t] * (self.k1 + 1) / (tf[t] + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            scores.append((i, s))
        scores.sort(key=lambda x: -x[1])
        return [(i, s) for i, s in scores[:top_k] if s > 0]


def _load_valid_chunks():
    """从 SQLite 取全部有效 chunk（元数据是过滤与溯源的唯一事实来源）"""
    conn = db.get_conn()
    rows = conn.execute(
        "SELECT c.chunk_id, c.doc_id, c.version, c.seq, c.text, c.section_path, c.ctype, c.tags, d.name "
        "FROM chunks c JOIN docs d ON c.doc_id=d.doc_id WHERE c.valid=1").fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------- 结构化路由：SQL 直查表 ----------
def structured_lookup(entities, expanded_query):
    """在结构化表中按实体匹配行，返回文本片段（无幻觉，不走向量）"""
    tables = db.get_valid_tables()
    hits = []
    q = expanded_query
    for t in tables:
        cols = json.loads(t["columns"])
        rows = json.loads(t["rows"])
        for r in rows:
            cells = dict(zip(cols, r))
            joined = " ".join(str(v) for v in r)
            if entities:
                matched = any(e in joined for e in entities)
            else:
                matched = False
            # 实体命中 且 查询关注列（含 hint 关键词的列）优先
            if matched:
                text = f"【{t['name']}·结构化数据】" + "；".join(
                    f"{k}：{v}" for k, v in cells.items() if str(v).strip())
                hits.append({"chunk_id": f"sql_{t['table_id']}_{abs(hash(json.dumps(r, ensure_ascii=False))) % 10**8}",
                             "text": text, "source": f"{t['name']}(结构化表)",
                             "section_path": t["name"], "version": t["version"],
                             "ctype": "structured", "doc_id": t["doc_id"], "seq": -1})
    return hits


# ---------- 多路召回 + 过滤 + 重排 ----------
def _vec_search(query, n=30):
    if collection.count() == 0:
        return []
    n = min(n, collection.count())
    vec = embed_texts([query], is_query=True)
    res = collection.query(query_embeddings=vec, n_results=n,
                           include=["documents", "metadatas", "distances"])
    out = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        out.append({"chunk_id": None, "text": doc, "vec_score": 1 - dist, "meta": meta})
    return out


def _tag_filter(cands, entities):
    """按实体标签过滤（标签来自章节标题+文档名）；过滤后结果不足则回退全量"""
    if not entities:
        return cands
    filtered = [c for c in cands
                if any(e in (c.get("tags_str") or "") for e in entities)]
    return filtered if len(filtered) >= 3 else cands


def search(q: str, top_k: int = 5) -> dict:
    pre = preprocess_query(q)
    valid = _load_valid_chunks()
    by_text = {c["text"]: c for c in valid}
    results_info = {"intent": pre["intent"], "route": [], "entities": pre["entities"],
                    "expansion": pre["expansion"], "structured": []}

    candidates = []      # 统一候选格式: {chunk_id, text, source, section_path, version, ctype, doc_id, seq, vec_score?}

    # 路由 1：结构化查询 -> SQL 直查表
    if pre["intent"] == "structured" or pre["entities"]:
        sql_hits = structured_lookup(pre["entities"], pre["expanded_query"])
        if sql_hits:
            results_info["route"].append("structured_sql")
            candidates.extend(sql_hits)
            results_info["structured"] = sql_hits

    # 路由 2：稀疏 BM25（专有名词/关键词）
    bm25 = BM25([tokenize(c["text"]) for c in valid])
    sparse_hits = bm25.search(tokenize(pre["expanded_query"]), top_k=30)
    results_info["route"].append("sparse_bm25")

    # 路由 3：稠密向量
    dense_hits = _vec_search(pre["expanded_query"], n=30)
    results_info["route"].append("dense_vector")

    # 合并稀疏 + 稠密（RRF）
    rrf = {}
    for rank, (i, s) in enumerate(sparse_hits):
        c = valid[i]
        rrf.setdefault(c["text"], {"c": c, "score": 0.0})["score"] += 1 / (60 + rank + 1)
    for rank, d in enumerate(dense_hits):
        c = by_text.get(d["text"])
        if c is None:
            continue                        # chroma 里有但已软删除 -> 丢弃
        rrf.setdefault(c["text"], {"c": c, "score": 0.0})["score"] += 1 / (60 + rank + 1)
    for v in rrf.values():
        c = v["c"]
        candidates.append({"chunk_id": c["chunk_id"], "text": c["text"], "source": c["name"],
                           "section_path": c["section_path"], "version": c["version"],
                           "ctype": c["ctype"], "doc_id": c["doc_id"], "seq": c["seq"],
                           "tags_str": c["tags"], "rrf": v["score"]})

    # 元数据过滤：valid 已在 SQL 保证；实体标签过滤
    candidates = _tag_filter(candidates, pre["entities"])

    # 去重
    seen, deduped = set(), []
    for c in candidates:
        if c["text"] in seen:
            continue
        seen.add(c["text"])
        deduped.append(c)

    if not deduped:
        return {**results_info, "results": [], "log_id": db.log_query(
            q, pre["intent"], "+".join(results_info["route"]), [])}

    # CrossEncoder 重排
    ranked = rerank(q, [c["text"] for c in deduped])
    results = []
    for idx, rr_score in ranked[:top_k]:
        c = deduped[idx]
        results.append({
            "chunk_id": c["chunk_id"], "text": c["text"], "source": c["source"],
            "section_path": c["section_path"], "version": c["version"],
            "ctype": c["ctype"], "score": round(rr_score, 4),
            "rrf": round(c.get("rrf", 0), 4),
            "citation": f"[{len(results)+1}] {c['source']} > {c['section_path']} (v{c['version']})",
        })

    log_id = db.log_query(q, pre["intent"], "+".join(results_info["route"]),
                          [r["chunk_id"] for r in results])
    return {**results_info, "results": results, "log_id": log_id}


# ---------- 上下文组装 + LLM（可选） ----------
def build_context(results, top_n=5):
    """组装带溯源的上下文：文档名、章节、版本、chunk_id"""
    parts = []
    for i, r in enumerate(results[:top_n], 1):
        parts.append(f"[{i}] 来源：{r['source']} > {r['section_path']}｜版本 v{r['version']}｜chunk_id={r['chunk_id']}\n{r['text']}")
    return "\n\n".join(parts)


SYSTEM_PROMPT = (
    "你是企业知识库问答助手。必须遵守：\n"
    "1. 仅根据提供的知识库片段回答，禁止编造；\n"
    "2. 每个论断后标注来源编号，如 [1]；\n"
    "3. 片段中没有的信息，明确回答“知识库中未找到”。"
)


def generate_answer(q: str, results) -> dict:
    """有 LLM 配置则生成带引用回答，否则返回溯源上下文"""
    import os
    context = build_context(results)
    base_url = os.environ.get("LLM_BASE_URL")
    api_key = os.environ.get("LLM_API_KEY")
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    if base_url and api_key and results:
        import httpx
        try:
            resp = httpx.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": model,
                      "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                   {"role": "user", "content": f"知识库片段：\n{context}\n\n问题：{q}"}],
                      "temperature": 0.1},
                timeout=60)
            answer = resp.json()["choices"][0]["message"]["content"]
            return {"answer": answer, "mode": "llm", "context": context}
        except Exception as e:
            return {"answer": None, "mode": "llm_error", "error": str(e), "context": context}
    return {"answer": None, "mode": "context_only",
            "context": context,
            "note": "未配置 LLM（设置环境变量 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL 后可自动生成回答）"}
