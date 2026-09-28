# -*- coding: utf-8 -*-
"""文档知识库 · RAG 工作流服务
主线1 文档入库&增量更新（指纹/软删除/评估门禁）
主线2 用户查询检索（多路召回+过滤+重排+溯源+可选LLM）
主线3 定时运维（物理清理/失败收集/备份）
"""
import shutil
import shutil as _sh
import json
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import db
import ingest
import retrieval
import evaluate

BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = BASE_DIR / "storage" / "uploads"
BACKUP_DIR = BASE_DIR / "storage" / "backups"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED = {".pdf", ".txt", ".md", ".docx"}

app = FastAPI(title="文档知识库 · RAG 工作流")


# ================= 启动：初始化 + 种子数据 =================
def seed_if_empty():
    if db.get_conn().execute("SELECT COUNT(*) c FROM docs").fetchone()["c"] > 0:
        return
    # 种子文档 1：员工手册
    demo1 = BASE_DIR / "员工手册.txt"
    if demo1.exists():
        ingest.ingest_document(demo1, "员工手册.txt")
    # 种子文档 2：武器参数表（演示结构化表格路由）
    demo2 = BASE_DIR / "武器参数表.md"
    if demo2.exists():
        ingest.ingest_document(demo2, "武器参数表.md")
    # 种子别名（Query 扩展）
    for entity, alias in [("大狙", "冥驹"), ("AWP", "大狙"), ("年假", "带薪年假"),
                          ("报销", "差旅报销"), ("公积金", "五险一金")]:
        db.add_alias(entity, alias)
    # 种子评估集
    for q, doc, kws in [
        ("出差住酒店一晚上限额是多少", "员工手册", ["600"]),
        ("密码多久需要更换一次", "员工手册", ["90"]),
        ("加班到很晚能报销打车费吗", "员工手册", ["打车", "餐补"]),
        ("大狙的价格是多少", "武器参数表", ["4750"]),
        ("冥驹的伤害是多少", "武器参数表", ["115"]),
        ("入职第一天要带哪些材料", "员工手册", ["身份证", "体检报告"]),
    ]:
        db.add_eval_item(q, doc, kws, source="manual")
    # 建立基线指标（gate=True 且无基线时自动写入）
    evaluate.run_eval(gate=True)


@app.on_event("startup")
def startup():
    db.init_db()
    seed_if_empty()


# ================= 主线1：文档接入 =================
@app.post("/api/upload")
async def upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """上传原始文档（对象存储），异步执行解析/分块/编码/评估门禁"""
    safe_name = Path(file.filename).name
    if not safe_name or Path(safe_name).suffix.lower() not in ALLOWED:
        raise HTTPException(400, f"仅支持 {'/'.join(sorted(ALLOWED))} 文件")
    save_path = STORAGE_DIR / f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{safe_name}"
    with save_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    result = ingest.ingest_document(save_path, safe_name)   # 指纹相同的会自动清理
    return result


@app.get("/api/docs")
def list_docs():
    conn = db.get_conn()
    rows = conn.execute("""
        SELECT d.doc_id, d.name, d.version, d.status, d.fingerprint, d.updated_at,
            SUM(CASE WHEN c.valid=1 THEN 1 ELSE 0 END) active_chunks,
            SUM(CASE WHEN c.valid=0 THEN 1 ELSE 0 END) retired_chunks
        FROM docs d LEFT JOIN chunks c ON c.doc_id=d.doc_id
        GROUP BY d.doc_id ORDER BY d.updated_at DESC""").fetchall()
    conn.close()
    return {"docs": [dict(r) for r in rows]}


@app.delete("/api/docs/{doc_id}")
def delete_doc(doc_id: str):
    doc = db.get_doc_by_id(doc_id)
    if not doc:
        raise HTTPException(404, "文档不存在")
    ingest.archive_document(doc_id)
    return {"ok": True, "message": f"已归档 {doc['name']}（软删除，历史版本可回滚）"}


@app.post("/api/docs/{doc_id}/rollback/{version}")
def rollback_doc(doc_id: str, version: int):
    """手动一键回滚到指定历史版本"""
    db.restore_version(doc_id, version)
    db.update_doc(doc_id, status="ready")
    return {"ok": True, "message": f"已回滚到 v{version}"}


# ================= 主线2：检索问答 =================
@app.get("/api/search")
def search_api(q: str, top_k: int = 5):
    if not q.strip():
        raise HTTPException(400, "查询不能为空")
    return retrieval.search(q.strip(), top_k=min(max(top_k, 1), 20))


@app.get("/api/answer")
def answer_api(q: str, top_k: int = 5):
    if not q.strip():
        raise HTTPException(400, "查询不能为空")
    res = retrieval.search(q.strip(), top_k=min(max(top_k, 1), 20))
    gen = retrieval.generate_answer(q.strip(), res["results"])
    # 回答写回日志
    if res.get("log_id") and gen.get("answer"):
        conn = db.get_conn()
        conn.execute("UPDATE query_logs SET answer=? WHERE id=?", (gen["answer"], res["log_id"]))
        conn.commit()
        conn.close()
    return {**res, "answer": gen.get("answer"), "answer_mode": gen["mode"],
            "context": gen["context"], "note": gen.get("note")}


class FeedbackIn(BaseModel):
    log_id: int
    feedback: str      # good / bad


@app.post("/api/feedback")
def feedback(body: FeedbackIn):
    row = db.set_feedback(body.log_id, body.feedback)
    if row is None:
        raise HTTPException(404, "日志不存在")
    # 主线3：失败案例自动加入评估集
    if body.feedback == "bad":
        db.add_eval_item(row["query"], None, [], source="auto")
        return {"ok": True, "message": "已记录不满意反馈，并自动加入评估集待人工标注"}
    return {"ok": True}


# ================= 别名（Query 扩展） =================
@app.get("/api/aliases")
def aliases_list():
    return {"aliases": db.get_aliases()}


class AliasIn(BaseModel):
    entity: str
    alias: str


@app.post("/api/aliases")
def aliases_add(body: AliasIn):
    db.add_alias(body.entity.strip(), body.alias.strip())
    return {"ok": True}


@app.delete("/api/aliases/{alias_id}")
def aliases_delete(alias_id: int):
    db.delete_alias(alias_id)
    return {"ok": True}


# ================= 评估 =================
class EvalIn(BaseModel):
    question: str
    expected_doc: str = None
    expected_keywords: list = []


@app.post("/api/eval")
def eval_add(body: EvalIn):
    db.add_eval_item(body.question.strip(), body.expected_doc, body.expected_keywords)
    return {"ok": True}


@app.get("/api/eval")
def eval_list():
    return {"items": db.get_eval_items()}


@app.delete("/api/eval/{item_id}")
def eval_delete(item_id: int):
    conn = db.get_conn()
    conn.execute("DELETE FROM qa_eval WHERE id=?", (item_id,))
    conn.commit()
    conn.close()
    return {"ok": True}


@app.post("/api/eval/run")
def eval_run():
    return evaluate.run_eval(gate=False)


@app.post("/api/eval/set_baseline")
def eval_baseline():
    r = evaluate.run_eval(gate=False)
    db.meta_set("baseline_metrics", json.dumps(r["metrics"], ensure_ascii=False))
    db.save_eval_run(r["metrics"], True, "baseline_set")
    return {"ok": True, "metrics": r["metrics"]}


@app.get("/api/eval/baseline")
def eval_baseline_get():
    raw = db.meta_get("baseline_metrics")
    return {"baseline": json.loads(raw) if raw else None}


# ================= 主线3：运维 =================
@app.post("/api/maintenance/cleanup")
def maintenance_cleanup(days: int = 7):
    """物理删除长期 valid=false 的 chunk（向量 + 元数据）"""
    stale = db.stale_invalid_chunks(days)
    if stale:
        ingest.collection.delete(ids=stale)
        db.hard_delete_chunks(stale)
    return {"ok": True, "removed": len(stale)}


@app.post("/api/maintenance/backup")
def maintenance_backup():
    """备份元数据与 chunk 快照（SQLite 单文件拷贝）"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = BACKUP_DIR / f"kb_backup_{ts}.sqlite3"
    src = db.DB_PATH
    src_bak = Path(str(src) + "-wal")
    _sh.copy(str(src), dest)
    if src_bak.exists():
        _sh.copy(src_bak, Path(str(dest) + "-wal"))
    return {"ok": True, "backup": dest.name}


@app.get("/api/maintenance/backups")
def maintenance_backups():
    return {"backups": sorted([p.name for p in BACKUP_DIR.glob("kb_backup_*.sqlite3")], reverse=True)}


@app.get("/api/logs")
def logs(limit: int = 50):
    return {"logs": db.recent_logs(limit)}


# ================= 页面 =================
@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
