# -*- coding: utf-8 -*-
"""数据层：SQLite 元数据表（PostgreSQL 的本地替身）
表：docs / chunks / structured_tables / aliases / qa_eval / query_logs / meta
约束：原始文档是唯一可信源；所有删除均为软删除（valid=0）
"""
import sqlite3
import json
import uuid
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "storage" / "kb.sqlite3"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def new_id() -> str:
    return uuid.uuid4().hex


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = get_conn()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS docs (
        doc_id      TEXT PRIMARY KEY,
        name        TEXT NOT NULL,
        fingerprint TEXT NOT NULL,
        version     INTEGER NOT NULL DEFAULT 1,
        status      TEXT NOT NULL DEFAULT 'processing',  -- processing/ready/failed/rolled_back/archived
        file_path   TEXT,
        created_at  TEXT, updated_at TEXT
    );
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id     TEXT PRIMARY KEY,
        doc_id       TEXT NOT NULL,
        version      INTEGER NOT NULL,
        seq          INTEGER NOT NULL,
        text         TEXT NOT NULL,          -- 拼接章节标题后的正文
        body         TEXT NOT NULL,          -- 不含标题前缀的正文
        section_path TEXT NOT NULL,          -- 父章节标题路径
        ctype        TEXT NOT NULL,          -- narrative / param
        tags         TEXT NOT NULL DEFAULT '[]',
        valid        INTEGER NOT NULL DEFAULT 1,
        created_at   TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id, version, valid);
    CREATE TABLE IF NOT EXISTS structured_tables (
        table_id  TEXT PRIMARY KEY,
        doc_id    TEXT NOT NULL,
        version   INTEGER NOT NULL,
        name      TEXT NOT NULL,
        columns   TEXT NOT NULL,             -- JSON array
        rows      TEXT NOT NULL,             -- JSON array of arrays
        valid     INTEGER NOT NULL DEFAULT 1
    );
    CREATE TABLE IF NOT EXISTS aliases (
        id     INTEGER PRIMARY KEY AUTOINCREMENT,
        entity TEXT NOT NULL,               -- 规范实体名
        alias  TEXT NOT NULL,               -- 别名/同义词
        UNIQUE(entity, alias)
    );
    CREATE TABLE IF NOT EXISTS qa_eval (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        question         TEXT NOT NULL,
        expected_doc     TEXT,              -- 期望命中的文档名
        expected_keywords TEXT NOT NULL DEFAULT '[]',  -- JSON array，答案应包含的关键词
        source           TEXT NOT NULL DEFAULT 'manual', -- manual / auto(线上不满意自动收集)
        created_at       TEXT
    );
    CREATE TABLE IF NOT EXISTS query_logs (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        ts         TEXT,
        query      TEXT,
        intent     TEXT,
        route      TEXT,
        top_chunks TEXT,
        answer     TEXT,
        feedback   TEXT                       -- good / bad / NULL
    );
    CREATE TABLE IF NOT EXISTS eval_runs (
        id      INTEGER PRIMARY KEY AUTOINCREMENT,
        ts      TEXT,
        metrics TEXT NOT NULL,              -- JSON
        passed  INTEGER,
        action  TEXT                        -- keep / rollback / baseline_set
    );
    CREATE TABLE IF NOT EXISTS meta (
        key   TEXT PRIMARY KEY,
        value TEXT
    );
    """)
    conn.commit()
    conn.close()


# ---------- meta ----------
def meta_get(key, default=None):
    conn = get_conn()
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    conn.close()
    return row["value"] if row else default


def meta_set(key, value):
    conn = get_conn()
    conn.execute("INSERT INTO meta(key,value) VALUES(?,?) "
                 "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
    conn.commit()
    conn.close()


# ---------- docs ----------
def get_doc_by_name(name):
    conn = get_conn()
    row = conn.execute("SELECT * FROM docs WHERE name=? ORDER BY updated_at DESC LIMIT 1",
                       (name,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_doc_by_id(doc_id):
    conn = get_conn()
    row = conn.execute("SELECT * FROM docs WHERE doc_id=?", (doc_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def create_doc(name, fingerprint, file_path):
    doc_id, ts = new_id(), now()
    conn = get_conn()
    conn.execute("INSERT INTO docs(doc_id,name,fingerprint,version,status,file_path,created_at,updated_at) "
                 "VALUES(?,?,?,?,?,?,?,?)",
                 (doc_id, name, fingerprint, 1, "processing", file_path, ts, ts))
    conn.commit()
    conn.close()
    return doc_id


def update_doc(doc_id, **fields):
    fields["updated_at"] = now()
    sets = ",".join(f"{k}=?" for k in fields)
    conn = get_conn()
    conn.execute(f"UPDATE docs SET {sets} WHERE doc_id=?", (*fields.values(), doc_id))
    conn.commit()
    conn.close()


# ---------- chunks ----------
def insert_chunks(rows):
    conn = get_conn()
    conn.executemany("INSERT INTO chunks(chunk_id,doc_id,version,seq,text,body,section_path,ctype,tags,valid,created_at) "
                     "VALUES(?,?,?,?,?,?,?,?,?,1,?)", rows)
    conn.commit()
    conn.close()


def soft_delete_chunks(doc_id, keep_version=None):
    """标记 valid=0；keep_version 指定版本不删（用于回滚时恢复旧版本）"""
    conn = get_conn()
    if keep_version is None:
        conn.execute("UPDATE chunks SET valid=0 WHERE doc_id=?", (doc_id,))
        conn.execute("UPDATE structured_tables SET valid=0 WHERE doc_id=?", (doc_id,))
    else:
        conn.execute("UPDATE chunks SET valid=0 WHERE doc_id=? AND version<>?", (doc_id, keep_version))
    conn.commit()
    conn.close()


def restore_version(doc_id, version):
    """回滚：恢复指定版本的 chunk"""
    conn = get_conn()
    conn.execute("UPDATE chunks SET valid=CASE WHEN version=? THEN 1 ELSE 0 END WHERE doc_id=?",
                 (version, doc_id))
    conn.execute("UPDATE structured_tables SET valid=CASE WHEN version=? THEN 1 ELSE 0 END WHERE doc_id=?",
                 (version, doc_id))
    conn.commit()
    conn.close()


def get_chunks(doc_id, version=None, valid_only=True):
    conn = get_conn()
    sql = "SELECT * FROM chunks WHERE doc_id=?"
    args = [doc_id]
    if version:
        sql += " AND version=?"
        args.append(version)
    if valid_only:
        sql += " AND valid=1"
    rows = conn.execute(sql, args).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def stale_invalid_chunks(days=7):
    """取超过 N 天前被软删除的 chunk（用于物理清理）"""
    conn = get_conn()
    rows = conn.execute(
        "SELECT chunk_id FROM chunks WHERE valid=0 AND created_at <= datetime('now', ?)",
        (f"-{int(days)} days",)).fetchall()
    conn.close()
    return [r["chunk_id"] for r in rows]


def hard_delete_chunks(chunk_ids):
    if not chunk_ids:
        return
    conn = get_conn()
    marks = ",".join("?" * len(chunk_ids))
    conn.execute(f"DELETE FROM chunks WHERE chunk_id IN ({marks})", chunk_ids)
    conn.commit()
    conn.close()


# ---------- structured tables ----------
def insert_tables(rows):
    conn = get_conn()
    conn.executemany("INSERT INTO structured_tables(table_id,doc_id,version,name,columns,rows,valid) "
                     "VALUES(?,?,?,?,?,?,1)", rows)
    conn.commit()
    conn.close()


def get_valid_tables():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM structured_tables WHERE valid=1").fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------- aliases ----------
def get_aliases():
    conn = get_conn()
    rows = conn.execute("SELECT id, entity, alias FROM aliases").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_alias(alias_id):
    conn = get_conn()
    conn.execute("DELETE FROM aliases WHERE id=?", (alias_id,))
    conn.commit()
    conn.close()


def add_alias(entity, alias):
    conn = get_conn()
    conn.execute("INSERT OR IGNORE INTO aliases(entity,alias) VALUES(?,?)", (entity, alias))
    conn.commit()
    conn.close()


# ---------- qa eval ----------
def add_eval_item(question, expected_doc, expected_keywords, source="manual"):
    conn = get_conn()
    conn.execute("INSERT INTO qa_eval(question,expected_doc,expected_keywords,source,created_at) "
                 "VALUES(?,?,?,?,?)",
                 (question, expected_doc, json.dumps(expected_keywords, ensure_ascii=False), source, now()))
    conn.commit()
    conn.close()


def get_eval_items(source=None):
    conn = get_conn()
    if source:
        rows = conn.execute("SELECT * FROM qa_eval WHERE source=?", (source,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM qa_eval").fetchall()
    conn.close()
    out = []
    for r in rows:
        d = dict(r)
        d["expected_keywords"] = json.loads(d["expected_keywords"])
        out.append(d)
    return out


# ---------- query logs ----------
def log_query(query, intent, route, top_chunks, answer=None):
    conn = get_conn()
    cur = conn.execute("INSERT INTO query_logs(ts,query,intent,route,top_chunks,answer) VALUES(?,?,?,?,?,?)",
                       (now(), query, intent, route, json.dumps(top_chunks, ensure_ascii=False), answer))
    conn.commit()
    log_id = cur.lastrowid
    conn.close()
    return log_id


def set_feedback(log_id, feedback):
    conn = get_conn()
    conn.execute("UPDATE query_logs SET feedback=? WHERE id=?", (feedback, log_id))
    conn.commit()
    row = conn.execute("SELECT query FROM query_logs WHERE id=?", (log_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def recent_logs(limit=50):
    conn = get_conn()
    rows = conn.execute("SELECT * FROM query_logs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    out = []
    for r in rows:
        d = dict(r)
        d["top_chunks"] = json.loads(d["top_chunks"] or "[]")
        out.append(d)
    return out


# ---------- eval runs ----------
def save_eval_run(metrics, passed, action):
    conn = get_conn()
    conn.execute("INSERT INTO eval_runs(ts,metrics,passed,action) VALUES(?,?,?,?)",
                 (now(), json.dumps(metrics, ensure_ascii=False), int(passed), action))
    conn.commit()
    conn.close()
