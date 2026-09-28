# -*- coding: utf-8 -*-
"""自动化评估：
- 对 QA 评估集逐题检索，计算 Recall@K（文档命中）、Keyword Recall、MRR、Precision@K
- 指标达标 -> 新索引生效；不达标 -> 告警并回滚，继续使用旧索引
"""
import json

import db
from retrieval import search

RECALL_DROP_LIMIT = 0.05     # 指标允许的最大跌幅
MRR_DROP_LIMIT = 0.05


def _judge_item(item, results, top_k=5):
    """单题判定：文档命中 + 关键词召回 + 首个命中排名"""
    top = results[:top_k]
    doc_hit_rank, kw_ranks = None, []
    for rank, r in enumerate(top, 1):
        text = r["text"]
        doc_ok = (not item["expected_doc"]) or (item["expected_doc"] in r["source"])
        kws = item["expected_keywords"]
        kw_ok = all(kw in text for kw in kws) if kws else doc_ok
        if doc_ok and kw_ok and doc_hit_rank is None:
            doc_hit_rank = rank
        if doc_ok and kw_ok:
            kw_ranks.append(rank)
    hit = doc_hit_rank is not None
    kw_recall = 0.0
    kws = item["expected_keywords"]
    if kws:
        found = 0
        for kw in kws:
            if any(kw in r["text"] for r in top):
                found += 1
        kw_recall = found / len(kws)
    return {"hit": hit, "rank": doc_hit_rank or 0, "kw_recall": kw_recall}


def run_eval(gate: bool = False, top_k: int = 5) -> dict:
    """执行评估。gate=True 时对比基线并决定是否放行/回滚"""
    items = db.get_eval_items()
    if not items:
        metrics = {"eval_size": 0, "recall_at_k": 0.0, "keyword_recall": 0.0,
                   "mrr": 0.0, "precision_at_k": 0.0, "top_k": top_k}
        return {"passed": True, "metrics": metrics, "action": "skip_no_evalset", "details": []}

    details, hits, kw_recalls, rr, prec = [], 0, [], 0.0, []
    for item in items:
        res = search(item["question"], top_k=top_k)
        j = _judge_item(item, res["results"], top_k)
        hits += j["hit"]
        kw_recalls.append(j["kw_recall"])
        rr += 1 / j["rank"] if j["rank"] else 0.0
        prec.append(len([r for r in res["results"] if j["hit"]]) / max(len(res["results"]), 1) if j["hit"] else 0.0)
        details.append({"question": item["question"], "hit": j["hit"],
                        "rank": j["rank"], "kw_recall": round(j["kw_recall"], 3),
                        "top_source": res["results"][0]["source"] if res["results"] else None})

    n = len(items)
    metrics = {"eval_size": n,
               "recall_at_k": round(hits / n, 4),
               "keyword_recall": round(sum(kw_recalls) / n, 4),
               "mrr": round(rr / n, 4),
               "precision_at_k": round(sum(prec) / n, 4),
               "top_k": top_k}

    if not gate:
        db.save_eval_run(metrics, True, "eval_only")
        return {"passed": True, "metrics": metrics, "action": "eval_only", "details": details}

    baseline = db.meta_get("baseline_metrics")
    if baseline is None:
        db.meta_set("baseline_metrics", json.dumps(metrics, ensure_ascii=False))
        db.save_eval_run(metrics, True, "baseline_set")
        return {"passed": True, "metrics": metrics, "action": "baseline_set", "details": details}

    base = json.loads(baseline)
    recall_drop = base["recall_at_k"] - metrics["recall_at_k"]
    mrr_drop = base["mrr"] - metrics["mrr"]
    passed = recall_drop <= RECALL_DROP_LIMIT and mrr_drop <= MRR_DROP_LIMIT
    action = "keep" if passed else "rollback"
    db.save_eval_run(metrics, passed, action)
    return {"passed": passed, "metrics": metrics, "baseline": base, "action": action,
            "recall_drop": round(recall_drop, 4), "mrr_drop": round(mrr_drop, 4),
            "details": details}
