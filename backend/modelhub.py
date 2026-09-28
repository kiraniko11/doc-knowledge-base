# -*- coding: utf-8 -*-
"""模型层：BGE-small-zh 稠密向量 + BGE-reranker 交叉编码精排（本地 ONNX 推理）
约束：Embedding 模型一旦选定不可随意更换（更换需全量重建索引）
"""
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
EMBED_DIR = BASE_DIR / "models" / "bge-small-zh"
RERANK_DIR = BASE_DIR / "models" / "bge-reranker-base"

# ---------- 稠密嵌入（bge-small-zh-v1.5, 384 维） ----------
_tokenizer = Tokenizer.from_file(str(EMBED_DIR / "tokenizer.json"))
_tokenizer.enable_truncation(max_length=512)
_tokenizer.enable_padding()
_session = ort.InferenceSession(str(EMBED_DIR / "model_quantized.onnx"),
                                providers=["CPUExecutionProvider"])
_INPUT_NAMES = {i.name for i in _session.get_inputs()}
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


def embed_texts(texts, is_query=False) -> list:
    """批量文本 -> 归一化向量（mean pooling）"""
    if is_query:
        texts = [QUERY_PREFIX + t for t in texts]
    encs = _tokenizer.encode_batch(texts)
    ids = np.array([e.ids for e in encs], dtype=np.int64)
    mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
    feeds = {"input_ids": ids, "attention_mask": mask}
    if "token_type_ids" in _INPUT_NAMES:
        feeds["token_type_ids"] = np.zeros_like(ids)
    out = _session.run(None, feeds)[0]
    mask_f = mask[:, :, None].astype(np.float32)
    summed = (out * mask_f).sum(axis=1)
    counts = np.clip(mask_f.sum(axis=1), 1e-9, None)
    vecs = summed / counts
    vecs = vecs / np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9, None)
    return vecs.tolist()


# ---------- CrossEncoder 重排（bge-reranker-base int8） ----------
_rr_tokenizer = Tokenizer.from_file(str(RERANK_DIR / "tokenizer.json"))
_rr_tokenizer.enable_truncation(max_length=512)
_rr_tokenizer.enable_padding()
_rr_session = ort.InferenceSession(str(RERANK_DIR / "model_int8.onnx"),
                                   providers=["CPUExecutionProvider"])
_RR_INPUT_NAMES = {i.name for i in _rr_session.get_inputs()}


def rerank(query: str, passages: list) -> list:
    """返回 [(原始下标, sigmoid 相关性分数)]，按分数降序"""
    if not passages:
        return []
    pairs = [(query, p) for p in passages]
    encs = _rr_tokenizer.encode_batch(pairs, add_special_tokens=True)
    ids = np.array([e.ids for e in encs], dtype=np.int64)
    mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
    feeds = {"input_ids": ids, "attention_mask": mask}
    if "token_type_ids" in _RR_INPUT_NAMES:
        feeds["token_type_ids"] = np.zeros_like(ids)
    logits = _rr_session.run(None, feeds)[0].reshape(-1)
    scores = 1.0 / (1.0 + np.exp(-logits))
    order = np.argsort(-scores)
    return [(int(i), float(scores[i])) for i in order]
