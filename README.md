# 文档知识库（RAG 工作流版）

本地文档知识库应用：上传 PDF / Word / TXT / Markdown → 自动解析、结构化分离、语义切片入向量库 → 自然语言检索（多路召回 + 重排 + 溯源）。带完整的增量入库、评估门禁、运维备份工作流。

![Python](https://img.shields.io/badge/Python-3.13+-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.1x-green) ![ChromaDB](https://img.shields.io/badge/ChromaDB-persistent-orange)

## 功能总览

### 主线 1：文档入库 & 增量更新
- SHA256 文档指纹四态判断：**新增 / 更新（版本+1，旧片段软删除）/ 无改动跳过 / 失败重试重建**
- 解析保留标题层级（md 标题栈、docx Heading 样式、txt 启发式）
- **表格结构化分离**：表格进 SQLite 结构化表，禁止直接入向量（数值查询走 SQL，无幻觉）
- 语义分块强制拼接父章节标题；叙事文本 600 字（重叠 15%）、参数/列表 200 字（低重叠）
- 向量只增写不改写；**上线前评估门禁**：指标跌幅 >0.05 自动回滚旧版本索引

### 主线 2：实时检索（RAG）
- Query 预处理：实体抽取（别名表）、意图三分类（数值查询/专有名词/开放问答）、别名扩展（冥驹↔大狙）
- 路由：数值查询→SQL 直查结构化表；专名→BM25 稀疏检索；普通问答→稠密+BM25 多路召回 RRF 合并
- `valid=true` 元数据过滤 + 去重 + CrossEncoder 重排 + 溯源引用（文档名>章节>版本>chunk_id）
- 可选 LLM 生成回答（配置 `LLM_BASE_URL` 环境变量启用，Prompt 强制引用来源）
- 日志埋点 + 用户反馈，点踩自动加入评估集

### 主线 3：定时运维
- 退役片段物理清理、SQLite 快照备份、手动版本回滚、失败案例自动收集

## 快速开始

### 1. 安装依赖

```bash
python -m venv .venv
.venv/Scripts/pip install fastapi "uvicorn[standard]" python-multipart pypdf python-docx chromadb onnxruntime tokenizers
# Windows 用 .venv\Scripts\pip，Linux/macOS 用 .venv/bin/pip
```

### 2. 下载本地模型（约 300MB，国内镜像秒下）

模型不放仓库里，需要手动下载到 `models/` 目录：

```bash
mkdir -p models/bge-small-zh models/bge-reranker-base

# 嵌入模型（24MB 量化版）
curl -L -o models/bge-small-zh/tokenizer.json https://hf-mirror.com/Xenova/bge-small-zh-v1.5/resolve/main/tokenizer.json
curl -L -o models/bge-small-zh/model_quantized.onnx https://hf-mirror.com/Xenova/bge-small-zh-v1.5/resolve/main/onnx/model_quantized.onnx

# 重排模型（int8 量化）
curl -L -o models/bge-reranker-base/tokenizer.json https://hf-mirror.com/Xenova/bge-reranker-base/resolve/main/tokenizer.json
curl -L -o models/bge-reranker-base/model_int8.onnx https://hf-mirror.com/Xenova/bge-reranker-base/resolve/main/onnx/model_int8.onnx
```

### 3. 启动

```bash
cd backend
../.venv/Scripts/python -m uvicorn app:app --host 127.0.0.1 --port 8765
# 或者双击项目根目录的 start.bat
```

打开 http://127.0.0.1:8765 即可使用（三页签：检索问答 / 文档管理 / 评估与运维）。

## 项目结构

```
doc-knowledge-base/
├── backend/
│   ├── app.py          # FastAPI 路由：上传/检索/评估/运维/别名/反馈
│   ├── db.py           # SQLite 元数据表：docs/chunks/structured/aliases/qa_eval/logs
│   ├── ingest.py       # 入库流水线：指纹→解析→结构化分离→分块→编码→门禁
│   ├── retrieval.py    # 检索工作流：预处理→路由→多路召回→重排→溯源组装
│   ├── evaluate.py     # 评估：Recall/Precision/MRR、基线对比、门禁回滚
│   ├── modelhub.py     # BGE 嵌入 + CrossEncoder 重排（ONNX 本地推理）
│   └── static/index.html
├── models/             # 本地模型（不入库，见上方下载命令）
├── storage/            # 运行时数据：uploads + chroma + sqlite（不入库）
├── start.bat           # Windows 一键启动
└── 武器参数表.md / 员工手册.txt   # 示例文档（首次启动自动种子入库）
```

## 关键设计约束

1. **原始文档是唯一可信源**，向量库/索引只是索引
2. 所有更新使用**软删除**，检索默认过滤 `valid=false`
3. Embedding 模型一旦选定不可随意更换（向量维度与语义空间绑定）
4. 表格类结构化数据禁止直接丢入向量，优先 SQL 查询
5. 上线必须经过 QA 评估，指标下跌禁止切换知识库版本

## 本地化替身说明

生产方案 → 本实现：MinIO/OSS → 本地 uploads 目录、PostgreSQL → SQLite、BGE-M3 稠密+稀疏 → bge-small-zh 稠密 + BM25、Marker → pypdf/docx。升级到真实组件时按模块替换 `db.py` / `modelhub.py` 即可。
