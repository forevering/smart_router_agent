# Embedding 模型加载指南

本文档以项目当前使用的 `paraphrase-multilingual-MiniLM-L12-v2` 为例，说明 embedding 模型加载时的注意事项和操作步骤。

---

## 1. 当前使用的模型概况

| 项目 | 值 |
|------|-----|
| 模型名称 | `paraphrase-multilingual-MiniLM-L12-v2` |
| 来源 | HuggingFace / Sentence-Transformers |
| 向量维度 | 384 |
| 运行方式 | 本地推理（无远程 API 调用） |
| 模型大小 | ~500 MB（首次加载到内存） |
| 多语言支持 | 50+ 语言（含中文、英文、德文） |

---

## 2. 加载前的注意事项

### 2.1 离线环境配置

在 import 任何 transformers/sentence-transformers 库 **之前**，必须设置以下环境变量，否则模型会尝试联网下载或校验：

```python
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
```

> **重要：** 这些变量必须在 `from sentence_transformers import SentenceTransformer` 之前设置，否则不生效。

### 2.2 模型文件必须预先下载到本地缓存

离线模式下，模型文件必须已经存在于本地缓存目录：
- Windows 默认缓存路径：`C:\Users\<用户名>\.cache\huggingface\hub\`
- Linux 默认缓存路径：`~/.cache/huggingface/hub/`

如果缓存目录中没有模型文件，离线加载会直接报错。

**首次下载模型的方法：**

```bash
# 方法 1：联网环境下直接 Python 加载（会自动下载到缓存）
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# 方法 2：使用 HuggingFace 镜像站（适用于国内网络环境）
set HF_ENDPOINT=https://hf-mirror.com
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# 方法 3：手动下载后放到缓存目录
# 从 https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 下载
```

### 2.3 依赖包版本要求

```
sentence-transformers>=2.2.0
torch>=1.11.0
transformers>=4.20.0
```

确保这些包已正确安装。可以用以下命令确认：

```bash
pip show sentence-transformers torch transformers
```

### 2.4 向量维度一致性

模型输出的向量维度 **必须** 与 Qdrant 向量数据库中创建的 collection 的 `vector_size` 一致。当前项目中所有 collection 统一使用 **384 维**：

| Collection | 用途 | 向量维度 |
|---|---|---|
| `tool_registry` | 工具语义检索 | 384 |
| `ltm_facts` | 长期记忆存取 | 384 |
| `kg_nodes` | 知识图谱节点检索 | 384 |

> **风险提示：** 如果更换模型（如换成 768 维的模型），必须同时清除并重建所有 Qdrant collection，否则会出现维度不匹配错误。

### 2.5 单例模式加载

模型加载耗时较长（首次约 3-10 秒）且占用 ~500MB 内存。应使用单例模式避免重复加载：

```python
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model
```

**不要** 在每次调用时都创建新的 `SentenceTransformer` 实例。

---

## 3. 一步步加载流程

### Step 1：确认模型文件存在

```python
import os
from pathlib import Path

cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
model_dir = cache_dir / "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"

if model_dir.exists():
    print("✓ 模型缓存已存在")
else:
    print("✗ 模型缓存不存在，需要先下载")
```

### Step 2：设置离线环境变量

```python
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
```

### Step 3：加载模型

```python
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384

model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print(f"模型加载完成: {EMBEDDING_MODEL_NAME} (dim={EMBEDDING_DIM})")
```

### Step 4：验证模型输出

```python
test_text = "这是一个测试句子"
embedding = model.encode(test_text)

assert embedding.shape == (384,), f"维度不匹配: 期望 384, 实际 {embedding.shape}"
print(f"✓ 输出维度正确: {embedding.shape}")
```

### Step 5：验证语义相似度功能

```python
from sentence_transformers.util import cos_sim

query = "天气怎么样"
sentences = ["今天天气很好", "我想吃苹果", "明天会下雨吗"]

query_emb = model.encode(query)
sent_embs = model.encode(sentences)

scores = cos_sim(query_emb, sent_embs)
print(f"相似度分数: {scores.tolist()}")
# 预期: "天气" 相关句子的分数明显高于无关句子
```

### Step 6：集成到项目中

```python
from config.config import get_embedding_model, QDRANT_PATH
from qdrant_client import QdrantClient

# 获取单例模型
embedding_model = get_embedding_model()

# 连接向量数据库
qdrant_client = QdrantClient(path=QDRANT_PATH)

# 初始化各组件
from memory.memory_store import MemoryStore
from memory.temporal_kg import TemporalKG
from tools.tool_registry import ToolRegistry

memory_store = MemoryStore(db_path, qdrant_client, embedding_model)
temporal_kg = TemporalKG(db_path, qdrant_client, embedding_model)
tool_registry = ToolRegistry(qdrant_client, embedding_model)
```

---

## 4. 常见问题排查

| 问题 | 原因 | 解决方法 |
|------|------|----------|
| `OSError: Can't load tokenizer` | 模型文件未下载到本地 | 联网环境下先运行一次加载，或手动下载模型文件 |
| `ConnectionError` | 离线变量设置晚于 import | 将 `os.environ` 设置移到文件最顶部 |
| 维度不匹配错误 | Qdrant collection 的 vector_size 与模型输出不一致 | 确保 collection 创建时指定 `size=384` |
| 内存不足 | 多次实例化模型 | 使用单例模式 `get_embedding_model()` |
| 中文效果差 | 使用了仅英文模型 | 当前模型 `multilingual` 已支持中文，无需更换 |
| 加载速度慢 | 首次加载正常现象 | 使用单例，仅首次加载慢，后续复用 |

---

## 5. 更换模型时的检查清单

如果需要更换 embedding 模型，需按以下步骤操作：

- [ ] 确认新模型的向量维度
- [ ] 更新 `config/config.py` 中的 `EMBEDDING_MODEL_NAME` 和 `EMBEDDING_DIM`
- [ ] 删除 `qdrant_data/` 下所有 collection 数据（维度变化时必须重建）
- [ ] 重新注册所有工具（tool_registry 需要重新 encode）
- [ ] 清空长期记忆 collection（ltm_facts 需要重新 encode）
- [ ] 清空知识图谱 collection（kg_nodes 需要重新 encode）
- [ ] 运行验证脚本确认新模型工作正常
- [ ] 重新评估相似度阈值（如 `min_score=0.3`、去重阈值 `0.95`）

---

## 6. 项目中的实际使用方式

模型在项目中的三个核心用途：

```
用户输入 → encode → 查询向量
                         ↓
              ┌──────────┼──────────┐
              ↓          ↓          ↓
        tool_registry  ltm_facts  kg_nodes
        (工具检索)    (记忆检索)  (图谱检索)
              ↓          ↓          ↓
         cosine sim  cosine sim  cosine sim
              ↓          ↓          ↓
         Top-K 工具   Top-K 记忆  种子实体
```

所有语义检索均使用 **余弦相似度 (Cosine Distance)**，由 Qdrant 在检索时自动计算。
