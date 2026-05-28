# Embedding Model Loading Guide

This document uses the project's current model `paraphrase-multilingual-MiniLM-L12-v2` as an example to explain the key considerations and step-by-step procedures for loading an embedding model.

---

## 1. Current Model Overview

| Item | Value |
|------|-------|
| Model Name | `paraphrase-multilingual-MiniLM-L12-v2` |
| Source | HuggingFace / Sentence-Transformers |
| Vector Dimension | 384 |
| Execution Mode | Local inference (no remote API calls) |
| Model Size | ~500 MB (first load into memory) |
| Multilingual Support | 50+ languages (including Chinese, English, German) |

---

## 2. Pre-Loading Considerations

### 2.1 Offline Environment Configuration

The following environment variables **must** be set before importing any transformers/sentence-transformers libraries, otherwise the model will attempt to connect to the internet for downloading or validation:

```python
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
```

> **Important:** These variables must be set before `from sentence_transformers import SentenceTransformer`, otherwise they will not take effect.

### 2.2 Model Files Must Be Pre-Downloaded to Local Cache

In offline mode, model files must already exist in the local cache directory:
- Windows default cache path: `C:\Users\<username>\.cache\huggingface\hub\`
- Linux default cache path: `~/.cache/huggingface/hub/`

If the model files are not in the cache directory, offline loading will fail immediately.

**Methods to download the model for the first time:**

```bash
# Method 1: Load directly in Python with internet access (auto-downloads to cache)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# Method 2: Use HuggingFace mirror (for restricted network environments)
set HF_ENDPOINT=https://hf-mirror.com
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# Method 3: Manual download and place in cache directory
# Download from https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

### 2.3 Dependency Version Requirements

```
sentence-transformers>=2.2.0
torch>=1.11.0
transformers>=4.20.0
```

Ensure these packages are correctly installed. Verify with:

```bash
pip show sentence-transformers torch transformers
```

### 2.4 Vector Dimension Consistency

The model's output vector dimension **must** match the `vector_size` of collections created in the Qdrant vector database. All collections in this project uniformly use **384 dimensions**:

| Collection | Purpose | Vector Dimension |
|---|---|---|
| `tool_registry` | Tool semantic search | 384 |
| `ltm_facts` | Long-term memory storage & retrieval | 384 |
| `kg_nodes` | Knowledge graph node search | 384 |

> **Risk Warning:** If you switch to a different model (e.g., one with 768 dimensions), you must clear and rebuild all Qdrant collections simultaneously, otherwise dimension mismatch errors will occur.

### 2.5 Singleton Loading Pattern

Model loading is time-consuming (approximately 3-10 seconds on first load) and consumes ~500MB of memory. Use the singleton pattern to avoid redundant loading:

```python
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model
```

**Do not** create a new `SentenceTransformer` instance on every call.

---

## 3. Step-by-Step Loading Procedure

### Step 1: Verify Model Files Exist

```python
import os
from pathlib import Path

cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
model_dir = cache_dir / "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"

if model_dir.exists():
    print("✓ Model cache exists")
else:
    print("✗ Model cache not found, download required")
```

### Step 2: Set Offline Environment Variables

```python
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
```

### Step 3: Load the Model

```python
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384

model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print(f"Model loaded: {EMBEDDING_MODEL_NAME} (dim={EMBEDDING_DIM})")
```

### Step 4: Verify Model Output

```python
test_text = "This is a test sentence"
embedding = model.encode(test_text)

assert embedding.shape == (384,), f"Dimension mismatch: expected 384, got {embedding.shape}"
print(f"✓ Output dimension correct: {embedding.shape}")
```

### Step 5: Verify Semantic Similarity

```python
from sentence_transformers.util import cos_sim

query = "How is the weather?"
sentences = ["The weather is great today", "I want to eat an apple", "Will it rain tomorrow?"]

query_emb = model.encode(query)
sent_embs = model.encode(sentences)

scores = cos_sim(query_emb, sent_embs)
print(f"Similarity scores: {scores.tolist()}")
# Expected: weather-related sentences score significantly higher than unrelated ones
```

### Step 6: Integrate into the Project

```python
from config.config import get_embedding_model, QDRANT_PATH
from qdrant_client import QdrantClient

# Get singleton model
embedding_model = get_embedding_model()

# Connect to vector database
qdrant_client = QdrantClient(path=QDRANT_PATH)

# Initialize components
from memory.memory_store import MemoryStore
from memory.temporal_kg import TemporalKG
from tools.tool_registry import ToolRegistry

memory_store = MemoryStore(db_path, qdrant_client, embedding_model)
temporal_kg = TemporalKG(db_path, qdrant_client, embedding_model)
tool_registry = ToolRegistry(qdrant_client, embedding_model)
```

---

## 4. Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| `OSError: Can't load tokenizer` | Model files not downloaded locally | Run the model load once with internet access, or manually download model files |
| `ConnectionError` | Offline variables set after import | Move `os.environ` settings to the very top of the file |
| Dimension mismatch error | Qdrant collection `vector_size` doesn't match model output | Ensure collection is created with `size=384` |
| Out of memory | Multiple model instantiations | Use singleton pattern `get_embedding_model()` |
| Poor Chinese performance | Using an English-only model | Current `multilingual` model already supports Chinese, no change needed |
| Slow loading | Normal for first load | Use singleton; only the first load is slow, subsequent calls reuse the instance |

---

## 5. Model Replacement Checklist

If you need to switch to a different embedding model, follow these steps:

- [ ] Confirm the new model's vector dimension
- [ ] Update `EMBEDDING_MODEL_NAME` and `EMBEDDING_DIM` in `config/config.py`
- [ ] Delete all collection data under `qdrant_data/` (must rebuild when dimensions change)
- [ ] Re-register all tools (tool_registry needs re-encoding)
- [ ] Clear long-term memory collection (ltm_facts needs re-encoding)
- [ ] Clear knowledge graph collection (kg_nodes needs re-encoding)
- [ ] Run validation scripts to confirm the new model works correctly
- [ ] Re-evaluate similarity thresholds (e.g., `min_score=0.3`, dedup threshold `0.95`)

---

## 6. Usage in the Project

The model serves three core purposes in the project:

```
User Input → encode → Query Vector
                          ↓
               ┌──────────┼──────────┐
               ↓          ↓          ↓
         tool_registry  ltm_facts  kg_nodes
         (Tool Search) (Memory)   (KG Search)
               ↓          ↓          ↓
          cosine sim  cosine sim  cosine sim
               ↓          ↓          ↓
          Top-K Tools  Top-K Facts  Seed Entities
```

All semantic retrieval uses **Cosine Distance**, computed automatically by Qdrant during search.
