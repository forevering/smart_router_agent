"""
Configuration Module
- Azure OpenAI LLM connection (via CNTLM proxy)
- Local Embedding model (sentence-transformers)
- Qdrant / SQLite storage paths
- Global constants

BASE_DIR points to the smart_router_agent package root directory; all file paths are constructed relative to it.
"""
import os

# Must set offline mode before importing any HuggingFace-related libraries
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import httpx
from langchain_openai import AzureChatOpenAI

# ==================== Path Configuration ====================
# BASE_DIR points to the smart_router_agent package root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "agent_data.db")            # SQLite: L1 conversations + KG relations
QDRANT_PATH = os.path.join(BASE_DIR, "qdrant_data")           # Qdrant local storage directory

# ==================== Proxy Configuration ====================
PROXY_URL = "http://127.0.0.1:3128"

# ==================== Azure OpenAI Configuration ====================
AZURE_ENDPOINT = "https://docparseai.openai.azure.com"
AZURE_DEPLOYMENT = "me-sales-agent"
API_VERSION = "2025-01-01-preview"
API_KEY = "40e765a1ded749b99509a9010b95b783"

# ==================== Embedding Configuration ====================
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384                                            # Output dimension of this model

# ==================== KG Retrieval Configuration ====================
MAX_HOPS = 2              # Hard cap for graph expansion, enforced at code level
SIMILARITY_THRESHOLD = 0.35  # Minimum threshold for semantic matching
TOP_K = 5                 # Retrieve top-k results


def get_llm(temperature: float = 0, max_tokens: int = 2000) -> AzureChatOpenAI:
    """
    Create an AzureChatOpenAI instance.
    Uses httpx to configure CNTLM proxy, equivalent to requests.proxies in request_openai 1_test.py.
    """
    http_client = httpx.Client(proxy=PROXY_URL)
    async_http_client = httpx.AsyncClient(proxy=PROXY_URL)
    return AzureChatOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        azure_deployment=AZURE_DEPLOYMENT,
        api_version=API_VERSION,
        api_key=API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
        http_client=http_client,
        http_async_client=async_http_client,
    )


# ==================== Embedding Model Singleton ====================
_embedding_model = None

def get_embedding_model():
    """
    Lazily load the SentenceTransformer model (singleton pattern).
    First call loads the model into memory (~500MB); subsequent calls reuse the instance.
    """
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print(f"  [Embedding] Model loaded: {EMBEDDING_MODEL_NAME} (dim={EMBEDDING_DIM})")
    return _embedding_model
