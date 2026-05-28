"""
配置模块
- Azure OpenAI LLM 连接（通过 CNTLM 代理）
- 本地 Embedding 模型 (sentence-transformers)
- Qdrant / SQLite 存储路径
- 全局常量

BASE_DIR 指向 smart_router_agent 包根目录，所有文件路径基于此拼接。
"""
import os

# 必须在任何 HuggingFace 相关库导入前设置离线模式
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import httpx
from langchain_openai import AzureChatOpenAI

# ==================== 路径配置 ====================
# BASE_DIR 指向 smart_router_agent 包根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "agent_data.db")            # SQLite: L1 对话 + KG 关系
QDRANT_PATH = os.path.join(BASE_DIR, "qdrant_data")           # Qdrant 本地存储目录

# ==================== 代理配置 ====================
PROXY_URL = "http://127.0.0.1:3128"

# ==================== Azure OpenAI 配置 ====================
AZURE_ENDPOINT = "https://docparseai.openai.azure.com"
AZURE_DEPLOYMENT = "me-sales-agent"
API_VERSION = "2025-01-01-preview"
API_KEY = "40e765a1ded749b99509a9010b95b783"

# ==================== Embedding 配置 ====================
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384                                            # 该模型输出维度

# ==================== KG 检索配置 ====================
MAX_HOPS = 2              # 图扩展硬上限，代码层面封死
SIMILARITY_THRESHOLD = 0.35  # 语义匹配最低阈值
TOP_K = 5                 # 检索 top-k 个结果


def get_llm(temperature: float = 0, max_tokens: int = 2000) -> AzureChatOpenAI:
    """
    创建 AzureChatOpenAI 实例。
    使用 httpx 配置 CNTLM 代理，与 request_openai 1_test.py 的 requests.proxies 等效。
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


# ==================== Embedding 模型单例 ====================
_embedding_model = None

def get_embedding_model():
    """
    懒加载 SentenceTransformer 模型（单例模式）。
    首次调用时加载模型到内存（约 500MB），后续调用直接复用。
    """
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print(f"  [Embedding] 模型加载完成: {EMBEDDING_MODEL_NAME} (dim={EMBEDDING_DIM})")
    return _embedding_model
