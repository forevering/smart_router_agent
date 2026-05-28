"""config 子包 — 配置、Prompt 加载、外部配置文件"""
from smart_router_agent.config.config import (
    BASE_DIR,
    DB_PATH,
    QDRANT_PATH,
    PROXY_URL,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIM,
    MAX_HOPS,
    SIMILARITY_THRESHOLD,
    TOP_K,
    get_llm,
    get_embedding_model,
)
from smart_router_agent.config.prompt_loader import PromptLoader
