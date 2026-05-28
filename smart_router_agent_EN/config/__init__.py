"""config subpackage — Configuration, prompt loading, external config files"""
from smart_router_agent_EN.config.config import (
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
from smart_router_agent_EN.config.prompt_loader import PromptLoader
