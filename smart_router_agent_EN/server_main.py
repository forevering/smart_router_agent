"""
Smart Router Agent - Production Service Entry Point

Reads mcp_servers_config.json for initialization without clearing DB/Qdrant.
Starts a single FastAPI service (admin + chat), runs in blocking mode.

Usage:
  python -m smart_router_agent_EN.server_main
  python -m smart_router_agent_EN.server_main --host 0.0.0.0 --port 8200
"""
import os
import sys
import json
import asyncio
import argparse

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import uvicorn
from qdrant_client import QdrantClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smart_router_agent_EN.config.config import get_llm, get_embedding_model, DB_PATH, QDRANT_PATH, BASE_DIR
from smart_router_agent_EN.memory.memory_store import MemoryStore
from smart_router_agent_EN.memory.temporal_kg import TemporalKG
from smart_router_agent_EN.tools.tool_registry import ToolRegistry
from smart_router_agent_EN.tools.mcp_client import MCPClientFactory
from smart_router_agent_EN.config.prompt_loader import PromptLoader
from smart_router_agent_EN.admin.admin_controller import AdminController
from smart_router_agent_EN.admin.admin_api import create_admin_app
from smart_router_agent_EN.core.graph import build_graph


CONFIG_FILE = os.path.join(BASE_DIR, "config", "mcp_servers_config.json")


async def init_app(host: str, port: int):
    print("=" * 70)
    print("  Smart Router Agent - Production Server")
    print("=" * 70)

    llm = get_llm(temperature=0, max_tokens=2000)
    print("  LLM ready")

    print("  Loading Embedding model...")
    embedding_model = get_embedding_model()

    qdrant_client = QdrantClient(path=QDRANT_PATH)
    print(f"  Qdrant ready ({QDRANT_PATH})")

    memory_store = MemoryStore(DB_PATH, qdrant_client, embedding_model)
    temporal_kg = TemporalKG(DB_PATH, qdrant_client, embedding_model)
    await memory_store.init_db()
    await temporal_kg.init_db()

    prompt_loader = PromptLoader()
    mcp_client = MCPClientFactory()
    tool_registry = ToolRegistry(qdrant_client, embedding_model)

    admin_ctrl = AdminController(tool_registry, mcp_client, prompt_loader)

    # Initialize from config file
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    await admin_ctrl.init_from_config(config_dict)

    graph = build_graph(
        memory_store, temporal_kg, llm,
        tool_registry, mcp_client, prompt_loader,
    )
    print("  LangGraph ready")

    app = create_admin_app(admin_ctrl, graph=graph)

    print(f"\n  Starting service: http://{host}:{port}")
    print(f"  Chat API: POST /api/chat")
    print(f"  Admin API: POST /admin/mcp/register, /admin/mcp/refresh, /admin/prompts/reload")
    print(f"  Health check: GET /admin/health")

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


def main():
    parser = argparse.ArgumentParser(description="Smart Router Agent Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8200)
    args = parser.parse_args()

    import platform
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(init_app(args.host, args.port))


if __name__ == "__main__":
    main()
