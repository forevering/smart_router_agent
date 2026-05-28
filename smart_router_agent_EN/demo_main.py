"""
Smart Router Agent — E2E Demo Entry Point

End-to-end integration verification:
1. Automatically starts mock_mcp_servers.py (real MCP SSE Server, port 8200)
2. Uses HTTP Transport configuration → MCPClientFactory → official SSE Client
3. Four-phase demo (cold start → routing → hot update → new capabilities take effect immediately)
4. Gracefully kills the background server subprocess in finally block regardless of success/failure

Usage:
  python -m smart_router_agent_EN.demo_main
"""
import os
import sys
import json
import shutil
import asyncio
import atexit
import subprocess

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
# Ensure localhost connections bypass corporate proxy
os.environ["NO_PROXY"] = os.environ.get("NO_PROXY", "") + ",127.0.0.1,localhost"

import httpx
from langchain_core.messages import HumanMessage
from qdrant_client import QdrantClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smart_router_agent_EN.config.config import get_llm, get_embedding_model, DB_PATH, QDRANT_PATH
from smart_router_agent_EN.memory.memory_store import MemoryStore
from smart_router_agent_EN.memory.temporal_kg import TemporalKG
from smart_router_agent_EN.tools.tool_registry import ToolRegistry
from smart_router_agent_EN.tools.mcp_client import MCPClientFactory
from smart_router_agent_EN.config.prompt_loader import PromptLoader
from smart_router_agent_EN.admin.admin_controller import AdminController
from smart_router_agent_EN.admin.admin_api import start_admin_server_in_background
from smart_router_agent_EN.core.graph import build_graph

# Python interpreter path (consistent with current process)
PYTHON = sys.executable
MCP_SERVER_MODULE = "smart_router_agent_EN.tools.mock_mcp_servers"
MCP_SERVER_PORT = 8200
MCP_SERVER_URL_BASE = f"http://127.0.0.1:{MCP_SERVER_PORT}"

# Demo configuration: force HTTP transport → real SSE
DEMO_CONFIG = {
    "mcpServers": {
        "Weather_Server": {
            "transport": "http",
            "url": f"{MCP_SERVER_URL_BASE}/weather/sse",
        },
        "Travel_Server": {
            "transport": "http",
            "url": f"{MCP_SERVER_URL_BASE}/travel/sse",
        },
    }
}


def start_mock_server() -> subprocess.Popen:
    """Start mock_mcp_servers.py subprocess"""
    print(f"  [Demo] Starting Mock MCP Server subprocess (port {MCP_SERVER_PORT})...")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.Popen(
        [PYTHON, "-m", MCP_SERVER_MODULE, "--port", str(MCP_SERVER_PORT)],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    atexit.register(lambda: kill_proc(proc))
    return proc


def kill_proc(proc: subprocess.Popen):
    """Gracefully kill subprocess"""
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


async def wait_for_server(url: str, timeout: float = 15.0):
    """Poll and wait for server health check to pass"""
    deadline = asyncio.get_event_loop().time() + timeout
    async with httpx.AsyncClient() as client:
        while asyncio.get_event_loop().time() < deadline:
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    print(f"  [Demo] Mock MCP Server is ready")
                    return
            except (httpx.ConnectError, httpx.RemoteProtocolError):
                pass
            await asyncio.sleep(0.5)
    raise TimeoutError(f"Mock MCP Server not ready within {timeout}s")


async def trigger_server_upgrade(server_name: str):
    """Call Mock Server's /admin/upgrade endpoint to simulate remote service upgrade"""
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{MCP_SERVER_URL_BASE}/admin/upgrade/{server_name}")
        resp.raise_for_status()
        data = resp.json()
        print(f"  [Demo] Remote upgrade: {data['message']}")


async def main():
    proc = start_mock_server()
    try:
        await _run_demo(proc)
    finally:
        print("\n  [Demo] Shutting down Mock MCP Server subprocess...")
        kill_proc(proc)


async def _run_demo(proc: subprocess.Popen):
    print("=" * 70)
    print("  Smart Router Agent — E2E Demo (Official MCP SDK)")
    print("=" * 70)

    # ================================================================
    # Phase 1: Cold Start
    # ================================================================
    print("\n" + "=" * 70)
    print("  🚀 Phase 1: Cold Start — Initializing all components")
    print("=" * 70)

    # Wait for Mock Server to be ready
    await wait_for_server(f"{MCP_SERVER_URL_BASE}/health")

    # Clean up old data
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"  Cleaned up old database: {DB_PATH}")
    if os.path.exists(QDRANT_PATH):
        shutil.rmtree(QDRANT_PATH, ignore_errors=True)
        print(f"  Cleaned up old Qdrant data: {QDRANT_PATH}")

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
    print("  LTM + KG ready")

    prompt_loader = PromptLoader()

    # Use MCPClientFactory (HTTP transport)
    mcp_client = MCPClientFactory()
    tool_registry = ToolRegistry(qdrant_client, embedding_model)

    admin_ctrl = AdminController(tool_registry, mcp_client, prompt_loader)
    await admin_ctrl.init_from_config(DEMO_CONFIG)

    start_admin_server_in_background(admin_ctrl, port=8100)

    graph = build_graph(
        memory_store, temporal_kg, llm,
        tool_registry, mcp_client, prompt_loader,
    )
    print("  LangGraph build complete")

    user_id = "user_001"
    thread_id = "thread_001"
    config = {"configurable": {"thread_id": thread_id}}

    # ================================================================
    # Round 1: Build Knowledge Graph
    # ================================================================
    print("\n" + "=" * 70)
    print("  📍 Round 1: User shares business trip plan (building knowledge graph)")
    print("=" * 70)

    user_msg_1 = "I'm going on a business trip to Shanghai next week, staying for about three days."
    print(f"\n  👤 User: {user_msg_1}\n")

    input_1 = {
        "messages": [HumanMessage(content=user_msg_1)],
        "user_id": user_id,
        "long_term_memory": [],
        "kg_context": [],
        "contextualized_query": "",
        "response": "",
    }
    async for event in graph.astream(input_1, config, stream_mode="updates"):
        pass

    print("\n" + "-" * 40)
    print("  📦 After Round 1: Storage state")
    print("-" * 40)

    ltm_results = memory_store.retrieve(user_id, "business trip Shanghai travel")
    print(f"\n  Long-term memory ({len(ltm_results)} items):")
    for item in ltm_results:
        print(f"    • [{item['score']:.3f}] {item['fact']}")

    kg_data = await temporal_kg.semantic_search(user_id, "business trip Shanghai travel")
    print(f"\n  Knowledge graph ({len(kg_data.get('subgraph_nodes', []))} entities, "
          f"{len(kg_data.get('subgraph_relations', []))} relations):")
    for node in kg_data.get("subgraph_nodes", []):
        print(f"    [Entity] {node['entity']} ({node.get('entity_type', '?')})")
    for rel in kg_data.get("subgraph_relations", []):
        print(f"    [Relation] {rel['source']} --[{rel['relation_type']}]--> {rel['target']}")

    # ================================================================
    # Phase 2: Normal Routing
    # ================================================================
    print("\n" + "=" * 70)
    print("  📍 Phase 2: Normal Routing — Weather inquiry")
    print("=" * 70)

    user_msg_2 = "I'm going to Shanghai next week, will the weather affect my travel plans?"
    print(f"\n  👤 User: {user_msg_2}\n")

    input_2 = {
        "messages": [HumanMessage(content=user_msg_2)],
        "user_id": user_id,
    }
    async for event in graph.astream(input_2, config, stream_mode="updates"):
        pass

    # ================================================================
    # Phase 3: Hot Update (without restarting Agent)
    # ================================================================
    print("\n" + "=" * 70)
    print("  🔧 Phase 3: Admin triggers hot update (no restart for Agent or Server)")
    print("=" * 70)

    # 3a. Simulate remote Weather_Server upgrade (add get_typhoon_warning)
    await trigger_server_upgrade("weather")

    # 3b. Simulate remote Travel_Server upgrade (add search_flight)
    await trigger_server_upgrade("travel")

    # 3c. Agent-side refresh, re-fetch tools/list to discover new tools
    await admin_ctrl.refresh_mcp_server("Weather_Server")
    await admin_ctrl.refresh_mcp_server("Travel_Server")

    # ================================================================
    # Phase 4: New Capabilities Take Effect Immediately
    # ================================================================
    print("\n" + "=" * 70)
    print("  📍 Phase 4: New capabilities take effect immediately — Flights + Typhoon")
    print("=" * 70)

    user_msg_3 = "Help me check flights to Shanghai, and also check if there are any typhoons?"
    print(f"\n  👤 User: {user_msg_3}\n")

    input_3 = {
        "messages": [HumanMessage(content=user_msg_3)],
        "user_id": user_id,
    }
    async for event in graph.astream(input_3, config, stream_mode="updates"):
        pass

    # Final storage state
    print("\n" + "-" * 40)
    print("  📦 Final storage state")
    print("-" * 40)

    ltm_results = memory_store.retrieve(user_id, "business trip Shanghai travel weather flight typhoon")
    print(f"\n  Long-term memory ({len(ltm_results)} items):")
    for item in ltm_results:
        print(f"    • [{item['score']:.3f}] {item['fact']}")

    kg_data = await temporal_kg.semantic_search(user_id, "business trip Shanghai travel weather flight")
    print(f"\n  Knowledge graph ({len(kg_data.get('subgraph_nodes', []))} entities, "
          f"{len(kg_data.get('subgraph_relations', []))} relations):")
    for node in kg_data.get("subgraph_nodes", []):
        print(f"    [Entity] {node['entity']} ({node.get('entity_type', '?')})")
    for rel in kg_data.get("subgraph_relations", []):
        props = json.dumps(rel.get("properties", {}), ensure_ascii=False)
        print(f"    [Relation] {rel['source']} --[{rel['relation_type']}]--> {rel['target']}"
              f"{' ' + props if props and props != '{}' else ''}")

    # Gracefully close connection pool
    await mcp_client.close_all()

    print("\n" + "=" * 70)
    print("  ✅ Four-phase E2E demo complete")
    print("=" * 70)


if __name__ == "__main__":
    import platform
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
