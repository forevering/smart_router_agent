"""
Smart Router Agent — E2E Demo 入口

端到端集成验证：
1. 自动启动 mock_mcp_servers.py (真实 MCP SSE Server, port 8200)
2. 使用 HTTP Transport 配置 → MCPClientFactory → 官方 SSE Client
3. 四阶段演示（冷启动 → 路由 → 热更新 → 新能力即时生效）
4. 无论成功/失败，finally 中优雅 kill 后台 Server 子进程

使用方式：
  python -m smart_router_agent.demo_main
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
# 确保 localhost 连接不走企业代理
os.environ["NO_PROXY"] = os.environ.get("NO_PROXY", "") + ",127.0.0.1,localhost"

import httpx
from langchain_core.messages import HumanMessage
from qdrant_client import QdrantClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smart_router_agent.config.config import get_llm, get_embedding_model, DB_PATH, QDRANT_PATH
from smart_router_agent.memory.memory_store import MemoryStore
from smart_router_agent.memory.temporal_kg import TemporalKG
from smart_router_agent.tools.tool_registry import ToolRegistry
from smart_router_agent.tools.mcp_client import MCPClientFactory
from smart_router_agent.config.prompt_loader import PromptLoader
from smart_router_agent.admin.admin_controller import AdminController
from smart_router_agent.admin.admin_api import start_admin_server_in_background
from smart_router_agent.core.graph import build_graph

# Python 解释器路径（与当前进程一致）
PYTHON = sys.executable
MCP_SERVER_MODULE = "smart_router_agent.tools.mock_mcp_servers"
MCP_SERVER_PORT = 8200
MCP_SERVER_URL_BASE = f"http://127.0.0.1:{MCP_SERVER_PORT}"

# Demo 配置：强制使用 HTTP transport → 真实 SSE
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
    """启动 mock_mcp_servers.py 子进程"""
    print(f"  [Demo] 启动 Mock MCP Server 子进程 (port {MCP_SERVER_PORT})...")
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
    """优雅 kill 子进程"""
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


async def wait_for_server(url: str, timeout: float = 15.0):
    """轮询等待 Server 健康检查通过"""
    deadline = asyncio.get_event_loop().time() + timeout
    async with httpx.AsyncClient() as client:
        while asyncio.get_event_loop().time() < deadline:
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    print(f"  [Demo] Mock MCP Server 已就绪")
                    return
            except (httpx.ConnectError, httpx.RemoteProtocolError):
                pass
            await asyncio.sleep(0.5)
    raise TimeoutError(f"Mock MCP Server 未在 {timeout}s 内就绪")


async def trigger_server_upgrade(server_name: str):
    """调用 Mock Server 的 /admin/upgrade 端点，模拟远端服务升级"""
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{MCP_SERVER_URL_BASE}/admin/upgrade/{server_name}")
        resp.raise_for_status()
        data = resp.json()
        print(f"  [Demo] 远端升级: {data['message']}")


async def main():
    proc = start_mock_server()
    try:
        await _run_demo(proc)
    finally:
        print("\n  [Demo] 关闭 Mock MCP Server 子进程...")
        kill_proc(proc)


async def _run_demo(proc: subprocess.Popen):
    print("=" * 70)
    print("  Smart Router Agent — E2E Demo (Official MCP SDK)")
    print("=" * 70)

    # ================================================================
    # 阶段 1：冷启动
    # ================================================================
    print("\n" + "=" * 70)
    print("  🚀 阶段 1：冷启动 — 初始化所有组件")
    print("=" * 70)

    # 等待 Mock Server 就绪
    await wait_for_server(f"{MCP_SERVER_URL_BASE}/health")

    # 清理旧数据
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"  已清理旧数据库: {DB_PATH}")
    if os.path.exists(QDRANT_PATH):
        shutil.rmtree(QDRANT_PATH, ignore_errors=True)
        print(f"  已清理旧 Qdrant 数据: {QDRANT_PATH}")

    llm = get_llm(temperature=0, max_tokens=2000)
    print("  LLM 就绪")

    print("  加载 Embedding 模型...")
    embedding_model = get_embedding_model()

    qdrant_client = QdrantClient(path=QDRANT_PATH)
    print(f"  Qdrant 就绪 ({QDRANT_PATH})")

    memory_store = MemoryStore(DB_PATH, qdrant_client, embedding_model)
    temporal_kg = TemporalKG(DB_PATH, qdrant_client, embedding_model)
    await memory_store.init_db()
    await temporal_kg.init_db()
    print("  LTM + KG 就绪")

    prompt_loader = PromptLoader()

    # 使用 MCPClientFactory (HTTP transport)
    mcp_client = MCPClientFactory()
    tool_registry = ToolRegistry(qdrant_client, embedding_model)

    admin_ctrl = AdminController(tool_registry, mcp_client, prompt_loader)
    await admin_ctrl.init_from_config(DEMO_CONFIG)

    start_admin_server_in_background(admin_ctrl, port=8100)

    graph = build_graph(
        memory_store, temporal_kg, llm,
        tool_registry, mcp_client, prompt_loader,
    )
    print("  LangGraph 构建完成")

    user_id = "user_001"
    thread_id = "thread_001"
    config = {"configurable": {"thread_id": thread_id}}

    # ================================================================
    # Round 1：建立知识图谱
    # ================================================================
    print("\n" + "=" * 70)
    print("  📍 Round 1：用户告知出差计划 (建立知识图谱)")
    print("=" * 70)

    user_msg_1 = "我下周要去上海出差，大概待三天。"
    print(f"\n  👤 用户: {user_msg_1}\n")

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
    print("  📦 Round 1 后：存储状态")
    print("-" * 40)

    ltm_results = memory_store.retrieve(user_id, "出差上海旅行")
    print(f"\n  长期记忆 ({len(ltm_results)} 条):")
    for item in ltm_results:
        print(f"    • [{item['score']:.3f}] {item['fact']}")

    kg_data = await temporal_kg.semantic_search(user_id, "出差上海旅行")
    print(f"\n  知识图谱 ({len(kg_data.get('subgraph_nodes', []))} 实体, "
          f"{len(kg_data.get('subgraph_relations', []))} 关系):")
    for node in kg_data.get("subgraph_nodes", []):
        print(f"    [实体] {node['entity']} ({node.get('entity_type', '?')})")
    for rel in kg_data.get("subgraph_relations", []):
        print(f"    [关系] {rel['source']} --[{rel['relation_type']}]--> {rel['target']}")

    # ================================================================
    # 阶段 2：正常路由
    # ================================================================
    print("\n" + "=" * 70)
    print("  📍 阶段 2：正常路由 — 天气询问")
    print("=" * 70)

    user_msg_2 = "我下周去上海，天气会影响我的旅行计划吗？"
    print(f"\n  👤 用户: {user_msg_2}\n")

    input_2 = {
        "messages": [HumanMessage(content=user_msg_2)],
        "user_id": user_id,
    }
    async for event in graph.astream(input_2, config, stream_mode="updates"):
        pass

    # ================================================================
    # 阶段 3：热更新（不重启 Agent）
    # ================================================================
    print("\n" + "=" * 70)
    print("  🔧 阶段 3：管理员触发热更新 (Agent + Server 均不重启)")
    print("=" * 70)

    # 3a. 模拟远端 Weather_Server 升级（新增 get_typhoon_warning）
    await trigger_server_upgrade("weather")

    # 3b. 模拟远端 Travel_Server 升级（新增 search_flight）
    await trigger_server_upgrade("travel")

    # 3c. Agent 端刷新，重新拉取 tools/list 发现新工具
    await admin_ctrl.refresh_mcp_server("Weather_Server")
    await admin_ctrl.refresh_mcp_server("Travel_Server")

    # ================================================================
    # 阶段 4：新能力即时生效
    # ================================================================
    print("\n" + "=" * 70)
    print("  📍 阶段 4：新能力即时生效 — 航班 + 台风")
    print("=" * 70)

    user_msg_3 = "帮我查一下去上海的航班，顺便看看有没有台风？"
    print(f"\n  👤 用户: {user_msg_3}\n")

    input_3 = {
        "messages": [HumanMessage(content=user_msg_3)],
        "user_id": user_id,
    }
    async for event in graph.astream(input_3, config, stream_mode="updates"):
        pass

    # 最终存储状态
    print("\n" + "-" * 40)
    print("  📦 最终存储状态")
    print("-" * 40)

    ltm_results = memory_store.retrieve(user_id, "出差上海旅行天气航班台风")
    print(f"\n  长期记忆 ({len(ltm_results)} 条):")
    for item in ltm_results:
        print(f"    • [{item['score']:.3f}] {item['fact']}")

    kg_data = await temporal_kg.semantic_search(user_id, "出差上海旅行天气航班")
    print(f"\n  知识图谱 ({len(kg_data.get('subgraph_nodes', []))} 实体, "
          f"{len(kg_data.get('subgraph_relations', []))} 关系):")
    for node in kg_data.get("subgraph_nodes", []):
        print(f"    [实体] {node['entity']} ({node.get('entity_type', '?')})")
    for rel in kg_data.get("subgraph_relations", []):
        props = json.dumps(rel.get("properties", {}), ensure_ascii=False)
        print(f"    [关系] {rel['source']} --[{rel['relation_type']}]--> {rel['target']}"
              f"{' ' + props if props and props != '{}' else ''}")

    # 优雅关闭连接池
    await mcp_client.close_all()

    print("\n" + "=" * 70)
    print("  ✅ 四阶段 E2E 演示完成")
    print("=" * 70)


if __name__ == "__main__":
    import platform
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
