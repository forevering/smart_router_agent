"""
LangGraph 图构建模块 (MCP + Tool Retrieval 架构)

构建 StateGraph，编排所有节点和边：

                  ┌─────────────────┐
                  │     START        │
                  └───────┬──────────┘
                         ╱ ╲
              ┌─────────╱   ╲─────────┐
              │ retrieve_ltm │ retrieve_kg │    ← 并行检索 LTM 和 KG
              └──────┬───────┘──────┬──────┘
                     └──────┬───────┘
                    ┌───────▼───────┐
                    │ query_rewriter│              ← 融合上下文改写 query
                    └───────┬───────┘
                    ┌───────▼───────┐
                    │ tool_retriever│              ← 向量检索 Top-K 工具
                    └───────┬───────┘
                    ┌───────▼───────┐
                    │    router     │              ← 动态绑定工具 + LLM 决策
                    └───────┬───────┘
                           ╱ ╲
            (有 tool_calls) ╱   ╲ (无 tool_calls)
                    ┌──────▼──────┐  ┌───────▼───────┐
                    │mcp_executor │  │   generator   │
                    └──────┬──────┘  └───────┬───────┘
                           │                  │
                    ┌──────▼──────┐           │
                    │   router    │← 回流     │
                    │  (循环边)    │           │
                    └─────────────┘           │
                                     ┌───────▼───────┐
                                     │   generator   │
                                     └───────┬───────┘
                                            ╱ ╲
                                 ┌─────────╱   ╲──────────┐
                                 │ summarizer  │ kg_manager │
                                 └──────┬──────┘──────┬─────┘
                                        └──────┬──────┘
                                       ┌───────▼───────┐
                                       │      END      │
                                       └───────────────┘

关键设计：
- router 输出 AIMessage 后通过 conditional_edge 分流
- 有 tool_calls → mcp_executor → 无条件回流 router
- 无 tool_calls → generator → summarizer + kg_manager (并行) → END
"""
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from smart_router_agent.core.state import AgentState
from smart_router_agent.memory.memory_store import MemoryStore
from smart_router_agent.memory.temporal_kg import TemporalKG
from smart_router_agent.tools.tool_registry import ToolRegistry
from smart_router_agent.tools.mcp_client import MCPClientAdapter
from smart_router_agent.config.prompt_loader import PromptLoader
from smart_router_agent.core.nodes import (
    make_retrieve_ltm,
    make_retrieve_kg,
    make_query_rewriter,
    make_tool_retriever,
    make_router,
    make_mcp_executor,
    make_generator,
    make_summarizer,
    make_kg_manager,
)


def _route_after_router(state: AgentState) -> str:
    """
    条件边：检查 Router 输出的最后一条 AIMessage 是否包含 tool_calls。
    - 有 tool_calls → "mcp_executor"
    - 无 tool_calls → "generator"
    """
    messages = state.get("messages", [])
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            if getattr(m, 'tool_calls', None):
                return "mcp_executor"
            break
    return "generator"


def build_graph(
    memory_store: MemoryStore,
    temporal_kg: TemporalKG,
    llm,
    tool_registry: ToolRegistry,
    mcp_client: MCPClientAdapter,
    prompt_loader: PromptLoader,
):
    """
    构建并编译 LangGraph StateGraph。

    Returns:
        编译后的 CompiledGraph，附带 MemorySaver checkpoint
    """
    # ---- 创建各节点函数（依赖注入） ----
    retrieve_ltm_fn = make_retrieve_ltm(memory_store)
    retrieve_kg_fn = make_retrieve_kg(temporal_kg)
    query_rewriter_fn = make_query_rewriter(llm, prompt_loader)
    tool_retriever_fn = make_tool_retriever(tool_registry)
    router_fn = make_router(llm, tool_registry, prompt_loader)
    mcp_executor_fn = make_mcp_executor(tool_registry, mcp_client)
    generator_fn = make_generator(llm, prompt_loader)
    summarizer_fn = make_summarizer(memory_store, llm, prompt_loader)
    kg_manager_fn = make_kg_manager(temporal_kg, llm, prompt_loader)

    # ---- 构建 StateGraph ----
    builder = StateGraph(AgentState)

    # 注册节点
    builder.add_node("retrieve_ltm", retrieve_ltm_fn)
    builder.add_node("retrieve_kg", retrieve_kg_fn)
    builder.add_node("query_rewriter", query_rewriter_fn)
    builder.add_node("tool_retriever", tool_retriever_fn)
    builder.add_node("router", router_fn)
    builder.add_node("mcp_executor", mcp_executor_fn)
    builder.add_node("generator", generator_fn)
    builder.add_node("summarizer", summarizer_fn)
    builder.add_node("kg_manager", kg_manager_fn)

    # ---- 编排边 ----

    # 1. 并行检索：START 同时触发 LTM 和 KG 检索
    builder.add_edge(START, "retrieve_ltm")
    builder.add_edge(START, "retrieve_kg")

    # 2. Fan-in：query_rewriter 等待 LTM 和 KG 都完成后执行
    builder.add_edge("retrieve_ltm", "query_rewriter")
    builder.add_edge("retrieve_kg", "query_rewriter")

    # 3. query_rewriter → tool_retriever → router
    builder.add_edge("query_rewriter", "tool_retriever")
    builder.add_edge("tool_retriever", "router")

    # 4. 条件边：router 输出后分流
    builder.add_conditional_edges(
        "router",
        _route_after_router,
        {"mcp_executor": "mcp_executor", "generator": "generator"},
    )

    # 5. mcp_executor 无条件回流到 router（支持多轮串行工具调用）
    builder.add_edge("mcp_executor", "router")

    # 6. 并行后处理：generator → summarizer + kg_manager
    builder.add_edge("generator", "summarizer")
    builder.add_edge("generator", "kg_manager")

    # 7. 汇入终点
    builder.add_edge("summarizer", END)
    builder.add_edge("kg_manager", END)

    # ---- 编译 ----
    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    return graph
