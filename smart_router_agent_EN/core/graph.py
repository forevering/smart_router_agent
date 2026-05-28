"""
LangGraph Graph Construction Module (MCP + Tool Retrieval Architecture)

Constructs the StateGraph, orchestrating all nodes and edges:

                  ┌─────────────────┐
                  │     START        │
                  └───────┬──────────┘
                         ╱ ╲
              ┌─────────╱   ╲─────────┐
              │ retrieve_ltm │ retrieve_kg │    ← Parallel retrieval of LTM and KG
              └──────┬───────┘──────┬──────┘
                     └──────┬───────┘
                    ┌───────▼───────┐
                    │ query_rewriter│              ← Fuse context and rewrite query
                    └───────┬───────┘
                    ┌───────▼───────┐
                    │ tool_retriever│              ← Vector search for Top-K tools
                    └───────┬───────┘
                    ┌───────▼───────┐
                    │    router     │              ← Dynamic tool binding + LLM decision
                    └───────┬───────┘
                           ╱ ╲
            (has tool_calls) ╱   ╲ (no tool_calls)
                    ┌──────▼──────┐  ┌───────▼───────┐
                    │mcp_executor │  │   generator   │
                    └──────┬──────┘  └───────┬───────┘
                           │                  │
                    ┌──────▼──────┐           │
                    │   router    │← loop back  │
                    │  (cycle edge)│           │
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

Key design decisions:
- router outputs AIMessage, then routes via conditional_edge
- has tool_calls → mcp_executor → unconditionally loops back to router
- no tool_calls → generator → summarizer + kg_manager (parallel) → END
"""
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from smart_router_agent_EN.core.state import AgentState
from smart_router_agent_EN.memory.memory_store import MemoryStore
from smart_router_agent_EN.memory.temporal_kg import TemporalKG
from smart_router_agent_EN.tools.tool_registry import ToolRegistry
from smart_router_agent_EN.tools.mcp_client import MCPClientAdapter
from smart_router_agent_EN.config.prompt_loader import PromptLoader
from smart_router_agent_EN.core.nodes import (
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
    Conditional edge: checks if the last AIMessage output by Router contains tool_calls.
    - has tool_calls → "mcp_executor"
    - no tool_calls → "generator"
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
    Build and compile the LangGraph StateGraph.

    Returns:
        Compiled CompiledGraph with MemorySaver checkpoint
    """
    # ---- Create node functions (dependency injection) ----
    retrieve_ltm_fn = make_retrieve_ltm(memory_store)
    retrieve_kg_fn = make_retrieve_kg(temporal_kg)
    query_rewriter_fn = make_query_rewriter(llm, prompt_loader)
    tool_retriever_fn = make_tool_retriever(tool_registry)
    router_fn = make_router(llm, tool_registry, prompt_loader)
    mcp_executor_fn = make_mcp_executor(tool_registry, mcp_client)
    generator_fn = make_generator(llm, prompt_loader)
    summarizer_fn = make_summarizer(memory_store, llm, prompt_loader)
    kg_manager_fn = make_kg_manager(temporal_kg, llm, prompt_loader)

    # ---- Build StateGraph ----
    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("retrieve_ltm", retrieve_ltm_fn)
    builder.add_node("retrieve_kg", retrieve_kg_fn)
    builder.add_node("query_rewriter", query_rewriter_fn)
    builder.add_node("tool_retriever", tool_retriever_fn)
    builder.add_node("router", router_fn)
    builder.add_node("mcp_executor", mcp_executor_fn)
    builder.add_node("generator", generator_fn)
    builder.add_node("summarizer", summarizer_fn)
    builder.add_node("kg_manager", kg_manager_fn)

    # ---- Orchestrate edges ----

    # 1. Parallel retrieval: START triggers both LTM and KG retrieval simultaneously
    builder.add_edge(START, "retrieve_ltm")
    builder.add_edge(START, "retrieve_kg")

    # 2. Fan-in: query_rewriter executes after both LTM and KG retrieval complete
    builder.add_edge("retrieve_ltm", "query_rewriter")
    builder.add_edge("retrieve_kg", "query_rewriter")

    # 3. query_rewriter → tool_retriever → router
    builder.add_edge("query_rewriter", "tool_retriever")
    builder.add_edge("tool_retriever", "router")

    # 4. Conditional edge: route after router output
    builder.add_conditional_edges(
        "router",
        _route_after_router,
        {"mcp_executor": "mcp_executor", "generator": "generator"},
    )

    # 5. mcp_executor unconditionally loops back to router (supports multi-round serial tool calls)
    builder.add_edge("mcp_executor", "router")

    # 6. Parallel post-processing: generator → summarizer + kg_manager
    builder.add_edge("generator", "summarizer")
    builder.add_edge("generator", "kg_manager")

    # 7. Converge to end
    builder.add_edge("summarizer", END)
    builder.add_edge("kg_manager", END)

    # ---- Compile ----
    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    return graph
