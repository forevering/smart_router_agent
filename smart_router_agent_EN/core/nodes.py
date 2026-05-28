"""
Graph Node Functions Module (MCP + Tool Retrieval Architecture)

Contains all node implementations for the LangGraph StateGraph:
1. retrieve_ltm       - Retrieve long-term memory
2. retrieve_kg        - Retrieve temporal knowledge graph
3. query_rewriter     - Rewrite user query by fusing context
4. tool_retriever     - Vector search for Top-K tools based on rewritten query
5. router             - Dynamically bind tools, LLM decides whether to invoke
6. mcp_executor       - Generic MCP client proxy, concurrent tool execution
7. generator          - Synthesize information and generate final response (streaming)
8. summarizer         - Async post-processing: extract facts and update LTM
9. kg_manager         - Async post-processing: extract entities/relations and update KG

Each node function is created via a factory function (make_*) that accepts external dependency injection.
"""
import json
import asyncio
from typing import Dict, Any, List

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from smart_router_agent_EN.core.state import AgentState
from smart_router_agent_EN.tools.tool_registry import ToolRegistry
from smart_router_agent_EN.tools.mcp_client import MCPClientAdapter
from smart_router_agent_EN.config.prompt_loader import PromptLoader


def _get_last_query(messages) -> str:
    """Extract the most recent user question from messages"""
    for m in reversed(messages):
        if getattr(m, 'type', '') == 'human':
            return m.content
    return ""


def _format_stm(messages, limit=10) -> str:
    lines = []
    for m in messages[-limit:]:
        role = getattr(m, 'type', 'unknown')
        lines.append(f"{role}: {m.content}")
    return "\n".join(lines) if lines else "(no conversation history)"


def _format_ltm(ltm) -> str:
    return "\n".join(f"- {f}" for f in ltm) if ltm else "(no long-term memory)"


def _format_kg(kg) -> str:
    return "\n".join(kg) if kg else "(knowledge graph is empty)"


def _parse_json_safe(text: str):
    """Safely parse JSON returned by LLM"""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        end_idx = len(lines) - 1
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip() == "```":
                end_idx = i
                break
        text = "\n".join(lines[1:end_idx]).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            start = text.find(start_char)
            end = text.rfind(end_char)
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    continue
        return None


# ================================================================
# Node 1: Retrieve Long-Term Memory (LTM)
# ================================================================
def make_retrieve_ltm(memory_store):
    async def retrieve_ltm(state: AgentState) -> Dict[str, Any]:
        user_id = state["user_id"]
        query = _get_last_query(state.get("messages", []))
        if not query:
            print(f"  [Retrieve] No user query, skipping LTM retrieval")
            return {"long_term_memory": []}
        results = memory_store.retrieve(user_id, query)
        facts = [item["fact"] for item in results]
        print(f"  [Retrieve] LTM semantic search complete ({len(facts)} relevant facts)")
        return {"long_term_memory": facts}
    return retrieve_ltm


# ================================================================
# Node 2: Retrieve Temporal Knowledge Graph (KG)
# ================================================================
def make_retrieve_kg(temporal_kg):
    async def retrieve_kg(state: AgentState) -> Dict[str, Any]:
        user_id = state["user_id"]
        query = _get_last_query(state.get("messages", []))
        if not query:
            print(f"  [Retrieve] No user query, skipping KG retrieval")
            return {"kg_context": []}
        kg_data = await temporal_kg.semantic_search(user_id, query)
        kg_strings = []
        for seed in kg_data.get("seed_entities", []):
            kg_strings.append(f"[Seed Entity] {seed['entity']} (similarity: {seed['score']:.3f})")
        for node in kg_data.get("subgraph_nodes", []):
            props_str = json.dumps(node.get("properties", {}), ensure_ascii=False)
            kg_strings.append(
                f"[Entity] {node['entity']} (type: {node.get('entity_type', '?')}"
                f"{', properties: ' + props_str if props_str and props_str != '{}' else ''})"
            )
        for rel in kg_data.get("subgraph_relations", []):
            props_str = json.dumps(rel.get("properties", {}), ensure_ascii=False)
            kg_strings.append(
                f"[Relation] {rel['source']} --[{rel['relation_type']}]--> {rel['target']}"
                f"{' (properties: ' + props_str + ')' if props_str and props_str != '{}' else ''}"
            )
        hops = kg_data.get("hops_used", 0)
        node_count = len(kg_data.get("subgraph_nodes", []))
        rel_count = len(kg_data.get("subgraph_relations", []))
        print(f"  [Retrieve] KG semantic search complete ({node_count} entities, {rel_count} relations, {hops}-hop expansion)")
        return {"kg_context": kg_strings}
    return retrieve_kg


# ================================================================
# Node 3: Query Rewriter (rewrite user query)
# ================================================================
def make_query_rewriter(llm, prompt_loader: PromptLoader):
    async def query_rewriter(state: AgentState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        ltm = state.get("long_term_memory", [])
        kg = state.get("kg_context", [])
        query = _get_last_query(messages)

        if not query:
            return {"contextualized_query": ""}

        prompt_text = prompt_loader.format(
            "query_rewriter",
            query=query,
            ltm_context=_format_ltm(ltm),
            kg_context=_format_kg(kg),
            stm_context=_format_stm(messages),
        )

        try:
            response = await llm.ainvoke(
                [HumanMessage(content=prompt_text)],
            )
            contextualized = response.content.strip()
        except Exception as e:
            print(f"  [QueryRewriter] Rewrite failed: {e}, using original query")
            contextualized = query

        print(f"  [QueryRewriter] Original: {query}")
        print(f"  [QueryRewriter] Rewritten: {contextualized}")
        return {"contextualized_query": contextualized}
    return query_rewriter


# ================================================================
# Node 4: Tool Retriever (vector search for Top-K tools)
# ================================================================
def make_tool_retriever(tool_registry: ToolRegistry):
    async def tool_retriever(state: AgentState) -> Dict[str, Any]:
        ctx_query = state.get("contextualized_query", "")
        if not ctx_query:
            ctx_query = _get_last_query(state.get("messages", []))
        if not ctx_query:
            print(f"  [ToolRetriever] No query text, skipping tool retrieval")
            return {}
        matched_tools = tool_registry.search(ctx_query, top_k=3)
        if matched_tools:
            names = [t.name for t in matched_tools]
            print(f"  [ToolRetriever] Retrieved Top-{len(matched_tools)} tools: {', '.join(names)}")
        else:
            print(f"  [ToolRetriever] No relevant tools found")
        return {}
    return tool_retriever


# ================================================================
# Node 5: Router (dynamic tool binding + LLM decision)
# ================================================================
def make_router(llm, tool_registry: ToolRegistry, prompt_loader: PromptLoader):
    async def router(state: AgentState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        ltm = state.get("long_term_memory", [])
        kg = state.get("kg_context", [])
        user_id = state.get("user_id", "unknown")

        query = _get_last_query(messages)
        ctx_query = state.get("contextualized_query", query)

        # Retrieve Top-K tools
        matched_tools = tool_registry.search(ctx_query, top_k=3)

        # Build tool schema list for bind_tools
        lc_tools = []
        for t in matched_tools:
            schema = t.schema
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            func_schema = {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.search_document,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }
            lc_tools.append(func_schema)

        if lc_tools:
            tool_names = [t["function"]["name"] for t in lc_tools]
            print(f"  [Router] Dynamically bound {len(lc_tools)} tools: {', '.join(tool_names)}")
        else:
            print(f"  [Router] No tools available to bind")

        # Build Router Prompt
        prompt_text = prompt_loader.format(
            "router",
            query=query,
            stm_context=_format_stm(messages),
            ltm_context=_format_ltm(ltm),
            kg_context=_format_kg(kg),
            user_id=user_id,
        )

        # bind_tools
        if lc_tools:
            llm_with_tools = llm.bind_tools(lc_tools)
        else:
            llm_with_tools = llm

        # Build message list to send to LLM
        # On first entry, only the prompt; when looping back from mcp_executor,
        # include tool interaction history
        llm_messages = [HumanMessage(content=prompt_text)]

        # Collect existing tool interaction records (AIMessage with tool_calls + ToolMessage)
        # This allows LLM to see previous tool call results, avoiding duplicate calls
        for m in messages:
            if isinstance(m, AIMessage) and getattr(m, 'tool_calls', None):
                llm_messages.append(m)
            elif isinstance(m, ToolMessage):
                llm_messages.append(m)

        try:
            response = await llm_with_tools.ainvoke(llm_messages)
        except Exception as e:
            print(f"  [Router] Decision failed: {e}")
            response = AIMessage(content="Routing decision failed, generating response directly.")

        tool_calls = getattr(response, 'tool_calls', None) or []
        if tool_calls:
            tc_names = [tc["name"] for tc in tool_calls]
            print(f"  [Router] Decision: calling tools {', '.join(tc_names)}")
        else:
            print(f"  [Router] Decision: no tool calls needed")

        return {"messages": [response]}
    return router


# ================================================================
# Node 6: MCP Executor (generic tool execution node)
# ================================================================
def make_mcp_executor(tool_registry: ToolRegistry, mcp_client: MCPClientAdapter):
    async def mcp_executor(state: AgentState) -> Dict[str, Any]:
        messages = state.get("messages", [])

        # Find the last AIMessage with tool_calls
        last_ai_msg = None
        for m in reversed(messages):
            if isinstance(m, AIMessage) and getattr(m, 'tool_calls', None):
                last_ai_msg = m
                break

        if not last_ai_msg or not last_ai_msg.tool_calls:
            print(f"  [MCPExecutor] No tool_calls, skipping")
            return {}

        tool_calls = last_ai_msg.tool_calls
        print(f"  [MCPExecutor] Executing {len(tool_calls)} tool calls concurrently...")

        async def execute_single(tc: dict) -> ToolMessage:
            tool_name = tc["name"]
            arguments = tc.get("args", {})
            tool_call_id = tc.get("id", "")

            server_name = tool_registry.get_server_for_tool(tool_name)
            if not server_name:
                content = json.dumps({"error": f"Unknown tool: {tool_name}"}, ensure_ascii=False)
                return ToolMessage(content=content, tool_call_id=tool_call_id, name=tool_name)

            try:
                result = await mcp_client.execute_tool(server_name, tool_name, arguments)
                content = json.dumps(result, ensure_ascii=False)
                print(f"  [MCPExecutor] {tool_name}@{server_name} call complete")
            except Exception as e:
                content = f"Execution failed, please verify parameters: {e}"
                print(f"  [MCPExecutor] {tool_name}@{server_name} call failed: {e}")

            return ToolMessage(content=content, tool_call_id=tool_call_id, name=tool_name)

        # asyncio.gather executes all independent tools concurrently
        tool_messages = await asyncio.gather(
            *[execute_single(tc) for tc in tool_calls]
        )

        return {"messages": list(tool_messages)}
    return mcp_executor


# ================================================================
# Node 7: Response Generator (streaming output)
# ================================================================
def make_generator(llm, prompt_loader: PromptLoader):
    async def generator(state: AgentState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        ltm = state.get("long_term_memory", [])
        kg = state.get("kg_context", [])

        query = _get_last_query(messages)

        # Collect ToolMessage content
        tool_results = {}
        for m in messages:
            if isinstance(m, ToolMessage):
                try:
                    tool_results[m.name] = json.loads(m.content)
                except (json.JSONDecodeError, AttributeError):
                    tool_results[getattr(m, 'name', 'unknown')] = m.content

        tool_text = json.dumps(tool_results, ensure_ascii=False, indent=2) if tool_results else "(no tool call results)"

        prompt_text = prompt_loader.format(
            "generator",
            query=query,
            stm_context=_format_stm(messages),
            ltm_context=_format_ltm(ltm),
            kg_context=_format_kg(kg),
            tool_results=tool_text,
        )

        print(f"\n  🤖 Assistant: ", end="", flush=True)
        full_response = ""
        async for chunk in llm.astream([HumanMessage(content=prompt_text)]):
            token = chunk.content
            if token:
                print(token, end="", flush=True)
                full_response += token
        print()

        return {
            "messages": [AIMessage(content=full_response)],
            "response": full_response,
        }
    return generator


# ================================================================
# Node 8: Summarizer / Long-Term Memory Update (async post-processing)
# ================================================================
def make_summarizer(memory_store, llm, prompt_loader: PromptLoader):
    async def summarizer(state: AgentState) -> Dict[str, Any]:
        user_id = state["user_id"]
        messages = state.get("messages", [])

        thread_id = "default"
        msg_ids = await memory_store.store_conversation(user_id, thread_id, messages)
        print(f"  [Background-LTM] L1 stored {len(msg_ids)} conversation messages")

        facts = await memory_store.extract_facts(messages, llm, prompt_loader)
        if facts:
            stored = memory_store.store_facts(user_id, facts, source_conversation_ids=msg_ids)
            print(f"  [Background-LTM] L2 extracted {len(facts)} facts, actually stored {stored} (after dedup)")
        else:
            print(f"  [Background-LTM] No new long-term memory facts found")

        return {}
    return summarizer


# ================================================================
# Node 9: Knowledge Graph Manager / KG Update (async post-processing)
# ================================================================
def make_kg_manager(temporal_kg, llm, prompt_loader: PromptLoader):
    async def kg_manager(state: AgentState) -> Dict[str, Any]:
        user_id = state["user_id"]
        messages = state.get("messages", [])
        await temporal_kg.extract_and_update(user_id, messages, llm, prompt_loader)
        print(f"  [Background-KG] Knowledge graph update process complete")
        return {}
    return kg_manager
