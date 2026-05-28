"""
Graph 节点函数模块 (MCP + Tool Retrieval 架构)

包含 LangGraph StateGraph 的所有节点实现：
1. retrieve_ltm       - 检索长期记忆
2. retrieve_kg        - 检索时序知识图谱
3. query_rewriter     - 融合上下文改写用户 query
4. tool_retriever     - 基于改写 query 向量检索 Top-K 工具
5. router             - 动态绑定工具，LLM 决策是否调用
6. mcp_executor       - 通用 MCP 客户端代理，并发执行工具
7. generator          - 整合信息生成最终回复（流式输出）
8. summarizer         - 异步后处理：提取事实更新 LTM
9. kg_manager         - 异步后处理：抽取实体/关系更新 KG

每个节点函数通过工厂函数（make_*）创建，接受外部依赖注入。
"""
import json
import asyncio
from typing import Dict, Any, List

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from smart_router_agent.core.state import AgentState
from smart_router_agent.tools.tool_registry import ToolRegistry
from smart_router_agent.tools.mcp_client import MCPClientAdapter
from smart_router_agent.config.prompt_loader import PromptLoader


def _get_last_query(messages) -> str:
    """从 messages 中提取最近的用户问题"""
    for m in reversed(messages):
        if getattr(m, 'type', '') == 'human':
            return m.content
    return ""


def _format_stm(messages, limit=10) -> str:
    lines = []
    for m in messages[-limit:]:
        role = getattr(m, 'type', 'unknown')
        lines.append(f"{role}: {m.content}")
    return "\n".join(lines) if lines else "(无历史对话)"


def _format_ltm(ltm) -> str:
    return "\n".join(f"- {f}" for f in ltm) if ltm else "(无长期记忆)"


def _format_kg(kg) -> str:
    return "\n".join(kg) if kg else "(知识图谱为空)"


def _parse_json_safe(text: str):
    """安全解析 LLM 返回的 JSON"""
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
# 节点 1: 检索长期记忆 (LTM)
# ================================================================
def make_retrieve_ltm(memory_store):
    async def retrieve_ltm(state: AgentState) -> Dict[str, Any]:
        user_id = state["user_id"]
        query = _get_last_query(state.get("messages", []))
        if not query:
            print(f"  [检索] 无用户问题，跳过 LTM 检索")
            return {"long_term_memory": []}
        results = memory_store.retrieve(user_id, query)
        facts = [item["fact"] for item in results]
        print(f"  [检索] 长期记忆语义检索完成 ({len(facts)} 条相关事实)")
        return {"long_term_memory": facts}
    return retrieve_ltm


# ================================================================
# 节点 2: 检索时序知识图谱 (KG)
# ================================================================
def make_retrieve_kg(temporal_kg):
    async def retrieve_kg(state: AgentState) -> Dict[str, Any]:
        user_id = state["user_id"]
        query = _get_last_query(state.get("messages", []))
        if not query:
            print(f"  [检索] 无用户问题，跳过 KG 检索")
            return {"kg_context": []}
        kg_data = await temporal_kg.semantic_search(user_id, query)
        kg_strings = []
        for seed in kg_data.get("seed_entities", []):
            kg_strings.append(f"[种子实体] {seed['entity']} (相似度: {seed['score']:.3f})")
        for node in kg_data.get("subgraph_nodes", []):
            props_str = json.dumps(node.get("properties", {}), ensure_ascii=False)
            kg_strings.append(
                f"[实体] {node['entity']} (类型: {node.get('entity_type', '?')}"
                f"{', 属性: ' + props_str if props_str and props_str != '{}' else ''})"
            )
        for rel in kg_data.get("subgraph_relations", []):
            props_str = json.dumps(rel.get("properties", {}), ensure_ascii=False)
            kg_strings.append(
                f"[关系] {rel['source']} --[{rel['relation_type']}]--> {rel['target']}"
                f"{' (属性: ' + props_str + ')' if props_str and props_str != '{}' else ''}"
            )
        hops = kg_data.get("hops_used", 0)
        node_count = len(kg_data.get("subgraph_nodes", []))
        rel_count = len(kg_data.get("subgraph_relations", []))
        print(f"  [检索] 知识图谱语义检索完成 ({node_count} 个实体, {rel_count} 条关系, {hops}-hop 扩展)")
        return {"kg_context": kg_strings}
    return retrieve_kg


# ================================================================
# 节点 3: Query Rewriter (改写用户查询)
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
            print(f"  [QueryRewriter] 改写失败: {e}，使用原始 query")
            contextualized = query

        print(f"  [QueryRewriter] 原始: {query}")
        print(f"  [QueryRewriter] 改写: {contextualized}")
        return {"contextualized_query": contextualized}
    return query_rewriter


# ================================================================
# 节点 4: Tool Retriever (向量检索 Top-K 工具)
# ================================================================
def make_tool_retriever(tool_registry: ToolRegistry):
    async def tool_retriever(state: AgentState) -> Dict[str, Any]:
        ctx_query = state.get("contextualized_query", "")
        if not ctx_query:
            ctx_query = _get_last_query(state.get("messages", []))
        if not ctx_query:
            print(f"  [ToolRetriever] 无查询文本，跳过工具检索")
            return {}
        matched_tools = tool_registry.search(ctx_query, top_k=3)
        if matched_tools:
            names = [t.name for t in matched_tools]
            print(f"  [ToolRetriever] 检索到 Top-{len(matched_tools)} 工具: {', '.join(names)}")
        else:
            print(f"  [ToolRetriever] 未检索到相关工具")
        return {}
    return tool_retriever


# ================================================================
# 节点 5: Router (动态绑定工具 + LLM 决策)
# ================================================================
def make_router(llm, tool_registry: ToolRegistry, prompt_loader: PromptLoader):
    async def router(state: AgentState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        ltm = state.get("long_term_memory", [])
        kg = state.get("kg_context", [])
        user_id = state.get("user_id", "unknown")

        query = _get_last_query(messages)
        ctx_query = state.get("contextualized_query", query)

        # 检索 Top-K 工具
        matched_tools = tool_registry.search(ctx_query, top_k=3)

        # 构建工具 schema 列表用于 bind_tools
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
            print(f"  [Router] 动态绑定 {len(lc_tools)} 个工具: {', '.join(tool_names)}")
        else:
            print(f"  [Router] 无可用工具绑定")

        # 构建 Router Prompt
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

        # 构建发送给 LLM 的消息列表
        # 首次进入时只有 prompt；从 mcp_executor 回流时需附加 tool 交互历史
        llm_messages = [HumanMessage(content=prompt_text)]

        # 收集已有的 tool 交互记录 (AIMessage with tool_calls + ToolMessage)
        # 这样 LLM 才能看到之前的工具调用结果，避免重复调用
        for m in messages:
            if isinstance(m, AIMessage) and getattr(m, 'tool_calls', None):
                llm_messages.append(m)
            elif isinstance(m, ToolMessage):
                llm_messages.append(m)

        try:
            response = await llm_with_tools.ainvoke(llm_messages)
        except Exception as e:
            print(f"  [Router] 决策失败: {e}")
            response = AIMessage(content="路由决策失败，将直接生成回复。")

        tool_calls = getattr(response, 'tool_calls', None) or []
        if tool_calls:
            tc_names = [tc["name"] for tc in tool_calls]
            print(f"  [Router] 决策: 调用工具 {', '.join(tc_names)}")
        else:
            print(f"  [Router] 决策: 无需调用工具")

        return {"messages": [response]}
    return router


# ================================================================
# 节点 6: MCP Executor (通用工具执行节点)
# ================================================================
def make_mcp_executor(tool_registry: ToolRegistry, mcp_client: MCPClientAdapter):
    async def mcp_executor(state: AgentState) -> Dict[str, Any]:
        messages = state.get("messages", [])

        # 找最后一条有 tool_calls 的 AIMessage
        last_ai_msg = None
        for m in reversed(messages):
            if isinstance(m, AIMessage) and getattr(m, 'tool_calls', None):
                last_ai_msg = m
                break

        if not last_ai_msg or not last_ai_msg.tool_calls:
            print(f"  [MCPExecutor] 无 tool_calls，跳过")
            return {}

        tool_calls = last_ai_msg.tool_calls
        print(f"  [MCPExecutor] 并发执行 {len(tool_calls)} 个工具调用...")

        async def execute_single(tc: dict) -> ToolMessage:
            tool_name = tc["name"]
            arguments = tc.get("args", {})
            tool_call_id = tc.get("id", "")

            server_name = tool_registry.get_server_for_tool(tool_name)
            if not server_name:
                content = json.dumps({"error": f"未知工具: {tool_name}"}, ensure_ascii=False)
                return ToolMessage(content=content, tool_call_id=tool_call_id, name=tool_name)

            try:
                result = await mcp_client.execute_tool(server_name, tool_name, arguments)
                content = json.dumps(result, ensure_ascii=False)
                print(f"  [MCPExecutor] {tool_name}@{server_name} 调用完成")
            except Exception as e:
                content = f"执行失败，请重新检查参数: {e}"
                print(f"  [MCPExecutor] {tool_name}@{server_name} 调用失败: {e}")

            return ToolMessage(content=content, tool_call_id=tool_call_id, name=tool_name)

        # asyncio.gather 并发执行所有独立工具
        tool_messages = await asyncio.gather(
            *[execute_single(tc) for tc in tool_calls]
        )

        return {"messages": list(tool_messages)}
    return mcp_executor


# ================================================================
# 节点 7: 回复生成器（流式输出）
# ================================================================
def make_generator(llm, prompt_loader: PromptLoader):
    async def generator(state: AgentState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        ltm = state.get("long_term_memory", [])
        kg = state.get("kg_context", [])

        query = _get_last_query(messages)

        # 收集 ToolMessage 内容
        tool_results = {}
        for m in messages:
            if isinstance(m, ToolMessage):
                try:
                    tool_results[m.name] = json.loads(m.content)
                except (json.JSONDecodeError, AttributeError):
                    tool_results[getattr(m, 'name', 'unknown')] = m.content

        tool_text = json.dumps(tool_results, ensure_ascii=False, indent=2) if tool_results else "(无工具调用结果)"

        prompt_text = prompt_loader.format(
            "generator",
            query=query,
            stm_context=_format_stm(messages),
            ltm_context=_format_ltm(ltm),
            kg_context=_format_kg(kg),
            tool_results=tool_text,
        )

        print(f"\n  🤖 助手: ", end="", flush=True)
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
# 节点 8: 摘要器 / 长期记忆更新（异步后处理）
# ================================================================
def make_summarizer(memory_store, llm, prompt_loader: PromptLoader):
    async def summarizer(state: AgentState) -> Dict[str, Any]:
        user_id = state["user_id"]
        messages = state.get("messages", [])

        thread_id = "default"
        msg_ids = await memory_store.store_conversation(user_id, thread_id, messages)
        print(f"  [后台-LTM] L1 存储 {len(msg_ids)} 条对话消息")

        facts = await memory_store.extract_facts(messages, llm, prompt_loader)
        if facts:
            stored = memory_store.store_facts(user_id, facts, source_conversation_ids=msg_ids)
            print(f"  [后台-LTM] L2 提取 {len(facts)} 条事实, 实际存储 {stored} 条 (去重后)")
        else:
            print(f"  [后台-LTM] 未发现新的长期记忆事实")

        return {}
    return summarizer


# ================================================================
# 节点 9: 知识图谱管理器 / KG 更新（异步后处理）
# ================================================================
def make_kg_manager(temporal_kg, llm, prompt_loader: PromptLoader):
    async def kg_manager(state: AgentState) -> Dict[str, Any]:
        user_id = state["user_id"]
        messages = state.get("messages", [])
        await temporal_kg.extract_and_update(user_id, messages, llm, prompt_loader)
        print(f"  [后台-KG] 知识图谱更新流程完成")
        return {}
    return kg_manager
