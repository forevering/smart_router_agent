"""
AdminController - 控制面管理类

提供 Agent 运行时的热管理能力，与数据面（对话路由）解耦：
- init_from_config: 从配置字典批量初始化所有 MCP Server
- register_new_mcp_server: 接入全新 MCP Server
- refresh_mcp_server: 全量刷新（Upsert ALL + Delete stale）
- reload_prompts: 热加载 Prompt 配置
"""
import asyncio
from typing import Dict, Any, List, Optional

from smart_router_agent.tools.tool_registry import ToolRegistry, ToolDefinition
from smart_router_agent.tools.mcp_client import MCPClientAdapter
from smart_router_agent.config.prompt_loader import PromptLoader


class AdminController:
    """
    管理控制端（控制面）。
    负责接收外部的变动通知，对 ToolRegistry 和 PromptLoader 进行热更新。
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        mcp_client: MCPClientAdapter,
        prompt_loader: PromptLoader,
    ):
        self._registry = tool_registry
        self._mcp_client = mcp_client
        self._prompt_loader = prompt_loader
        self._servers_config: Dict[str, Dict[str, Any]] = {}

    async def init_from_config(self, config_dict: Dict[str, Any]):
        """
        从配置字典批量初始化所有 MCP Server。
        config_dict 格式:
        {
            "mcpServers": {
                "Server_Name": {"transport": "mock"|"http", "url": "...", ...},
                ...
            }
        }
        """
        servers = config_dict.get("mcpServers", {})
        print(f"  [AdminController] 从配置初始化 {len(servers)} 个 MCP Server...")
        for server_name, conn_cfg in servers.items():
            await self.register_new_mcp_server(server_name, conn_cfg)
        print(f"  [AdminController] 配置初始化完成")

    async def register_new_mcp_server(
        self, server_name: str, connection_config: Dict[str, Any]
    ):
        """
        接入全新 MCP Server。
        1. 注册连接到 MCP Client
        2. 拉取 tools/list
        3. Upsert 工具 Embedding 到 Vector DB
        """
        print(f"\n  [AdminController] >>> 注册 MCP Server '{server_name}'")
        self._servers_config[server_name] = connection_config
        self._mcp_client.register_server(server_name, connection_config)

        tools_list = await self._mcp_client.fetch_tools_list(server_name)
        print(f"  [AdminController] 拉取到 {len(tools_list)} 个工具")

        tool_defs = self._tools_list_to_defs(server_name, tools_list)
        self._registry.register_batch(tool_defs)
        print(f"  [AdminController] <<< '{server_name}' 注册完成，{len(tool_defs)} 个工具已就绪\n")

    async def refresh_mcp_server(self, server_name: str):
        """
        全量刷新 Server 的工具列表：
        1. 重新拉取 tools/list
        2. Upsert ALL 远端工具（覆盖 schema 变更）
        3. Delete 本地有但远端已移除的 stale 工具
        """
        print(f"\n  [AdminController] >>> 刷新 MCP Server '{server_name}'")

        tools_list = await self._mcp_client.fetch_tools_list(server_name)
        print(f"  [AdminController] 拉取到 {len(tools_list)} 个工具")

        remote_names = {t["name"] for t in tools_list}

        # Upsert ALL（register_batch 内部是 upsert 语义）
        tool_defs = self._tools_list_to_defs(server_name, tools_list)
        if tool_defs:
            self._registry.register_batch(tool_defs)
            print(f"  [AdminController] Upsert {len(tool_defs)} 个工具")

        # Delete stale
        existing_tools = self._registry.get_all_tools_for_server(server_name)
        stale = [t for t in existing_tools if t.name not in remote_names]
        for t in stale:
            self._registry.remove_tool(t.name, server_name)
            print(f"  [AdminController] 删除 stale 工具: {t.name}")

        print(f"  [AdminController] <<< '{server_name}' 刷新完成: "
              f"upsert {len(tool_defs)}, 删除 {len(stale)}\n")

    def reload_prompts(self):
        """热加载 Prompt 配置"""
        print(f"  [AdminController] >>> 重新加载 Prompt 配置")
        self._prompt_loader.reload()
        print(f"  [AdminController] <<< Prompt 热加载完成")

    @property
    def servers_config(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._servers_config)

    @staticmethod
    def _tools_list_to_defs(server_name: str, tools_list: List[Dict[str, Any]]) -> List[ToolDefinition]:
        return [
            ToolDefinition(
                name=t["name"],
                mcp_server_name=server_name,
                schema=t.get("inputSchema", {}),
                search_document=t.get("search_document", t.get("description", "")),
            )
            for t in tools_list
        ]
