"""
AdminController - Control Plane Management Class

Provides hot management capabilities for the Agent runtime, decoupled from the data plane (conversation routing):
- init_from_config: Batch-initialize all MCP Servers from a config dictionary
- register_new_mcp_server: Connect a new MCP Server
- refresh_mcp_server: Full refresh (Upsert ALL + Delete stale)
- reload_prompts: Hot-reload Prompt configuration
"""
import asyncio
from typing import Dict, Any, List, Optional

from smart_router_agent_EN.tools.tool_registry import ToolRegistry, ToolDefinition
from smart_router_agent_EN.tools.mcp_client import MCPClientAdapter
from smart_router_agent_EN.config.prompt_loader import PromptLoader


class AdminController:
    """
    Admin control endpoint (control plane).
    Responsible for receiving external change notifications and performing hot updates
    on ToolRegistry and PromptLoader.
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
        Batch-initialize all MCP Servers from a config dictionary.
        config_dict format:
        {
            "mcpServers": {
                "Server_Name": {"transport": "mock"|"http", "url": "...", ...},
                ...
            }
        }
        """
        servers = config_dict.get("mcpServers", {})
        print(f"  [AdminController] Initializing {len(servers)} MCP Server(s) from config...")
        for server_name, conn_cfg in servers.items():
            await self.register_new_mcp_server(server_name, conn_cfg)
        print(f"  [AdminController] Config initialization complete")

    async def register_new_mcp_server(
        self, server_name: str, connection_config: Dict[str, Any]
    ):
        """
        Connect a new MCP Server.
        1. Register connection to MCP Client
        2. Fetch tools/list
        3. Upsert tool embeddings to Vector DB
        """
        print(f"\n  [AdminController] >>> Registering MCP Server '{server_name}'")
        self._servers_config[server_name] = connection_config
        self._mcp_client.register_server(server_name, connection_config)

        tools_list = await self._mcp_client.fetch_tools_list(server_name)
        print(f"  [AdminController] Fetched {len(tools_list)} tools")

        tool_defs = self._tools_list_to_defs(server_name, tools_list)
        self._registry.register_batch(tool_defs)
        print(f"  [AdminController] <<< '{server_name}' registration complete, {len(tool_defs)} tools ready\n")

    async def refresh_mcp_server(self, server_name: str):
        """
        Full refresh of a Server's tool list:
        1. Re-fetch tools/list
        2. Upsert ALL remote tools (covering schema changes)
        3. Delete local tools that are no longer on the remote (stale)
        """
        print(f"\n  [AdminController] >>> Refreshing MCP Server '{server_name}'")

        tools_list = await self._mcp_client.fetch_tools_list(server_name)
        print(f"  [AdminController] Fetched {len(tools_list)} tools")

        remote_names = {t["name"] for t in tools_list}

        # Upsert ALL (register_batch uses upsert semantics internally)
        tool_defs = self._tools_list_to_defs(server_name, tools_list)
        if tool_defs:
            self._registry.register_batch(tool_defs)
            print(f"  [AdminController] Upsert {len(tool_defs)} tools")

        # Delete stale
        existing_tools = self._registry.get_all_tools_for_server(server_name)
        stale = [t for t in existing_tools if t.name not in remote_names]
        for t in stale:
            self._registry.remove_tool(t.name, server_name)
            print(f"  [AdminController] Deleted stale tool: {t.name}")

        print(f"  [AdminController] <<< '{server_name}' refresh complete: "
              f"upsert {len(tool_defs)}, deleted {len(stale)}\n")

    def reload_prompts(self):
        """Hot-reload Prompt configuration"""
        print(f"  [AdminController] >>> Reloading Prompt configuration")
        self._prompt_loader.reload()
        print(f"  [AdminController] <<< Prompt hot-reload complete")

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
