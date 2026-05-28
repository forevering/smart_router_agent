"""
MCP Client Adapter Layer — Pure Official Protocol Network Communication Middleware

Implemented based on the official MCP SDK (mcp.client.sse / mcp.client.session).
Contains no hardcoded business data; only handles MCP protocol communication and connection pool management.

Includes:
- MCPClientAdapter: Abstract base class (register_server / fetch_tools_list / execute_tool)
- HttpMCPClient: Production-grade implementation based on official SSE Client
    - Lazy Init: register_server only stores config; connection established on first call
    - Idle TTL: Background coroutine periodically scans; idle connections are actively closed
    - Anti-avalanche Retry (Jitter Retry): Random sleep before reconnection after disconnect
- MCPClientFactory: Transport router (http/sse → HttpMCPClient, stdio → NotImplementedError)
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from abc import ABC, abstractmethod
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

from mcp.client.session import ClientSession
from mcp.client.sse import sse_client

# Ensure localhost connections bypass corporate proxy
import os as _os
if "127.0.0.1" not in _os.environ.get("NO_PROXY", ""):
    _os.environ["NO_PROXY"] = _os.environ.get("NO_PROXY", "") + ",127.0.0.1,localhost"

logger = logging.getLogger(__name__)


# ================================================================
# Abstract Base Class
# ================================================================

class MCPClientAdapter(ABC):
    """MCP client unified interface base class; upper-layer code depends only on this abstraction."""

    @abstractmethod
    def register_server(self, server_name: str, connection_config: Dict[str, Any]):
        """Register MCP Server connection config (does not establish connection)"""
        ...

    @abstractmethod
    async def fetch_tools_list(self, server_name: str) -> List[Dict[str, Any]]:
        """
        Fetch tools/list from MCP Server.
        Returns: Native Python Dict list, each element contains:
            - name: str
            - description: str
            - inputSchema: dict (JSON Schema)
        """
        ...

    @abstractmethod
    async def execute_tool(
        self, server_name: str, tool_name: str, arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Send tools/call RPC request to MCP Server"""
        ...


# ================================================================
# Connection Pool Entry
# ================================================================

class _SessionEntry:
    """A single session entry in the connection pool"""
    __slots__ = ("session", "exit_stack", "last_active")

    def __init__(self, session: ClientSession, exit_stack: AsyncExitStack):
        self.session = session
        self.exit_stack = exit_stack
        self.last_active: float = time.monotonic()

    def touch(self):
        self.last_active = time.monotonic()


# ================================================================
# Official SSE Implementation
# ================================================================

class HttpMCPClient(MCPClientAdapter):
    """
    Production-grade SSE Client implementation based on the official MCP SDK.
    Uses mcp.client.sse.sse_client to establish SSE long connections,
    and mcp.client.session.ClientSession for JSON-RPC handshake and calls.

    Connection pool management strategy (production-grade / K8s-friendly):
    - Lazy Init: register_server only saves config; connection established on first RPC
    - Idle TTL: Background periodic scan; connections idle beyond IDLE_TTL are actively closed
    - Anti-avalanche: Random jitter sleep before retry on disconnect
    """

    IDLE_TTL: float = 600.0       # Reclaim after 10 minutes idle
    CLEANUP_INTERVAL: float = 60.0  # Scan every 60 seconds

    def __init__(self):
        self._connections: Dict[str, Dict[str, Any]] = {}   # server_name → config
        self._sessions: Dict[str, _SessionEntry] = {}       # server_name → session
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None

    # ---- Register (store config only) ----

    def register_server(self, server_name: str, connection_config: Dict[str, Any]):
        self._connections[server_name] = connection_config

    # ---- Lazy Init: establish connection on demand ----

    async def _ensure_session(self, server_name: str) -> ClientSession:
        async with self._lock:
            entry = self._sessions.get(server_name)
            if entry is not None:
                entry.touch()
                return entry.session

            config = self._connections.get(server_name)
            if not config:
                raise ValueError(f"Server '{server_name}' not registered")

            url = config["url"]
            headers = config.get("headers")
            logger.info(f"[MCP] Lazy-init SSE → {server_name} ({url})")

            stack = AsyncExitStack()
            try:
                streams = await stack.enter_async_context(
                    sse_client(url, headers=headers, timeout=10, sse_read_timeout=600)
                )
                read_stream, write_stream = streams
                session = await stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
                )
                await session.initialize()
            except BaseException:
                await stack.aclose()
                raise

            entry = _SessionEntry(session, stack)
            self._sessions[server_name] = entry

            # Start background cleanup (on first connection)
            self._ensure_cleanup_running()

            logger.info(f"[MCP] Session ready: {server_name}")
            return session

    # ---- Idle TTL Reclamation ----

    def _ensure_cleanup_running(self):
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._idle_cleanup_loop())

    async def _idle_cleanup_loop(self):
        while True:
            await asyncio.sleep(self.CLEANUP_INTERVAL)
            now = time.monotonic()
            async with self._lock:
                stale = [
                    name for name, entry in self._sessions.items()
                    if now - entry.last_active > self.IDLE_TTL
                ]
                for name in stale:
                    logger.info(f"[MCP] Idle TTL exceeded, closing: {name}")
                    try:
                        await self._sessions[name].exit_stack.aclose()
                    except Exception:
                        pass
                    del self._sessions[name]

    # ---- Anti-avalanche Retry ----

    async def _with_retry(self, server_name: str, operation):
        """Execute operation; if connection is broken, retry once with jitter"""
        for attempt in range(2):
            try:
                session = await self._ensure_session(server_name)
                return await operation(session)
            except Exception as exc:
                if attempt == 0:
                    logger.warning(
                        f"[MCP] {server_name} call failed ({exc}), "
                        f"evicting session and retrying with jitter..."
                    )
                    async with self._lock:
                        entry = self._sessions.pop(server_name, None)
                        if entry:
                            try:
                                await entry.exit_stack.aclose()
                            except Exception:
                                pass
                    jitter = random.uniform(0.1, 0.5)
                    await asyncio.sleep(jitter)
                else:
                    raise

    # ---- Public Interface ----

    async def fetch_tools_list(self, server_name: str) -> List[Dict[str, Any]]:
        async def _op(session: ClientSession):
            result = await session.list_tools()
            return self._map_tools(result)
        return await self._with_retry(server_name, _op)

    async def execute_tool(
        self, server_name: str, tool_name: str, arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        async def _op(session: ClientSession):
            result = await session.call_tool(tool_name, arguments)
            return self._map_call_result(result)
        return await self._with_retry(server_name, _op)

    # ---- Data Format Stripping (SDK strong types → native Dict) ----

    @staticmethod
    def _map_tools(result) -> List[Dict[str, Any]]:
        """Convert ListToolsResult to native Dict list"""
        tools = []
        for tool in result.tools:
            tools.append({
                "name": tool.name,
                "description": tool.description or "",
                "inputSchema": tool.inputSchema if tool.inputSchema else {},
            })
        return tools

    @staticmethod
    def _map_call_result(result) -> Dict[str, Any]:
        """Convert CallToolResult to native Dict"""
        texts = []
        for item in result.content:
            if hasattr(item, "text"):
                texts.append(item.text)
        combined = "\n".join(texts)
        try:
            return json.loads(combined)
        except (json.JSONDecodeError, TypeError):
            return {"result": combined}

    # ---- Graceful Shutdown ----

    async def close_all(self):
        """Close all active connections"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
        async with self._lock:
            for name, entry in list(self._sessions.items()):
                try:
                    await entry.exit_stack.aclose()
                except Exception:
                    pass
            self._sessions.clear()


# ================================================================
# Transport Router (Factory)
# ================================================================

class MCPClientFactory(MCPClientAdapter):
    """
    Routes requests to the appropriate Client implementation by transport type:
    - "http" / "sse" → HttpMCPClient (official SSE protocol)
    - "stdio" → NotImplementedError (reserved)
    Upper-layer code calls through the MCPClientAdapter interface, completely transparent.
    """

    def __init__(self):
        self._http = HttpMCPClient()
        self._routing: Dict[str, str] = {}  # server_name → transport

    def register_server(self, server_name: str, connection_config: Dict[str, Any]):
        transport = connection_config.get("transport", "http")
        if transport == "stdio":
            raise NotImplementedError(
                f"Stdio transport not yet implemented (server: {server_name}). "
                "Reserved for future local tool support."
            )
        self._routing[server_name] = transport
        self._http.register_server(server_name, connection_config)

    async def fetch_tools_list(self, server_name: str) -> List[Dict[str, Any]]:
        return await self._http.fetch_tools_list(server_name)

    async def execute_tool(
        self, server_name: str, tool_name: str, arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        return await self._http.execute_tool(server_name, tool_name, arguments)

    async def close_all(self):
        """Gracefully close all underlying connections"""
        await self._http.close_all()
