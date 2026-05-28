"""
MCP Client 适配层 — 纯官方协议网络通信中间件

基于官方 MCP SDK (mcp.client.sse / mcp.client.session) 实现。
不含任何硬编码业务数据，仅负责 MCP 协议通信和连接池管理。

包含：
- MCPClientAdapter: 抽象基类 (register_server / fetch_tools_list / execute_tool)
- HttpMCPClient: 基于官方 SSE Client 的生产级实现
    - 懒加载 (Lazy Init): register_server 仅存配置，首次调用时建连
    - 空闲回收 (Idle TTL): 后台协程定期扫描，超时连接主动关闭
    - 防雪崩重试 (Jitter Retry): 断连后随机休眠再重连
- MCPClientFactory: 传输路由器 (http/sse → HttpMCPClient, stdio → NotImplementedError)
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

# 确保 localhost 连接不走企业代理
import os as _os
if "127.0.0.1" not in _os.environ.get("NO_PROXY", ""):
    _os.environ["NO_PROXY"] = _os.environ.get("NO_PROXY", "") + ",127.0.0.1,localhost"

logger = logging.getLogger(__name__)


# ================================================================
# 抽象基类
# ================================================================

class MCPClientAdapter(ABC):
    """MCP 客户端统一接口基类，上层代码仅依赖此抽象。"""

    @abstractmethod
    def register_server(self, server_name: str, connection_config: Dict[str, Any]):
        """注册 MCP Server 的连接配置（不建连）"""
        ...

    @abstractmethod
    async def fetch_tools_list(self, server_name: str) -> List[Dict[str, Any]]:
        """
        从 MCP Server 拉取 tools/list。
        Returns: 原生 Python Dict 列表，每个元素包含：
            - name: str
            - description: str
            - inputSchema: dict (JSON Schema)
        """
        ...

    @abstractmethod
    async def execute_tool(
        self, server_name: str, tool_name: str, arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """向 MCP Server 发起 tools/call RPC 调用"""
        ...


# ================================================================
# 连接池条目
# ================================================================

class _SessionEntry:
    """连接池中的一个 Session 条目"""
    __slots__ = ("session", "exit_stack", "last_active")

    def __init__(self, session: ClientSession, exit_stack: AsyncExitStack):
        self.session = session
        self.exit_stack = exit_stack
        self.last_active: float = time.monotonic()

    def touch(self):
        self.last_active = time.monotonic()


# ================================================================
# 官方 SSE 实现
# ================================================================

class HttpMCPClient(MCPClientAdapter):
    """
    基于官方 MCP SDK 的 SSE Client 实现。
    通过 mcp.client.sse.sse_client 建立 SSE 长连接，
    通过 mcp.client.session.ClientSession 完成 JSON-RPC 握手和调用。

    连接池管理策略 (生产级 / K8s 友好)：
    - 懒加载: register_server 仅保存配置，首次 RPC 时才建连
    - 空闲回收: 后台定时扫描，闲置超过 IDLE_TTL 的连接主动关闭
    - 防雪崩: 断连重试前引入随机 Jitter 休眠
    """

    IDLE_TTL: float = 600.0       # 空闲 10 分钟回收
    CLEANUP_INTERVAL: float = 60.0  # 每 60 秒扫描一次

    def __init__(self):
        self._connections: Dict[str, Dict[str, Any]] = {}   # server_name → config
        self._sessions: Dict[str, _SessionEntry] = {}       # server_name → session
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None

    # ---- 注册（仅存配置） ----

    def register_server(self, server_name: str, connection_config: Dict[str, Any]):
        self._connections[server_name] = connection_config

    # ---- 懒加载：按需建连 ----

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

            # 启动后台清理（首次建连时）
            self._ensure_cleanup_running()

            logger.info(f"[MCP] Session ready: {server_name}")
            return session

    # ---- 空闲回收 ----

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

    # ---- 防雪崩重试 ----

    async def _with_retry(self, server_name: str, operation):
        """执行操作，如果连接断开则 jitter 重试一次"""
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

    # ---- 公开接口 ----

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

    # ---- 数据格式剥离 (SDK 强类型 → 原生 Dict) ----

    @staticmethod
    def _map_tools(result) -> List[Dict[str, Any]]:
        """将 ListToolsResult 转为原生 Dict 列表"""
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
        """将 CallToolResult 转为原生 Dict"""
        texts = []
        for item in result.content:
            if hasattr(item, "text"):
                texts.append(item.text)
        combined = "\n".join(texts)
        try:
            return json.loads(combined)
        except (json.JSONDecodeError, TypeError):
            return {"result": combined}

    # ---- 优雅关闭 ----

    async def close_all(self):
        """关闭所有活跃连接"""
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
# 传输路由器（工厂）
# ================================================================

class MCPClientFactory(MCPClientAdapter):
    """
    按 transport 类型将请求路由到对应的 Client 实现：
    - "http" / "sse" → HttpMCPClient (官方 SSE 协议)
    - "stdio" → NotImplementedError (预留)
    上层代码通过 MCPClientAdapter 接口调用，完全无感知。
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
        """优雅关闭底层所有连接"""
        await self._http.close_all()
