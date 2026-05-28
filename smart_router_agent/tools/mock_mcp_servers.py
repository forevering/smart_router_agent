"""
Mock MCP Servers — DDD 领域分组的真实 MCP 服务端

使用官方 MCP SDK (mcp.server.Server + SseServerTransport) 实现。
单端口 (8200) 多路径挂载，两个独立 MCP Server：

  Weather_Server:  /weather/sse, /weather/messages
  Travel_Server:   /travel/sse, /travel/messages

管理端 POST /admin/upgrade/{weather|travel}  模拟远端服务升级（动态注册工具）。

使用方式：
  python -m smart_router_agent.mock_mcp_servers
  python -m smart_router_agent.mock_mcp_servers --port 8200
"""
from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any, Callable, Sequence

import uvicorn
from fastapi import FastAPI, Request
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import TextContent, Tool
from starlette.responses import Response


# ================================================================
# 可动态扩展的 MCP Server 封装
# ================================================================

class DynamicMCPServer:
    """
    封装 mcp.server.Server + 动态工具注册。
    支持运行时 add_tool()，后续 list_tools 立即可见。
    """

    def __init__(self, name: str):
        self.server = Server(name)
        self._tools: dict[str, Tool] = {}
        self._handlers: dict[str, Callable] = {}

        @self.server.list_tools()
        async def _list_tools() -> list[Tool]:
            return list(self._tools.values())

        @self.server.call_tool()
        async def _call_tool(name: str, arguments: dict | None = None) -> Sequence[TextContent]:
            handler = self._handlers.get(name)
            if not handler:
                return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}, ensure_ascii=False))]
            result = await handler(**(arguments or {}))
            if isinstance(result, str):
                text = result
            else:
                text = json.dumps(result, ensure_ascii=False)
            return [TextContent(type="text", text=text)]

    def add_tool(self, tool: Tool, handler: Callable):
        self._tools[tool.name] = tool
        self._handlers[tool.name] = handler


# ================================================================
# 工具执行函数
# ================================================================

async def exec_get_weather(location: str = "未知", date_range: str = "next_week", **kw) -> str:
    await asyncio.sleep(0.3)
    forecasts = {
        "上海": [
            {"date": "2026-05-26", "weather": "多云", "temp_high": 26, "temp_low": 18, "rain_prob": "20%"},
            {"date": "2026-05-27", "weather": "小雨", "temp_high": 23, "temp_low": 17, "rain_prob": "75%"},
            {"date": "2026-05-28", "weather": "中雨", "temp_high": 21, "temp_low": 16, "rain_prob": "90%"},
            {"date": "2026-05-29", "weather": "阴转多云", "temp_high": 24, "temp_low": 17, "rain_prob": "30%"},
            {"date": "2026-05-30", "weather": "晴", "temp_high": 28, "temp_low": 19, "rain_prob": "5%"},
        ],
        "北京": [
            {"date": "2026-05-26", "weather": "晴", "temp_high": 30, "temp_low": 16, "rain_prob": "5%"},
            {"date": "2026-05-27", "weather": "晴转多云", "temp_high": 28, "temp_low": 15, "rain_prob": "10%"},
            {"date": "2026-05-28", "weather": "多云", "temp_high": 26, "temp_low": 14, "rain_prob": "15%"},
        ],
    }
    forecast = forecasts.get(location, forecasts["上海"])
    rainy = [f["date"] for f in forecast if int(f["rain_prob"].replace("%", "")) > 50]
    advisory = f"{'、'.join(rainy)} 有较大降雨概率，建议携带雨具" if rainy else "预报期内天气总体良好，适合出行"
    return json.dumps(
        {"service": "get_weather", "location": location, "period": date_range, "forecast": forecast, "advisory": advisory},
        ensure_ascii=False,
    )


async def exec_get_typhoon_warning(region: str = "未知", **kw) -> str:
    await asyncio.sleep(0.2)
    return json.dumps({
        "service": "get_typhoon_warning", "region": region,
        "warnings": [{
            "typhoon_name": "台风'海棠'", "current_position": "东海海域",
            "expected_landing": "预计5月21日在浙江沿海登陆", "impact_level": "黄色预警",
            "affected_areas": ["上海", "浙江", "江苏南部"],
            "advisory": "建议密切关注气象部门最新预警信息，做好防台准备",
        }],
    }, ensure_ascii=False)


async def exec_get_schedule(user_id: str = "unknown", date_range: str = "next_week", **kw) -> str:
    await asyncio.sleep(0.2)
    return json.dumps({
        "service": "get_schedule", "user_id": user_id, "period": date_range,
        "events": [
            {"date": "2026-05-26", "time": "09:00-12:00", "title": "客户拜访 - 张总", "location": "上海市浦东新区陆家嘴"},
            {"date": "2026-05-27", "time": "14:00-16:00", "title": "项目评审会议", "location": "上海办公室"},
            {"date": "2026-05-28", "time": "10:00-11:30", "title": "供应商洽谈", "location": "上海市闵行区"},
            {"date": "2026-05-29", "time": "自由安排", "title": "返程", "location": ""},
        ],
    }, ensure_ascii=False)


async def exec_search_flight(departure: str = "未知", destination: str = "未知", date: str = "未知", **kw) -> str:
    await asyncio.sleep(0.3)
    return json.dumps({
        "service": "search_flight", "departure": departure, "destination": destination, "date": date,
        "flights": [
            {"flight_no": "MU5101", "departure_time": "07:00", "arrival_time": "09:15", "price": 680, "airline": "东方航空"},
            {"flight_no": "CA1501", "departure_time": "10:30", "arrival_time": "12:45", "price": 820, "airline": "中国国航"},
            {"flight_no": "FM9101", "departure_time": "14:00", "arrival_time": "16:10", "price": 590, "airline": "上海航空"},
            {"flight_no": "CZ3501", "departure_time": "18:30", "arrival_time": "20:40", "price": 720, "airline": "南方航空"},
        ],
    }, ensure_ascii=False)


# ================================================================
# 工具定义 (JSON Schema)
# ================================================================

TOOL_GET_WEATHER = Tool(
    name="get_weather",
    description=(
        "查询指定地点的天气预报，返回未来数天的天气、气温和降雨概率。"
        "典型用例：用户询问某城市天气怎样；用户出差旅行前想了解目的地天气；"
        "用户关心下周天气是否影响出行计划。"
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "查询天气的城市名称，如'上海'、'北京'"},
            "date_range": {"type": "string", "description": "时间范围，如'next_week'、'tomorrow'", "default": "next_week"},
        },
        "required": ["location"],
    },
)

TOOL_GET_TYPHOON_WARNING = Tool(
    name="get_typhoon_warning",
    description=(
        "查询指定地区的台风预警信息，包括台风名称、位置、预计登陆时间、影响等级。"
        "典型用例：用户想知道某地区有没有台风；用户关心台风是否影响出行；"
        "用户询问台风预警信息和防台建议。"
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "region": {"type": "string", "description": "查询台风预警的地区，如'上海'、'浙江'"},
        },
        "required": ["region"],
    },
)

TOOL_GET_SCHEDULE = Tool(
    name="get_schedule",
    description=(
        "查询用户的日程安排和会议信息。"
        "典型用例：用户想知道自己下周有什么安排；用户询问某天有没有会议；"
        "用户想了解出差期间的日程安排。"
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "user_id": {"type": "string", "description": "用户ID"},
            "date_range": {"type": "string", "description": "时间范围", "default": "next_week"},
        },
        "required": ["user_id"],
    },
)

TOOL_SEARCH_FLIGHT = Tool(
    name="search_flight",
    description=(
        "搜索航班信息，包括航班号、出发/到达时间、价格等。"
        "典型用例：用户想查询去某个城市的航班；用户要订机票出差旅行；"
        "用户想比较不同航班的价格和时间。"
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "departure": {"type": "string", "description": "出发城市"},
            "destination": {"type": "string", "description": "目的城市"},
            "date": {"type": "string", "description": "出发日期，如'2026-05-18'"},
        },
        "required": ["departure", "destination"],
    },
)


# ================================================================
# 构建 FastAPI 应用
# ================================================================

# ---- 创建两个 DynamicMCPServer ----
weather_server = DynamicMCPServer("Weather_Server")
travel_server = DynamicMCPServer("Travel_Server")

# 冷启动时注册的工具
weather_server.add_tool(TOOL_GET_WEATHER, exec_get_weather)
travel_server.add_tool(TOOL_GET_SCHEDULE, exec_get_schedule)

# ---- SSE 传输层 (路径相对于子应用 mount point) ----
#weather_sse = SseServerTransport("/messages")  #"/weather/messages"
weather_sse = SseServerTransport("/weather/messages")
#travel_sse = SseServerTransport("/messages")   #"/travel/messages"   
travel_sse = SseServerTransport("/travel/messages")

# ---- FastAPI 应用 ----
app = FastAPI(title="Mock MCP Servers (DDD Gateway)")


# ---- SSE 端点必须绕过 FastAPI/Starlette 中间件，直接作为 raw ASGI app 挂载 ----
def _create_sse_app(sse_transport: SseServerTransport, mcp_server: DynamicMCPServer):
    """为一个 MCP Server 创建 raw ASGI 应用（绕过 Route/Starlette 中间件对 SSE 流的拦截）"""
    async def asgi_app(scope, receive, send):
        if scope["type"] != "http":
            return
        path = scope.get("path", "")
        if path.endswith("/sse"):
            async with sse_transport.connect_sse(scope, receive, send) as streams:
                await mcp_server.server.run(
                    streams[0], streams[1],
                    mcp_server.server.create_initialization_options(),
                )
        elif path.endswith("/messages") and scope.get("method", "GET") == "POST":
            await sse_transport.handle_post_message(scope, receive, send)
        else:
            response = Response("Not Found", status_code=404)
            await response(scope, receive, send)
    return asgi_app

app.mount("/weather", _create_sse_app(weather_sse, weather_server))
app.mount("/travel", _create_sse_app(travel_sse, travel_server))


@app.post("/admin/upgrade/{server_name}")
async def upgrade_handler(server_name: str):
    if server_name == "weather":
        weather_server.add_tool(TOOL_GET_TYPHOON_WARNING, exec_get_typhoon_warning)
        return {"status": "ok", "message": "Weather_Server upgraded: +get_typhoon_warning"}
    elif server_name == "travel":
        travel_server.add_tool(TOOL_SEARCH_FLIGHT, exec_search_flight)
        return {"status": "ok", "message": "Travel_Server upgraded: +search_flight"}
    else:
        return {"status": "error", "message": f"Unknown: {server_name}"}


@app.get("/health")
async def health_handler():
    return {"status": "healthy", "servers": ["Weather_Server", "Travel_Server"]}


def main():
    parser = argparse.ArgumentParser(description="Mock MCP Servers")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8200)
    args = parser.parse_args()

    print(f"  [MockMCPServers] Starting: http://{args.host}:{args.port}")
    print(f"  [MockMCPServers] Weather SSE: /weather/sse")
    print(f"  [MockMCPServers] Travel  SSE: /travel/sse")
    print(f"  [MockMCPServers] Admin: POST /admin/upgrade/{{weather|travel}}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
