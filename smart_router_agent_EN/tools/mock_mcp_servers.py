"""
Mock MCP Servers — DDD Domain-Grouped Real MCP Server Implementations

Implemented using the official MCP SDK (mcp.server.Server + SseServerTransport).
Single port (8200), multi-path mount with two independent MCP Servers:

  Weather_Server:  /weather/sse, /weather/messages
  Travel_Server:   /travel/sse, /travel/messages

Admin endpoint POST /admin/upgrade/{weather|travel} simulates remote service upgrade (dynamic tool registration).

Usage:
  python -m smart_router_agent_EN.mock_mcp_servers
  python -m smart_router_agent_EN.mock_mcp_servers --port 8200
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
# Dynamically Extensible MCP Server Wrapper
# ================================================================

class DynamicMCPServer:
    """
    Wraps mcp.server.Server + dynamic tool registration.
    Supports runtime add_tool(); subsequent list_tools calls see new tools immediately.
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
# Tool Execution Functions
# ================================================================

async def exec_get_weather(location: str = "unknown", date_range: str = "next_week", **kw) -> str:
    await asyncio.sleep(0.3)
    forecasts = {
        "Shanghai": [
            {"date": "2026-05-26", "weather": "Cloudy", "temp_high": 26, "temp_low": 18, "rain_prob": "20%"},
            {"date": "2026-05-27", "weather": "Light rain", "temp_high": 23, "temp_low": 17, "rain_prob": "75%"},
            {"date": "2026-05-28", "weather": "Moderate rain", "temp_high": 21, "temp_low": 16, "rain_prob": "90%"},
            {"date": "2026-05-29", "weather": "Overcast to cloudy", "temp_high": 24, "temp_low": 17, "rain_prob": "30%"},
            {"date": "2026-05-30", "weather": "Sunny", "temp_high": 28, "temp_low": 19, "rain_prob": "5%"},
        ],
        "Beijing": [
            {"date": "2026-05-26", "weather": "Sunny", "temp_high": 30, "temp_low": 16, "rain_prob": "5%"},
            {"date": "2026-05-27", "weather": "Sunny to cloudy", "temp_high": 28, "temp_low": 15, "rain_prob": "10%"},
            {"date": "2026-05-28", "weather": "Cloudy", "temp_high": 26, "temp_low": 14, "rain_prob": "15%"},
        ],
    }
    forecast = forecasts.get(location, forecasts["Shanghai"])
    rainy = [f["date"] for f in forecast if int(f["rain_prob"].replace("%", "")) > 50]
    advisory = f"High probability of rain on {', '.join(rainy)}; recommend bringing an umbrella" if rainy else "Weather is generally good during the forecast period, suitable for travel"
    return json.dumps(
        {"service": "get_weather", "location": location, "period": date_range, "forecast": forecast, "advisory": advisory},
        ensure_ascii=False,
    )


async def exec_get_typhoon_warning(region: str = "unknown", **kw) -> str:
    await asyncio.sleep(0.2)
    return json.dumps({
        "service": "get_typhoon_warning", "region": region,
        "warnings": [{
            "typhoon_name": "Typhoon 'Haitang'", "current_position": "East China Sea",
            "expected_landing": "Expected to make landfall along the Zhejiang coast on May 21", "impact_level": "Yellow warning",
            "affected_areas": ["Shanghai", "Zhejiang", "Southern Jiangsu"],
            "advisory": "Please closely monitor the latest warning information from meteorological departments and prepare for the typhoon",
        }],
    }, ensure_ascii=False)


async def exec_get_schedule(user_id: str = "unknown", date_range: str = "next_week", **kw) -> str:
    await asyncio.sleep(0.2)
    return json.dumps({
        "service": "get_schedule", "user_id": user_id, "period": date_range,
        "events": [
            {"date": "2026-05-26", "time": "09:00-12:00", "title": "Client visit - Mr. Zhang", "location": "Lujiazui, Pudong New Area, Shanghai"},
            {"date": "2026-05-27", "time": "14:00-16:00", "title": "Project review meeting", "location": "Shanghai office"},
            {"date": "2026-05-28", "time": "10:00-11:30", "title": "Supplier negotiation", "location": "Minhang District, Shanghai"},
            {"date": "2026-05-29", "time": "Flexible", "title": "Return trip", "location": ""},
        ],
    }, ensure_ascii=False)


async def exec_search_flight(departure: str = "unknown", destination: str = "unknown", date: str = "unknown", **kw) -> str:
    await asyncio.sleep(0.3)
    return json.dumps({
        "service": "search_flight", "departure": departure, "destination": destination, "date": date,
        "flights": [
            {"flight_no": "MU5101", "departure_time": "07:00", "arrival_time": "09:15", "price": 680, "airline": "China Eastern Airlines"},
            {"flight_no": "CA1501", "departure_time": "10:30", "arrival_time": "12:45", "price": 820, "airline": "Air China"},
            {"flight_no": "FM9101", "departure_time": "14:00", "arrival_time": "16:10", "price": 590, "airline": "Shanghai Airlines"},
            {"flight_no": "CZ3501", "departure_time": "18:30", "arrival_time": "20:40", "price": 720, "airline": "China Southern Airlines"},
        ],
    }, ensure_ascii=False)


# ================================================================
# Tool Definitions (JSON Schema)
# ================================================================

TOOL_GET_WEATHER = Tool(
    name="get_weather",
    description=(
        "Query weather forecast for a specified location, returning weather conditions, temperature, and rain probability for the coming days. "
        "Typical use cases: user asks about weather in a city; user wants to check destination weather before a business trip or travel; "
        "user is concerned whether next week's weather will affect travel plans."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name to query weather for, e.g. 'Shanghai', 'Beijing'"},
            "date_range": {"type": "string", "description": "Time range, e.g. 'next_week', 'tomorrow'", "default": "next_week"},
        },
        "required": ["location"],
    },
)

TOOL_GET_TYPHOON_WARNING = Tool(
    name="get_typhoon_warning",
    description=(
        "Query typhoon warning information for a specified region, including typhoon name, position, expected landfall time, and impact level. "
        "Typical use cases: user wants to know if there is a typhoon in a region; user is concerned whether a typhoon will affect travel; "
        "user asks about typhoon warning information and preparation advice."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "region": {"type": "string", "description": "Region to query for typhoon warnings, e.g. 'Shanghai', 'Zhejiang'"},
        },
        "required": ["region"],
    },
)

TOOL_GET_SCHEDULE = Tool(
    name="get_schedule",
    description=(
        "Query user's schedule and meeting information. "
        "Typical use cases: user wants to know what's planned for next week; user asks if there are meetings on a certain day; "
        "user wants to know the schedule during a business trip."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "user_id": {"type": "string", "description": "User ID"},
            "date_range": {"type": "string", "description": "Time range", "default": "next_week"},
        },
        "required": ["user_id"],
    },
)

TOOL_SEARCH_FLIGHT = Tool(
    name="search_flight",
    description=(
        "Search flight information including flight number, departure/arrival times, and prices. "
        "Typical use cases: user wants to find flights to a city; user needs to book a ticket for a business trip or travel; "
        "user wants to compare prices and times of different flights."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "departure": {"type": "string", "description": "Departure city"},
            "destination": {"type": "string", "description": "Destination city"},
            "date": {"type": "string", "description": "Departure date, e.g. '2026-05-18'"},
        },
        "required": ["departure", "destination"],
    },
)


# ================================================================
# Build FastAPI Application
# ================================================================

# ---- Create two DynamicMCPServers ----
weather_server = DynamicMCPServer("Weather_Server")
travel_server = DynamicMCPServer("Travel_Server")

# Tools registered at cold start
weather_server.add_tool(TOOL_GET_WEATHER, exec_get_weather)
travel_server.add_tool(TOOL_GET_SCHEDULE, exec_get_schedule)

# ---- SSE transport layer (paths relative to sub-app mount point) ----
#weather_sse = SseServerTransport("/messages")  #"/weather/messages"
weather_sse = SseServerTransport("/weather/messages")
#travel_sse = SseServerTransport("/messages")   #"/travel/messages"   
travel_sse = SseServerTransport("/travel/messages")

# ---- FastAPI application ----
app = FastAPI(title="Mock MCP Servers (DDD Gateway)")


# ---- SSE endpoints must bypass FastAPI/Starlette middleware, mounted directly as raw ASGI app ----
def _create_sse_app(sse_transport: SseServerTransport, mcp_server: DynamicMCPServer):
    """Create a raw ASGI app for an MCP Server (bypasses Route/Starlette middleware interception of SSE streams)"""
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
