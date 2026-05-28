sequenceDiagram
    participant OPS as 运维工程师
    participant AGENT as Agent (FastAPI)
    participant CTRL as AdminController
    participant MCP_CLI as MCPClientFactory
    participant MCP_SRV as 外部 MCP Server

    Note over OPS,MCP_SRV: === 方案 A: 刷新工具 ===
    OPS->>MCP_SRV: (业务团队升级并重启 Server)
    OPS->>AGENT: POST /admin/mcp/refresh<br/>{"server_name": "Weather_Server"}
    AGENT->>CTRL: refresh_mcp_server("Weather_Server")
    CTRL->>MCP_CLI: fetch_tools_list("Weather_Server")
    MCP_CLI->>MCP_SRV: SSE → tools/list (JSON-RPC)
    MCP_SRV-->>MCP_CLI: [get_weather, get_air_quality]
    MCP_CLI-->>CTRL: 工具列表
    CTRL->>CTRL: Upsert ALL to Qdrant
    CTRL->>CTRL: Delete Stale from Qdrant
    CTRL-->>AGENT: 200 OK
    AGENT-->>OPS: {"status": "ok"}

    Note over OPS,MCP_SRV: === 验证 ===
    OPS->>AGENT: POST /api/chat<br/>{"message": "空气质量如何？"}
    AGENT->>AGENT: RAG Top-K → get_air_quality
    AGENT->>MCP_CLI: execute_tool("Weather_Server", "get_air_quality", {...})
    MCP_CLI->>MCP_SRV: SSE → tools/call
    MCP_SRV-->>MCP_CLI: result
    AGENT-->>OPS: 综合回复