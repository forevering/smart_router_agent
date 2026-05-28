import os
os.environ["NO_PROXY"] = "127.0.0.1,localhost"

import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client

async def main():
    # Using the weather service as an example
    server_url = "http://127.0.0.1:8200/weather/sse"
    print(f"[Client] Connecting to MCP Server: {server_url} ...")

    # 1. Establish SSE transport layer connection
    # sse_client automatically handles GET requests for the stream and processes subsequent POST callback endpoints
    async with sse_client(server_url) as (read_stream, write_stream):
        
        # 2. Establish MCP client session
        async with ClientSession(read_stream, write_stream) as session:
            
            # 3. Must send initialization handshake first (Initialize Handshake)
            print("[Client] Sending initialization handshake...")
            await session.initialize()
            print("[Client] ✅ Handshake successful!\n")

            # ==========================================
            # Action A: Request available tool list from server (List Tools)
            # ==========================================
            print("[Client] Requesting tool list...")
            tools_response = await session.list_tools()
            
            print("[Client] Discovered tools:")
            for tool in tools_response.tools:
                print(f"  - 🛠️ {tool.name}: {tool.description[:40]}...")
            print("-" * 40)

            # ==========================================
            # Action B: Call a specific tool (Call Tool)
            # ==========================================
            target_tool = "get_weather"
            arguments = {
                "location": "Shanghai",
                "date_range": "next_week"
            }
            
            print(f"[Client] Calling tool '{target_tool}' with arguments: {arguments}")
            
            # Initiate call and wait for result
            result = await session.call_tool(target_tool, arguments)
            
            # Parse and print return results (MCP protocol specifies a Content list is returned)
            print("[Client] 🎯 Tool call complete! Results:")
            for content in result.content:
                if content.type == "text":
                    print(f"  📝 {content.text}")
                else:
                    print(f"  📦 [{content.type}] (non-text data)")

if __name__ == "__main__":
    asyncio.run(main())