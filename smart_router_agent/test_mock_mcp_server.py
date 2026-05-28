import os
os.environ["NO_PROXY"] = "127.0.0.1,localhost"

import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client

async def main():
    # 我们以天气服务为例
    server_url = "http://127.0.0.1:8200/weather/sse"
    print(f"[Client] 正在连接到 MCP Server: {server_url} ...")

    # 1. 建立 SSE 传输层连接
    # sse_client 会自动处理 GET 请求获取流，并处理后续的 POST 回调端点
    async with sse_client(server_url) as (read_stream, write_stream):
        
        # 2. 建立 MCP 客户端会话
        async with ClientSession(read_stream, write_stream) as session:
            
            # 3. 必须先发送初始化握手 (Initialize Handshake)
            print("[Client] 正在发送初始化握手协议...")
            await session.initialize()
            print("[Client] ✅ 握手成功！\n")

            # ==========================================
            # 动作 A：向服务端请求可用工具列表 (List Tools)
            # ==========================================
            print("[Client] 正在请求工具列表...")
            tools_response = await session.list_tools()
            
            print("[Client] 发现以下工具:")
            for tool in tools_response.tools:
                print(f"  - 🛠️ {tool.name}: {tool.description[:40]}...")
            print("-" * 40)

            # ==========================================
            # 动作 B：调用具体的工具 (Call Tool)
            # ==========================================
            target_tool = "get_weather"
            arguments = {
                "location": "上海",
                "date_range": "next_week"
            }
            
            print(f"[Client] 准备调用工具 '{target_tool}'，参数: {arguments}")
            
            # 发起调用并等待结果
            result = await session.call_tool(target_tool, arguments)
            
            # 解析并打印返回结果 (MCP 协议规定返回的是一个 Content 列表)
            print("[Client] 🎯 工具调用完成！返回结果：")
            for content in result.content:
                if content.type == "text":
                    print(f"  📝 {content.text}")
                else:
                    print(f"  📦 [{content.type}] (非文本数据)")

if __name__ == "__main__":
    asyncio.run(main())