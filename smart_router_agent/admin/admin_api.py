"""
Admin + Chat REST API - FastAPI 包装层

为 AdminController 提供 HTTP 管理接口，同时提供 /api/chat 对话接口。
"""
import asyncio
import threading
import uuid
from typing import Optional, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from smart_router_agent.admin.admin_controller import AdminController


# ---- Request Models ----
class RegisterServerRequest(BaseModel):
    server_name: str
    connection_config: dict


class RefreshServerRequest(BaseModel):
    server_name: str


class ChatRequest(BaseModel):
    message: str
    user_id: str = "default_user"
    thread_id: Optional[str] = None


# ---- FastAPI App Factory ----
def create_admin_app(
    admin_controller: AdminController,
    graph: Any = None,
) -> FastAPI:
    """创建 FastAPI 应用（管理 + 对话）"""
    app = FastAPI(title="Smart Router Agent API", version="2.0.0")

    @app.post("/admin/mcp/register")
    async def register_mcp_server(req: RegisterServerRequest):
        try:
            await admin_controller.register_new_mcp_server(
                req.server_name, req.connection_config
            )
            return {"status": "ok", "message": f"Server '{req.server_name}' registered"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/admin/mcp/refresh")
    async def refresh_mcp_server(req: RefreshServerRequest):
        try:
            await admin_controller.refresh_mcp_server(req.server_name)
            return {"status": "ok", "message": f"Server '{req.server_name}' refreshed"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/admin/prompts/reload")
    async def reload_prompts():
        try:
            admin_controller.reload_prompts()
            return {"status": "ok", "message": "Prompts reloaded"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/admin/health")
    async def health():
        return {"status": "healthy"}

    if graph is not None:
        from langchain_core.messages import HumanMessage

        @app.post("/api/chat")
        async def chat(req: ChatRequest):
            thread_id = req.thread_id or str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}
            input_state = {
                "messages": [HumanMessage(content=req.message)],
                "user_id": req.user_id,
            }
            final_state = None
            async for event in graph.astream(input_state, config, stream_mode="updates"):
                final_state = event
            # 获取最终 response
            state = await graph.aget_state(config)
            response = state.values.get("response", "")
            return {
                "thread_id": thread_id,
                "response": response,
            }

    return app


def start_admin_server_in_background(
    admin_controller: AdminController,
    host: str = "127.0.0.1",
    port: int = 8100,
    graph: Any = None,
):
    """在后台线程中启动 FastAPI 服务器。非阻塞。"""
    app = create_admin_app(admin_controller, graph=graph)

    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        config = uvicorn.Config(app, host=host, port=port, log_level="warning")
        server = uvicorn.Server(config)
        loop.run_until_complete(server.serve())

    thread = threading.Thread(target=_run, daemon=True, name="admin-api")
    thread.start()
    print(f"  [AdminAPI] HTTP 服务已启动: http://{host}:{port}")
    return thread
