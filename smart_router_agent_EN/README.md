# Smart Router Agent

> A cloud-native AI gateway built on LangGraph and the official MCP protocol, featuring a three-layer memory architecture (STM + LTM + Temporal KG) and million-scale tool semantic routing capabilities.

---

## Core Features

| Feature | Description |
|---------|-------------|
| **Domain-Driven Design (DDD)** | Organized into five domain packages: `config / memory / tools / admin / core`, with clear responsibilities and boundaries |
| **Three-Layer Memory Architecture** | STM (LangGraph Checkpoint auto-managed) + LTM (Qdrant semantic fact store) + Temporal KG (SQLite temporal graph + Qdrant semantic entry) |
| **RAG for Tools** | Contextualized Query → Qdrant vector recall Top-K candidate tools → LLM decision binding, natively supports million-scale tool inventory |
| **Official MCP Protocol (SSE/JSON-RPC)** | SSE persistent connection Client based on `mcp` Python SDK v1.27+, fully compliant with the MCP open protocol |
| **Production-Grade Connection Pool** | Lazy Init + Idle TTL Reclamation (600s) + Anti-Stampede Jitter Retry |
| **Control Plane / Data Plane Decoupling** | AdminController (CRUD) fully decoupled from LangGraph data plane, supports zero-downtime hot-swapping |
| **Stateless Tool Registry** | Full tool metadata + Embeddings persisted to Qdrant payload, auto-recovers on process restart |
| **Zero-Downtime Hot-Swap** | Remote MCP Server upgrade → Agent calls `refresh` to discover new tools, no restart needed |

---

## Project Structure

```
smart_router_agent/
├── __init__.py                  # Package entry
├── demo_main.py                 # Demo entry: auto-spawns subprocess, 4-phase E2E scenario
├── server_main.py               # Production entry: persistent FastAPI service
│
├── config/                      # 🔧 Configuration Layer
│   ├── __init__.py
│   ├── config.py                # Global config: Azure OpenAI / Embedding / paths / constants
│   ├── prompt_loader.py         # Prompt template loader (hot-reload + thread-safe)
│   ├── prompts.yaml             # All prompt template definitions
│   └── mcp_servers_config.json  # MCP Server connection config (used in production mode)
│
├── memory/                      # 🧠 Memory Layer
│   ├── __init__.py
│   ├── memory_store.py          # Dual-layer LTM: L1 SQLite conversations + L2 Qdrant facts
│   └── temporal_kg.py           # Temporal Knowledge Graph: semantic seed + bounded N-hop expansion
│
├── tools/                       # 🔌 Tools Layer
│   ├── __init__.py
│   ├── tool_registry.py         # Stateless tool registry (Qdrant-persisted)
│   ├── mcp_client.py            # Official MCP SDK SSE Client + connection pool
│   └── mock_mcp_servers.py      # Mock MCP Server (DDD-partitioned, for dev/test)
│
├── admin/                       # 🛡️ Control Plane
│   ├── __init__.py
│   ├── admin_controller.py      # Hot management core: register/refresh/prompt hot-reload
│   └── admin_api.py             # FastAPI REST interface (management + chat)
│
├── core/                        # ⚙️ Data Plane (LangGraph Graph)
│   ├── __init__.py
│   ├── state.py                 # AgentState TypedDict definition
│   ├── nodes.py                 # All 9 Graph node factory functions
│   └── graph.py                 # StateGraph orchestration + conditional edges + compilation
│
├── qdrant_data/                 # Qdrant local file storage (auto-generated)
└── agent_data.db                # SQLite database (auto-generated)
```

---

## Quick Start

### Prerequisites

| Dependency | Version Requirement |
|------------|---------------------|
| Python | ≥ 3.10 |
| mcp (MCP SDK) | ≥ 1.27 |
| langchain-openai | ≥ 0.3 |
| langgraph | ≥ 0.4 |
| fastapi | ≥ 0.100 |
| uvicorn | ≥ 0.30 |
| qdrant-client | ≥ 1.9 |
| sentence-transformers | ≥ 2.2 |
| aiosqlite | ≥ 0.19 |
| pyyaml | ≥ 6.0 |
| httpx | ≥ 0.27 |

```bash
# Create conda environment
conda create -n py310 python=3.10 -y
conda activate py310

# Install dependencies
pip install mcp fastapi uvicorn httpx langchain-openai langchain-core \
    langgraph aiosqlite sentence-transformers qdrant-client pyyaml
```

### Demo Mode (demo_main.py)

```bash
cd <project_root>   # Parent directory containing smart_router_agent/

# Windows (enterprise network requires proxy bypass)
$env:NO_PROXY = ".bosch.com,127.0.0.1,localhost"
python -m smart_router_agent.demo_main
```

**Execution Flow:**
1. Automatically starts `smart_router_agent.tools.mock_mcp_servers` as a subprocess (port 8200)
2. Polls `/health` waiting for Server readiness
3. Four-phase scenario:
   - **Phase 1 — Cold Start**: Initialize LLM / Embedding / Qdrant / LTM / KG, register MCP Servers, build LangGraph
   - **Phase 2 — Normal Routing**: User asks about weather → semantic tool retrieval → LLM decision → MCP call → response generation
   - **Phase 3 — Hot Update**: Call Mock Server's `/admin/upgrade` → Agent executes `refresh_mcp_server`
   - **Phase 4 — New Capability Verification**: New tools (search_flight, get_typhoon_warning) immediately available
4. `finally` block gracefully kills subprocess

### Production Mode (server_main.py)

```bash
python -m smart_router_agent.server_main --host 0.0.0.0 --port 8100
```

- Reads `config/mcp_servers_config.json` for MCP Server initialization
- Does not clear existing DB / Qdrant data (incremental operation)
- Provides REST API:
  - `POST /api/chat` — Chat interface
  - `POST /admin/mcp/register` — Register new MCP Server
  - `POST /admin/mcp/refresh` — Refresh existing Server tools
  - `POST /admin/prompts/reload` — Prompt hot-reload
  - `GET /admin/health` — Health check

---

## Detailed Architecture Documentation

- [ARCHITECTURE_EN.md](./ARCHITECTURE_EN.md) — Core Architecture Design Deep Dive
- [OPERATIONS_EN.md](./OPERATIONS_EN.md) — Operations & Hot-Swap Manual

---

## Key Configuration

### config/mcp_servers_config.json

```json
{
  "mcpServers": {
    "Weather_Server": {
      "transport": "http",
      "url": "http://127.0.0.1:8200/weather/sse"
    },
    "Travel_Server": {
      "transport": "http",
      "url": "http://127.0.0.1:8200/travel/sse"
    }
  }
}
```

### config/config.py Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BASE_DIR` | Auto-inferred | smart_router_agent package root directory |
| `PROXY_URL` | `http://127.0.0.1:3128` | CNTLM local proxy |
| `EMBEDDING_MODEL_NAME` | `paraphrase-multilingual-MiniLM-L12-v2` | Multilingual embedding model |
| `EMBEDDING_DIM` | 384 | Vector dimension |
| `MAX_HOPS` | 2 | KG graph expansion hard limit |
| `SIMILARITY_THRESHOLD` | 0.35 | Minimum semantic matching threshold |
| `TOP_K` | 5 | Number of Top-K retrieval results |

---

## License

Internal Use Only — Bosch Group.

# Smart Router Agent

> 基于 LangGraph 与官方 MCP 协议构建的、具备三层记忆架构（STM + LTM + Temporal KG）和百万级工具语义路由能力的云原生 AI 网关。

---

## 核心特性

| 特性 | 说明 |
|------|------|
| **领域驱动设计 (DDD)** | 按 `config / memory / tools / admin / core` 五大领域分包，职责清晰、边界明确 |
| **三层记忆架构** | STM (LangGraph Checkpoint 自动管理) + LTM (Qdrant 语义事实库) + Temporal KG (SQLite 时序图 + Qdrant 语义入口) |
| **RAG for Tools** | Contextualized Query → Qdrant 向量召回 Top-K 候选工具 → LLM 决策绑定，天然支持百万级工具规模 |
| **官方 MCP 协议 (SSE/JSON-RPC)** | 基于 `mcp` Python SDK v1.27+ 的 SSE 长连接 Client，完全合规于 MCP 开放协议 |
| **生产级长连接池** | 懒加载 (Lazy Init) + 空闲回收 (Idle TTL 600s) + 防雪崩重试 (Jitter Retry) |
| **控制面与数据面解耦** | AdminController (CRUD) 与 LangGraph 数据面完全解耦，支持零停机热插拔 |
| **无状态工具注册表** | 全量工具元数据 + Embedding 持久化到 Qdrant payload，进程重启自动恢复 |
| **零停机热插拔** | 远端 MCP Server 升级 → Agent 调用 `refresh` 即可发现新工具，无需重启 |

---

## 目录结构

```
smart_router_agent/
├── __init__.py                  # 包入口
├── demo_main.py                 # 演示入口：自动拉起子进程，四阶段 E2E 剧本
├── server_main.py               # 生产入口：常驻 FastAPI 服务
│
├── config/                      # 🔧 配置层
│   ├── __init__.py
│   ├── config.py                # 全局配置：Azure OpenAI / Embedding / 路径 / 常量
│   ├── prompt_loader.py         # Prompt 模板加载器（热加载 + 线程安全）
│   ├── prompts.yaml             # 所有 Prompt 模板定义
│   └── mcp_servers_config.json  # MCP Server 连接配置（生产模式使用）
│
├── memory/                      # 🧠 记忆层
│   ├── __init__.py
│   ├── memory_store.py          # 双层 LTM：L1 SQLite 对话 + L2 Qdrant 事实
│   └── temporal_kg.py           # 时序知识图谱：语义种子 + 受限 N-hop 扩展
│
├── tools/                       # 🔌 工具层
│   ├── __init__.py
│   ├── tool_registry.py         # 无状态工具注册中心（Qdrant 持久化）
│   ├── mcp_client.py            # 官方 MCP SDK SSE Client + 连接池
│   └── mock_mcp_servers.py      # Mock MCP Server（DDD 分域，用于开发测试）
│
├── admin/                       # 🛡️ 控制面
│   ├── __init__.py
│   ├── admin_controller.py      # 热管理核心：注册/刷新/Prompt 热加载
│   └── admin_api.py             # FastAPI REST 接口（管理 + 对话）
│
├── core/                        # ⚙️ 数据面（LangGraph 图）
│   ├── __init__.py
│   ├── state.py                 # AgentState TypedDict 定义
│   ├── nodes.py                 # 全部 9 个 Graph 节点工厂函数
│   └── graph.py                 # StateGraph 编排 + 条件边 + 编译
│
├── qdrant_data/                 # Qdrant 本地文件存储（自动生成）
└── agent_data.db                # SQLite 数据库（自动生成）
```

---

## 快速开始

### 环境准备

| 依赖 | 版本要求 |
|------|----------|
| Python | ≥ 3.10 |
| mcp (MCP SDK) | ≥ 1.27 |
| langchain-openai | ≥ 0.3 |
| langgraph | ≥ 0.4 |
| fastapi | ≥ 0.100 |
| uvicorn | ≥ 0.30 |
| qdrant-client | ≥ 1.9 |
| sentence-transformers | ≥ 2.2 |
| aiosqlite | ≥ 0.19 |
| pyyaml | ≥ 6.0 |
| httpx | ≥ 0.27 |

```bash
# 创建 conda 环境
conda create -n py310 python=3.10 -y
conda activate py310

# 安装依赖
pip install mcp fastapi uvicorn httpx langchain-openai langchain-core \
    langgraph aiosqlite sentence-transformers qdrant-client pyyaml
```

### 演示模式 (demo_main.py)

```bash
cd <project_root>   # 包含 smart_router_agent/ 目录的父级

# Windows (企业网络需 bypass proxy)
$env:NO_PROXY = ".bosch.com,127.0.0.1,localhost"
python -m smart_router_agent.demo_main
```

**执行流程：**
1. 自动以子进程启动 `smart_router_agent.tools.mock_mcp_servers` (port 8200)
2. 轮询 `/health` 等待 Server 就绪
3. 四阶段剧本：
   - **阶段 1 — 冷启动**：初始化 LLM / Embedding / Qdrant / LTM / KG，注册 MCP Server，构建 LangGraph
   - **阶段 2 — 正常路由**：用户提问天气 → 工具语义检索 → LLM 决策 → MCP 调用 → 生成回复
   - **阶段 3 — 热更新**：调用 Mock Server 的 `/admin/upgrade` → Agent 执行 `refresh_mcp_server`
   - **阶段 4 — 新能力验证**：新工具 (search_flight, get_typhoon_warning) 即时可用
4. `finally` 块优雅 kill 子进程

### 生产模式 (server_main.py)

```bash
python -m smart_router_agent.server_main --host 0.0.0.0 --port 8100
```

- 读取 `config/mcp_servers_config.json` 进行 MCP Server 初始化
- 不清空已有 DB / Qdrant 数据（增量运行）
- 提供 REST API：
  - `POST /api/chat` — 对话接口
  - `POST /admin/mcp/register` — 注册新 MCP Server
  - `POST /admin/mcp/refresh` — 刷新现有 Server 工具
  - `POST /admin/prompts/reload` — Prompt 热加载
  - `GET /admin/health` — 健康检查

---

## 详细架构文档

- [ARCHITECTURE.md](./ARCHITECTURE.md) — 核心架构设计深度解读
- [OPERATIONS.md](./OPERATIONS.md) — 运维与热插拔操作手册

---

## 关键配置

### config/mcp_servers_config.json

```json
{
  "mcpServers": {
    "Weather_Server": {
      "transport": "http",
      "url": "http://127.0.0.1:8200/weather/sse"
    },
    "Travel_Server": {
      "transport": "http",
      "url": "http://127.0.0.1:8200/travel/sse"
    }
  }
}
```

### config/config.py 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `BASE_DIR` | 自动推断 | smart_router_agent 包根目录 |
| `PROXY_URL` | `http://127.0.0.1:3128` | CNTLM 本地代理 |
| `EMBEDDING_MODEL_NAME` | `paraphrase-multilingual-MiniLM-L12-v2` | 多语言 Embedding 模型 |
| `EMBEDDING_DIM` | 384 | 向量维度 |
| `MAX_HOPS` | 2 | KG 图扩展硬上限 |
| `SIMILARITY_THRESHOLD` | 0.35 | 语义匹配最低阈值 |
| `TOP_K` | 5 | 检索 Top-K 结果数 |

---

## License

Internal Use Only — Bosch Group.
