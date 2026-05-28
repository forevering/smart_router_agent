"""
Smart Router Agent - MCP + Tool Retrieval 智能路由 Agent
基于 LangGraph 构建，支持 STM / LTM / Temporal KG 三层记忆架构
集成 ToolRegistry + MCP Client + AdminController 热插拔管理

DDD 目录结构：
  config/   - 配置、Prompt 加载、prompts.yaml、mcp_servers_config.json
  memory/   - 长期记忆存储、时序知识图谱
  tools/    - 工具注册中心、MCP 客户端、Mock MCP 服务端
  admin/    - 管理控制面、REST API
  core/     - LangGraph 图、节点逻辑、状态定义
"""
