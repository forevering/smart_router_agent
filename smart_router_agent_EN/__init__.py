"""
Smart Router Agent - MCP + Tool Retrieval Intelligent Routing Agent
Built on LangGraph, supporting a three-layer memory architecture: STM / LTM / Temporal KG
Integrates ToolRegistry + MCP Client + AdminController for hot-plug management

DDD directory structure:
  config/   - Configuration, prompt loading, prompts.yaml, mcp_servers_config.json
  memory/   - Long-term memory storage, temporal knowledge graph
  tools/    - Tool registry, MCP client, Mock MCP server
  admin/    - Admin control plane, REST API
  core/     - LangGraph graph, node logic, state definitions
"""
