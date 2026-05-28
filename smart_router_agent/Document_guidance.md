README.md — 项目概述、核心特性表、DDD 树状目录说明、Quick Start（演示模式 + 生产模式）、关键配置参考
ARCHITECTURE.md — 三层记忆架构详解（STM/LTM/Temporal KG）、RAG for Tools 两阶段路由流程、MCP SSE 通信机制与连接池策略、LangGraph 节点拓扑与条件边、数据持久化拓扑、生产防坑指南
OPERATIONS.md — 零停机热插拔操作手册（方案 A 刷新 / 方案 B 全新接入），含完整 curl 命令示例、Prompt 热加载、对话接口用法、故障排查 Checklist、Mermaid 时序图
Embedding_Model_Guide.md — Embedding 模型加载指南，以项目当前使用的 paraphrase-multilingual-MiniLM-L12-v2 为例，涵盖离线环境配置、本地缓存确认、单例加载模式、维度一致性检查、一步步加载流程、常见问题排查及更换模型检查清单