"""
Tool Registry 模块 (无状态化 — 全量持久化到 Qdrant)

所有工具的注册中心，支持：
- Tool 元数据管理 (name, mcp_server_name, schema, search_document)
- search_document Embedding 向量化并持久化到 Qdrant
- 完整工具定义（含序列化 schema）存入 Qdrant payload
- 基于 Embedding 的 Top-K 语义检索
- 所有查询直接走 Qdrant，无内存字典，重启后数据不丢失
"""
import json
import uuid
import threading
from typing import Dict, List, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    ScrollRequest,
)


class ToolDefinition:
    """单个工具的完整定义"""

    def __init__(
        self,
        name: str,
        mcp_server_name: str,
        schema: Dict[str, Any],
        search_document: str,
    ):
        self.name = name
        self.mcp_server_name = mcp_server_name
        self.schema = schema
        self.search_document = search_document

    def to_payload(self) -> Dict[str, Any]:
        """序列化为 Qdrant payload"""
        return {
            "tool_name": self.name,
            "mcp_server_name": self.mcp_server_name,
            "schema_json": json.dumps(self.schema, ensure_ascii=False),
            "search_document": self.search_document,
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "ToolDefinition":
        """从 Qdrant payload 反序列化"""
        return cls(
            name=payload["tool_name"],
            mcp_server_name=payload["mcp_server_name"],
            schema=json.loads(payload.get("schema_json", "{}")),
            search_document=payload.get("search_document", ""),
        )


def _tool_point_id(tool_name: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"tool:{tool_name}"))


class ToolRegistry:
    """
    工具注册中心（彻底无状态化）。
    所有工具元数据 + Embedding 持久化在 Qdrant 中，
    重启后自动从 Qdrant 恢复，无需重新注册。
    """

    COLLECTION_NAME = "tool_registry"

    def __init__(self, qdrant_client: QdrantClient, embedding_model):
        self._qdrant = qdrant_client
        self._embedding_model = embedding_model
        self._embedding_dim = embedding_model.get_sentence_embedding_dimension()
        self._lock = threading.RLock()
        self._init_collection()

    def _init_collection(self):
        existing = [c.name for c in self._qdrant.get_collections().collections]
        if self.COLLECTION_NAME not in existing:
            self._qdrant.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self._embedding_dim,
                    distance=Distance.COSINE,
                ),
            )

    # ================================================================
    # 写操作
    # ================================================================

    def register(self, tool: ToolDefinition):
        """注册单个工具（Embedding + 全量 payload 持久化到 Qdrant）"""
        with self._lock:
            self._upsert_embedding(tool)
        print(f"  [ToolRegistry] 已注册工具: {tool.name} (server: {tool.mcp_server_name})")

    def register_batch(self, tools: List[ToolDefinition]):
        """批量注册/更新工具"""
        if not tools:
            return
        with self._lock:
            points = []
            for t in tools:
                embedding = self._embedding_model.encode(t.search_document).tolist()
                points.append(PointStruct(
                    id=_tool_point_id(t.name),
                    vector=embedding,
                    payload=t.to_payload(),
                ))
            self._qdrant.upsert(collection_name=self.COLLECTION_NAME, points=points)
        print(f"  [ToolRegistry] 批量注册/更新 {len(tools)} 个工具")

    def unregister(self, tool_name: str):
        """从 Qdrant 中移除工具"""
        with self._lock:
            try:
                self._qdrant.delete(
                    collection_name=self.COLLECTION_NAME,
                    points_selector=[_tool_point_id(tool_name)],
                )
                print(f"  [ToolRegistry] 已移除工具: {tool_name}")
            except Exception:
                pass

    # ================================================================
    # 读操作 — 全部直接查 Qdrant
    # ================================================================

    def get_tool(self, tool_name: str) -> Optional[ToolDefinition]:
        """按名称精确查询工具"""
        results = self._qdrant.scroll(
            collection_name=self.COLLECTION_NAME,
            scroll_filter=Filter(
                must=[FieldCondition(key="tool_name", match=MatchValue(value=tool_name))]
            ),
            limit=1,
        )[0]
        if results:
            return ToolDefinition.from_payload(results[0].payload)
        return None

    def get_server_for_tool(self, tool_name: str) -> Optional[str]:
        """查询工具所属的 MCP Server"""
        t = self.get_tool(tool_name)
        return t.mcp_server_name if t else None

    def get_all_tools_for_server(self, server_name: str) -> List[ToolDefinition]:
        """获取某个 Server 下的所有工具"""
        results = self._qdrant.scroll(
            collection_name=self.COLLECTION_NAME,
            scroll_filter=Filter(
                must=[FieldCondition(key="mcp_server_name", match=MatchValue(value=server_name))]
            ),
            limit=100,
        )[0]
        return [ToolDefinition.from_payload(p.payload) for p in results]

    def remove_tool(self, tool_name: str, server_name: str):
        """从 Qdrant 中删除指定工具"""
        point_id = self._tool_point_id(tool_name, server_name)
        self._qdrant.delete(
            collection_name=self.COLLECTION_NAME,
            points_selector=[point_id],
        )

    def search(self, query: str, top_k: int = 3) -> List[ToolDefinition]:
        """基于 query Embedding 语义检索 Top-K 工具"""
        query_embedding = self._embedding_model.encode(query).tolist()
        results = self._qdrant.query_points(
            collection_name=self.COLLECTION_NAME,
            query=query_embedding,
            limit=top_k,
        ).points
        return [ToolDefinition.from_payload(hit.payload) for hit in results]

    # ================================================================
    # 内部
    # ================================================================

    def _upsert_embedding(self, tool: ToolDefinition):
        embedding = self._embedding_model.encode(tool.search_document).tolist()
        self._qdrant.upsert(
            collection_name=self.COLLECTION_NAME,
            points=[PointStruct(
                id=_tool_point_id(tool.name),
                vector=embedding,
                payload=tool.to_payload(),
            )],
        )
