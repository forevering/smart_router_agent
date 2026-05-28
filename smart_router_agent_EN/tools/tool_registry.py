"""
Tool Registry Module (Stateless — fully persisted to Qdrant)

Central registry for all tools, supporting:
- Tool metadata management (name, mcp_server_name, schema, search_document)
- search_document embedding vectorization and persistence to Qdrant
- Complete tool definitions (including serialized schema) stored in Qdrant payload
- Embedding-based Top-K semantic retrieval
- All queries go directly through Qdrant; no in-memory dictionaries; data persists across restarts
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
    """Complete definition of a single tool"""

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
        """Serialize to Qdrant payload"""
        return {
            "tool_name": self.name,
            "mcp_server_name": self.mcp_server_name,
            "schema_json": json.dumps(self.schema, ensure_ascii=False),
            "search_document": self.search_document,
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "ToolDefinition":
        """Deserialize from Qdrant payload"""
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
    Tool Registry (fully stateless).
    All tool metadata + embeddings are persisted in Qdrant;
    automatically recovers from Qdrant on restart without re-registration.
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
    # Write Operations
    # ================================================================

    def register(self, tool: ToolDefinition):
        """Register a single tool (Embedding + full payload persisted to Qdrant)"""
        with self._lock:
            self._upsert_embedding(tool)
        print(f"  [ToolRegistry] Registered tool: {tool.name} (server: {tool.mcp_server_name})")

    def register_batch(self, tools: List[ToolDefinition]):
        """Batch register/update tools"""
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
        print(f"  [ToolRegistry] Batch registered/updated {len(tools)} tools")

    def unregister(self, tool_name: str):
        """Remove a tool from Qdrant"""
        with self._lock:
            try:
                self._qdrant.delete(
                    collection_name=self.COLLECTION_NAME,
                    points_selector=[_tool_point_id(tool_name)],
                )
                print(f"  [ToolRegistry] Removed tool: {tool_name}")
            except Exception:
                pass

    # ================================================================
    # Read Operations — all go directly through Qdrant
    # ================================================================

    def get_tool(self, tool_name: str) -> Optional[ToolDefinition]:
        """Exact query for a tool by name"""
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
        """Query the MCP Server a tool belongs to"""
        t = self.get_tool(tool_name)
        return t.mcp_server_name if t else None

    def get_all_tools_for_server(self, server_name: str) -> List[ToolDefinition]:
        """Get all tools under a given Server"""
        results = self._qdrant.scroll(
            collection_name=self.COLLECTION_NAME,
            scroll_filter=Filter(
                must=[FieldCondition(key="mcp_server_name", match=MatchValue(value=server_name))]
            ),
            limit=100,
        )[0]
        return [ToolDefinition.from_payload(p.payload) for p in results]

    def remove_tool(self, tool_name: str, server_name: str):
        """Delete the specified tool from Qdrant"""
        point_id = self._tool_point_id(tool_name, server_name)
        self._qdrant.delete(
            collection_name=self.COLLECTION_NAME,
            points_selector=[point_id],
        )

    def search(self, query: str, top_k: int = 3) -> List[ToolDefinition]:
        """Semantic retrieval of Top-K tools based on query embedding"""
        query_embedding = self._embedding_model.encode(query).tolist()
        results = self._qdrant.query_points(
            collection_name=self.COLLECTION_NAME,
            query=query_embedding,
            limit=top_k,
        ).points
        return [ToolDefinition.from_payload(hit.payload) for hit in results]

    # ================================================================
    # Internal
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
