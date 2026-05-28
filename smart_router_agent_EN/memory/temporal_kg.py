"""
Temporal Knowledge Graph (Temporal KG) Persistent Storage Module

Architecture:
  Semantic Entry Layer → Qdrant kg_nodes collection (node embeddings for semantic seed matching)
  Relation + Temporal Layer → SQLite kg_nodes / kg_relations tables (structured storage, supports N-hop expansion)

Retrieval Strategy (semantic seed + bounded graph expansion):
  1. query → embedding → Qdrant top-k matching entry entities (score > threshold)
  2. Entry entities → SQLite 1-hop expansion (with temporal filtering)
  3. If 1-hop neighbors contain entities with high similarity to query → auto-append 1-hop (total 2-hop)
  4. Hard cap MAX_HOPS = 2, enforced at code level

Schema Conventions:
  - active_start: system time at storage
  - active_end: defaults to NULL (currently valid); set to current time when invalidated
  - Query rule: active_start <= reference_time < (active_end or ∞)
  - Physical deletion of old data is strictly prohibited; only logical invalidation
"""
import json
import uuid
import aiosqlite
from datetime import datetime
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage

from smart_router_agent_EN.config.config import MAX_HOPS, SIMILARITY_THRESHOLD, TOP_K


def _parse_json_safe(text: str):
    """Safely parse JSON returned by LLM, handling markdown code blocks"""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        end_idx = len(lines) - 1
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip() == "```":
                end_idx = i
                break
        text = "\n".join(lines[1:end_idx]).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            start = text.find(start_char)
            end = text.rfind(end_char)
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    continue
        return None


class TemporalKG:
    """
    Temporal Knowledge Graph.
    Semantic entry: Qdrant kg_nodes collection
    Relation storage: SQLite kg_nodes + kg_relations tables
    """

    COLLECTION_NAME = "kg_nodes"

    def __init__(self, db_path: str, qdrant_client: QdrantClient, embedding_model):
        """
        Args:
            db_path: SQLite database file path
            qdrant_client: Shared Qdrant client instance (shared with MemoryStore)
            embedding_model: SentenceTransformer model instance
        """
        self.db_path = db_path
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_model.get_sentence_embedding_dimension()
        self.qdrant = qdrant_client

    async def init_db(self):
        """Initialize SQLite tables and Qdrant collection"""
        async with aiosqlite.connect(self.db_path) as db:
            # ---- SQLite node table ----
            await db.execute("""
                CREATE TABLE IF NOT EXISTS kg_nodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    entity TEXT NOT NULL,
                    entity_type TEXT,
                    properties TEXT,
                    active_start TEXT NOT NULL,
                    active_end TEXT
                )
            """)
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_kgn_user ON kg_nodes(user_id)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_kgn_entity ON kg_nodes(user_id, entity)"
            )

            # ---- SQLite relation table ----
            await db.execute("""
                CREATE TABLE IF NOT EXISTS kg_relations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    target TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    properties TEXT,
                    active_start TEXT NOT NULL,
                    active_end TEXT
                )
            """)
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_kgr_user ON kg_relations(user_id)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_kgr_source ON kg_relations(user_id, source)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_kgr_target ON kg_relations(user_id, target)"
            )
            await db.commit()

        # ---- Qdrant collection (KG node semantic entry) ----
        existing = [c.name for c in self.qdrant.get_collections().collections]
        if self.COLLECTION_NAME not in existing:
            self.qdrant.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )

    # ================================================================
    # CRUD Operations
    # ================================================================

    async def add_node(
        self,
        user_id: str,
        entity: str,
        entity_type: str = "unknown",
        properties: Optional[Dict] = None,
    ):
        """
        Insert a new node into SQLite + Qdrant.
        SQLite: structured storage (temporal management)
        Qdrant: embedding semantic index (retrieval entry)
        """
        now = datetime.now().isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            # Idempotency check
            cursor = await db.execute(
                "SELECT id FROM kg_nodes WHERE user_id = ? AND entity = ? AND active_end IS NULL",
                (user_id, entity)
            )
            if await cursor.fetchone():
                return  # Active node already exists

            await db.execute(
                "INSERT INTO kg_nodes (user_id, entity, entity_type, properties, active_start, active_end) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, entity, entity_type, json.dumps(properties or {}, ensure_ascii=False), now, None)
            )
            await db.commit()

        # Synchronous write to Qdrant (embedding semantic index)
        embedding = self.embedding_model.encode(entity).tolist()
        # Use deterministic ID: hash of user_id + entity to ensure idempotency
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{user_id}:{entity}"))
        self.qdrant.upsert(
            collection_name=self.COLLECTION_NAME,
            points=[PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "user_id": user_id,
                    "entity": entity,
                    "entity_type": entity_type,
                },
            )],
        )

    async def add_relation(
        self,
        user_id: str,
        source: str,
        target: str,
        relation_type: str,
        properties: Optional[Dict] = None,
    ):
        """
        Insert a new relation into SQLite.
        Conflict handling: when same source + relation_type but different target, old relation is logically invalidated.
        Physical deletion of old data is strictly prohibited.
        """
        now = datetime.now().isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            # ---- Conflict detection ----
            cursor = await db.execute(
                "SELECT id, target FROM kg_relations "
                "WHERE user_id = ? AND source = ? AND relation_type = ? AND active_end IS NULL",
                (user_id, source, relation_type)
            )
            for row_id, existing_target in await cursor.fetchall():
                if existing_target != target:
                    await db.execute(
                        "UPDATE kg_relations SET active_end = ? WHERE id = ?",
                        (now, row_id)
                    )

            # ---- Idempotency check ----
            cursor = await db.execute(
                "SELECT id FROM kg_relations "
                "WHERE user_id = ? AND source = ? AND target = ? AND relation_type = ? AND active_end IS NULL",
                (user_id, source, target, relation_type)
            )
            if await cursor.fetchone():
                await db.commit()
                return

            # ---- Insert ----
            await db.execute(
                "INSERT INTO kg_relations "
                "(user_id, source, target, relation_type, properties, active_start, active_end) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (user_id, source, target, relation_type,
                 json.dumps(properties or {}, ensure_ascii=False), now, None)
            )
            await db.commit()

    async def invalidate_node(self, user_id: str, entity: str):
        """Logically invalidate the specified node (SQLite active_end + Qdrant retain but mark)"""
        now = datetime.now().isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE kg_nodes SET active_end = ? "
                "WHERE user_id = ? AND entity = ? AND active_end IS NULL",
                (now, user_id, entity)
            )
            await db.commit()
        # Remove the node's semantic entry from Qdrant (invalidated nodes should not be retrievable)
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{user_id}:{entity}"))
        try:
            self.qdrant.delete(
                collection_name=self.COLLECTION_NAME,
                points_selector=[point_id],
            )
        except Exception:
            pass  # Node may not exist in Qdrant

    async def invalidate_relation(
        self, user_id: str, source: str, target: str, relation_type: str
    ):
        """Logically invalidate the specified relation"""
        now = datetime.now().isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE kg_relations SET active_end = ? "
                "WHERE user_id = ? AND source = ? AND target = ? AND relation_type = ? AND active_end IS NULL",
                (now, user_id, source, target, relation_type)
            )
            await db.commit()

    # ================================================================
    # Query: Full Snapshot (for LLM context / debugging)
    # ================================================================

    async def query_snapshot(
        self, user_id: str, reference_time: Optional[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Query the full KG snapshot valid at reference_time.
        Used for LLM context building and debug display.
        """
        if reference_time is None:
            reference_time = datetime.now().isoformat()

        results = {"nodes": [], "relations": []}
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT entity, entity_type, properties FROM kg_nodes "
                "WHERE user_id = ? AND active_start <= ? AND (active_end IS NULL OR active_end > ?)",
                (user_id, reference_time, reference_time)
            )
            for row in await cursor.fetchall():
                results["nodes"].append({
                    "entity": row[0],
                    "entity_type": row[1],
                    "properties": json.loads(row[2]) if row[2] else {}
                })

            cursor = await db.execute(
                "SELECT source, target, relation_type, properties FROM kg_relations "
                "WHERE user_id = ? AND active_start <= ? AND (active_end IS NULL OR active_end > ?)",
                (user_id, reference_time, reference_time)
            )
            for row in await cursor.fetchall():
                results["relations"].append({
                    "source": row[0],
                    "target": row[1],
                    "relation_type": row[2],
                    "properties": json.loads(row[3]) if row[3] else {}
                })
        return results

    # ================================================================
    # Query: Semantic Seed + Bounded N-hop Graph Expansion
    # ================================================================

    async def semantic_search(
        self,
        user_id: str,
        query: str,
        reference_time: Optional[str] = None,
        top_k: int = TOP_K,
        min_score: float = SIMILARITY_THRESHOLD,
    ) -> Dict[str, Any]:
        """
        Core retrieval method: semantic seed + bounded graph expansion.

        Flow:
        1. embedding(query) → Qdrant top-k matching entry entities
        2. Entry entities → SQLite 1-hop expansion (with temporal filtering)
        3. If 1-hop neighbors have high-similarity entities → append 1-hop (total 2-hop)
        4. Hard cap MAX_HOPS = 2

        Args:
            user_id: User ID
            query: Current query text
            reference_time: Reference time (defaults to current time)
            top_k: Entry entity top-k
            min_score: Minimum similarity threshold

        Returns:
            {
                "seed_entities": [{"entity": ..., "score": ...}],
                "subgraph_nodes": [...],
                "subgraph_relations": [...],
                "hops_used": 1 or 2
            }
        """
        if reference_time is None:
            reference_time = datetime.now().isoformat()

        # ---- Step 1: Qdrant semantic matching for entry entities ----
        query_embedding = self.embedding_model.encode(query).tolist()
        qdrant_results = self.qdrant.query_points(
            collection_name=self.COLLECTION_NAME,
            query=query_embedding,
            query_filter=Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            ),
            limit=top_k,
        ).points

        seed_entities = [
            {"entity": hit.payload["entity"], "score": hit.score}
            for hit in qdrant_results
            if hit.score >= min_score
        ]

        if not seed_entities:
            return {
                "seed_entities": [],
                "subgraph_nodes": [],
                "subgraph_relations": [],
                "hops_used": 0,
            }

        # ---- Step 2: 1-hop expansion ----
        seed_names = {s["entity"] for s in seed_entities}
        hop1_relations, hop1_neighbors = await self._expand_one_hop(
            user_id, seed_names, reference_time
        )

        all_relations = list(hop1_relations)
        all_entity_names = seed_names | hop1_neighbors
        hops_used = 1

        # ---- Step 3: Check if 1-hop neighbors have high-similarity entities, decide whether to append 2-hop ----
        if hop1_neighbors and hops_used < MAX_HOPS:
            # Compute similarity between 1-hop neighbor names and query
            neighbor_names = list(hop1_neighbors - seed_names)
            if neighbor_names:
                neighbor_embeddings = self.embedding_model.encode(neighbor_names)
                import numpy as np
                query_emb_np = np.array(query_embedding)
                scores = np.dot(neighbor_embeddings, query_emb_np) / (
                    np.linalg.norm(neighbor_embeddings, axis=1) * np.linalg.norm(query_emb_np)
                )
                # Find neighbors exceeding threshold
                expand_entities = set()
                for name, score in zip(neighbor_names, scores):
                    if score >= min_score:
                        expand_entities.add(name)

                if expand_entities:
                    hop2_relations, hop2_neighbors = await self._expand_one_hop(
                        user_id, expand_entities, reference_time
                    )
                    all_relations.extend(hop2_relations)
                    all_entity_names |= hop2_neighbors
                    hops_used = 2

        # ---- Deduplicate relations ----
        seen_rels = set()
        unique_relations = []
        for rel in all_relations:
            key = (rel["source"], rel["target"], rel["relation_type"])
            if key not in seen_rels:
                seen_rels.add(key)
                unique_relations.append(rel)

        # ---- Get node info for all entities ----
        subgraph_nodes = await self._get_nodes_by_names(
            user_id, all_entity_names, reference_time
        )

        return {
            "seed_entities": seed_entities,
            "subgraph_nodes": subgraph_nodes,
            "subgraph_relations": unique_relations,
            "hops_used": hops_used,
        }

    async def _expand_one_hop(
        self,
        user_id: str,
        entity_names: set,
        reference_time: str,
    ) -> tuple:
        """
        Expand 1 hop from the given entity set (bidirectional: outgoing + incoming edges).

        Returns:
            (relations_list, neighbor_entity_names_set)
        """
        relations = []
        neighbors = set()

        if not entity_names:
            return relations, neighbors

        # Build IN clause placeholders
        placeholders = ",".join(["?" for _ in entity_names])
        names_list = list(entity_names)
        time_params = [reference_time, reference_time]

        async with aiosqlite.connect(self.db_path) as db:
            # ---- Outgoing edges: source IN entity_names ----
            cursor = await db.execute(
                f"SELECT source, target, relation_type, properties FROM kg_relations "
                f"WHERE user_id = ? AND source IN ({placeholders}) "
                f"AND active_start <= ? AND (active_end IS NULL OR active_end > ?)",
                [user_id] + names_list + time_params
            )
            for row in await cursor.fetchall():
                rel = {
                    "source": row[0], "target": row[1],
                    "relation_type": row[2],
                    "properties": json.loads(row[3]) if row[3] else {}
                }
                relations.append(rel)
                neighbors.add(row[0])
                neighbors.add(row[1])

            # ---- Incoming edges: target IN entity_names ----
            cursor = await db.execute(
                f"SELECT source, target, relation_type, properties FROM kg_relations "
                f"WHERE user_id = ? AND target IN ({placeholders}) "
                f"AND active_start <= ? AND (active_end IS NULL OR active_end > ?)",
                [user_id] + names_list + time_params
            )
            for row in await cursor.fetchall():
                rel = {
                    "source": row[0], "target": row[1],
                    "relation_type": row[2],
                    "properties": json.loads(row[3]) if row[3] else {}
                }
                relations.append(rel)
                neighbors.add(row[0])
                neighbors.add(row[1])

        return relations, neighbors

    async def _get_nodes_by_names(
        self,
        user_id: str,
        entity_names: set,
        reference_time: str,
    ) -> List[Dict[str, Any]]:
        """Get valid node info for the specified entity names"""
        if not entity_names:
            return []

        placeholders = ",".join(["?" for _ in entity_names])
        names_list = list(entity_names)

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                f"SELECT entity, entity_type, properties FROM kg_nodes "
                f"WHERE user_id = ? AND entity IN ({placeholders}) "
                f"AND active_start <= ? AND (active_end IS NULL OR active_end > ?)",
                [user_id] + names_list + [reference_time, reference_time]
            )
            return [
                {
                    "entity": row[0],
                    "entity_type": row[1],
                    "properties": json.loads(row[2]) if row[2] else {}
                }
                for row in await cursor.fetchall()
            ]

    # ================================================================
    # LLM-Driven Extraction and Update
    # ================================================================

    async def extract_and_update(
        self, user_id: str, messages: list, llm: AzureChatOpenAI, prompt_loader=None
    ):
        """
        Use LLM to extract entities and relations from the current conversation turn,
        and update the temporal knowledge graph.
        Updates both SQLite (structured) and Qdrant (semantic index).
        """
        existing_kg = await self.query_snapshot(user_id)

        conversation_lines = []
        for m in messages:
            role = getattr(m, 'type', 'unknown')
            content = getattr(m, 'content', str(m))
            conversation_lines.append(f"{role}: {content}")
        conversation = "\n".join(conversation_lines)

        if prompt_loader:
            prompt = prompt_loader.format(
                "kg_manager",
                existing_kg=json.dumps(existing_kg, ensure_ascii=False, indent=2),
                conversation=conversation,
            )
        else:
            prompt = (
                "You are a knowledge graph extraction expert. Please extract entities and relations only from what the [user (human)] said.\n"
                "Do not extract information from assistant (ai/assistant) replies.\n\n"
                f"Existing knowledge graph:\n{json.dumps(existing_kg, ensure_ascii=False, indent=2)}\n\n"
                f"Current conversation:\n{conversation}\n\n"
                "Please output only the following JSON format, no other text:\n"
                "{\n"
                '  "nodes": [\n'
                '    {"entity": "entity_name", "entity_type": "person|location|event|organization|time|other", "properties": {}}\n'
                "  ],\n"
                '  "relations": [\n'
                '    {"source": "entity1", "target": "entity2", "relation_type": "relation_type", "properties": {}}\n'
                "  ],\n"
                '  "invalidations": [\n'
                '    {"source": "entity1", "target": "entity2", "relation_type": "relation_type"}\n'
                "  ]\n"
                "}\n\n"
                "Notes:\n"
                "- Only extract facts explicitly stated by the user (human); do not extract content from assistant suggestions\n"
                "- Use 'user' to represent the user entity itself\n"
                "- Entity and relation naming should be concise and consistent\n"
                "- Properties can include supplementary information such as time and location\n"
                "- Only add invalidations when knowledge has actually changed\n"
                "- If there is no new information to extract, return empty arrays\n"
                "- You MUST output all entity names, relation types, and property values in English only. Do NOT use Chinese."
            )

        try:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            result = _parse_json_safe(response.content)
            if not isinstance(result, dict):
                print(f"  [KG] Extraction result parsing failed")
                return
        except Exception as e:
            print(f"  [KG] Extraction failed: {e}")
            return

        # Step 1: Invalidation operations
        invalidations = result.get("invalidations", [])
        for inv in invalidations:
            await self.invalidate_relation(
                user_id, inv["source"], inv["target"], inv["relation_type"]
            )

        # Step 2: Insert new nodes (SQLite + Qdrant)
        new_nodes = result.get("nodes", [])
        for node in new_nodes:
            await self.add_node(
                user_id,
                node["entity"],
                node.get("entity_type", "unknown"),
                node.get("properties")
            )

        # Step 3: Insert new relations (SQLite)
        new_relations = result.get("relations", [])
        for rel in new_relations:
            await self.add_relation(
                user_id,
                rel["source"], rel["target"],
                rel["relation_type"],
                rel.get("properties")
            )

        print(
            f"  [KG] Update complete: "
            f"invalidated {len(invalidations)} relations, "
            f"added {len(new_nodes)} nodes, "
            f"added {len(new_relations)} relations"
        )
