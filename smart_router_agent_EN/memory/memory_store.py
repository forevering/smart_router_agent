"""
Dual-Layer Long Term Memory (LTM) Persistent Storage Module

Architecture:
  L1 (Raw Conversations)  → SQLite conversations table
  L2 (Extracted Facts)    → Qdrant ltm_facts collection + embedding semantic index

L1 Responsibilities:
  - Store complete conversation records (role + content + timestamp)
  - Support precise queries by thread_id / time range
  - Serve as provenance source for L2 facts

L2 Responsibilities:
  - Use LLM to extract atomized facts from conversations
  - Store each fact as embedding vector + metadata (user_id, source_conversation_id, created_at)
  - Support semantic similarity top-k retrieval + metadata filter

Retrieval Flow:
  query → embedding(query) → Qdrant semantic top-k (user_id filter) → related facts
                                                                    ↓ (on demand)
                                         via source_conversation_id → L1 raw conversations
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


def _parse_json_safe(text: str):
    """Safely parse JSON returned by LLM, handling markdown code block wrapping"""
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
        for start_char, end_char in [("[", "]"), ("{", "}")]:
            start = text.find(start_char)
            end = text.rfind(end_char)
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    continue
        return None


class MemoryStore:
    """
    Dual-layer long-term memory storage.
    L1: SQLite conversations table (raw conversation archive + provenance)
    L2: Qdrant ltm_facts collection (facts + embedding semantic index)
    """

    COLLECTION_NAME = "ltm_facts"

    def __init__(self, db_path: str, qdrant_client: QdrantClient, embedding_model):
        """
        Args:
            db_path: SQLite database file path
            qdrant_client: Shared Qdrant client instance
            embedding_model: SentenceTransformer model instance
        """
        self.db_path = db_path
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_model.get_sentence_embedding_dimension()
        self.qdrant = qdrant_client

    async def init_db(self):
        """Initialize SQLite L1 table and Qdrant L2 collection"""
        # ---- L1: SQLite conversations table ----
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    thread_id TEXT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_conv_thread ON conversations(thread_id)"
            )
            await db.commit()

        # ---- L2: Qdrant collection ----
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
    # L1: Raw Conversation Storage and Retrieval
    # ================================================================

    async def store_conversation(
        self, user_id: str, thread_id: str, messages: list
    ) -> List[str]:
        """
        Store current turn's conversation messages into L1 conversations table.

        Args:
            user_id: User ID
            thread_id: Conversation thread ID
            messages: LangChain Message object list

        Returns:
            List of stored message IDs (used for L2 fact provenance)
        """
        now = datetime.now().isoformat()
        msg_ids = []
        async with aiosqlite.connect(self.db_path) as db:
            for m in messages:
                msg_id = str(uuid.uuid4())
                role = getattr(m, 'type', 'unknown')
                content = getattr(m, 'content', str(m))
                await db.execute(
                    "INSERT OR IGNORE INTO conversations (id, user_id, thread_id, role, content, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (msg_id, user_id, thread_id, role, content, now)
                )
                msg_ids.append(msg_id)
            await db.commit()
        return msg_ids

    async def retrieve_conversations(
        self, user_id: str, thread_id: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, str]]:
        """
        Retrieve user's raw conversation records.

        Args:
            user_id: User ID
            thread_id: Optional, specify thread
            limit: Maximum number of records to return

        Returns:
            Conversation record list [{role, content, created_at}, ...]
        """
        async with aiosqlite.connect(self.db_path) as db:
            if thread_id:
                cursor = await db.execute(
                    "SELECT role, content, created_at FROM conversations "
                    "WHERE user_id = ? AND thread_id = ? ORDER BY created_at DESC LIMIT ?",
                    (user_id, thread_id, limit)
                )
            else:
                cursor = await db.execute(
                    "SELECT role, content, created_at FROM conversations "
                    "WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                    (user_id, limit)
                )
            rows = await cursor.fetchall()
            return [{"role": row[0], "content": row[1], "created_at": row[2]} for row in rows]

    # ================================================================
    # L2: Fact Extraction + Qdrant Storage + Semantic Retrieval
    # ================================================================

    async def extract_facts(self, messages: list, llm: AzureChatOpenAI, prompt_loader=None) -> List[str]:
        """
        Use LLM to extract atomized facts from conversation messages.

        Args:
            messages: LangChain Message object list
            llm: AzureChatOpenAI instance
            prompt_loader: PromptLoader instance (optional; if provided, reads prompt from YAML)

        Returns:
            List of extracted fact strings
        """
        conversation_lines = []
        for m in messages:
            role = getattr(m, 'type', 'unknown')
            content = getattr(m, 'content', str(m))
            conversation_lines.append(f"{role}: {content}")
        conversation = "\n".join(conversation_lines)

        if prompt_loader:
            prompt = prompt_loader.format("summarizer", conversation=conversation)
        else:
            prompt = (
                "You are an information extraction expert. Please extract key factual information about the [user] from the following conversation.\n\n"
                "Focus on what the user (human) said, and extract:\n"
                "- User's plans (e.g., business trips, travel, meetings)\n"
                "- User's preferences (e.g., dietary, transportation mode)\n"
                "- User's identity information (e.g., company, position, city)\n"
                "- Interpersonal relationships mentioned by the user\n"
                "- Any personal information worth remembering long-term\n\n"
                "Requirements:\n"
                "1. Each fact must be an atomized, self-contained declarative sentence\n"
                "2. Only focus on what the user said; do not extract from assistant (ai/assistant) replies\n"
                "3. Even if there is only one fact, extract it\n"
                "4. You MUST output in English only. Do NOT use Chinese or any other language.\n\n"
                f"Conversation content:\n{conversation}\n\n"
                "Please output only in JSON array format, no other text. If there are facts, extract them, e.g.:\n"
                '["User is going on a business trip to Shanghai next week", "User plans to stay in Shanghai for three days"]'
            )

        try:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            facts = _parse_json_safe(response.content)
            if isinstance(facts, list):
                return [str(f) for f in facts if f]
            return []
        except Exception as e:
            print(f"  [LTM] Fact extraction failed: {e}")
            return []

    def store_facts(
        self,
        user_id: str,
        facts: List[str],
        source_conversation_ids: Optional[List[str]] = None,
    ) -> int:
        """
        Store extracted facts into Qdrant L2 with embedding and metadata.
        Synchronous method (Qdrant local client is synchronous and extremely fast).

        Auto-deduplication: for the same user_id, if a fact with similarity > 0.95 already exists, skip it.

        Args:
            user_id: User ID
            facts: List of fact strings
            source_conversation_ids: Provenance conversation ID list (corresponds 1:1 with facts)

        Returns:
            Number of new facts actually stored
        """
        if not facts:
            return 0

        now = datetime.now().isoformat()
        stored_count = 0

        for i, fact in enumerate(facts):
            # Compute embedding
            embedding = self.embedding_model.encode(fact).tolist()

            # Deduplication check: query Qdrant for highly similar existing facts
            existing = self.qdrant.query_points(
                collection_name=self.COLLECTION_NAME,
                query=embedding,
                query_filter=Filter(
                    must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
                ),
                limit=1,
            ).points
            if existing and existing[0].score > 0.95:
                continue  # Highly duplicated, skip

            # Build point
            source_id = source_conversation_ids[i] if source_conversation_ids and i < len(source_conversation_ids) else ""
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "user_id": user_id,
                    "fact": fact,
                    "source_conversation_id": source_id,
                    "created_at": now,
                },
            )
            self.qdrant.upsert(
                collection_name=self.COLLECTION_NAME,
                points=[point],
            )
            stored_count += 1

        return stored_count

    def retrieve(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        min_score: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve user's long-term memory facts based on semantic similarity (L2 layer).

        Args:
            user_id: User ID
            query: Current query text
            top_k: Return at most k results
            min_score: Minimum similarity threshold

        Returns:
            [{fact, score, source_conversation_id, created_at}, ...]
            Sorted by similarity in descending order
        """
        query_embedding = self.embedding_model.encode(query).tolist()

        results = self.qdrant.query_points(
            collection_name=self.COLLECTION_NAME,
            query=query_embedding,
            query_filter=Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            ),
            limit=top_k,
        ).points

        return [
            {
                "fact": hit.payload["fact"],
                "score": hit.score,
                "source_conversation_id": hit.payload.get("source_conversation_id", ""),
                "created_at": hit.payload.get("created_at", ""),
            }
            for hit in results
            if hit.score >= min_score
        ]

    async def retrieve_source_conversation(
        self, conversation_id: str
    ) -> Optional[Dict[str, str]]:
        """
        On-demand backtracking: retrieve original conversation from L1 by source_conversation_id.

        Args:
            conversation_id: ID from the conversations table

        Returns:
            {role, content, created_at} or None
        """
        if not conversation_id:
            return None
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT role, content, created_at FROM conversations WHERE id = ?",
                (conversation_id,)
            )
            row = await cursor.fetchone()
            if row:
                return {"role": row[0], "content": row[1], "created_at": row[2]}
            return None
