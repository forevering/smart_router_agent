"""
双层长期记忆 (Long Term Memory) 持久化存储模块

架构：
  L1 (原始对话)  → SQLite conversations 表
  L2 (提取事实)  → Qdrant ltm_facts collection + embedding 语义索引

L1 职责：
  - 存储完整对话记录（角色 + 内容 + 时间戳）
  - 支持按 thread_id / 时间范围精确查询
  - 作为 L2 事实的溯源来源

L2 职责：
  - 使用 LLM 从对话中提取原子化事实
  - 每条事实存 embedding 向量 + metadata (user_id, source_conversation_id, created_at)
  - 支持语义相似度 top-k 检索 + metadata filter

检索流程：
  query → embedding(query) → Qdrant 语义 top-k (user_id filter) → 相关事实
                                                                    ↓ (按需)
                                         通过 source_conversation_id → L1 原始对话
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
    """安全解析 LLM 返回的 JSON，处理 markdown 代码块包裹"""
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
    双层长期记忆存储。
    L1: SQLite conversations 表 (原始对话存档 + 溯源)
    L2: Qdrant ltm_facts collection (事实 + embedding 语义索引)
    """

    COLLECTION_NAME = "ltm_facts"

    def __init__(self, db_path: str, qdrant_client: QdrantClient, embedding_model):
        """
        Args:
            db_path: SQLite 数据库文件路径
            qdrant_client: 共享的 Qdrant 客户端实例
            embedding_model: SentenceTransformer 模型实例
        """
        self.db_path = db_path
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_model.get_sentence_embedding_dimension()
        self.qdrant = qdrant_client

    async def init_db(self):
        """初始化 SQLite L1 表和 Qdrant L2 collection"""
        # ---- L1: SQLite conversations 表 ----
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
    # L1: 原始对话存取
    # ================================================================

    async def store_conversation(
        self, user_id: str, thread_id: str, messages: list
    ) -> List[str]:
        """
        将本轮对话消息存入 L1 conversations 表。

        Args:
            user_id: 用户 ID
            thread_id: 会话线程 ID
            messages: LangChain Message 对象列表

        Returns:
            存储的消息 ID 列表（用于 L2 fact 溯源）
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
        检索用户的原始对话记录。

        Args:
            user_id: 用户 ID
            thread_id: 可选，指定线程
            limit: 最大返回条数

        Returns:
            对话记录列表 [{role, content, created_at}, ...]
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
    # L2: 事实提取 + Qdrant 存储 + 语义检索
    # ================================================================

    async def extract_facts(self, messages: list, llm: AzureChatOpenAI, prompt_loader=None) -> List[str]:
        """
        使用 LLM 从对话消息中提取原子化事实。

        Args:
            messages: LangChain Message 对象列表
            llm: AzureChatOpenAI 实例
            prompt_loader: PromptLoader 实例（可选，传入则从 YAML 读取 prompt）

        Returns:
            提取的事实字符串列表
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
                "你是一个信息提取专家。请从以下对话中提取关于【用户】的关键事实信息。\n\n"
                "重点关注用户（human）说的话，从中提取：\n"
                "- 用户的计划（如出差、旅行、会议）\n"
                "- 用户的偏好（如饮食、出行方式）\n"
                "- 用户的身份信息（如公司、职位、所在城市）\n"
                "- 用户提到的人物关系\n"
                "- 任何值得长期记忆的用户个人信息\n\n"
                "要求：\n"
                "1. 每条事实必须是原子化的、自包含的陈述句\n"
                "2. 只关注用户自己说的内容，不要从助手(ai/assistant)的回复中提取\n"
                "3. 即使只有一条事实，也要提取\n"
                "4. 用中文输出\n\n"
                f"对话内容：\n{conversation}\n\n"
                "请仅输出 JSON 数组格式，不要有其他文字。如果有事实就提取，例如：\n"
                '[\"用户下周要去上海出差\", \"用户计划在上海待三天\"]'
            )

        try:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            facts = _parse_json_safe(response.content)
            if isinstance(facts, list):
                return [str(f) for f in facts if f]
            return []
        except Exception as e:
            print(f"  [LTM] 事实提取失败: {e}")
            return []

    def store_facts(
        self,
        user_id: str,
        facts: List[str],
        source_conversation_ids: Optional[List[str]] = None,
    ) -> int:
        """
        将提取的事实存入 Qdrant L2，附带 embedding 和 metadata。
        同步方法（Qdrant local client 是同步的，操作极快）。

        自动去重：对同一 user_id，如果已存在相似度 > 0.95 的事实，跳过。

        Args:
            user_id: 用户 ID
            facts: 事实字符串列表
            source_conversation_ids: 溯源对话 ID 列表（与 facts 一一对应）

        Returns:
            实际存储的新事实数量
        """
        if not facts:
            return 0

        now = datetime.now().isoformat()
        stored_count = 0

        for i, fact in enumerate(facts):
            # 计算 embedding
            embedding = self.embedding_model.encode(fact).tolist()

            # 去重检查：查 Qdrant 中是否有高度相似的已有事实
            existing = self.qdrant.query_points(
                collection_name=self.COLLECTION_NAME,
                query=embedding,
                query_filter=Filter(
                    must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
                ),
                limit=1,
            ).points
            if existing and existing[0].score > 0.95:
                continue  # 高度重复，跳过

            # 构建 point
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
        基于语义相似度检索用户的长期记忆事实（L2 层）。

        Args:
            user_id: 用户 ID
            query: 当前查询文本
            top_k: 返回最多 k 条
            min_score: 最低相似度阈值

        Returns:
            [{fact, score, source_conversation_id, created_at}, ...]
            按相似度降序排列
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
        按需回溯：根据 source_conversation_id 从 L1 获取原始对话。

        Args:
            conversation_id: conversations 表的 id

        Returns:
            {role, content, created_at} 或 None
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
