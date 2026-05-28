"""
时序知识图谱 (Temporal Knowledge Graph) 持久化存储模块

架构：
  语义入口层 → Qdrant kg_nodes collection (节点 embedding，用于语义种子匹配)
  关系+时序层 → SQLite kg_nodes / kg_relations 表 (结构化存储，支持 N-hop 扩展)

检索策略（语义种子 + 受限图扩展）：
  1. query → embedding → Qdrant top-k 匹配入口实体 (score > threshold)
  2. 入口实体 → SQLite 1-hop 扩展 (带时序过滤)
  3. 如果 1-hop 邻居中存在与 query 高相似度的实体 → 自动追加 1-hop (总共 2-hop)
  4. 硬上限 MAX_HOPS = 2，代码层面封死

Schema 规范：
  - active_start: 存储时的系统时间
  - active_end: 默认 NULL (当前有效)，失效时设为当前时间
  - 查询规则: active_start <= reference_time < (active_end or ∞)
  - 严禁物理删除旧数据，只做逻辑失效
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

from smart_router_agent.config.config import MAX_HOPS, SIMILARITY_THRESHOLD, TOP_K


def _parse_json_safe(text: str):
    """安全解析 LLM 返回的 JSON，处理 markdown 代码块"""
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
    时序知识图谱。
    语义入口: Qdrant kg_nodes collection
    关系存储: SQLite kg_nodes + kg_relations 表
    """

    COLLECTION_NAME = "kg_nodes"

    def __init__(self, db_path: str, qdrant_client: QdrantClient, embedding_model):
        """
        Args:
            db_path: SQLite 数据库文件路径
            qdrant_client: 共享的 Qdrant 客户端实例（与 MemoryStore 共享）
            embedding_model: SentenceTransformer 模型实例
        """
        self.db_path = db_path
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_model.get_sentence_embedding_dimension()
        self.qdrant = qdrant_client

    async def init_db(self):
        """初始化 SQLite 表和 Qdrant collection"""
        async with aiosqlite.connect(self.db_path) as db:
            # ---- SQLite 节点表 ----
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

            # ---- SQLite 关系表 ----
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

        # ---- Qdrant collection（KG 节点语义入口）----
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
    # CRUD 操作
    # ================================================================

    async def add_node(
        self,
        user_id: str,
        entity: str,
        entity_type: str = "unknown",
        properties: Optional[Dict] = None,
    ):
        """
        插入新节点到 SQLite + Qdrant。
        SQLite: 结构化存储 (时序管理)
        Qdrant: embedding 语义索引 (检索入口)
        """
        now = datetime.now().isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            # 幂等检查
            cursor = await db.execute(
                "SELECT id FROM kg_nodes WHERE user_id = ? AND entity = ? AND active_end IS NULL",
                (user_id, entity)
            )
            if await cursor.fetchone():
                return  # 已存在有效节点

            await db.execute(
                "INSERT INTO kg_nodes (user_id, entity, entity_type, properties, active_start, active_end) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, entity, entity_type, json.dumps(properties or {}, ensure_ascii=False), now, None)
            )
            await db.commit()

        # 同步写入 Qdrant（embedding 语义索引）
        embedding = self.embedding_model.encode(entity).tolist()
        # 使用确定性 ID：基于 user_id + entity 的哈希，保证幂等
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
        插入新关系到 SQLite。
        冲突处理：同 source + relation_type 但不同 target 时，旧关系逻辑失效。
        严禁物理删除旧数据。
        """
        now = datetime.now().isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            # ---- 冲突检测 ----
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

            # ---- 幂等检查 ----
            cursor = await db.execute(
                "SELECT id FROM kg_relations "
                "WHERE user_id = ? AND source = ? AND target = ? AND relation_type = ? AND active_end IS NULL",
                (user_id, source, target, relation_type)
            )
            if await cursor.fetchone():
                await db.commit()
                return

            # ---- 插入 ----
            await db.execute(
                "INSERT INTO kg_relations "
                "(user_id, source, target, relation_type, properties, active_start, active_end) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (user_id, source, target, relation_type,
                 json.dumps(properties or {}, ensure_ascii=False), now, None)
            )
            await db.commit()

    async def invalidate_node(self, user_id: str, entity: str):
        """将指定节点逻辑失效（SQLite active_end + Qdrant 中保留但标记）"""
        now = datetime.now().isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE kg_nodes SET active_end = ? "
                "WHERE user_id = ? AND entity = ? AND active_end IS NULL",
                (now, user_id, entity)
            )
            await db.commit()
        # Qdrant 中删除该节点的语义入口（已失效的节点不应被检索到）
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{user_id}:{entity}"))
        try:
            self.qdrant.delete(
                collection_name=self.COLLECTION_NAME,
                points_selector=[point_id],
            )
        except Exception:
            pass  # 节点可能不存在于 Qdrant

    async def invalidate_relation(
        self, user_id: str, source: str, target: str, relation_type: str
    ):
        """将指定关系逻辑失效"""
        now = datetime.now().isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE kg_relations SET active_end = ? "
                "WHERE user_id = ? AND source = ? AND target = ? AND relation_type = ? AND active_end IS NULL",
                (now, user_id, source, target, relation_type)
            )
            await db.commit()

    # ================================================================
    # 查询：全量快照（用于 LLM 上下文 / 调试）
    # ================================================================

    async def query_snapshot(
        self, user_id: str, reference_time: Optional[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        查询 reference_time 时刻有效的 KG 全量快照。
        用于 LLM 上下文构建和调试展示。
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
    # 查询：语义种子 + 受限 N-hop 图扩展
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
        核心检索方法：语义种子 + 受限图扩展。

        流程：
        1. embedding(query) → Qdrant top-k 匹配入口实体
        2. 入口实体 → SQLite 1-hop 扩展（带时序过滤）
        3. 1-hop 邻居中如有高相似度实体 → 追加 1-hop（总共 2-hop）
        4. 硬上限 MAX_HOPS = 2

        Args:
            user_id: 用户 ID
            query: 当前查询文本
            reference_time: 参考时间（默认当前时间）
            top_k: 入口实体 top-k
            min_score: 最低相似度阈值

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

        # ---- Step 1: Qdrant 语义匹配入口实体 ----
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

        # ---- Step 2: 1-hop 扩展 ----
        seed_names = {s["entity"] for s in seed_entities}
        hop1_relations, hop1_neighbors = await self._expand_one_hop(
            user_id, seed_names, reference_time
        )

        all_relations = list(hop1_relations)
        all_entity_names = seed_names | hop1_neighbors
        hops_used = 1

        # ---- Step 3: 检查 1-hop 邻居是否有高相似度实体，决定是否追加 2-hop ----
        if hop1_neighbors and hops_used < MAX_HOPS:
            # 对 1-hop 邻居名称计算与 query 的相似度
            neighbor_names = list(hop1_neighbors - seed_names)
            if neighbor_names:
                neighbor_embeddings = self.embedding_model.encode(neighbor_names)
                import numpy as np
                query_emb_np = np.array(query_embedding)
                scores = np.dot(neighbor_embeddings, query_emb_np) / (
                    np.linalg.norm(neighbor_embeddings, axis=1) * np.linalg.norm(query_emb_np)
                )
                # 找到超过阈值的邻居
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

        # ---- 去重关系 ----
        seen_rels = set()
        unique_relations = []
        for rel in all_relations:
            key = (rel["source"], rel["target"], rel["relation_type"])
            if key not in seen_rels:
                seen_rels.add(key)
                unique_relations.append(rel)

        # ---- 获取所有实体的节点信息 ----
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
        从指定实体集合出发，扩展 1 跳（双向：出边 + 入边）。

        Returns:
            (relations_list, neighbor_entity_names_set)
        """
        relations = []
        neighbors = set()

        if not entity_names:
            return relations, neighbors

        # 构建 IN 子句的占位符
        placeholders = ",".join(["?" for _ in entity_names])
        names_list = list(entity_names)
        time_params = [reference_time, reference_time]

        async with aiosqlite.connect(self.db_path) as db:
            # ---- 出边：source IN entity_names ----
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

            # ---- 入边：target IN entity_names ----
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
        """获取指定实体名称的有效节点信息"""
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
    # LLM 驱动的抽取与更新
    # ================================================================

    async def extract_and_update(
        self, user_id: str, messages: list, llm: AzureChatOpenAI, prompt_loader=None
    ):
        """
        使用 LLM 从本轮对话中抽取实体与关系，并更新时序知识图谱。
        同时更新 SQLite (结构化) 和 Qdrant (语义索引)。
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
                "你是一个知识图谱抽取专家。请仅从【用户(human)说的话】中抽取实体和关系。\n"
                "不要从助手(ai/assistant)的回复中抽取信息。\n\n"
                f"已有知识图谱：\n{json.dumps(existing_kg, ensure_ascii=False, indent=2)}\n\n"
                f"本轮对话：\n{conversation}\n\n"
                "请仅输出以下 JSON 格式，不要有其他文字：\n"
                "{\n"
                '  "nodes": [\n'
                '    {"entity": "实体名", "entity_type": "person|location|event|organization|time|other", "properties": {}}\n'
                "  ],\n"
                '  "relations": [\n'
                '    {"source": "实体1", "target": "实体2", "relation_type": "关系类型", "properties": {}}\n'
                "  ],\n"
                '  "invalidations": [\n'
                '    {"source": "实体1", "target": "实体2", "relation_type": "关系类型"}\n'
                "  ]\n"
                "}\n\n"
                "注意：\n"
                "- 只提取用户(human)明确陈述的事实，不要提取助手建议的内容\n"
                "- 用 '用户' 表示用户本人这个实体\n"
                "- entity 和 relation 的命名要简洁一致\n"
                "- properties 中可以包含时间、地点等补充信息\n"
                "- 只在知识确实发生变化时才添加 invalidations\n"
                "- 如果没有新信息可提取，返回空数组"
            )

        try:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            result = _parse_json_safe(response.content)
            if not isinstance(result, dict):
                print(f"  [KG] 抽取结果解析失败")
                return
        except Exception as e:
            print(f"  [KG] 抽取失败: {e}")
            return

        # Step 1: 失效操作
        invalidations = result.get("invalidations", [])
        for inv in invalidations:
            await self.invalidate_relation(
                user_id, inv["source"], inv["target"], inv["relation_type"]
            )

        # Step 2: 插入新节点（SQLite + Qdrant）
        new_nodes = result.get("nodes", [])
        for node in new_nodes:
            await self.add_node(
                user_id,
                node["entity"],
                node.get("entity_type", "unknown"),
                node.get("properties")
            )

        # Step 3: 插入新关系（SQLite）
        new_relations = result.get("relations", [])
        for rel in new_relations:
            await self.add_relation(
                user_id,
                rel["source"], rel["target"],
                rel["relation_type"],
                rel.get("properties")
            )

        print(
            f"  [KG] 更新完成: "
            f"失效 {len(invalidations)} 条关系, "
            f"新增 {len(new_nodes)} 个节点, "
            f"新增 {len(new_relations)} 条关系"
        )
