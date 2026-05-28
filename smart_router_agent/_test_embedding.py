"""测试本地 embedding 模型：中文语义相似度"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 国内镜像（如代理可用则可忽略）

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

print("加载模型...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print(f"模型加载完成, 维度: {model.get_sentence_embedding_dimension()}")

# 测试中文语义相似度
texts = [
    "我下周要去上海出差",
    "用户计划最近去上海",
    "明天天气怎么样",
    "旅行计划受天气影响",
]
query = "下周的旅行安排"

print(f"\nQuery: {query}\n")
q_emb = model.encode(query)
t_embs = model.encode(texts)

for text, score in zip(texts, cos_sim(q_emb, t_embs)[0]):
    print(f"  {score.item():.4f}  {text}")
