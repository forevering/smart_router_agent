"""Test local embedding model: Chinese semantic similarity"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # China mirror (can be ignored if proxy is available)

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

print("Loading model...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print(f"Model loaded, dimension: {model.get_sentence_embedding_dimension()}")

# Test Chinese semantic similarity
texts = [
    "I'm going to Shanghai on a business trip next week",
    "User plans to go to Shanghai recently",
    "What's the weather like tomorrow",
    "Travel plans affected by weather",
]
query = "Next week's travel arrangements"

print(f"\nQuery: {query}\n")
q_emb = model.encode(query)
t_embs = model.encode(texts)

for text, score in zip(texts, cos_sim(q_emb, t_embs)[0]):
    print(f"  {score.item():.4f}  {text}")
