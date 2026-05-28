import requests

# 尝试常见的 embedding deployment 名称
candidates = [
    "text-embedding-ada-002",
    "text-embedding-3-small",
    "text-embedding-3-large",
    "embedding",
    "ada",
]

for name in candidates:
    url = f"https://docparseai.openai.azure.com/openai/deployments/{name}/embeddings?api-version=2025-01-01-preview"
    try:
        resp = requests.post(
            url,
            headers={"Content-Type": "application/json", "api-key": "40e765a1ded749b99509a9010b95b783"},
            json={"input": "test"},
            proxies={"http": "http://127.0.0.1:3128", "https": "http://127.0.0.1:3128"},
            timeout=15,
        )
        if resp.status_code == 200:
            dim = len(resp.json()["data"][0]["embedding"])
            print(f"  OK: {name} (dim={dim})")
        else:
            print(f"  {resp.status_code}: {name} -> {resp.text[:100]}")
    except Exception as e:
        print(f"  ERR: {name} -> {e}")
