import requests
resp = requests.get(
    'https://docparseai.openai.azure.com/openai/deployments?api-version=2025-01-01-preview',
    headers={'api-key': '40e765a1ded749b99509a9010b95b783'},
    proxies={'http': 'http://127.0.0.1:3128', 'https': 'http://127.0.0.1:3128'},
    timeout=30
)
print('Status:', resp.status_code)
if resp.status_code == 200:
    data = resp.json()
    for d in data.get('data', []):
        print(f"  {d['id']} -> {d.get('model', 'unknown')}")
else:
    print(resp.text[:500])
