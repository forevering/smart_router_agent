"""
测试 langchain-openai AzureChatOpenAI 通过 CNTLM 代理调用 Azure OpenAI
"""
import httpx
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage

# --- 代理配置 (与 request_openai 1_test.py 一致) ---
PROXY_URL = "http://127.0.0.1:3128"

http_client = httpx.Client(proxy=PROXY_URL)
async_http_client = httpx.AsyncClient(proxy=PROXY_URL)

# --- LLM 配置 ---
llm = AzureChatOpenAI(
    azure_endpoint="https://docparseai.openai.azure.com",
    azure_deployment="me-sales-agent",
    api_version="2025-01-01-preview",
    api_key="40e765a1ded749b99509a9010b95b783",
    temperature=0,
    max_tokens=200,
    http_client=http_client,
    http_async_client=async_http_client,
)

# --- 测试1: 同步调用 ---
print("=== 测试1: 同步调用 ===")
response = llm.invoke([HumanMessage(content="What is the capital of France? Answer in one sentence.")])
print(f"Response: {response.content}")
print(f"Token usage: {response.response_metadata.get('token_usage', 'N/A')}")

# --- 测试2: 流式调用 ---
print("\n=== 测试2: 流式调用 ===")
print("Streaming: ", end="", flush=True)
for chunk in llm.stream([HumanMessage(content="What is 2+2? Answer in one word.")]):
    print(chunk.content, end="", flush=True)
print("\n")

print("All tests passed!")
