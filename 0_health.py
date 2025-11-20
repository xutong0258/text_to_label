import requests

# 预测Issue场景
response = requests.get(
    "http://localhost:8000/health",
)

result = response.json()
print(f"预测场景: {result}")