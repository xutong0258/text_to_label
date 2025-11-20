import requests

# 预测Issue场景
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "title": "System hang during stress test",
        "description": "The system becomes unresponsive when running CPU and GPU stress testing."
    }
)

result = response.json()
print(f"预测场景: {result['predicted_scene']}")
print(f"置信度: {result['confidence']:.2%}")