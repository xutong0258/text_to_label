import requests

# 提交用户反馈
response = requests.post(
    "http://localhost:8000/feedback",
    json={
        "title": "System hang during stress test",
        "description": "The system becomes unresponsive...",
        "predicted_label": 4,
        "correct_label": 2,
        "user_id": "user_123"
    }
)

print(response.json()['message'])