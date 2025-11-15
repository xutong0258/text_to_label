# Issue场景分类系统

## 📋 项目简介

这是一个基于机器学习的Issue场景自动分类系统，能够根据用户填报的Issue信息（标题和描述）自动预测最可能的场景类型，支持7种场景分类。

## 🎯 支持的场景

| 标签 | 场景名称 |
|-----|---------|
| 0 | performance（性能问题） |
| 1 | battery life（电池续航） |
| 2 | 单双烤 |
| 3 | BSOD（蓝屏死机） |
| 4 | hang（系统挂起） |
| 5 | abnormal reboot（异常重启） |
| 6 | abnormal shutdown（异常关机） |

## 🚀 快速开始
需要使用uv管理python环境

### 1. 安装依赖

```bash
uv sync
```

### 2. 训练模型

```bash
uv run train_model.py
```

训练完成后会生成：
- `models/issue_classifier.pkl` - 分类器模型
- `models/tfidf_vectorizer.pkl` - 特征提取器
- `confusion_matrix.png` - 混淆矩阵可视化

### 3. 启动Web服务

```bash
uv run app.py
```

服务启动后访问：http://localhost:8000

## 📖 API使用示例

### 预测接口

```python
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
```

### 反馈接口

```python
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
```

## 📚 文档

详细的算法说明和使用指南请参考：[算法说明文档.md](./算法说明文档.md)

## 🔧 项目结构

```
├── config.py                  # 配置文件
├── train_model.py             # 模型训练脚本
├── app.py                     # Web服务
├── requirements.txt           # 依赖包列表
├── README.md                  # 本文件
├── 算法说明文档.md            # 详细算法文档
├── issue_data.csv             # 训练数据集
├── models/                    # 模型文件目录
│   ├── issue_classifier.pkl
│   └── tfidf_vectorizer.pkl
└── feedback.db                # 反馈数据库
```

## 🛠️ 技术栈

- **机器学习**: scikit-learn, TF-IDF, Logistic Regression
- **文本处理**: 
  - 英文分词：nltk（含停用词过滤）
  - 中文分词：jieba
  - 智能语言检测和混合处理
- **Web框架**: FastAPI
- **数据处理**: pandas, numpy
- **数据库**: SQLite

## 📈 模型性能

- 采用TF-IDF特征提取 + 逻辑回归分类
- 训练集/测试集：80%/20%
- 预期准确率：75-85%（取决于数据质量）

## 🔄 持续学习

系统支持收集用户反馈，可以定期重新训练模型以提升性能：

1. 收集反馈数据（通过`/feedback`接口）
2. 定期重新运行`train_model.py`
3. 重启Web服务加载新模型

## 📝 许可证

本项目仅供内部使用。

