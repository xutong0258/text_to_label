# -*- coding: utf-8 -*-
"""
配置文件
"""

# 场景标签映射
LABEL_MAPPING = {
    0: "performance",
    1: "battery life",
    2: "单双烤",
    3: "BSOD",
    4: "hang",
    5: "abnormal reboot",
    6: "abnormal shutdown"
}

# 反向映射
LABEL_REVERSE_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}

# 模型相关配置
MODEL_DIR = "models"
MODEL_PATH = f"{MODEL_DIR}/issue_classifier.pkl"
VECTORIZER_PATH = f"{MODEL_DIR}/tfidf_vectorizer.pkl"

# 数据相关配置
DATA_PATH = "issue_data.csv"
FEEDBACK_DB_PATH = "feedback.db"

# 训练参数
TEST_SIZE = 0.2
RANDOM_STATE = 42

