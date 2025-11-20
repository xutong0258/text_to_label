# -*- coding: utf-8 -*-
"""
训练模型脚本
用于训练issue场景分类模型，支持多种算法对比
"""

import os
import re
import pandas as pd
import numpy as np
import jieba
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    f1_score
)
from config import (
    LABEL_MAPPING, 
    MODEL_DIR, 
    MODEL_PATH, 
    VECTORIZER_PATH,
    DATA_PATH,
    TEST_SIZE,
    RANDOM_STATE
)
from utils.logger_util import *

# 下载nltk所需数据（首次运行需要）
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("正在下载nltk punkt分词器...")
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    logger.info("正在下载nltk停用词表...")
    nltk.download('stopwords', quiet=True)


class IssueClassifier:
    """Issue场景分类器"""
    
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.label_mapping = LABEL_MAPPING
        # 英文停用词
        try:
            self.english_stopwords = set(stopwords.words('english'))
        except:
            self.english_stopwords = set()
    
    @staticmethod
    def is_chinese_char(char):
        """判断是否为中文字符"""
        return '\u4e00' <= char <= '\u9fff'
    
    def detect_language_ratio(self, text):
        """
        检测文本中中文字符的比例
        
        Args:
            text: 输入文本
            
        Returns:
            中文字符比例（0-1之间）
        """
        if not text:
            return 0
        
        chinese_count = sum(1 for char in text if self.is_chinese_char(char))
        total_count = len(text)
        
        return chinese_count / total_count if total_count > 0 else 0
    
    def preprocess_text(self, text):
        """
        混合语言文本预处理（支持英文+中文）
        
        策略：
        - 英文部分：使用nltk分词，转小写，去停用词
        - 中文部分：使用jieba分词
        - 混合文本：两种分词器结合使用
        
        Args:
            text: 输入文本
            
        Returns:
            处理后的文本
        """
        if pd.isna(text):
            return ""
        
        # 转换为字符串
        text = str(text)
        
        # 检测中文比例
        chinese_ratio = self.detect_language_ratio(text)
        
        # 如果中文比例很高（>50%），主要使用jieba分词
        if chinese_ratio > 0.5:
            # 中文为主，使用jieba分词
            words = jieba.cut(text)
            return " ".join(words)
        
        # 否则使用混合策略
        # 1. 先用正则分离中文和非中文部分
        tokens = []
        
        # 分割文本为中文块和非中文块
        parts = re.split(r'([\u4e00-\u9fff]+)', text)
        
        for part in parts:
            if not part.strip():
                continue
            
            # 判断是否为中文
            if self.is_chinese_char(part[0]) if part else False:
                # 中文部分用jieba分词
                chinese_words = jieba.cut(part)
                tokens.extend(chinese_words)
            else:
                # 英文部分处理
                # 转小写
                part = part.lower()
                # 移除特殊字符，保留字母数字
                part = re.sub(r'[^a-z0-9\s]', ' ', part)
                # 清理多余空格
                part = re.sub(r'\s+', ' ', part).strip()
                
                if part:
                    # 使用nltk分词
                    try:
                        english_words = word_tokenize(part)
                        # 过滤停用词
                        english_words = [
                            w for w in english_words 
                            if w not in self.english_stopwords and len(w) > 1
                        ]
                        tokens.extend(english_words)
                    except:
                        # 如果nltk分词失败，按空格分割
                        words = part.split()
                        tokens.extend([w for w in words if len(w) > 1])
        
        return " ".join(tokens)
    
    def prepare_data(self, df, include_feedback=True):
        """
        准备训练数据
        
        Args:
            df: 数据框
            include_feedback: 是否包含反馈数据
            
        Returns:
            X: 特征文本列表
            y: 标签列表
        """
        logger.info("=" * 60)
        logger.info("数据准备阶段")
        logger.info("=" * 60)
        
        # 合并title和description
        logger.info("正在合并title和description字段...")
        df['combined_text'] = df['title'].fillna('') + " " + df['description'].fillna('')
        
        # 预处理文本
        logger.info("正在进行文本预处理（混合语言分词：英文用nltk，中文用jieba）...")
        df['processed_text'] = df['combined_text'].apply(self.preprocess_text)
        
        X = df['processed_text'].tolist()
        y = df['label'].tolist()
        
        logger.info(f"数据样本数: {len(X)}")
        logger.info(f"标签分布:\n{df['label'].value_counts().sort_index()}")
        
        return X, y
    
    def train(self, X_train, y_train, model_type='logistic_regression'):
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            model_type: 模型类型
        """
        logger.info("=" * 60)
        logger.info("特征提取阶段")
        logger.info("=" * 60)
        
        # TF-IDF特征提取
        logger.info("正在使用TF-IDF进行特征提取...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),  # 使用unigram和bigram
            min_df=2,
            max_df=0.8
        )
        
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        logger.info(f"TF-IDF特征维度: {X_train_tfidf.shape}")
        logger.info()
        
        logger.info("=" * 60)
        logger.info("模型训练阶段")
        logger.info("=" * 60)
        
        # 选择模型
        if model_type == 'logistic_regression':
            logger.info("使用模型: 逻辑回归 (Logistic Regression)")
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
                class_weight='balanced'  # 处理类别不平衡
            )
        elif model_type == 'random_forest':
            logger.info("使用模型: 随机森林 (Random Forest)")
            self.model = RandomForestClassifier(
                n_estimators=200,
                random_state=RANDOM_STATE,
                class_weight='balanced',
                max_depth=20
            )
        elif model_type == 'gradient_boosting':
            logger.info("使用模型: 梯度提升树 (Gradient Boosting)")
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                max_depth=10
            )
        elif model_type == 'svm':
            logger.info("使用模型: 支持向量机 (SVM)")
            self.model = SVC(
                kernel='rbf',
                random_state=RANDOM_STATE,
                class_weight='balanced',
                probability=True
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 训练模型
        logger.info("正在训练模型...")
        self.model.fit(X_train_tfidf, y_train)
        logger.info("模型训练完成！")
        logger.info()
        
    def evaluate(self, X_test, y_test):
        """
        评估模型
        
        Args:
            X_test: 测试特征
            y_test: 测试标签
            
        Returns:
            评估指标字典
        """
        logger.info("=" * 60)
        logger.info("模型评估阶段")
        logger.info("=" * 60)
        
        # 转换测试数据
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # 预测
        y_pred = self.model.predict(X_test_tfidf)
        
        # 计算准确率
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        logger.info(f"测试集准确率: {accuracy:.4f}")
        logger.info(f"测试集F1分数: {f1:.4f}")
        logger.info()
        
        # 详细分类报告
        logger.info("分类报告:")
        logger.info("-" * 60)
        target_names = [self.label_mapping[i] for i in sorted(self.label_mapping.keys())]
        logger.info(classification_report(y_test, y_pred, target_names=target_names))
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_test': y_test
        }
    
    def plot_confusion_matrix(self, cm, save_path='confusion_matrix.png'):
        """
        绘制混淆矩阵
        
        Args:
            cm: 混淆矩阵
            save_path: 保存路径
        """
        plt.figure(figsize=(12, 10))
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 绘制热力图
        labels = [self.label_mapping[i] for i in sorted(self.label_mapping.keys())]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels)
        
        plt.title('混淆矩阵 (Confusion Matrix)', fontsize=16, pad=20)
        plt.ylabel('真实标签 (True Label)', fontsize=12)
        plt.xlabel('预测标签 (Predicted Label)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"混淆矩阵已保存到: {save_path}")
        plt.close()
    
    def save_model(self):
        """保存模型和向量化器"""
        # 创建模型目录
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # 保存模型
        joblib.dump(self.model, MODEL_PATH)
        logger.info(f"模型已保存到: {MODEL_PATH}")
        
        # 保存向量化器
        joblib.dump(self.vectorizer, VECTORIZER_PATH)
        logger.info(f"向量化器已保存到: {VECTORIZER_PATH}")
    
    def load_model(self):
        """加载模型和向量化器"""
        if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
            raise FileNotFoundError("模型文件不存在，请先训练模型")
        
        self.model = joblib.load(MODEL_PATH)
        self.vectorizer = joblib.load(VECTORIZER_PATH)
        logger.info("模型加载成功！")
    
    def predict(self, title, description):
        """
        预测单个样本
        
        Args:
            title: 标题
            description: 描述
            
        Returns:
            预测结果字典
        """
        # 合并文本
        combined_text = f"{title} {description}"
        processed_text = self.preprocess_text(combined_text)
        
        # 转换特征
        X_tfidf = self.vectorizer.transform([processed_text])
        
        # 预测
        label = self.model.predict(X_tfidf)[0]
        probabilities = self.model.predict_proba(X_tfidf)[0]
        
        # 构造结果
        result = {
            'predicted_label': int(label),
            'predicted_scene': self.label_mapping[label],
            'confidence': float(probabilities[label]),
            'all_probabilities': {
                self.label_mapping[i]: float(prob) 
                for i, prob in enumerate(probabilities)
            }
        }
        
        return result


def compare_models(X_train, X_test, y_train, y_test):
    """
    比较不同模型的性能
    
    Args:
        X_train: 训练特征
        X_test: 测试特征
        y_train: 训练标签
        y_test: 测试标签
    """
    logger.info("\n" + "=" * 60)
    logger.info("模型对比实验")
    logger.info("=" * 60 + "\n")
    
    models = {
        'Logistic Regression': 'logistic_regression',
        'Random Forest': 'random_forest',
        'Gradient Boosting': 'gradient_boosting',
        'SVM': 'svm'
    }
    
    results = {}
    
    for model_name, model_type in models.items():
        logger.info(f"\n{'=' * 60}")
        logger.info(f"训练模型: {model_name}")
        logger.info(f"{'=' * 60}\n")
        
        classifier = IssueClassifier()
        classifier.train(X_train, y_train, model_type=model_type)
        metrics = classifier.evaluate(X_test, y_test)
        
        results[model_name] = {
            'accuracy': metrics['accuracy'],
            'f1_score': metrics['f1_score']
        }
    
    # 打印对比结果
    logger.info("\n" + "=" * 60)
    logger.info("模型对比结果")
    logger.info("=" * 60)
    logger.info(f"{'模型':<25} {'准确率':<15} {'F1分数':<15}")
    logger.info("-" * 60)
    for model_name, metric in results.items():
        logger.info(f"{model_name:<25} {metric['accuracy']:<15.4f} {metric['f1_score']:<15.4f}")
    
    # 找出最佳模型
    best_model = max(results.items(), key=lambda x: x[1]['f1_score'])
    logger.info(f"\n最佳模型: {best_model[0]} (F1={best_model[1]['f1_score']:.4f})")
    
    return results


def main():
    """主函数"""
    logger.info("\n" + "=" * 60)
    logger.info("Issue场景分类模型训练")
    logger.info("=" * 60 + "\n")
    
    # 加载数据
    logger.info("正在加载数据...")
    df = pd.read_csv(DATA_PATH)
    logger.info(f"数据集大小: {df.shape}")
    logger.info(f"数据集列名: {df.columns.tolist()}")
    
    # 创建分类器
    classifier = IssueClassifier()
    
    # 准备数据
    X, y = classifier.prepare_data(df)
    
    # 划分训练集和测试集
    logger.info("=" * 60)
    logger.info("划分训练集和测试集")
    logger.info("=" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE,
        stratify=y  # 分层采样，保持类别比例
    )
    logger.info(f"训练集大小: {len(X_train)}")
    logger.info(f"测试集大小: {len(X_test)}")
    logger.info()
    
    # 模型对比实验
    compare_models(X_train, X_test, y_train, y_test)
    
    # 训练最终模型（使用逻辑回归，平衡速度和效果）
    logger.info("\n" + "=" * 60)
    logger.info("训练最终部署模型")
    logger.info("=" * 60 + "\n")
    
    final_classifier = IssueClassifier()
    final_classifier.train(X_train, y_train, model_type='logistic_regression')
    
    # 评估
    metrics = final_classifier.evaluate(X_test, y_test)
    
    # 绘制混淆矩阵
    final_classifier.plot_confusion_matrix(metrics['confusion_matrix'])
    
    # 保存模型
    logger.info("\n" + "=" * 60)
    logger.info("保存模型")
    logger.info("=" * 60)
    final_classifier.save_model()
    
    # 测试预测功能
    logger.info("\n" + "=" * 60)
    logger.info("测试预测功能")
    logger.info("=" * 60)
    
    test_case = {
        'title': 'System hang when running stress test',
        'description': 'The system becomes unresponsive during CPU and GPU stress testing.'
    }
    
    result = final_classifier.predict(test_case['title'], test_case['description'])
    logger.info(f"\n测试用例:")
    logger.info(f"  Title: {test_case['title']}")
    logger.info(f"  Description: {test_case['description']}")
    logger.info(f"\n预测结果:")
    logger.info(f"  场景: {result['predicted_scene']}")
    logger.info(f"  置信度: {result['confidence']:.4f}")
    logger.info(f"  所有场景概率:")
    for scene, prob in result['all_probabilities'].items():
        logger.info(f"    {scene}: {prob:.4f}")
    
    logger.info("\n" + "=" * 60)
    logger.info("训练完成！")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

