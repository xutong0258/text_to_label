# -*- coding: utf-8 -*-
"""
Issue场景分类Web服务
提供预测和反馈收集功能
"""

import sqlite3
from datetime import datetime
from typing import Optional
from contextlib import contextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from train_model import IssueClassifier
from config import LABEL_MAPPING, LABEL_REVERSE_MAPPING, FEEDBACK_DB_PATH


# ============================================================================
# 数据模型定义
# ============================================================================

class PredictRequest(BaseModel):
    """预测请求模型"""
    title: str = Field(..., description="Issue标题", example="System crashes during stress test")
    description: str = Field(..., description="Issue描述", example="The system crashes when running CPU stress test for more than 30 minutes")


class PredictResponse(BaseModel):
    """预测响应模型"""
    predicted_label: int = Field(..., description="预测的标签（0-6）")
    predicted_scene: str = Field(..., description="预测的场景名称")
    confidence: float = Field(..., description="预测置信度")
    all_probabilities: dict = Field(..., description="所有场景的概率分布")


class FeedbackRequest(BaseModel):
    """反馈请求模型"""
    title: str = Field(..., description="Issue标题")
    description: str = Field(..., description="Issue描述")
    predicted_label: int = Field(..., description="系统预测的标签", ge=0, le=6)
    correct_label: int = Field(..., description="用户修正后的正确标签", ge=0, le=6)
    user_id: Optional[str] = Field(None, description="用户ID（可选）")


class FeedbackResponse(BaseModel):
    """反馈响应模型"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")
    feedback_id: int = Field(..., description="反馈记录ID")


# ============================================================================
# 数据库管理
# ============================================================================

class FeedbackDatabase:
    """反馈数据库管理类"""
    
    def __init__(self, db_path: str = FEEDBACK_DB_PATH):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """初始化数据库表"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    predicted_label INTEGER NOT NULL,
                    correct_label INTEGER NOT NULL,
                    user_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """获取数据库连接的上下文管理器"""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()
    
    def add_feedback(self, title: str, description: str, 
                    predicted_label: int, correct_label: int,
                    user_id: Optional[str] = None) -> int:
        """
        添加反馈记录
        
        Args:
            title: 标题
            description: 描述
            predicted_label: 预测标签
            correct_label: 正确标签
            user_id: 用户ID
            
        Returns:
            反馈记录ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO feedback (title, description, predicted_label, correct_label, user_id)
                VALUES (?, ?, ?, ?, ?)
            """, (title, description, predicted_label, correct_label, user_id))
            conn.commit()
            return cursor.lastrowid
    
    def get_all_feedback(self):
        """
        获取所有反馈记录
        
        Returns:
            反馈记录列表
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, title, description, predicted_label, correct_label, user_id, created_at
                FROM feedback
                ORDER BY created_at DESC
            """)
            columns = [desc[0] for desc in cursor.description]
            results = cursor.fetchall()
            return [dict(zip(columns, row)) for row in results]
    
    def get_feedback_count(self):
        """获取反馈记录总数"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM feedback")
            return cursor.fetchone()[0]
    
    def get_feedback_stats(self):
        """
        获取反馈统计信息
        
        Returns:
            统计信息字典
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 总记录数
            cursor.execute("SELECT COUNT(*) FROM feedback")
            total_count = cursor.fetchone()[0]
            
            # 预测错误的记录数
            cursor.execute("SELECT COUNT(*) FROM feedback WHERE predicted_label != correct_label")
            error_count = cursor.fetchone()[0]
            
            # 各场景的错误分布
            cursor.execute("""
                SELECT predicted_label, COUNT(*) as count
                FROM feedback
                WHERE predicted_label != correct_label
                GROUP BY predicted_label
            """)
            error_by_label = {row[0]: row[1] for row in cursor.fetchall()}
            
            return {
                'total_feedback': total_count,
                'incorrect_predictions': error_count,
                'accuracy': 1 - (error_count / total_count) if total_count > 0 else None,
                'error_by_scene': {
                    LABEL_MAPPING.get(label, f"Unknown-{label}"): count 
                    for label, count in error_by_label.items()
                }
            }


# ============================================================================
# FastAPI应用
# ============================================================================

# 创建FastAPI应用
app = FastAPI(
    title="Issue场景分类服务",
    description="基于机器学习的Issue场景自动分类系统，支持预测和用户反馈收集",
    version="1.0.0"
)

# 全局变量
classifier: Optional[IssueClassifier] = None
feedback_db: Optional[FeedbackDatabase] = None


@app.on_event("startup")
async def startup_event():
    """应用启动时加载模型和初始化数据库"""
    global classifier, feedback_db
    
    print("=" * 60)
    print("正在启动Issue场景分类服务...")
    print("=" * 60)
    
    # 加载模型
    try:
        classifier = IssueClassifier()
        classifier.load_model()
        print("✓ 模型加载成功")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        print("请先运行 train_model.py 训练模型")
        raise
    
    # 初始化数据库
    try:
        feedback_db = FeedbackDatabase()
        print("✓ 数据库初始化成功")
    except Exception as e:
        print(f"✗ 数据库初始化失败: {e}")
        raise
    
    print("=" * 60)
    print("服务启动完成！")
    print("=" * 60)


@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "Issue场景分类服务",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "predict": "/predict - 预测Issue场景",
            "feedback": "/feedback - 提交用户反馈",
            "stats": "/stats - 查看反馈统计",
            "health": "/health - 健康检查",
            "docs": "/docs - API文档"
        }
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    if classifier is None or classifier.model is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    feedback_count = feedback_db.get_feedback_count() if feedback_db else 0
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "feedback_count": feedback_count,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    预测Issue场景
    
    Args:
        request: 预测请求，包含title和description
        
    Returns:
        预测结果，包含场景标签、名称、置信度和概率分布
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        # 执行预测
        result = classifier.predict(request.title, request.description)
        
        return PredictResponse(
            predicted_label=result['predicted_label'],
            predicted_scene=result['predicted_scene'],
            confidence=result['confidence'],
            all_probabilities=result['all_probabilities']
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """
    提交用户反馈
    
    当用户认为预测结果不正确时，可以提交正确的标签
    这些反馈数据会被保存，用于后续模型迭代
    
    Args:
        request: 反馈请求，包含原始数据、预测标签和正确标签
        
    Returns:
        反馈提交结果
    """
    if feedback_db is None:
        raise HTTPException(status_code=503, detail="数据库未初始化")
    
    try:
        # 验证标签有效性
        if request.predicted_label not in LABEL_MAPPING:
            raise HTTPException(status_code=400, detail=f"无效的预测标签: {request.predicted_label}")
        
        if request.correct_label not in LABEL_MAPPING:
            raise HTTPException(status_code=400, detail=f"无效的正确标签: {request.correct_label}")
        
        # 保存反馈
        feedback_id = feedback_db.add_feedback(
            title=request.title,
            description=request.description,
            predicted_label=request.predicted_label,
            correct_label=request.correct_label,
            user_id=request.user_id
        )
        
        # 构造响应消息
        if request.predicted_label == request.correct_label:
            message = "感谢您的反馈！预测结果与您的选择一致。"
        else:
            predicted_scene = LABEL_MAPPING[request.predicted_label]
            correct_scene = LABEL_MAPPING[request.correct_label]
            message = f"感谢您的反馈！我们会学习这个案例（预测: {predicted_scene} → 正确: {correct_scene}）。"
        
        return FeedbackResponse(
            success=True,
            message=message,
            feedback_id=feedback_id
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"反馈提交失败: {str(e)}")


@app.get("/stats")
async def get_stats():
    """
    获取反馈统计信息
    
    Returns:
        反馈统计数据
    """
    if feedback_db is None:
        raise HTTPException(status_code=503, detail="数据库未初始化")
    
    try:
        stats = feedback_db.get_feedback_stats()
        return {
            "success": True,
            "data": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")


@app.get("/feedback/list")
async def list_feedback(limit: int = 100):
    """
    获取反馈记录列表
    
    Args:
        limit: 返回记录数量限制
        
    Returns:
        反馈记录列表
    """
    if feedback_db is None:
        raise HTTPException(status_code=503, detail="数据库未初始化")
    
    try:
        all_feedback = feedback_db.get_all_feedback()
        
        # 应用限制
        limited_feedback = all_feedback[:limit]
        
        # 添加场景名称
        for item in limited_feedback:
            item['predicted_scene'] = LABEL_MAPPING.get(item['predicted_label'], 'Unknown')
            item['correct_scene'] = LABEL_MAPPING.get(item['correct_label'], 'Unknown')
        
        return {
            "success": True,
            "count": len(limited_feedback),
            "total": len(all_feedback),
            "data": limited_feedback
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取反馈列表失败: {str(e)}")


@app.get("/labels")
async def get_labels():
    """
    获取所有场景标签映射
    
    Returns:
        标签映射字典
    """
    return {
        "success": True,
        "labels": LABEL_MAPPING,
        "reverse_labels": LABEL_REVERSE_MAPPING
    }


# ============================================================================
# 主程序
# ============================================================================

def main():
    """启动Web服务"""
    print("\n" + "=" * 60)
    print("Issue场景分类Web服务")
    print("=" * 60 + "\n")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()

