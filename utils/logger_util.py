# -*- coding: utf-8 -*-
# @Date     : 2025/9/26 11:46
# @Author   : q275343119
# @File     : logger_util.py
# @Description:

import os
import sys
from pathlib import Path
from typing import Literal
from loguru import logger

# 基于 Loguru 的日志配置模块, 支持简单验证和单例模式

# file = os.path.abspath(__file__)
path_dir = os.path.dirname(__file__)

VALID_LEVELS = {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}

_logger_configured = False

# "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
# "<level>{level: <5}</level> | "
def setup_logger(
    log_dir: str | None = None,
    log_format: str = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "        
        "<cyan>{file}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    ),
    level: Literal[
        "TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"
    ] = "INFO",
    rotation: str = "1 day",  # 每天切割
    retention: str = "7 days",  # 保留 7 天
    enqueue: bool = True,
    backtrace: bool = True,
    diagnose: bool = True,
    stdout: bool = True,
    file_output: bool = False,
):
    """
    初始化全局日志模块。

    参数:
    - log_dir: 日志文件目录 (可选, 如果 file_output=True 则推荐提供)
    - log_format: 日志输出格式 (默认带有时间、级别和位置)
    - level: 日志级别 (默认 "INFO", 支持 TRACE 到 CRITICAL)
    - rotation: 日志文件切割规则 (默认每天)
    - retention: 日志文件保留规则 (默认7天)
    - enqueue: 是否异步写入 (默认True, 避免阻塞)
    - backtrace: 是否记录异常回溯 (默认True)
    - diagnose: 是否诊断异常 (默认True)
    - stdout: 是否输出到控制台 (默认True)
    - file_output: 是否输出到文件 (默认False)

    示例:
    # 基础配置: 只控制台输出
    setup_logger(level="DEBUG")
    logger.info("这是一个信息日志")

    # 带文件输出
    setup_logger(log_dir="./logs", file_output=True, rotation="500 MB")
    logger.error("这是一个错误日志")

    # 注意: 配置后可全局使用 logger
    """

    # 使用单例模式, 确保日志配置只执行一次, 避免重复调用清除 handlers
    global _logger_configured
    if _logger_configured:
        logger.warning("Logger already configured, skipping...")
        return logger

    # 验证日志级别, 确保配置有效, 避免无效输入导致日志失败
    if level not in VALID_LEVELS:
        raise ValueError(f"Invalid log level: {level}")

    logger.remove()

    # 添加控制台输出, 支持彩色以提高开发调试的可读性
    if stdout:
        logger.add(
            sys.stdout,
            format=log_format,
            level=level,
            backtrace=backtrace,
            diagnose=diagnose,
            enqueue=enqueue,
            colorize=True,  # 关键: 启用彩色输出
        )

    # 添加文件输出, 支持旋转和保留, 适用于生产监控
    if file_output and log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_path = Path(log_dir) / "app-{time:YYYY-MM-DD}.log"
        logger.add(
            log_path,
            format=log_format,
            level=level,
            rotation=rotation,
            retention=retention,
            encoding="utf-8",
            enqueue=enqueue,
            backtrace=backtrace,
            diagnose=diagnose,
            colorize=False,  # 文件中不需要彩色
        )

    _logger_configured = True
    return logger


logging = setup_logger("./logs", level="INFO")

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("BSOD 日志分析工具启动")
    logger.info("=" * 60)
