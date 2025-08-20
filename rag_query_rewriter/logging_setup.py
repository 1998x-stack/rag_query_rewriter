"""Configure loguru logging for the package."""
from __future__ import annotations

import sys
from typing import Optional
from loguru import logger


def setup_logging(level: str = "INFO", file_path: Optional[str] = None) -> None:
    """Setup global logging sinks.

    中文说明：
        - 控制台彩色日志 + 可选文件日志（滚动/保留）。
    """
    logger.remove()
    logger.add(
        sys.stdout,
        level=level,
        enqueue=True,
        colorize=True,
        format=("<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                "<level>{message}</level>")
    )
    if file_path:
        logger.add(
            file_path,
            level=level,
            rotation="20 MB",
            retention="14 days",
            enqueue=True,
            encoding="utf-8",
        )
