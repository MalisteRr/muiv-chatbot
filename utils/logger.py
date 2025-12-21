"""
Настройка системы логирования
Логирование в консоль и файл
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Настройка системы логирования
    
    Args:
        level: Уровень логирования
        
    Returns:
        Настроенный логгер
    """
    # Создать директорию для логов
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Имя файла лога с датой
    log_filename = log_dir / f"muiv_bot_{datetime.now().strftime('%Y%m%d')}.log"
    
    # Формат логирования
    log_format = logging.Formatter(
        fmt='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Корневой логгер
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Очистить существующие обработчики
    root_logger.handlers.clear()
    
    # Обработчик для консоли
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(log_format)
    root_logger.addHandler(console_handler)
    
    # Обработчик для файла с ротацией
    file_handler = RotatingFileHandler(
        log_filename,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)  # В файл пишем все
    file_handler.setFormatter(log_format)
    root_logger.addHandler(file_handler)
    
    # Отдельный файл для ошибок
    error_filename = log_dir / f"muiv_bot_errors_{datetime.now().strftime('%Y%m%d')}.log"
    error_handler = RotatingFileHandler(
        error_filename,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(log_format)
    root_logger.addHandler(error_handler)
    
    # Логгер для внешних библиотек (уменьшаем шум)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("aiogram").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.INFO)
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Система логирования инициализирована")
    logger.info(f"Уровень: {logging.getLevelName(level)}")
    logger.info(f"Файл логов: {log_filename}")
    logger.info(f"Файл ошибок: {error_filename}")
    logger.info("=" * 60)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Получить логгер для модуля
    
    Args:
        name: Имя модуля
        
    Returns:
        Логгер
    """
    return logging.getLogger(name)