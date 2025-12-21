"""
Главный файл запуска бота
Поддерживает как PostgreSQL так и SQLite
"""

import asyncio
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent))

# Загрузка переменных окружения
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ Loaded .env from {env_path}")
else:
    print(f"⚠️  .env file not found at {env_path}")
    print("   Using environment variables or defaults")

from bot.dispatcher import dp, bot
from database.init_db import init_db, close_db
from utils.logger import setup_logging

# Настройка логирования
logger = setup_logging()


async def on_startup():
    """Действия при запуске бота"""
    logger.info("=" * 60)
    logger.info("🤖 Запуск чат-бота для абитуриентов МУИВ")
    logger.info("=" * 60)
    
    try:
        # Инициализация базы данных
        await init_db()
        logger.info("✅ База данных инициализирована")
    except Exception as e:
        logger.error(f"❌ Ошибка инициализации БД: {e}")
        logger.error("   Проверьте настройки DATABASE_URL в .env файле")
        raise
    
    logger.info("✅ Все системы готовы к работе!")
    logger.info("=" * 60)


async def on_shutdown():
    """Действия при остановке бота"""
    logger.info("=" * 60)
    logger.info("🛑 Остановка бота...")
    
    # Закрытие соединений с БД
    await close_db()
    
    logger.info("✅ Все соединения закрыты")
    logger.info("=" * 60)


async def main():
    """Главная функция запуска"""
    try:
        # Регистрация обработчиков событий
        dp.startup.register(on_startup)
        dp.shutdown.register(on_shutdown)
        
        # Удаление webhook (для polling режима)
        await bot.delete_webhook(drop_pending_updates=True)
        
        logger.info("🚀 Запуск polling режима...")
        
        # Запуск бота
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
        
    except Exception as e:
        logger.critical(f"💥 Критическая ошибка: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("⌨️  Бот остановлен пользователем (Ctrl+C)")
    except Exception as e:
        logger.critical(f"💥 Критическая ошибка при запуске: {e}", exc_info=True)
        sys.exit(1)