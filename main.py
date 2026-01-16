"""
Главный файл запуска бота
Поддерживает как PostgreSQL так и SQLite
С интеграцией RuBERT классификатора и собственной LSTM модели
"""

import asyncio
import logging
import sys
import os
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
        logger.info("🔧 Инициализация базы данных...")
        db_type = os.getenv("DB_TYPE", "sqlite")
        
        if db_type == "sqlite":
            db_path = os.getenv("SQLITE_DB_PATH", "data/bot.db")
            logger.info(f"   📁 Используется SQLite: {db_path}")
        else:
            logger.info(f"   🐘 Используется PostgreSQL")
        
        await init_db()
        logger.info("✅ База данных инициализирована")
        logger.info("   📊 Таблицы созданы (users, chat_history, feedback, analytics, faq)")
        
        # Перезагрузка FAQ из JSON
        try:
            from database.init_db import load_faq_from_json
            faq_path = Path(__file__).parent / "database" / "faq_61.json"
            
            if faq_path.exists():
                logger.info(f"🔄 Перезагрузка FAQ из {faq_path}...")
                count = await load_faq_from_json(str(faq_path))
                logger.info(f"✅ FAQ перезагружен: {count} вопросов")
            else:
                logger.warning(f"⚠️ Файл FAQ не найден: {faq_path}")
        except Exception as e:
            logger.error(f"❌ Ошибка перезагрузки FAQ: {e}")
    except Exception as e:
        logger.error(f"❌ Ошибка инициализации БД: {e}")
        logger.error("   Проверьте настройки DATABASE_URL в .env файле")
        raise
    
    # ========== ИНИЦИАЛИЗАЦИЯ RUBERT ==========
    try:
        from ml.intent_classifier import init_classifier
        
        # Путь к модели RuBERT
        model_path = os.getenv("RUBERT_MODEL_PATH", "ml/models/rubert_final/final_model")
        confidence_threshold = float(os.getenv("RUBERT_THRESHOLD", "0.7"))
        
        logger.info(f"🤖 Загружаю RuBERT модель из {model_path}...")
        
        init_classifier(
            model_path=model_path,
            confidence_threshold=confidence_threshold
        )
        
        logger.info("✅ RuBERT классификатор успешно загружен!")
        logger.info(f"   📊 Порог уверенности: {confidence_threshold}")
        
    except FileNotFoundError as e:
        logger.warning(f"⚠️ RuBERT модель не найдена: {e}")
        logger.warning(f"   Проверьте путь к модели в .env: RUBERT_MODEL_PATH")
        logger.warning("   Бот будет работать без RuBERT (только DeepSeek API)")
    except ImportError as e:
        logger.warning(f"⚠️ Библиотеки для RuBERT не установлены: {e}")
        logger.warning("   Установите: pip install transformers torch")
        logger.warning("   Бот будет работать без RuBERT (только DeepSeek API)")
    except Exception as e:
        logger.warning(f"⚠️ Не удалось загрузить RuBERT: {e}", exc_info=True)
        logger.warning("   Бот будет работать без RuBERT (только DeepSeek API)")
    # ==========================================
    
    # ========== ИНИЦИАЛИЗАЦИЯ LSTM МОДЕЛИ ==========
    try:
        from ml.custom_lstm_classifier import init_custom_classifier
        
        # Путь к LSTM модели
        lstm_model_path = os.getenv("LSTM_MODEL_PATH", "ml/models/lstm_classifier_balanced")
        lstm_threshold = float(os.getenv("LSTM_THRESHOLD", "0.7"))
        
        # Проверяем существование папки с моделью
        if Path(lstm_model_path).exists():
            logger.info(f"🧠 Загружаю собственную LSTM модель из {lstm_model_path}...")
            
            init_custom_classifier(
                model_path=lstm_model_path,
                confidence_threshold=lstm_threshold
            )
            
            logger.info("✅ LSTM модель успешно загружена!")
            logger.info(f"   📊 Порог уверенности: {lstm_threshold}")
        else:
            logger.info(f"ℹ️ LSTM модель не найдена: {lstm_model_path}")
            logger.info("   Используется только RuBERT классификатор")
        
    except ImportError as e:
        logger.debug(f"ℹ️ Модуль custom_lstm_classifier не установлен: {e}")
    except Exception as e:
        logger.warning(f"⚠️ Не удалось загрузить LSTM модель: {e}")
        logger.warning("   Используется только RuBERT классификатор")
    # ===========================================================
    
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
