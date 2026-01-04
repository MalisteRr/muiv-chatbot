"""
Инициализация бота и диспетчера
Регистрация всех обработчиков
"""

import logging
from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode
from config import config
from bot.handlers import rating_handlers as rating_handlers

logger = logging.getLogger(__name__)

# Инициализация бота
bot = Bot(
    token=config.bot.token,
    parse_mode=ParseMode.HTML
)

# Инициализация диспетчера
dp = Dispatcher()

# Импорт и регистрация роутеров
try:
    from bot.handlers.common import router as common_router
    from bot.handlers.user import router as user_router
    from bot.handlers.admin import router as admin_router
    
    # Регистрация роутеров (порядок важен!)
    dp.include_router(admin_router)
    dp.include_router(common_router)
    dp.include_router(rating_handlers.router)
    dp.include_router(user_router)
    logger.info("✅ Все роутеры зарегистрированы")
    
except Exception as e:
    logger.error(f"❌ Ошибка регистрации роутеров: {e}", exc_info=True)
    raise


logger.info("Dispatcher и Bot инициализированы")
