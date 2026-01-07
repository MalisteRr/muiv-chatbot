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
    from bot.handlers.auth_handlers import router as auth_router
    from bot.handlers.user import router as user_router
    from bot.handlers.admin import router as admin_router
    from bot.handlers.moderator import router as moderator_router
    
    # ========== ПРАВИЛЬНЫЙ ПОРЯДОК РОУТЕРОВ ==========
    # 
    # ВАЖНО: Роутеры обрабатываются СВЕРХУ ВНИЗ!
    # Первый подходящий обработчик перехватывает сообщение.
    #
    # Порядок:
    # 1. auth_router - обработка авторизации (/admin, /moderator)
    # 2. common_router - команды /start, /help (должны работать всегда)
    # 3. admin_router - кнопки и команды админа (с @require_role)
    # 4. moderator_router - кнопки и команды модератора (с @require_role)
    # 5. rating_handlers - обработка рейтингов
    # 6. user_router - ГЛАВНЫЙ обработчик текста (проверка пароля + вопросы)
    #
    # user_router ПОСЛЕДНИЙ потому что:
    # - Он имеет широкий @router.message(F.text) который ловит ВСЁ
    # - Внутри него проверка пароля (is_waiting_for_password)
    # - Если поставить раньше - перехватит команды админа/модератора
    
    dp.include_router(auth_router)          # 1. Авторизация
    dp.include_router(common_router)        # 2. Общие команды (/start, /help)
    dp.include_router(admin_router)         # 3. Админ (кнопки с @require_role)
    dp.include_router(moderator_router)     # 4. Модератор (кнопки с @require_role)
    dp.include_router(rating_handlers.router)  # 5. Рейтинги
    dp.include_router(user_router)          # 6. ПОСЛЕДНИЙ - вопросы + проверка пароля
    
    logger.info("✅ Все роутеры зарегистрированы в правильном порядке")
    
except Exception as e:
    logger.error(f"❌ Ошибка регистрации роутеров: {e}", exc_info=True)
    raise


logger.info("Dispatcher и Bot инициализированы")
