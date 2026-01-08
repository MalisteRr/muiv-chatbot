"""
Обработчики для модераторов
Урезанная панель управления с основными функциями мониторинга
"""

import logging
from datetime import datetime, timedelta
from aiogram import Router, F
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.filters import Command


from utils.auth_system import require_role, start_password_prompt, check_password, is_waiting_for_password, logout
from bot.keyboards import get_moderator_keyboard  # Импортируем клавиатуру!
from database.crud import (
    get_analytics_by_period,
    get_popular_questions,
    get_low_rated_messages,
    get_rating_statistics,
    export_analytics_csv,
    get_user_ratings
)

logger = logging.getLogger(__name__)
router = Router()


# ========== ЛОГИН/ЛОГАУТ ==========

@router.message(Command("moderator"))
async def cmd_moderator_start(message: Message):
    """Начало авторизации модератора или вход для админа"""
    
    user_id = message.from_user.id
    
    # Проверяем текущую роль
    from utils.auth_system import get_user_role
    from database.crud import get_user_info
    
    current_role = get_user_role(user_id)
    
    # Проверяем есть ли права админа в БД
    db_role = 'user'
    try:
        user_info = await get_user_info(user_id)
        if user_info:
            db_role = user_info.get('role', 'user')
    except Exception as e:
        logger.error(f"Ошибка проверки БД: {e}")
    
    # СЛУЧАЙ 1: Уже авторизован как модератор
    if current_role == 'moderator':
        await show_moderator_panel(message)
        return
    
    # СЛУЧАЙ 2: Админ (из БД или сессии) - входит БЕЗ ПАРОЛЯ
    if current_role == 'admin' or db_role == 'admin':
        logger.info(f"✅ Админ {user_id} входит в панель модератора без пароля")
        await show_moderator_panel(message)
        return
    
    # СЛУЧАЙ 3: Обычный пользователь - требуется пароль
    if start_password_prompt(user_id, 'moderator'):
        await message.answer(
            "🔐 <b>Вход в панель модератора</b>\n\n"
            "Введите пароль модератора:",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="❌ Отмена", callback_data="cancel_auth")]
            ])
        )
    else:
        await message.answer("❌ Ошибка авторизации")


@router.message(F.text == "🚪 Выйти из модератора")
async def cmd_moderator_logout(message: Message):
    """Выход из модератора"""
    user_id = message.from_user.id
    logout(user_id)
    
    from bot.keyboards import get_main_keyboard
    
    await message.answer(
        "👋 Вы вышли из панели модератора",
        reply_markup=get_main_keyboard()
    )


# ========== ПРОВЕРКА ПАРОЛЯ ==========

async def check_if_password_input(message: Message) -> bool:
    """
    Проверяет не является ли сообщение вводом пароля
    
    Returns:
        True если это ввод пароля (и он обработан)
        False если это обычное сообщение
        
    ИСПОЛЬЗОВАНИЕ:
    В основном обработчике вопросов добавь в начале:
    
    if await check_if_password_input(message):
        return  # Это был пароль, выходим
    
    # Дальше обработка обычного вопроса...
    """
    user_id = message.from_user.id
    
    # Проверяем ожидает ли пользователь ввода пароля
    if not is_waiting_for_password(user_id):
        return False  # Это НЕ пароль, обычное сообщение
    
    # Это попытка ввода пароля
    password = message.text.strip()
    role = check_password(user_id, password)
    
    if role:
        # Пароль верный
        await message.answer(
            f"✅ <b>Авторизация успешна!</b>\n\n"
            f"Вы вошли как: <b>{role}</b>"
        )
        
        # Показываем панель модератора
        await show_moderator_panel(message)
    else:
        # Пароль неверный
        await message.answer(
            "❌ <b>Неверный пароль!</b>\n\n"
            "Попробуйте еще раз или нажмите /moderator для повтора"
        )
    
    return True  # Это был ввод пароля, обработан


# ========== ГЛАВНАЯ ПАНЕЛЬ ==========

async def show_moderator_panel(message: Message):
    """Показать главную панель модератора"""
    
    logger.info(f"Показываю панель модератора для пользователя {message.from_user.id}")
    
    moderator_text = """🛡️ <b>Панель модератора</b>

<b>Доступные команды:</b>

📊 /mod_stats - Статистика за 7 дней
⭐ /ratings - Рейтинги пользователей
❓ /mod_popular - Топ-10 популярных вопросов
👎 /mod_low_rated - Низкие оценки
📥 /mod_export - Экспорт данных в CSV
🚪 /logout - Выход из панели

<b>Используйте кнопки ниже ⬇️</b>"""
    
    # ОДНО сообщение с Reply клавиатурой
    keyboard = get_moderator_keyboard()
    logger.info(f"Клавиатура создана: {keyboard is not None}")
    
    await message.answer(
        moderator_text,
        reply_markup=keyboard
    )
    
    logger.info(f"Панель модератора отправлена для {message.from_user.id}")


# ==================== ОБРАБОТЧИКИ КНОПОК ====================

@router.message(F.text == "📊 Статистика")
@require_role(["moderator", "admin"])
async def handle_stats_button(message: Message):
    """Кнопка статистики"""
    await cmd_mod_stats(message)


@router.message(F.text == "⭐ Рейтинги")
@require_role(["moderator", "admin"])
async def handle_ratings_button(message: Message):
    """Кнопка рейтингов"""
    await cmd_mod_ratings(message)


@router.message(F.text == "❓ Популярные")
@require_role(["moderator", "admin"])
async def handle_popular_button(message: Message):
    """Кнопка популярных вопросов"""
    await cmd_mod_popular(message)


@router.message(F.text == "👎 Низкие оценки")
@require_role(["moderator", "admin"])
async def handle_low_rated_button(message: Message):
    """Кнопка низких оценок"""
    await cmd_mod_low_rated(message)


@router.message(F.text == "📥 Экспорт")
@require_role(["moderator", "admin"])
async def handle_export_button(message: Message):
    """Кнопка экспорта"""
    await cmd_mod_export(message)


@router.message(F.text == "🔙 Главное меню")
@require_role(["moderator", "admin"])
async def handle_back_button(message: Message):
    """Вернуться в главное меню (как пользователь)"""
    from bot.keyboards import get_main_keyboard
    
    logout(message.from_user.id)
    
    await message.answer(
        "Вы вернулись в главное меню.\n"
        "Для входа в панель модератора используйте /moderator",
        reply_markup=get_main_keyboard()
    )


# ==================== СТАТИСТИКА ====================

@router.message(Command("mod_stats"))
@require_role(["moderator", "admin"])
async def cmd_mod_stats(message: Message):
    """Статистика за последние 7 дней"""
    
    try:
        # Получаем аналитику за 7 дней
        analytics = await get_analytics_by_period(days=7)
        
        if not analytics:
            await message.answer("⚠️ Нет данных за последние 7 дней")
            return
        
        text = (
            "📊 <b>Статистика за 7 дней</b>\n\n"
            f"👥 Пользователей: {analytics.get('total_users', 0)}\n"
            f"💬 Сообщений: {analytics.get('total_messages', 0)}\n"
            f"✅ Найдено в БД: {analytics.get('found_in_db', 0)}\n"
            f"❌ Не найдено: {analytics.get('not_found', 0)}\n"
            f"📈 Процент попадания: {analytics.get('hit_rate', 0):.1f}%\n\n"
            f"⭐ Средний рейтинг: {analytics.get('avg_rating', 0):.2f}/5\n"
            f"👍 Положительных: {analytics.get('positive_feedback', 0)}\n"
            f"👎 Отрицательных: {analytics.get('negative_feedback', 0)}"
        )
        
        await message.answer(text)
        
    except Exception as e:
        logger.error(f"Ошибка получения статистики: {e}")
        await message.answer("❌ Ошибка получения статистики")


# ==================== РЕЙТИНГИ ====================

@router.message(Command("ratings"))
@require_role(["moderator", "admin"])
async def cmd_mod_ratings(message: Message):
    """Статистика по рейтингам"""
    
    try:
        # Получаем статистику за 7 дней
        stats = await get_rating_statistics(days=7)
        
        if not stats or stats.get('total_ratings', 0) == 0:
            await message.answer("⚠️ Нет рейтингов за последние 7 дней")
            return
        
        text = (
            "⭐ <b>Рейтинги за 7 дней</b>\n\n"
            f"📊 Всего оценок: {stats.get('total_ratings', 0)}\n"
            f"⭐ Средний рейтинг: {stats.get('avg_rating', 0):.2f}/5\n\n"
            f"👍 Положительных (4-5): {stats.get('positive', 0)}\n"
            f"👎 Отрицательных (1-2): {stats.get('negative', 0)}\n"
        )
        
        # Типы отзывов
        feedback_types = stats.get('feedback_types', {})
        if feedback_types:
            text += "\n<b>Типы отзывов:</b>\n"
            type_names = {
                'good': '✅ Полезно',
                'bad': '❌ Бесполезно',
                'no_info': '❓ Нет информации',
                'unclear': '😕 Непонятно',
                'incorrect': '⚠️ Неверно'
            }
            for ftype, count in feedback_types.items():
                name = type_names.get(ftype, ftype)
                text += f"• {name}: {count}\n"
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="👎 Низкие оценки", callback_data="mod_low_rated_inline"),
                InlineKeyboardButton(text="📥 Экспорт", callback_data="mod_export_ratings")
            ]
        ])
        
        await message.answer(text, reply_markup=keyboard)
        
    except Exception as e:
        logger.error(f"Ошибка получения рейтингов: {e}")
        await message.answer("❌ Ошибка получения рейтингов")


# ==================== ПОПУЛЯРНЫЕ ВОПРОСЫ ====================

@router.message(Command("mod_popular"))
@require_role(["moderator", "admin"])
async def cmd_mod_popular(message: Message):
    """Топ-10 популярных вопросов"""
    
    try:
        questions = await get_popular_questions(limit=10, days=30)
        
        if not questions:
            await message.answer("⚠️ Нет данных о популярных вопросах")
            return
        
        text = "❓ <b>Топ-10 популярных вопросов (30 дней)</b>\n\n"
        
        for i, q in enumerate(questions, 1):
            question = q.get('question', 'N/A')
            count = q.get('count', 0)
            category = q.get('category', 'Без категории')
            
            # Обрезаем длинный вопрос
            if len(question) > 60:
                question = question[:60] + "..."
            
            text += f"{i}. <b>{question}</b>\n"
            text += f"   📊 Запросов: {count} | 🏷 {category}\n\n"
        
        await message.answer(text)
        
    except Exception as e:
        logger.error(f"Ошибка получения популярных вопросов: {e}")
        await message.answer("❌ Ошибка получения данных")


# ==================== НИЗКИЕ ОЦЕНКИ ====================

@router.message(Command("mod_low_rated"))
@require_role(["moderator", "admin"])
async def cmd_mod_low_rated(message: Message):
    """Сообщения с низкими оценками"""
    
    try:
        messages_data = await get_low_rated_messages(limit=5)
        
        if not messages_data:
            await message.answer("✅ Нет сообщений с низкими оценками!")
            return
        
        text = "👎 <b>Сообщения с низкими оценками (последние 5)</b>\n\n"
        
        for msg in messages_data:
            question = msg.get('user_question', 'N/A')
            rating = msg.get('rating', 0)
            feedback_type = msg.get('feedback_type', '')
            comment = msg.get('comment', '')
            created = msg.get('created_at', '')
            
            # Защита от None
            if question is None:
                question = 'N/A'
            if comment is None:
                comment = ''
            
            # Обрезаем длинный вопрос
            if len(question) > 80:
                question = question[:80] + "..."
            
            text += f"❓ <i>{question}</i>\n"
            text += f"⭐ Оценка: {rating}/5"
            
            if feedback_type:
                type_emoji = {
                    'bad': '❌',
                    'no_info': '❓',
                    'unclear': '😕',
                    'incorrect': '⚠️'
                }
                emoji = type_emoji.get(feedback_type, '📝')
                text += f" | {emoji} {feedback_type}"
            
            text += f"\n📅 {created[:16] if created else 'N/A'}\n"
            
            if comment:
                comment_short = comment[:100] + "..." if len(comment) > 100 else comment
                text += f"💬 <i>{comment_short}</i>\n"
            
            text += "\n"
        
        text += "💡 <i>Используйте эти данные для улучшения базы знаний</i>"
        
        await message.answer(text)
        
    except Exception as e:
        logger.error(f"Ошибка получения низко оценённых сообщений: {e}")
        await message.answer("❌ Ошибка получения данных")


# ==================== ЭКСПОРТ ДАННЫХ ====================

@router.message(Command("mod_export"))
@require_role(["moderator", "admin"])
async def cmd_mod_export(message: Message):
    """Экспорт данных в CSV"""
    
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="📊 Аналитика", callback_data="mod_export_analytics"),
            InlineKeyboardButton(text="⭐ Рейтинги", callback_data="mod_export_ratings")
        ]
    ])
    
    await message.answer(
        "📥 <b>Экспорт данных</b>\n\n"
        "Выберите тип данных для экспорта:",
        reply_markup=keyboard
    )


# ==================== CALLBACK HANDLERS ====================
# Оставлены только для inline кнопок внутри команд (экспорт и т.д.)

@router.callback_query(F.data == "mod_low_rated_inline")
async def handle_mod_low_rated_inline(callback):
    """Показать низкие оценки из inline кнопки"""
    await callback.answer()
    await cmd_mod_low_rated(callback.message)


@router.callback_query(F.data == "mod_export_analytics")
async def handle_mod_export_analytics(callback):
    """Экспорт аналитики"""
    try:
        await callback.message.answer("⏳ Подготовка файла аналитики...")
        
        csv_data = await export_analytics_csv(days=30)
        
        if not csv_data:
            await callback.message.answer("⚠️ Нет данных для экспорта")
            return
        
        filename = f"analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        from io import BytesIO
        from aiogram.types import BufferedInputFile
        
        file = BufferedInputFile(
            csv_data.encode('utf-8-sig'),
            filename=filename
        )
        
        await callback.message.answer_document(
            file,
            caption="📊 Аналитика за последние 30 дней"
        )
        
    except Exception as e:
        logger.error(f"Ошибка экспорта аналитики: {e}")
        await callback.message.answer("❌ Ошибка экспорта данных")


@router.callback_query(F.data == "mod_export_ratings")
async def handle_mod_export_ratings(callback):
    """Экспорт рейтингов"""
    try:
        await callback.message.answer("⏳ Подготовка файла рейтингов...")
        
        csv_data = await get_user_ratings(days=30)
        
        if not csv_data:
            await callback.message.answer("⚠️ Нет данных для экспорта")
            return
        
        filename = f"ratings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        from io import BytesIO
        from aiogram.types import BufferedInputFile
        
        file = BufferedInputFile(
            csv_data.encode('utf-8-sig'),
            filename=filename
        )
        
        await callback.message.answer_document(
            file,
            caption="⭐ Рейтинги за последние 30 дней"
        )
        
    except Exception as e:
        logger.error(f"Ошибка экспорта рейтингов: {e}")
        await callback.message.answer("❌ Ошибка экспорта данных")


@router.callback_query(F.data == "cancel_auth")
async def handle_cancel_auth(callback):
    """Отмена авторизации"""
    from utils.auth_system import cancel_password_prompt
    
    cancel_password_prompt(callback.from_user.id)
    
    await callback.message.delete()
    await callback.message.answer("❌ Авторизация отменена")


# ==================== LOGOUT ====================

@router.message(Command("logout"))
async def cmd_logout(message: Message):
    """Выход из сессии"""
    user_id = message.from_user.id
    
    from utils.auth_system import get_user_role
    current_role = get_user_role(user_id)
    
    if current_role == 'user':
        await message.answer("⚠️ Вы не авторизованы")
        return
    
    logout(user_id)
    
    from bot.keyboards import get_main_keyboard
    
    await message.answer(
        f"👋 Вы вышли из роли <b>{current_role}</b>",
        reply_markup=get_main_keyboard()
    )
