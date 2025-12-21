"""
Обработчики команд для администраторов
Управление ботом и статистика
"""

import logging
from datetime import datetime, timedelta
from aiogram import Router, F
from aiogram.filters import Command
from aiogram.types import Message

from config import config
from database.crud import (
    get_total_stats,
    get_popular_questions,
    get_unanswered_questions,
    get_recent_users,
    export_analytics_csv
)
from bot.keyboards import get_admin_keyboard
from utils.helpers import is_admin

logger = logging.getLogger(__name__)
router = Router(name='admin')


@router.message(Command("admin"))
async def cmd_admin_panel(message: Message):
    """
    Админ-панель
    Доступна только администраторам
    """
    if not is_admin(message.from_user.id):
        await message.answer("❌ У вас нет доступа к административной панели.")
        return
    
    admin_text = """🔐 **Административная панель МУИВ Bot**

**Доступные команды:**

📊 `/stats_full` - Полная статистика бота
📈 `/analytics` - Аналитика за период
💬 `/popular` - Популярные вопросы
❌ `/unanswered` - Вопросы без ответов
👥 `/users` - Список последних пользователей
📥 `/export` - Экспорт данных
🔄 `/reload_kb` - Перезагрузить базу знаний
📢 `/broadcast` - Рассылка сообщения
🛠️ `/debug` - Режим отладки

**Настройки:**
⚙️ `/set_welcome` - Изменить приветствие
⚙️ `/set_model` - Сменить AI модель
⚙️ `/maintenance` - Режим обслуживания"""
    
    await message.answer(
        admin_text,
        parse_mode="Markdown",
        reply_markup=get_admin_keyboard()
    )
    
    logger.info(f"Администратор {message.from_user.id} открыл админ-панель")


@router.message(Command("stats_full"))
async def cmd_full_stats(message: Message):
    """Полная статистика бота"""
    if not is_admin(message.from_user.id):
        return
    
    await message.answer("⏳ Собираю статистику...")
    
    try:
        stats = await get_total_stats()
        
        stats_text = f"""📊 **Полная статистика бота**

**Пользователи:**
👥 Всего: {stats['total_users']}
🆕 Новых за сегодня: {stats['new_today']}
📅 Новых за неделю: {stats['new_week']}
💬 Активных за сутки: {stats['active_today']}

**Сообщения:**
📨 Всего обработано: {stats['total_messages']}
📈 Сегодня: {stats['messages_today']}
📊 За неделю: {stats['messages_week']}

**База знаний:**
📚 Всего вопросов в FAQ: {stats['total_faq']}
✅ Категорий: {stats['total_categories']}
🏷️ Уникальных keywords: {stats['total_keywords']}

**Эффективность:**
✅ Найдено ответов: {stats['found_answers']} ({stats['success_rate']:.1f}%)
❌ Не найдено: {stats['not_found']} ({100 - stats['success_rate']:.1f}%)
⭐ Средняя оценка: {stats['avg_rating']:.2f}/5

**Система:**
🤖 AI модель: {config.ai.model}
💾 База данных: PostgreSQL
⚡ Uptime: {stats['uptime']}

_Обновлено: {datetime.now().strftime('%d.%m.%Y %H:%M')}_"""
        
        await message.answer(stats_text, parse_mode="Markdown")
        
    except Exception as e:
        logger.error(f"Ошибка получения статистики: {e}", exc_info=True)
        await message.answer("❌ Ошибка при получении статистики")


@router.message(Command("analytics"))
async def cmd_analytics(message: Message):
    """Аналитика за период"""
    if not is_admin(message.from_user.id):
        return
    
    # TODO: Добавить выбор периода (сегодня/неделя/месяц)
    # Пока показываем за последние 7 дней
    
    try:
        from database.crud import get_analytics_by_period
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        analytics = await get_analytics_by_period(start_date, end_date)
        
        analytics_text = f"""📈 **Аналитика за последние 7 дней**

**Активность по дням:**
{analytics['daily_activity']}

**Топ-5 категорий:**
{analytics['top_categories']}

**Пиковые часы:**
{analytics['peak_hours']}

**Конверсия:**
Успешных ответов: {analytics['conversion_rate']:.1f}%
Среднее время ответа: {analytics['avg_response_time']:.2f}с

**Удовлетворенность:**
😊 Положительных: {analytics['positive_feedback']}%
😐 Нейтральных: {analytics['neutral_feedback']}%
😔 Негативных: {analytics['negative_feedback']}%"""
        
        await message.answer(analytics_text, parse_mode="Markdown")
        
    except Exception as e:
        logger.error(f"Ошибка аналитики: {e}", exc_info=True)
        await message.answer("❌ Ошибка при получении аналитики")


@router.message(Command("popular"))
async def cmd_popular_questions(message: Message):
    """Популярные вопросы"""
    if not is_admin(message.from_user.id):
        return
    
    try:
        popular = await get_popular_questions(limit=10)
        
        if not popular:
            await message.answer("📊 Статистика вопросов пока пуста")
            return
        
        text = "🔥 **Топ-10 популярных вопросов:**\n\n"
        
        for i, item in enumerate(popular, 1):
            text += f"{i}. `{item['question'][:60]}...`\n"
            text += f"   Спрашивали: {item['count']} раз\n"
            text += f"   Категория: {item['category']}\n\n"
        
        await message.answer(text, parse_mode="Markdown")
        
    except Exception as e:
        logger.error(f"Ошибка получения популярных вопросов: {e}", exc_info=True)
        await message.answer("❌ Ошибка при получении данных")


@router.message(Command("unanswered"))
async def cmd_unanswered_questions(message: Message):
    """Вопросы без ответов"""
    if not is_admin(message.from_user.id):
        return
    
    try:
        unanswered = await get_unanswered_questions(limit=20)
        
        if not unanswered:
            await message.answer("✅ Все вопросы успешно обработаны!")
            return
        
        text = "❌ **Вопросы без ответа в базе знаний:**\n\n"
        text += "_Эти вопросы нужно добавить в FAQ_\n\n"
        
        for i, item in enumerate(unanswered, 1):
            text += f"{i}. `{item['question'][:70]}...`\n"
            text += f"   Дата: {item['timestamp']}\n"
            text += f"   Пользователь: {item['user_id']}\n\n"
            
            if i >= 10:  # Ограничим вывод
                text += f"_...и еще {len(unanswered) - 10} вопросов_"
                break
        
        await message.answer(text, parse_mode="Markdown")
        
    except Exception as e:
        logger.error(f"Ошибка получения необработанных вопросов: {e}", exc_info=True)
        await message.answer("❌ Ошибка при получении данных")


@router.message(Command("users"))
async def cmd_recent_users(message: Message):
    """Список последних пользователей"""
    if not is_admin(message.from_user.id):
        return
    
    try:
        users = await get_recent_users(limit=15)
        
        if not users:
            await message.answer("👥 Список пользователей пуст")
            return
        
        text = "👥 **Последние пользователи:**\n\n"
        
        for user in users:
            status = "🟢" if user['is_active'] else "⚪"
            text += f"{status} `{user['user_id']}` - {user['name']}\n"
            text += f"   Сообщений: {user['messages_count']}\n"
            text += f"   Последнее: {user['last_activity']}\n\n"
        
        await message.answer(text, parse_mode="Markdown")
        
    except Exception as e:
        logger.error(f"Ошибка получения пользователей: {e}", exc_info=True)
        await message.answer("❌ Ошибка при получении данных")


@router.message(Command("export"))
async def cmd_export_data(message: Message):
    """Экспорт данных в CSV"""
    if not is_admin(message.from_user.id):
        return
    
    await message.answer("⏳ Формирую отчет...")
    
    try:
        # Экспорт аналитики в CSV
        csv_data = await export_analytics_csv()
        
        if csv_data:
            # Отправка файла
            from aiogram.types import BufferedInputFile
            
            filename = f"muiv_bot_analytics_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
            file = BufferedInputFile(csv_data.encode('utf-8'), filename=filename)
            
            await message.answer_document(
                document=file,
                caption="📊 Отчет по аналитике бота"
            )
        else:
            await message.answer("❌ Нет данных для экспорта")
        
    except Exception as e:
        logger.error(f"Ошибка экспорта данных: {e}", exc_info=True)
        await message.answer("❌ Ошибка при экспорте данных")


@router.message(Command("reload_kb"))
async def cmd_reload_knowledge_base(message: Message):
    """Перезагрузка базы знаний"""
    if not is_admin(message.from_user.id):
        return
    
    await message.answer("⏳ Перезагружаю базу знаний...")
    
    try:
        # TODO: Реализовать перезагрузку кэша/индекса FAQ
        from ml.knowledge_base import reload_knowledge_base
        
        result = await reload_knowledge_base()
        
        if result['success']:
            await message.answer(
                f"✅ База знаний перезагружена!\n\n"
                f"📚 Загружено записей: {result['count']}\n"
                f"🏷️ Категорий: {result['categories']}"
            )
        else:
            await message.answer(f"❌ Ошибка: {result['error']}")
        
    except Exception as e:
        logger.error(f"Ошибка перезагрузки базы знаний: {e}", exc_info=True)
        await message.answer("❌ Ошибка при перезагрузке")


@router.message(Command("broadcast"))
async def cmd_broadcast(message: Message):
    """Рассылка сообщения всем пользователям"""
    if not is_admin(message.from_user.id):
        return
    
    # TODO: Реализовать механизм рассылки
    await message.answer(
        "📢 **Рассылка сообщений**\n\n"
        "Эта функция в разработке.\n"
        "Для рассылки используйте команду:\n"
        "`/broadcast_send <текст сообщения>`",
        parse_mode="Markdown"
    )


@router.message(Command("debug"))
async def cmd_toggle_debug(message: Message):
    """Переключение режима отладки"""
    if not is_admin(message.from_user.id):
        return
    
    config.debug = not config.debug
    
    status = "включен" if config.debug else "выключен"
    emoji = "🔍" if config.debug else "🔒"
    
    await message.answer(
        f"{emoji} Режим отладки **{status}**",
        parse_mode="Markdown"
    )
    
    logger.info(f"Режим отладки переключен: {config.debug}")