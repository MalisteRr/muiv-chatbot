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
    get_rating_statistics,
    get_low_rated_messages,
    export_analytics_csv
)
from bot.keyboards import get_admin_keyboard
from utils.auth_system import require_role  # новый декоратор

logger = logging.getLogger(__name__)
router = Router(name='admin')


# ========== КОМАНДЫ ==========

@router.message(Command("admin"))
@require_role("admin")
async def cmd_admin_panel(message: Message):
    """Админ-панель"""
    admin_text = """🔐 **Админ-панель Бота**

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
@require_role("admin")
async def cmd_full_stats(message: Message):
    """Полная статистика бота"""
    await message.answer("⏳ Собираю статистику...")
    try:
        stats = await get_total_stats()
        import html
        stats_text = (
            "<b>📊 Полная статистика бота</b>\n\n"
            f"<b>Пользователи:</b>\n"
            f"👥 Всего: {html.escape(str(stats.get('total_users', 'N/A')))}\n"
            f"🆕 Новых за сегодня: {html.escape(str(stats.get('new_today', 'N/A')))}\n"
            f"📅 Новых за неделю: {html.escape(str(stats.get('new_week', 'N/A')))}\n"
            f"💬 Активных за сутки: {html.escape(str(stats.get('active_today', 'N/A')))}\n\n"
            f"<b>Сообщения:</b>\n"
            f"📨 Всего обработано: {html.escape(str(stats.get('total_messages', 'N/A')))}\n"
            f"📈 Сегодня: {html.escape(str(stats.get('messages_today', 'N/A')))}\n"
            f"📊 За неделю: {html.escape(str(stats.get('messages_week', 'N/A')))}\n\n"
            f"<b>База знаний:</b>\n"
            f"📚 Всего вопросов в FAQ: {html.escape(str(stats.get('total_faq', 'N/A')))}\n"
            f"✅ Категорий: {html.escape(str(stats.get('total_categories', 'N/A')))}\n"
            f"🏷️ Уникальных keywords: {html.escape(str(stats.get('total_keywords', 'N/A')))}\n\n"
            f"<b>Эффективность:</b>\n"
            f"✅ Найдено ответов: {html.escape(str(stats.get('found_answers', 'N/A')))} ({html.escape(str(round(stats.get('success_rate', 0), 1)))}%)\n"
            f"❌ Не найдено: {html.escape(str(stats.get('not_found', 'N/A')))} ({html.escape(str(round(100 - stats.get('success_rate', 0), 1)))}%)\n"
            f"⭐ Средняя оценка: {html.escape(str(round(stats.get('avg_rating', 0), 2)))} /5\n\n"
            f"<b>Система:</b>\n"
            f"🤖 AI модель: {html.escape(str(getattr(config.ai, 'model', 'N/A')))}\n"
            f"💾 База данных: PostgreSQL\n"
            f"⚡ Uptime: {html.escape(str(stats.get('uptime', 'N/A')))}\n\n"
            f"<i>Обновлено: {html.escape(datetime.now().strftime('%d.%m.%Y %H:%M'))}</i>"
        )
        await message.answer(stats_text, parse_mode="HTML")
    except Exception as e:
        logger.error(f"Ошибка получения статистики: {e}", exc_info=True)
        await message.answer("❌ Ошибка при получении статистики")


@router.message(Command("analytics"))
@require_role("admin")
async def cmd_analytics(message: Message):
    """Аналитика за период"""
    try:
        from database.crud import get_analytics_by_period
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        analytics = await get_analytics_by_period(start_date, end_date)
        import html
        analytics_text = (
            "<b>📈 Аналитика за последние 7 дней</b>\n\n"
            f"<b>Активность по дням:</b>\n{html.escape(str(analytics.get('daily_activity', 'N/A')))}\n\n"
            f"<b>Топ-5 категорий:</b>\n{html.escape(str(analytics.get('top_categories', 'N/A')))}\n\n"
            f"<b>Пиковые часы:</b>\n{html.escape(str(analytics.get('peak_hours', 'N/A')))}\n\n"
            f"<b>Конверсия:</b>\nУспешных ответов: {html.escape(str(round(analytics.get('conversion_rate', 0), 1)))}%\n"
            f"Среднее время ответа: {html.escape(str(round(analytics.get('avg_response_time', 0), 2)))}с\n\n"
            f"<b>Удовлетворенность:</b>\n😊 Положительных: {html.escape(str(analytics.get('positive_feedback', 'N/A')))}%\n"
            f"😐 Нейтральных: {html.escape(str(analytics.get('neutral_feedback', 'N/A')))}%\n"
            f"😔 Негативных: {html.escape(str(analytics.get('negative_feedback', 'N/A')))}%"
        )
        await message.answer(analytics_text, parse_mode="HTML")
    except Exception as e:
        logger.error(f"Ошибка аналитики: {e}", exc_info=True)
        await message.answer("❌ Ошибка при получении аналитики")


@router.message(Command("popular"))
@require_role("admin")
async def cmd_popular_questions(message: Message):
    """Популярные вопросы"""
    try:
        popular = await get_popular_questions(limit=10)
        if not popular:
            await message.answer("📊 Статистика вопросов пока пуста")
            return
        import html
        text = "<b>🔥 Топ-10 популярных вопросов:</b>\n\n"
        for i, item in enumerate(popular, 1):
            q = html.escape(str(item.get('question', ''))[:60])
            count = html.escape(str(item.get('count', '0')))
            category = html.escape(str(item.get('category', '')))
            text += f"{i}. <code>{q}...</code>\n"
            text += f"   Спрашивали: {count} раз\n"
            text += f"   Категория: {category}\n\n"
        await message.answer(text, parse_mode="HTML")
    except Exception as e:
        logger.error(f"Ошибка получения популярных вопросов: {e}", exc_info=True)
        await message.answer("❌ Ошибка при получении данных")


@router.message(Command("unanswered"))
@require_role("admin")
async def cmd_unanswered_questions(message: Message):
    """Вопросы без ответов"""
    try:
        unanswered = await get_unanswered_questions(limit=20)
        if not unanswered:
            await message.answer("✅ Все вопросы успешно обработаны!")
            return
        import html
        text = "<b>❌ Вопросы без ответа в базе знаний:</b>\n\n<i>Эти вопросы нужно добавить в FAQ</i>\n\n"
        for i, item in enumerate(unanswered, 1):
            q = html.escape(str(item.get('question', ''))[:70])
            timestamp = html.escape(str(item.get('timestamp', '')))
            user_id = html.escape(str(item.get('user_id', '')))
            text += f"{i}. <code>{q}...</code>\n"
            text += f"   Дата: {timestamp}\n"
            text += f"   Пользователь: {user_id}\n\n"
            if i >= 10:
                text += f"<i>...и еще {len(unanswered) - 10} вопросов</i>"
                break
        await message.answer(text, parse_mode="HTML")
    except Exception as e:
        logger.error(f"Ошибка получения необработанных вопросов: {e}", exc_info=True)
        await message.answer("❌ Ошибка при получении данных")


@router.message(Command("users"))
@require_role("admin")
async def cmd_recent_users(message: Message):
    """Список последних пользователей"""
    try:
        users = await get_recent_users(limit=15)
        if not users:
            await message.answer("👥 Список пользователей пуст")
            return
        import html
        text = "<b>👥 Последние пользователи:</b>\n\n"
        for user in users:
            status = "🟢" if user.get('is_active') else "⚪"
            user_id = html.escape(str(user.get('user_id', 'N/A')))
            name = html.escape(str(user.get('name', '')))
            messages_count = html.escape(str(user.get('messages_count', 0)))
            last_activity = html.escape(str(user.get('last_activity', 'N/A')))
            text += f"{status} <code>{user_id}</code> - {name}\n"
            text += f"   Сообщений: {messages_count}\n"
            text += f"   Последнее: {last_activity}\n\n"
        await message.answer(text, parse_mode="HTML")
    except Exception as e:
        logger.error(f"Ошибка получения пользователей: {e}", exc_info=True)
        await message.answer("❌ Ошибка при получении данных")


@router.message(Command("export"))
@require_role("admin")
async def cmd_export_data(message: Message):
    """Экспорт данных в CSV"""
    await message.answer("⏳ Формирую отчет...")
    try:
        csv_data = await export_analytics_csv()
        if csv_data:
            from aiogram.types import BufferedInputFile
            filename = f"muiv_bot_analytics_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
            file_bytes = csv_data.encode('utf-8-sig')
            file = BufferedInputFile(file_bytes, filename=filename)
            await message.answer_document(document=file, caption="📊 Отчет по аналитике бота")
        else:
            await message.answer("❌ Нет данных для экспорта")
    except Exception as e:
        logger.error(f"Ошибка экспорта данных: {e}", exc_info=True)
        await message.answer("❌ Ошибка при экспорте данных")


@router.message(Command("reload_kb"))
@require_role("admin")
async def cmd_reload_knowledge_base(message: Message):
    """Перезагрузка базы знаний"""
    await message.answer("⏳ Перезагружаю базу знаний...")
    try:
        from ml.knowledge_base import reload_knowledge_base
        result = await reload_knowledge_base()
        if result['success']:
            await message.answer(
                f"✅ База знаний перезагружена!\n\n"
                f"📚 Загружено записей: {result['count']}\n"
                f"🏷️ Категорий: {result['categories']}"
            )
        else:
            import html
            await message.answer("❌ Ошибка: " + html.escape(str(result.get('error', ''))), parse_mode="HTML")
    except Exception as e:
        logger.error(f"Ошибка перезагрузки базы знаний: {e}", exc_info=True)
        await message.answer("❌ Ошибка при перезагрузке")


@router.message(Command("broadcast"))
@require_role("admin")
async def cmd_broadcast(message: Message):
    """Рассылка сообщения всем пользователям"""
    await message.answer(
        "📢 <b>Рассылка сообщений</b>\n\n"
        "Эта функция в разработке.\n"
        "Для рассылки используйте команду:\n"
        "<code>/broadcast_send &lt;текст сообщения&gt;</code>",
        parse_mode="HTML"
    )


@router.message(Command("debug"))
@require_role("admin")
async def cmd_toggle_debug(message: Message):
    """Переключение режима отладки"""
    config.debug = not config.debug
    status = "включен" if config.debug else "выключен"
    emoji = "🔍" if config.debug else "🔒"
    import html
    await message.answer(f"{emoji} Режим отладки <b>{html.escape(status)}</b>", parse_mode="HTML")
    logger.info(f"Режим отладки переключен: {config.debug}")


@router.message(Command("ratings"))
@require_role("admin")
async def cmd_ratings_stats(message: Message):
    """Статистика по рейтингам ответов"""
    try:
        args = message.text.split()
        days = int(args[1]) if len(args) > 1 and args[1].isdigit() else 7
        stats = await get_rating_statistics(days=days)
        if not stats or stats['total_ratings'] == 0:
            await message.answer(f"📊 Нет оценок за последние {days} дней")
            return
        total = stats['total_ratings']
        avg = stats['avg_rating']
        positive = stats['positive']
        negative = stats['negative']
        neutral = total - positive - negative
        positive_pct = (positive / total * 100) if total > 0 else 0
        negative_pct = (negative / total * 100) if total > 0 else 0
        neutral_pct = (neutral / total * 100) if total > 0 else 0
        text = f"""📊 <b>Статистика рейтингов за {days} дней</b>

📈 <b>Общая статистика:</b>
• Всего оценок: {total}
• Средний рейтинг: {avg} ⭐
• Положительных: {positive} ({positive_pct:.1f}%) 👍
• Нейтральных: {neutral} ({neutral_pct:.1f}%) 😐
• Отрицательных: {negative} ({negative_pct:.1f}%) 👎

"""
        if stats.get('feedback_types'):
            text += "<b>Причины плохих оценок:</b>\n"
            for feedback_type, count in stats['feedback_types'].items():
                emoji = {
                    'bad_no_info': '❌',
                    'bad_unclear': '🤔',
                    'bad_incorrect': '📊',
                    'bad': '👎'
                }.get(feedback_type, '•')
                type_name = {
                    'bad_no_info': 'Нет нужной информации',
                    'bad_unclear': 'Ответ непонятен',
                    'bad_incorrect': 'Информация неточная',
                    'bad': 'Не указана причина',
                    'good': 'Положительные'
                }.get(feedback_type, feedback_type)
                text += f"{emoji} {type_name}: {count}\n"
        await message.answer(text, parse_mode="HTML")
    except Exception as e:
        logger.error(f"Ошибка получения статистики рейтингов: {e}", exc_info=True)
        await message.answer("❌ Ошибка при получении статистики")


@router.message(Command("bad_rated"))
@require_role("admin")
async def cmd_low_rated_messages(message: Message):
    """Список сообщений с плохими оценками"""
    try:
        args = message.text.split()
        limit = int(args[1]) if len(args) > 1 and args[1].isdigit() else 10
        messages_list = await get_low_rated_messages(limit=limit)
        if not messages_list:
            await message.answer("✅ Нет плохо оценённых сообщений")
            return
        text = f"👎 <b>Последние {len(messages_list)} плохо оценённых ответов:</b>\n\n"
        for idx, msg in enumerate(messages_list, 1):
            import html
            user_q = html.escape(str(msg.get('user_question') or 'N/A')[:100])
            bot_ans = html.escape(str(msg.get('bot_response') or 'N/A')[:100])
            rating = msg['rating']
            feedback_type = msg['feedback_type'] or 'не указано'
            comment = html.escape(str(msg.get('comment') or ''))
            date = html.escape(str(msg.get('created_at', 'N/A'))[:16])
            text += f"<b>{idx}.</b> Рейтинг: {rating}⭐\n"
            text += f"   Дата: {date}\n"
            text += f"   Вопрос: {user_q}...\n"
            text += f"   Ответ: {bot_ans}...\n"
            if comment:
                text += f"   💬 Причина: {comment}\n"
            text += "\n"
            if len(text) > 3500:
                await message.answer(text, parse_mode="HTML")
                text = ""
        if text:
            await message.answer(text, parse_mode="HTML")
    except Exception as e:
        logger.error(f"Ошибка получения плохо оценённых сообщений: {e}", exc_info=True)
        await message.answer("❌ Ошибка при получении данных")


@router.message(Command("rating_export"))
@require_role("admin")
async def cmd_export_ratings(message: Message):
    """Экспорт всех рейтингов в CSV"""
    try:
        import io, csv
        from datetime import datetime
        messages_list = await get_low_rated_messages(limit=1000)  # экспорт последних 1000
        if not messages_list:
            await message.answer("❌ Нет данных для экспорта")
            return
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["id", "user_question", "bot_response", "rating", "feedback_type", "comment", "date"])
        for msg in messages_list:
            writer.writerow([
                msg.get("id"),
                msg.get("user_question"),
                msg.get("bot_response"),
                msg.get("rating"),
                msg.get("feedback_type"),
                msg.get("comment"),
                msg.get("created_at")
            ])
        output.seek(0)
        from aiogram.types import BufferedInputFile
        filename = f"low_rated_messages_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        file = BufferedInputFile(output.read().encode("utf-8-sig"), filename=filename)
        await message.answer_document(file, caption="📤 Экспорт низко оценённых сообщений")
    except Exception as e:
        logger.error(f"Ошибка экспорта рейтингов: {e}", exc_info=True)
        await message.answer("❌ Ошибка при экспорте")


# ========== ОБРАБОТЧИКИ КНОПОК ==========

@router.message(F.text == "📊 Статистика")
@require_role("admin")
async def handle_stats_button(message: Message):
    await cmd_full_stats(message)

@router.message(F.text == "📈 Аналитика")
@require_role("admin")
async def handle_analytics_button(message: Message):
    await cmd_analytics(message)

@router.message(F.text == "🔥 Популярные")
@require_role("admin")
async def handle_popular_button(message: Message):
    await cmd_popular_questions(message)

@router.message(F.text == "❌ Без ответов")
@require_role("admin")
async def handle_unanswered_button(message: Message):
    await cmd_unanswered_questions(message)

@router.message(F.text == "👥 Пользователи")
@require_role("admin")
async def handle_users_button(message: Message):
    await cmd_recent_users(message)

@router.message(F.text == "📥 Экспорт")
@require_role("admin")
async def handle_export_button(message: Message):
    await cmd_export_data(message)
    
@router.message(F.text == "📤 Экспорт рейтингов")
@require_role("admin")
async def handle_export_ratings_button(message: Message):
    await cmd_export_ratings(message)

@router.message(F.text == "🔄 Reload KB")
@require_role("admin")
async def handle_reload_button(message: Message):
    await cmd_reload_knowledge_base(message)

@router.message(F.text == "🔙 Главное меню")
@require_role("admin")
async def handle_back_button(message: Message):
    from bot.keyboards import get_main_keyboard
    await message.answer("Вы вернулись в главное меню", reply_markup=get_main_keyboard())
