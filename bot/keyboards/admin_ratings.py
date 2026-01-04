"""
Админские команды для работы с рейтингами
Добавление к bot/handlers/admin.py

ИНСТРУКЦИЯ:
Добавь эти функции в файл bot/handlers/admin.py
Импорты добавь в начало:
from database.crud_ratings import get_rating_statistics, get_low_rated_messages
"""

from aiogram import Router
from aiogram.types import Message
from aiogram.filters import Command
import logging

from database.crud_ratings import get_rating_statistics, get_low_rated_messages

logger = logging.getLogger(__name__)


@router.message(Command("ratings"))
async def cmd_ratings_stats(message: Message):
    """
    Статистика по рейтингам ответов
    Команда: /ratings [days]
    """
    if not is_admin(message.from_user.id):
        return
    
    try:
        # Получить период из команды (по умолчанию 7 дней)
        args = message.text.split()
        days = int(args[1]) if len(args) > 1 and args[1].isdigit() else 7
        
        # Получить статистику
        stats = await get_rating_statistics(days=days)
        
        if not stats or stats['total_ratings'] == 0:
            await message.answer(f"📊 Нет оценок за последние {days} дней")
            return
        
        # Формирование ответа
        total = stats['total_ratings']
        avg = stats['avg_rating']
        positive = stats['positive']
        negative = stats['negative']
        neutral = total - positive - negative
        
        # Процентные показатели
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
        
        # Детализация по типам отзывов
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
async def cmd_low_rated_messages(message: Message):
    """
    Список сообщений с плохими оценками
    Команда: /bad_rated [limit]
    """
    if not is_admin(message.from_user.id):
        return
    
    try:
        # Получить лимит из команды (по умолчанию 10)
        args = message.text.split()
        limit = int(args[1]) if len(args) > 1 and args[1].isdigit() else 10
        
        # Получить низко оценённые сообщения
        messages_list = await get_low_rated_messages(limit=limit)
        
        if not messages_list:
            await message.answer("✅ Нет плохо оценённых сообщений")
            return
        
        text = f"👎 <b>Последние {len(messages_list)} плохо оценённых ответов:</b>\n\n"
        
        for idx, msg in enumerate(messages_list, 1):
            user_q = (msg['user_question'] or 'N/A')[:100]
            bot_ans = (msg['bot_response'] or 'N/A')[:100]
            rating = msg['rating']
            feedback_type = msg['feedback_type'] or 'не указано'
            comment = msg['comment'] or ''
            date = msg['created_at'][:16] if msg['created_at'] else 'N/A'
            
            text += f"<b>{idx}.</b> Рейтинг: {rating}⭐\n"
            text += f"   Дата: {date}\n"
            text += f"   Вопрос: {user_q}...\n"
            text += f"   Ответ: {bot_ans}...\n"
            
            if comment:
                text += f"   💬 Причина: {comment}\n"
            
            text += "\n"
            
            # Telegram ограничение на длину сообщения
            if len(text) > 3500:
                await message.answer(text, parse_mode="HTML")
                text = ""
        
        if text:
            await message.answer(text, parse_mode="HTML")
        
    except Exception as e:
        logger.error(f"Ошибка получения плохо оценённых сообщений: {e}", exc_info=True)
        await message.answer("❌ Ошибка при получении данных")


@router.message(Command("rating_export"))
async def cmd_export_ratings(message: Message):
    """
    Экспорт всех рейтингов в CSV
    Команда: /rating_export
    """
    if not is_admin(message.from_user.id):
        return
    
    try:
        import io
        import csv
        from datetime import datetime
        
        # Получаем все плохо оценённые сообщения
        messages_list = await get_low_rated_messages(limit=1000)
        
        if not messages_list:
            await message.answer("Нет данных для экспорта")
            return
        
        # Создаём CSV
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Заголовки
        writer.writerow([
            'ID', 'User ID', 'Rating', 'Feedback Type', 'Comment',
            'User Question', 'Bot Response', 'Date'
        ])
        
        # Данные
        for msg in messages_list:
            writer.writerow([
                msg['id'],
                msg['user_id'],
                msg['rating'],
                msg['feedback_type'] or '',
                msg['comment'] or '',
                msg['user_question'] or '',
                msg['bot_response'] or '',
                msg['created_at'] or ''
            ])
        
        # Отправляем файл
        csv_data = output.getvalue()
        csv_bytes = io.BytesIO(csv_data.encode('utf-8-sig'))  # UTF-8 with BOM для Excel
        csv_bytes.name = f"ratings_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        from aiogram.types import BufferedInputFile
        document = BufferedInputFile(csv_bytes.read(), filename=csv_bytes.name)
        
        await message.answer_document(
            document=document,
            caption=f"📊 Экспорт рейтингов ({len(messages_list)} записей)"
        )
        
        logger.info(f"Экспорт рейтингов выполнен админом {message.from_user.id}")
        
    except Exception as e:
        logger.error(f"Ошибка экспорта рейтингов: {e}", exc_info=True)
        await message.answer("❌ Ошибка при экспорте данных")


@router.message(Command("rating_notify"))
async def cmd_rating_notifications(message: Message):
    """
    Настройка уведомлений о плохих оценках
    Команда: /rating_notify [on/off]
    
    TODO: Реализовать сохранение настроек в БД
    Пока просто показываем текущий статус
    """
    if not is_admin(message.from_user.id):
        return
    
    await message.answer("""⚙️ <b>Настройка уведомлений о рейтингах</b>

<i>Функция в разработке</i>

Планируется:
• Автоматические уведомления при оценке ≤2⭐
• Еженедельная сводка по рейтингам
• Настройка порога уведомлений

Используйте команды:
/ratings - статистика
/bad_rated - список плохих оценок
/rating_export - экспорт в CSV""", parse_mode="HTML")