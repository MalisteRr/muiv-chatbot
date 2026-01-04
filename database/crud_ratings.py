"""
CRUD функции для работы с рейтингами и обратной связью
Дополнение к существующему crud.py
"""

import logging
from typing import Optional, Dict, List
from datetime import datetime, timedelta
from database.init_db import get_sqlite

logger = logging.getLogger(__name__)


# ==================== РЕЙТИНГИ И ОБРАТНАЯ СВЯЗЬ ====================

async def save_rating(
    user_id: int,
    chat_message_id: int,
    rating: int,
    feedback_type: Optional[str] = None,
    comment: Optional[str] = None
) -> bool:
    """
    Сохранить рейтинг ответа
    
    Args:
        user_id: ID пользователя
        chat_message_id: ID сообщения из chat_history
        rating: Оценка (1-5) или (1=хорошо, 0=плохо для простой системы)
        feedback_type: Тип отзыва (good/bad/no_info/unclear/incorrect)
        comment: Текстовый комментарий пользователя
        
    Returns:
        True если успешно
    """
    db = get_sqlite()
    if not db:
        logger.error("БД не подключена")
        return False
    
    try:
        # Проверяем существует ли уже рейтинг для этого сообщения
        async with db.execute("""
            SELECT id FROM feedback 
            WHERE user_id = ? AND chat_message_id = ?
        """, (user_id, chat_message_id)) as cursor:
            existing = await cursor.fetchone()
        
        if existing:
            # Обновляем существующий рейтинг
            await db.execute("""
                UPDATE feedback 
                SET rating = ?, feedback_type = ?, comment = ?, created_at = CURRENT_TIMESTAMP
                WHERE user_id = ? AND chat_message_id = ?
            """, (rating, feedback_type, comment, user_id, chat_message_id))
            logger.info(f"Рейтинг обновлён для пользователя {user_id}, сообщение {chat_message_id}")
        else:
            # Создаём новый рейтинг
            await db.execute("""
                INSERT INTO feedback (user_id, chat_message_id, rating, feedback_type, comment)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, chat_message_id, rating, feedback_type, comment))
            logger.info(f"Новый рейтинг от пользователя {user_id}, сообщение {chat_message_id}")
        
        await db.commit()
        
        # Обновляем статистику пользователя
        await update_user_rating_stats(user_id)
        
        return True
        
    except Exception as e:
        logger.error(f"Ошибка сохранения рейтинга: {e}", exc_info=True)
        return False


async def update_user_rating_stats(user_id: int) -> bool:
    """
    Обновить средний рейтинг пользователя в user_stats
    
    Args:
        user_id: ID пользователя
        
    Returns:
        True если успешно
    """
    db = get_sqlite()
    if not db:
        return False
    
    try:
        # Вычисляем средний рейтинг
        async with db.execute("""
            SELECT AVG(rating) FROM feedback WHERE user_id = ?
        """, (user_id,)) as cursor:
            avg_rating = await cursor.fetchone()
            avg_rating = avg_rating[0] if avg_rating[0] else 0
        
        # Обновляем в user_stats
        await db.execute("""
            UPDATE user_stats 
            SET avg_rating = ?, last_updated = CURRENT_TIMESTAMP
            WHERE user_id = ?
        """, (avg_rating, user_id))
        
        await db.commit()
        return True
        
    except Exception as e:
        logger.error(f"Ошибка обновления статистики рейтинга: {e}")
        return False


async def get_rating_statistics(days: int = 7) -> Dict:
    """
    Получить статистику по рейтингам за период
    
    Args:
        days: Количество дней назад
        
    Returns:
        Словарь со статистикой
    """
    db = get_sqlite()
    if not db:
        return {}
    
    try:
        # Общая статистика
        async with db.execute("""
            SELECT 
                COUNT(*) as total_ratings,
                AVG(rating) as avg_rating,
                SUM(CASE WHEN rating >= 4 THEN 1 ELSE 0 END) as positive,
                SUM(CASE WHEN rating <= 2 THEN 1 ELSE 0 END) as negative
            FROM feedback
            WHERE created_at >= datetime('now', '-' || ? || ' days')
        """, (days,)) as cursor:
            row = await cursor.fetchone()
        
        # Статистика по типам отзывов
        async with db.execute("""
            SELECT feedback_type, COUNT(*) as count
            FROM feedback
            WHERE created_at >= datetime('now', '-' || ? || ' days')
              AND feedback_type IS NOT NULL
            GROUP BY feedback_type
        """, (days,)) as cursor:
            feedback_types = await cursor.fetchall()
        
        feedback_breakdown = {ft[0]: ft[1] for ft in feedback_types} if feedback_types else {}
        
        return {
            'total_ratings': row[0] if row else 0,
            'avg_rating': round(row[1], 2) if row and row[1] else 0,
            'positive': row[2] if row else 0,
            'negative': row[3] if row else 0,
            'feedback_types': feedback_breakdown,
            'period_days': days
        }
        
    except Exception as e:
        logger.error(f"Ошибка получения статистики рейтингов: {e}")
        return {}


async def get_low_rated_messages(limit: int = 10) -> List[Dict]:
    """
    Получить сообщения с низким рейтингом для улучшения
    
    Args:
        limit: Максимальное количество результатов
        
    Returns:
        Список сообщений с плохими оценками
    """
    db = get_sqlite()
    if not db:
        return []
    
    try:
        async with db.execute("""
            SELECT 
                f.id,
                f.user_id,
                f.chat_message_id,
                f.rating,
                f.feedback_type,
                f.comment,
                f.created_at,
                ch.message as user_question,
                ch.bot_response
            FROM feedback f
            LEFT JOIN chat_history ch ON f.chat_message_id = ch.id
            WHERE f.rating <= 2
            ORDER BY f.created_at DESC
            LIMIT ?
        """, (limit,)) as cursor:
            rows = await cursor.fetchall()
        
        results = []
        for row in rows:
            results.append({
                'id': row[0],
                'user_id': row[1],
                'chat_message_id': row[2],
                'rating': row[3],
                'feedback_type': row[4],
                'comment': row[5],
                'created_at': row[6],
                'user_question': row[7],
                'bot_response': row[8]
            })
        
        return results
        
    except Exception as e:
        logger.error(f"Ошибка получения низко оценённых сообщений: {e}")
        return []


async def get_user_ratings(user_id: int, limit: int = 20) -> List[Dict]:
    """
    Получить историю рейтингов пользователя
    
    Args:
        user_id: ID пользователя
        limit: Максимальное количество
        
    Returns:
        Список рейтингов пользователя
    """
    db = get_sqlite()
    if not db:
        return []
    
    try:
        async with db.execute("""
            SELECT 
                f.rating,
                f.feedback_type,
                f.comment,
                f.created_at,
                ch.message as question
            FROM feedback f
            LEFT JOIN chat_history ch ON f.chat_message_id = ch.id
            WHERE f.user_id = ?
            ORDER BY f.created_at DESC
            LIMIT ?
        """, (user_id, limit)) as cursor:
            rows = await cursor.fetchall()
        
        results = []
        for row in rows:
            results.append({
                'rating': row[0],
                'feedback_type': row[1],
                'comment': row[2],
                'created_at': row[3],
                'question': row[4]
            })
        
        return results
        
    except Exception as e:
        logger.error(f"Ошибка получения рейтингов пользователя: {e}")
        return []