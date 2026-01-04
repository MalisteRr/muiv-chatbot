"""
Обработчики callback для системы рейтинга
Обработка нажатий на кнопки 👍/👎 и детальной оценки
"""

import logging
from aiogram import Router, F
from aiogram.types import CallbackQuery
from aiogram.exceptions import TelegramBadRequest

# Предполагаем что эти функции будут добавлены в crud.py
# Пока импортируем из отдельного файла
from database.crud_ratings import save_rating, get_rating_statistics

logger = logging.getLogger(__name__)
router = Router(name='rating')


@router.callback_query(F.data.startswith("rate_good_"))
async def handle_good_rating(callback: CallbackQuery):
    """
    Обработка положительной оценки 👍
    """
    try:
        # Извлекаем ID сообщения из callback_data
        message_id = int(callback.data.split("_")[-1])
        user_id = callback.from_user.id
        
        # Сохраняем рейтинг (5 звёзд для "полезно")
        success = await save_rating(
            user_id=user_id,
            chat_message_id=message_id,
            rating=5,
            feedback_type="good"
        )
        
        if success:
            # Редактируем сообщение - убираем кнопки
            try:
                await callback.message.edit_reply_markup(reply_markup=None)
            except TelegramBadRequest:
                pass  # Сообщение уже изменено или удалено
            
            # Отправляем благодарность
            await callback.answer(
                "✅ Спасибо за оценку! Рад что смог помочь 😊",
                show_alert=False
            )
            
            logger.info(f"Положительная оценка от пользователя {user_id} для сообщения {message_id}")
        else:
            await callback.answer(
                "❌ Ошибка сохранения оценки. Попробуйте позже.",
                show_alert=True
            )
    
    except Exception as e:
        logger.error(f"Ошибка обработки положительной оценки: {e}", exc_info=True)
        await callback.answer("❌ Произошла ошибка", show_alert=True)


@router.callback_query(F.data.startswith("rate_bad_"))
async def handle_bad_rating(callback: CallbackQuery):
    """
    Обработка отрицательной оценки 👎
    Предлагаем уточнить причину
    """
    try:
        # Извлекаем ID сообщения
        message_id = int(callback.data.split("_")[-1])
        user_id = callback.from_user.id
        
        # Сохраняем базовый рейтинг (1 звезда для "не помогло")
        success = await save_rating(
            user_id=user_id,
            chat_message_id=message_id,
            rating=1,
            feedback_type="bad"
        )
        
        if success:
            # Импортируем клавиатуру для уточнения причины
            from bot.rating_keyboards import get_feedback_reason_keyboard
            
            # Редактируем сообщение - показываем кнопки с причинами
            try:
                await callback.message.edit_reply_markup(
                    reply_markup=get_feedback_reason_keyboard(message_id, "bad")
                )
            except TelegramBadRequest:
                pass
            
            await callback.answer(
                "Помогите улучшить ответы - укажите причину",
                show_alert=False
            )
            
            logger.info(f"Отрицательная оценка от пользователя {user_id} для сообщения {message_id}")
        else:
            await callback.answer(
                "❌ Ошибка сохранения оценки",
                show_alert=True
            )
    
    except Exception as e:
        logger.error(f"Ошибка обработки отрицательной оценки: {e}", exc_info=True)
        await callback.answer("❌ Произошла ошибка", show_alert=True)


@router.callback_query(F.data.startswith("reason_"))
async def handle_feedback_reason(callback: CallbackQuery):
    """
    Обработка уточнения причины плохой оценки
    """
    try:
        # Парсим callback_data: reason_TYPE_MESSAGEID
        parts = callback.data.split("_")
        reason_type = parts[1]  # Нет инфы / Непонятный ответ / Неверный ответ / Пропуск
        message_id = int(parts[-1])
        user_id = callback.from_user.id
        
        # Маппинг типов причин
        reason_mapping = {
            'нет информации': 'Нет нужной информации',
            'непонятно': 'Ответ непонятен',
            'Неверно': 'Информация неточная',
            'Пропустить': None
        }
        
        feedback_comment = reason_mapping.get(reason_type)
        
        if reason_type != 'skip':
            # Обновляем рейтинг с указанием причины
            await save_rating(
                user_id=user_id,
                chat_message_id=message_id,
                rating=1,
                feedback_type=f"bad_{reason_type}",
                comment=feedback_comment
            )
        
        # Убираем кнопки
        try:
            await callback.message.edit_reply_markup(reply_markup=None)
        except TelegramBadRequest:
            pass
        
        # Благодарим за обратную связь
        if reason_type == 'skip':
            await callback.answer("Спасибо за оценку!", show_alert=False)
        else:
            await callback.answer(
                "✅ Спасибо! Мы учтём ваш отзыв для улучшения ответов.",
                show_alert=True
            )
        
        logger.info(f"Причина плохой оценки от {user_id}: {reason_type}")
    
    except Exception as e:
        logger.error(f"Ошибка обработки причины: {e}", exc_info=True)
        await callback.answer("❌ Произошла ошибка", show_alert=True)


@router.callback_query(F.data.startswith("stars_"))
async def handle_star_rating(callback: CallbackQuery):
    """
    Обработка детальной оценки (1-5 звёзд)
    """
    try:
        # Парсим: stars_RATING_MESSAGEID
        parts = callback.data.split("_")
        stars = int(parts[1])
        message_id = int(parts[-1])
        user_id = callback.from_user.id
        
        # Сохраняем оценку
        success = await save_rating(
            user_id=user_id,
            chat_message_id=message_id,
            rating=stars,
            feedback_type="stars"
        )
        
        if success:
            # Убираем кнопки
            try:
                await callback.message.edit_reply_markup(reply_markup=None)
            except TelegramBadRequest:
                pass
            
            # Благодарим с emoji в зависимости от оценки
            if stars >= 4:
                message = f"⭐ Спасибо за {stars} звёзд! Рад что помог! 😊"
            elif stars == 3:
                message = f"⭐ Спасибо за оценку! Постараемся стать лучше!"
            else:
                message = f"Спасибо за оценку. Мы работаем над улучшением ответов."
            
            await callback.answer(message, show_alert=False)
            
            logger.info(f"Оценка {stars} звёзд от пользователя {user_id}")
        else:
            await callback.answer("❌ Ошибка сохранения", show_alert=True)
    
    except Exception as e:
        logger.error(f"Ошибка обработки звёздной оценки: {e}", exc_info=True)
        await callback.answer("❌ Произошла ошибка", show_alert=True)


@router.callback_query(F.data.startswith("thanks_"))
async def handle_thanks(callback: CallbackQuery):
    """
    Обработка кнопки "Спасибо" после положительной оценки
    """
    try:
        # Просто убираем кнопки
        try:
            await callback.message.edit_reply_markup(reply_markup=None)
        except TelegramBadRequest:
            pass
        
        await callback.answer("❤️ Всегда рад помочь!", show_alert=False)
    
    except Exception as e:
        logger.error(f"Ошибка обработки благодарности: {e}")
        await callback.answer()