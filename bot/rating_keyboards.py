"""
Inline клавиатуры для рейтинга ответов
Кнопки обратной связи после каждого ответа
"""

from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton


def get_rating_keyboard(message_id: int) -> InlineKeyboardMarkup:
    """
    Клавиатура для оценки ответа
    
    Args:
        message_id: ID сообщения бота для привязки рейтинга
        
    Returns:
        Inline клавиатура с кнопками оценки
    """
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(
                text="👍 Полезно",
                callback_data=f"rate_good_{message_id}"
            ),
            InlineKeyboardButton(
                text="👎 Не помогло",
                callback_data=f"rate_bad_{message_id}"
            )
        ]
    ])
    
    return keyboard


def get_detailed_rating_keyboard(message_id: int) -> InlineKeyboardMarkup:
    """
    Детальная оценка (1-5 звёзд)
    
    Args:
        message_id: ID сообщения бота
        
    Returns:
        Inline клавиатура с оценками 1-5
    """
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="⭐", callback_data=f"stars_1_{message_id}"),
            InlineKeyboardButton(text="⭐⭐", callback_data=f"stars_2_{message_id}"),
            InlineKeyboardButton(text="⭐⭐⭐", callback_data=f"stars_3_{message_id}"),
        ],
        [
            InlineKeyboardButton(text="⭐⭐⭐⭐", callback_data=f"stars_4_{message_id}"),
            InlineKeyboardButton(text="⭐⭐⭐⭐⭐", callback_data=f"stars_5_{message_id}"),
        ]
    ])
    
    return keyboard


def get_feedback_reason_keyboard(message_id: int, rating_type: str) -> InlineKeyboardMarkup:
    """
    Уточнение причины негативной оценки
    
    Args:
        message_id: ID сообщения бота
        rating_type: Тип оценки (good/bad)
        
    Returns:
        Inline клавиатура с причинами
    """
    if rating_type == "bad":
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="❌ Нет нужной информации",
                    callback_data=f"reason_no_info_{message_id}"
                )
            ],
            [
                InlineKeyboardButton(
                    text="🤔 Ответ непонятен",
                    callback_data=f"reason_unclear_{message_id}"
                )
            ],
            [
                InlineKeyboardButton(
                    text="📊 Информация неточная",
                    callback_data=f"reason_incorrect_{message_id}"
                )
            ],
            [
                InlineKeyboardButton(
                    text="⏭️ Пропустить",
                    callback_data=f"reason_skip_{message_id}"
                )
            ]
        ])
    else:
        # Для положительной оценки можно добавить опциональную благодарность
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="✅ Спасибо!",
                    callback_data=f"thanks_{message_id}"
                )
            ]
        ])
    
    return keyboard