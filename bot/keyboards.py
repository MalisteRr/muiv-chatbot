"""
Клавиатуры для Telegram бота
Reply и Inline клавиатуры
"""

from aiogram.types import (
    ReplyKeyboardMarkup,
    KeyboardButton,
    InlineKeyboardMarkup,
    InlineKeyboardButton
)


def get_main_keyboard() -> ReplyKeyboardMarkup:
    """
    Главная клавиатура для пользователей
    Категории вопросов + дополнительные функции
    """
    keyboard = ReplyKeyboardMarkup(
        keyboard=[
            # Первая строка - популярные темы
            [
                KeyboardButton(text="📚 Документы"),
                KeyboardButton(text="💰 Стоимость")
            ],
            # Вторая строка - бюджет и общежитие
            [
                KeyboardButton(text="🎓 Бюджет"),
                KeyboardButton(text="🏠 Общежитие")
            ],
            # Третья строка - особые условия
            [
                KeyboardButton(text="📝 Без ЕГЭ"),
                KeyboardButton(text="🏫 Формы обучения")
            ],
            # Четвертая строка - контакты и помощь
            [
                KeyboardButton(text="📞 Контакты"),
                KeyboardButton(text="❓ Помощь")
            ]
        ],
        resize_keyboard=True,
        input_field_placeholder="Задайте ваш вопрос или выберите тему...",
        one_time_keyboard=False
    )
    return keyboard


def get_admin_keyboard() -> ReplyKeyboardMarkup:
    """
    Клавиатура для администраторов
    Быстрый доступ к админ-функциям
    """
    keyboard = ReplyKeyboardMarkup(
        keyboard=[
            [
                KeyboardButton(text="📊 Статистика"),
                KeyboardButton(text="📈 Аналитика")
            ],
            [
                KeyboardButton(text="🔥 Популярные"),
                KeyboardButton(text="❌ Без ответов")
            ],
            [
                KeyboardButton(text="👥 Пользователи"),
                KeyboardButton(text="📥 Экспорт")
            ],
            [
                KeyboardButton(text="🔄 Reload KB"),
                KeyboardButton(text="🔙 Главное меню")
            ]
        ],
        resize_keyboard=True,
        one_time_keyboard=False
    )
    return keyboard


def get_categories_inline() -> InlineKeyboardMarkup:
    """
    Inline клавиатура с категориями FAQ
    Для быстрого выбора темы
    """
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="📚 Документы",
                    callback_data="cat_documents"
                ),
                InlineKeyboardButton(
                    text="💰 Стоимость",
                    callback_data="cat_cost"
                )
            ],
            [
                InlineKeyboardButton(
                    text="🎓 Бюджет",
                    callback_data="cat_budget"
                ),
                InlineKeyboardButton(
                    text="🏠 Общежитие",
                    callback_data="cat_dormitory"
                )
            ],
            [
                InlineKeyboardButton(
                    text="📝 Без ЕГЭ",
                    callback_data="cat_no_ege"
                ),
                InlineKeyboardButton(
                    text="🏫 Формы обучения",
                    callback_data="cat_forms"
                )
            ],
            [
                InlineKeyboardButton(
                    text="📋 Все категории",
                    callback_data="cat_all"
                )
            ]
        ]
    )
    return keyboard


def get_feedback_keyboard() -> InlineKeyboardMarkup:
    """
    Клавиатура для оценки ответа
    Собирает обратную связь от пользователей
    """
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="👍 Полезно",
                    callback_data="feedback_positive"
                ),
                InlineKeyboardButton(
                    text="👎 Не помогло",
                    callback_data="feedback_negative"
                )
            ],
            [
                InlineKeyboardButton(
                    text="📝 Оставить комментарий",
                    callback_data="feedback_comment"
                )
            ]
        ]
    )
    return keyboard


def get_rating_keyboard() -> InlineKeyboardMarkup:
    """
    Клавиатура с рейтингом (1-5 звезд)
    Для детальной оценки качества ответа
    """
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="⭐", callback_data="rating_1"),
                InlineKeyboardButton(text="⭐⭐", callback_data="rating_2"),
                InlineKeyboardButton(text="⭐⭐⭐", callback_data="rating_3"),
                InlineKeyboardButton(text="⭐⭐⭐⭐", callback_data="rating_4"),
                InlineKeyboardButton(text="⭐⭐⭐⭐⭐", callback_data="rating_5")
            ]
        ]
    )
    return keyboard


def get_admin_actions_inline(user_id: int) -> InlineKeyboardMarkup:
    """
    Inline клавиатура с действиями администратора
    
    Args:
        user_id: ID пользователя для действий
    """
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🚫 Заблокировать",
                    callback_data=f"admin_block_{user_id}"
                ),
                InlineKeyboardButton(
                    text="✅ Разблокировать",
                    callback_data=f"admin_unblock_{user_id}"
                )
            ],
            [
                InlineKeyboardButton(
                    text="📊 История",
                    callback_data=f"admin_history_{user_id}"
                ),
                InlineKeyboardButton(
                    text="📈 Статистика",
                    callback_data=f"admin_stats_{user_id}"
                )
            ],
            [
                InlineKeyboardButton(
                    text="💬 Отправить сообщение",
                    callback_data=f"admin_message_{user_id}"
                )
            ]
        ]
    )
    return keyboard


def get_confirmation_keyboard(action: str, data: str) -> InlineKeyboardMarkup:
    """
    Клавиатура подтверждения действия
    
    Args:
        action: Название действия
        data: Данные для callback
    """
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="✅ Подтвердить",
                    callback_data=f"confirm_{action}_{data}"
                ),
                InlineKeyboardButton(
                    text="❌ Отмена",
                    callback_data="cancel"
                )
            ]
        ]
    )
    return keyboard


def get_pagination_keyboard(
    current_page: int,
    total_pages: int,
    callback_prefix: str
) -> InlineKeyboardMarkup:
    """
    Клавиатура пагинации
    
    Args:
        current_page: Текущая страница
        total_pages: Всего страниц
        callback_prefix: Префикс для callback_data
    """
    buttons = []
    
    # Кнопка "Назад"
    if current_page > 1:
        buttons.append(
            InlineKeyboardButton(
                text="◀️ Назад",
                callback_data=f"{callback_prefix}_page_{current_page - 1}"
            )
        )
    
    # Информация о странице
    buttons.append(
        InlineKeyboardButton(
            text=f"{current_page}/{total_pages}",
            callback_data="pagination_info"
        )
    )
    
    # Кнопка "Вперед"
    if current_page < total_pages:
        buttons.append(
            InlineKeyboardButton(
                text="Вперед ▶️",
                callback_data=f"{callback_prefix}_page_{current_page + 1}"
            )
        )
    
    keyboard = InlineKeyboardMarkup(inline_keyboard=[buttons])
    return keyboard