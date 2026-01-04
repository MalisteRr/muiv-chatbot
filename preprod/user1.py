"""
Обработчики пользовательских запросов
Обработка вопросов через AI и базу знаний
"""

import logging
import asyncio
from aiogram import Router, F
from aiogram.types import Message

from bot.keyboards import get_main_keyboard
from bot.dispatcher import bot
from ml.chat_manager import ChatManager
from database.crud import save_chat_message, log_question_analytics, create_or_update_user
from config import config

logger = logging.getLogger(__name__)
router = Router(name='user')

# Инициализация менеджера чата
chat_manager = ChatManager()


async def process_user_question(message: Message):
    """
    Универсальная функция обработки вопросов пользователя
    
    Args:
        message: Сообщение от пользователя
    """
    user_id = message.from_user.id
    user_name = message.from_user.full_name
    question = message.text
    
    # Создать/обновить пользователя перед любыми операциями
    await create_or_update_user(
        user_id=user_id,
        username=message.from_user.username,
        first_name=message.from_user.first_name,
        last_name=message.from_user.last_name
    )
    
    # Показать индикатор "печатает..."
    typing_task = asyncio.create_task(keep_typing(message.chat.id))
    
    logger.info(f"Вопрос от пользователя {user_id} ({user_name}): {question[:100]}...")
    
    try:
        # Получить ответ через ChatManager (AI + база знаний)
        response_data = await chat_manager.get_response(user_id, question)
        
        # Остановить индикатор печати
        typing_task.cancel()
        
        answer = response_data['answer']
        found_in_db = response_data['found_in_db']
        sources_used = response_data.get('sources', [])
        
        # Логирование для аналитики (в фоне, не блокируем ответ)
        asyncio.create_task(log_question_analytics(
            user_id=user_id,
            question=question,
            found_answer=found_in_db,
            sources_count=len(sources_used)
        ))
        
        # Сохранить в историю чата (в фоне)
        asyncio.create_task(save_chat_message(
            user_id=user_id,
            user_name=user_name,
            message=question,
            bot_response=answer,
            source='telegram',
            found_in_db=found_in_db
        ))
        
        # Отправить ответ пользователю
        await message.answer(
            answer,
            parse_mode="Markdown",
            reply_markup=get_main_keyboard()
        )
        
        # Дополнительная информация для отладки (только для админов)
        if user_id in config.bot.admin_ids and config.debug:
            debug_info = f"\n\n_🔍 Debug: Найдено источников: {len(sources_used)}, В БД: {found_in_db}_"
            await message.answer(debug_info, parse_mode="Markdown")
        
        logger.info(
            f"Ответ отправлен пользователю {user_id}. "
            f"Найдено в БД: {found_in_db}, Источников: {len(sources_used)}"
        )
        
    except asyncio.CancelledError:
        # Задача была отменена - это нормально
        pass
    except Exception as e:
        # Остановить индикатор печати при ошибке
        typing_task.cancel()
        
        logger.error(f"Ошибка при обработке вопроса от {user_id}: {e}", exc_info=True)
        
        error_message = """😔 Извините, произошла техническая ошибка при обработке вашего запроса.

Пожалуйста, попробуйте:
• Переформулировать вопрос
• Или свяжитесь напрямую с приемной комиссией:

📞 8 (800) 550-03-63 (бесплатно)
✉️ pk@muiv.ru

Мы работаем над устранением проблемы."""
        
        await message.answer(error_message, parse_mode="Markdown")


async def keep_typing(chat_id: int):
    """
    Поддерживает индикатор "печатает..." во время обработки
    
    Args:
        chat_id: ID чата
    """
    try:
        while True:
            await bot.send_chat_action(chat_id, "typing")
            await asyncio.sleep(4)  # Обновлять каждые 4 секунды
    except asyncio.CancelledError:
        # Задача отменена - всё хорошо
        pass


# ========== ОБРАБОТЧИКИ КНОПОК КАТЕГОРИЙ ==========

@router.message(F.text.in_([
    "📚 Документы",
    "💰 Стоимость",
    "🎓 Бюджет",
    "🏠 Общежитие",
    "📝 Без ЕГЭ",
    "🏫 Формы обучения"
]))
async def handle_category_buttons(message: Message):
    """
    Обработчик кнопок категорий
    Все вопросы обрабатываются через AI для естественных ответов
    """
    await process_user_question(message)


# ========== ОБРАБОТЧИК ПРОИЗВОЛЬНОГО ТЕКСТА ==========

@router.message(F.text)
async def handle_text_message(message: Message):
    """
    Обработчик всех текстовых сообщений
    Главный обработчик вопросов пользователей
    """
    # Игнорировать команды (они обрабатываются отдельно)
    if message.text.startswith('/'):
        return
    
    await process_user_question(message)


# ========== ОБРАБОТЧИКИ ДРУГИХ ТИПОВ СООБЩЕНИЙ ==========

@router.message(F.photo)
async def handle_photo(message: Message):
    """Обработчик фото (пока не поддерживается)"""
    await message.answer(
        "📷 К сожалению, я пока не умею обрабатывать фотографии.\n\n"
        "Пожалуйста, опишите ваш вопрос текстом или выберите тему из меню.",
        reply_markup=get_main_keyboard()
    )


@router.message(F.document)
async def handle_document(message: Message):
    """Обработчик документов (пока не поддерживается)"""
    await message.answer(
        "📄 К сожалению, я пока не умею обрабатывать документы.\n\n"
        "Пожалуйста, опишите ваш вопрос текстом или выберите тему из меню.",
        reply_markup=get_main_keyboard()
    )


@router.message(F.voice)
async def handle_voice(message: Message):
    """Обработчик голосовых сообщений (пока не поддерживается)"""
    await message.answer(
        "🎤 К сожалению, я пока не умею обрабатывать голосовые сообщения.\n\n"
        "Пожалуйста, напишите ваш вопрос текстом или выберите тему из меню.",
        reply_markup=get_main_keyboard()
    )


@router.message(F.sticker)
async def handle_sticker(message: Message):
    """Обработчик стикеров"""
    sticker_responses = [
        "😊 Отличный стикер! Но задайте вопрос текстом - я смогу вам помочь!",
        "😄 Спасибо за стикер! Чем могу помочь?",
        "👍 Понял настроение! Что вы хотите узнать о поступлении?"
    ]
    
    import random
    await message.answer(
        random.choice(sticker_responses),
        reply_markup=get_main_keyboard()
    )
