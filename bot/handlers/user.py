"""
Обработчики пользовательских запросов
Обработка вопросов через AI и базу знаний
С улучшенным UX - показ прогресса обработки
"""

import logging
import asyncio
import random
from aiogram import Router, F
from aiogram.types import Message

from bot.keyboards import get_main_keyboard
from bot.rating_keyboards import get_rating_keyboard
from bot.dispatcher import bot
from ml.chat_manager import ChatManager
from database.crud import save_chat_message, log_question_analytics, create_or_update_user
from config import config

logger = logging.getLogger(__name__)
router = Router(name='user')

# Инициализация менеджера чата
chat_manager = ChatManager()


# Набор сообщений для разных этапов обработки
PROCESSING_MESSAGES = {
    'start': [
        "🔍 Ищу информацию в базе знаний...",
        "🤔 Анализирую ваш вопрос...",
        "📚 Просматриваю базу данных университета...",
        "🎯 Подбираю наиболее точный ответ...",
    ],
    'searching': [
        "🔎 Поиск релевантной информации...",
        "📖 Изучаю документацию МУИВ...",
        "💡 Формирую ответ на основе официальных данных...",
    ],
    'ai_processing': [
        "🤖 Генерирую персонализированный ответ...",
        "✨ Обрабатываю запрос через AI...",
        "📝 Формулирую понятный ответ...",
    ]
}


async def send_progress_message(chat_id: int, stage: str = 'start') -> Message:
    """
    Отправить сообщение о прогрессе обработки
    
    Args:
        chat_id: ID чата
        stage: Этап обработки (start, searching, ai_processing)
        
    Returns:
        Отправленное сообщение
    """
    messages = PROCESSING_MESSAGES.get(stage, PROCESSING_MESSAGES['start'])
    text = random.choice(messages)
    
    return await bot.send_message(chat_id, text)


async def update_progress_message(message: Message, new_text: str):
    """
    Обновить сообщение о прогрессе
    
    Args:
        message: Сообщение для обновления
        new_text: Новый текст
    """
    try:
        await message.edit_text(new_text)
    except Exception as e:
        # Если редактирование не удалось (сообщение слишком старое и т.д.)
        logger.debug(f"Не удалось обновить прогресс-сообщение: {e}")


async def process_user_question(message: Message, show_progress: bool = True):
    """
    Универсальная функция обработки вопросов пользователя
    С показом прогресса обработки
    
    Args:
        message: Сообщение от пользователя
        show_progress: Показывать ли прогресс-сообщения
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
    
    progress_msg = None
    typing_task = None
    
    logger.info(f"Вопрос от пользователя {user_id} ({user_name}): {question[:100]}...")
    
    try:
        # ЭТАП 1: Отправить начальное сообщение о прогрессе
        if show_progress:
            progress_msg = await send_progress_message(message.chat.id, 'start')
            await asyncio.sleep(0.5)  # Небольшая задержка для читаемости
        
        # Показать индикатор "печатает..."
        typing_task = asyncio.create_task(keep_typing(message.chat.id))
        
        # ЭТАП 2: Поиск в базе знаний
        if show_progress and progress_msg:
            await update_progress_message(
                progress_msg,
                random.choice(PROCESSING_MESSAGES['searching'])
            )
        
        # Небольшая задержка перед AI запросом (для UX)
        await asyncio.sleep(0.3)
        
        # ЭТАП 3: Обработка через AI
        if show_progress and progress_msg:
            await update_progress_message(
                progress_msg,
                random.choice(PROCESSING_MESSAGES['ai_processing'])
            )
        
        # Получить ответ через ChatManager (AI + база знаний)
        response_data = await chat_manager.get_response(user_id, question)
        
        # Остановить индикатор печати
        if typing_task:
            typing_task.cancel()
        
        answer = response_data['answer']
        found_in_db = response_data['found_in_db']
        sources_used = response_data.get('sources', [])
        
        # ЭТАП 4: Удалить прогресс-сообщение
        if progress_msg:
            try:
                await progress_msg.delete()
            except Exception:
                pass  # Не критично если не удалось удалить
        
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
        
        # ЭТАП 5: Отправить финальный ответ пользователю
        
        # Send AI answer as plain text to avoid accidental Markdown parsing
        bot_message = await message.answer(
        answer,
        reply_markup=get_main_keyboard()
        )
        # Добавить кнопки рейтинга (отдельное сообщение)
        await message.answer(
        "💭 Был ли ответ полезен?",
        reply_markup=get_rating_keyboard(bot_message.message_id)
        )

        # Основная клавиатура
        await message.answer(
        "Выберите тему или задайте другой вопрос:",
        reply_markup=get_main_keyboard()
        )
        # Дополнительная информация для отладки (только для админов)
        if user_id in config.bot.admin_ids and config.debug:
            debug_info = f"\n\n🔍 Debug: Найдено источников: {len(sources_used)}, В БД: {found_in_db}"
            await message.answer(debug_info)
        
        logger.info(
            f"Ответ отправлен пользователю {user_id}. "
            f"Найдено в БД: {found_in_db}, Источников: {len(sources_used)}"
        )
        
    except asyncio.CancelledError:
        # Задача была отменена - это нормально
        pass
    except Exception as e:
        # Остановить индикатор печати при ошибке
        if typing_task:
            typing_task.cancel()
        
        # Удалить прогресс-сообщение при ошибке
        if progress_msg:
            try:
                await progress_msg.delete()
            except Exception:
                pass
        
        logger.error(f"Ошибка при обработке вопроса от {user_id}: {e}", exc_info=True)
        
        error_message = """😔 Извините, произошла техническая ошибка при обработке вашего запроса.

Пожалуйста, попробуйте:
• Переформулировать вопрос
• Или свяжитесь напрямую с приемной комиссией:

📞 8 (800) 550-03-63 (бесплатно)
✉️ pk@muiv.ru

Мы работаем над устранением проблемы."""
        
        await message.answer(error_message)


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
    Явное сопоставление кнопок с категориями FAQ
    """
    # Карта: текст кнопки → категория в FAQ
    category_map = {
        "📚 Документы": "Документы",
        "💰 Стоимость": "Стоимость",
        "🎓 Бюджет": "Бюджет",
        "🏠 Общежитие": "Общежитие",
        "📝 Без ЕГЭ": "Без ЕГЭ",
        "🏫 Формы обучения": "Обучение"
    }
    
    button_text = message.text
    category = category_map.get(button_text)
    
    if not category:
        # Если категория не найдена - обрабатываем как обычный вопрос
        await process_user_question(message, show_progress=True)
        return
    
    try:
        # Прогресс
        progress_msg = await message.answer("⏳ Ищу информацию...")
        
        # Получаем ответ из FAQ по категории
        from database.crud import get_faq_answer_by_category
        
        faq_result = await get_faq_answer_by_category(category)
        
        if faq_result:
            # Извлекаем текст ответа из словаря
            faq_answer = faq_result.get('answer') if isinstance(faq_result, dict) else faq_result
            
            await progress_msg.delete()
            
            # Отправляем ответ
            sent_message = await message.answer(faq_answer)
            
            # Показываем клавиатуру рейтинга
            await message.answer(
                "💭 Был ли ответ полезен?",
                reply_markup=get_rating_keyboard(sent_message.message_id)
            )
            
            # Сохраняем в историю
            from database.crud import save_chat_message
            await save_chat_message(
                user_id=message.from_user.id,
                user_name=message.from_user.full_name, 
                message=button_text,  
                bot_response=faq_answer,
                source='telegram',
                found_in_db=True
            )

        else:
            # Если в FAQ нет - используем AI
            await progress_msg.delete()
            await process_user_question(message, show_progress=False)
            
    except Exception as e:
        logger.error(f"Ошибка обработки кнопки категории: {e}")
        await message.answer("❌ Произошла ошибка. Попробуйте еще раз.")



# ========== ОБРАБОТЧИК ПРОИЗВОЛЬНОГО ТЕКСТА ==========

@router.message(F.text)
async def handle_text_message(message: Message):
    """
    Обработчик всех текстовых сообщений
    Главный обработчик вопросов пользователей
    С показом прогресса для длинных запросов
    """
    # ========== ПРОВЕРКА ПАРОЛЯ МОДЕРАТОРА/АДМИНА ==========
    from utils.auth_system import is_waiting_for_password, check_password
    
    if is_waiting_for_password(message.from_user.id):
        password = message.text.strip()
        role = check_password(message.from_user.id, password)
        
        if role:
            await message.answer(
                f"✅ <b>Авторизация успешна!</b>\n\n"
                f"Вы вошли как: <b>{role}</b>"
            )
            
            logger.info(f"Пользователь {message.from_user.id} авторизован как {role}, показываю панель...")
            
            # Показать соответствующую панель
            try:
                if role == 'admin':
                    from bot.handlers.admin import cmd_admin_panel
                    logger.info("Вызываю cmd_admin_panel...")
                    await cmd_admin_panel(message)
                elif role == 'moderator':
                    from bot.handlers.moderator import show_moderator_panel
                    logger.info("Вызываю show_moderator_panel...")
                    await show_moderator_panel(message)
                    logger.info("show_moderator_panel выполнена успешно")
            except Exception as e:
                logger.error(f"Ошибка при показе панели {role}: {e}", exc_info=True)
                await message.answer(f"❌ Ошибка при загрузке панели: {e}")
        else:
            await message.answer(
                "❌ <b>Неверный пароль!</b>\n\n"
                "Попробуйте еще раз или используйте команду для повтора:\n"
                "• /admin - для входа как админ\n"
                "• /moderator - для входа как модератор"
            )
        
        return  # ВАЖНО: Выходим, не обрабатываем как обычный вопрос
    # =========================================================
    
    # Игнорировать команды (они обрабатываются отдельно)
    if message.text.startswith('/'):
        return
    
    # Игнорировать админ кнопки (пусть admin.py обработает)
    admin_buttons = [
        '📊 Статистика', '📈 Аналитика', '🔥 Популярные',
        '❌ Без ответов', '👥 Пользователи', '📥 Экспорт',
        '🔄 Reload KB', '🔙 Главное меню'
    ]
    if message.text in admin_buttons:
        return
    
    # Показываем прогресс для всех текстовых вопросов
    show_progress = len(message.text) > 10  # Показывать прогресс если вопрос длиннее 10 символов
    
    await process_user_question(message, show_progress=show_progress)


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
    
    await message.answer(
        random.choice(sticker_responses),
        reply_markup=get_main_keyboard()
    )
