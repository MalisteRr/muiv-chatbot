"""
ПАТЧ для bot/handlers/user.py
Добавляет кнопки рейтинга к ответам бота

ИНСТРУКЦИЯ:
1. Добавь импорт в начало файла (после других импортов):
   from bot.rating_keyboards import get_rating_keyboard

2. Найди строку ~77-82 где отправляется ответ:
   await message.answer(
       answer,
       parse_mode="Markdown",
       reply_markup=get_main_keyboard()
   )

3. Замени на:
   # Отправить ответ пользователю с кнопками рейтинга
   bot_message = await message.answer(
       answer,
       parse_mode="Markdown",
       reply_markup=get_main_keyboard()
   )
   
   # Добавить кнопки для оценки ответа (отдельным сообщением)
   rating_message = await message.answer(
       "💭 Был ли ответ полезен?",
       reply_markup=get_rating_keyboard(bot_message.message_id)
   )

ИЛИ вариант 2 (inline кнопки в том же сообщении):

3. Замени на:
   # Сохраняем ID сообщения в БД для привязки рейтинга
   bot_message = await message.answer(
       answer,
       parse_mode="Markdown"
   )
   
   # Добавляем inline кнопки рейтинга
   await bot_message.edit_reply_markup(
       reply_markup=get_rating_keyboard(bot_message.message_id)
   )
   
   # Обычная клавиатура внизу
   await message.answer(
       "Выберите тему:",
       reply_markup=get_main_keyboard()
   )
"""

# ========== ПОЛНЫЙ КОД ФУНКЦИИ С РЕЙТИНГОМ ==========

async def process_user_question_with_rating(message: Message):
    """
    Универсальная функция обработки вопросов пользователя
    С добавлением кнопок рейтинга
    
    Args:
        message: Сообщение от пользователя
    """
    user_id = message.from_user.id
    user_name = message.from_user.full_name
    question = message.text
    
    # Импорты (добавить в начало файла user.py)
    from bot.rating_keyboards import get_rating_keyboard
    
    # Создать/обновить пользователя
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
        # Получить ответ через ChatManager
        response_data = await chat_manager.get_response(user_id, question)
        
        # Остановить индикатор печати
        typing_task.cancel()
        
        answer = response_data['answer']
        found_in_db = response_data['found_in_db']
        sources_used = response_data.get('sources', [])
        
        # Сохранить в историю чата (получаем ID сообщения)
        from database.crud import save_chat_message_with_id
        
        # ВАЖНО: Нужно модифицировать save_chat_message чтобы возвращал ID
        # Или использовать стандартную функцию и потом получить последний ID
        
        # Логирование для аналитики (в фоне)
        asyncio.create_task(log_question_analytics(
            user_id=user_id,
            question=question,
            found_answer=found_in_db,
            sources_count=len(sources_used)
        ))
        
        # Отправить ответ пользователю
        bot_message = await message.answer(
            answer,
            parse_mode="Markdown"
        )
        
        # Сохранить в историю (теперь знаем message_id бота)
        asyncio.create_task(save_chat_message(
            user_id=user_id,
            user_name=user_name,
            message=question,
            bot_response=answer,
            source='telegram',
            found_in_db=found_in_db
        ))
        
        # ДОБАВЛЕНО: Кнопки рейтинга (отдельное сообщение)
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
            debug_info = f"\n\n_🔍 Debug: Источников: {len(sources_used)}, В БД: {found_in_db}_"
            await message.answer(debug_info, parse_mode="Markdown")
        
        logger.info(
            f"Ответ отправлен пользователю {user_id}. "
            f"Найдено в БД: {found_in_db}, Источников: {len(sources_used)}"
        )
        
    except asyncio.CancelledError:
        pass
    except Exception as e:
        typing_task.cancel()
        logger.error(f"Ошибка при обработке вопроса от {user_id}: {e}", exc_info=True)
        
        error_message = """😔 Извините, произошла техническая ошибка.

Пожалуйста, попробуйте:
• Переформулировать вопрос
• Связаться с приёмной комиссией:

📞 8 (800) 550-03-63 (бесплатно)
✉️ pk@muiv.ru"""
        
        await message.answer(error_message, parse_mode="Markdown")