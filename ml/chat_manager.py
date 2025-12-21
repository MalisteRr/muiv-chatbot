"""
Менеджер диалогов с использованием AI
Управление контекстом, интеграция с базой знаний и LLM
ИСПРАВЛЕНО: убраны ID вопросов из контекста
"""

import logging
from typing import Dict, List, Optional
from openai import AsyncOpenAI

from config import (
    config,
    SYSTEM_PROMPT,
    CHAT_HISTORY_LIMIT,
    CONTEXT_MESSAGES_LIMIT
)
from database.crud import search_faq_by_keywords
from utils.text_processing import extract_keywords

logger = logging.getLogger(__name__)


class ChatManager:
    """
    Управление диалогами с пользователями
    Интеграция AI модели с базой знаний
    """
    
    def __init__(self):
        """Инициализация менеджера чата"""
        # OpenAI клиент
        self.client = AsyncOpenAI(
            api_key=config.ai.api_key,
            base_url=config.ai.base_url
        )
        
        # История диалогов пользователей {user_id: [messages]}
        self.chat_history: Dict[int, List[Dict]] = {}
        
        logger.info(f"ChatManager инициализирован. Модель: {config.ai.model}")
    
    
    def _get_user_history(self, user_id: int) -> List[Dict]:
        """
        Получить историю диалога пользователя
        
        Args:
            user_id: ID пользователя
            
        Returns:
            Список сообщений из истории
        """
        if user_id not in self.chat_history:
            self.chat_history[user_id] = []
        
        # Возвращаем последние N сообщений для контекста
        return self.chat_history[user_id][-CONTEXT_MESSAGES_LIMIT:]
    
    
    def _add_to_history(self, user_id: int, role: str, content: str):
        """
        Добавить сообщение в историю
        
        Args:
            user_id: ID пользователя
            role: Роль (user/assistant)
            content: Содержимое сообщения
        """
        if user_id not in self.chat_history:
            self.chat_history[user_id] = []
        
        self.chat_history[user_id].append({
            "role": role,
            "content": content
        })
        
        # Ограничиваем размер истории
        if len(self.chat_history[user_id]) > CHAT_HISTORY_LIMIT:
            self.chat_history[user_id] = self.chat_history[user_id][-CHAT_HISTORY_LIMIT:]
    
    
    def clear_history(self, user_id: int):
        """
        Очистить историю диалога пользователя
        
        Args:
            user_id: ID пользователя
        """
        if user_id in self.chat_history:
            del self.chat_history[user_id]
            logger.info(f"История пользователя {user_id} очищена")
    
    
    async def _get_context_from_kb(self, question: str) -> tuple[str, bool, list]:
        """
        Получить контекст из базы знаний
        
        Args:
            question: Вопрос пользователя
            
        Returns:
            Tuple: (контекст, найдено_ли, список_источников)
        """
        # Извлечь ключевые слова из вопроса
        keywords = extract_keywords(question)
        
        if not keywords:
            keywords = [question]
        
        logger.info(f"Поиск в БД по ключевым словам: {keywords}")
        
        # Поиск в базе знаний
        results = await search_faq_by_keywords(keywords)
        
        if not results:
            logger.info("Релевантная информация в БД не найдена")
            return ("", False, [])
        
        # ИСПРАВЛЕНО: Формирование контекста БЕЗ упоминания ID вопросов
        context_parts = []
        sources = []
        
        for idx, result in enumerate(results, 1):
            # Создаем естественный контекст без ID
            # Вместо "вопрос 25" используем просто нумерацию для внутренней структуры
            context_parts.append(
                f"📌 Тема: {result['question']}\n"
                f"{result['answer']}"
            )
            
            # Сохраняем информацию об источниках (для логирования, не для AI)
            sources.append({
                'id': result.get('id', idx),
                'category': result.get('category', 'Общее'),
                'question': result['question']
            })
        
        # ИСПРАВЛЕНО: Используем более естественное разделение
        context = "\n\n---\n\n".join(context_parts)
        
        logger.info(f"Найдено {len(results)} релевантных записей в БД")
        
        return (context, True, sources)
    
    
    async def _generate_ai_response(
        self,
        question: str,
        context: str,
        history: List[Dict]
    ) -> str:
        """
        Генерация ответа через AI модель
        
        Args:
            question: Вопрос пользователя
            context: Контекст из базы знаний
            history: История диалога
            
        Returns:
            Сгенерированный ответ
        """
        # Формирование сообщений для модели
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        
        # Добавляем историю диалога
        messages.extend(history)
        
        # ИСПРАВЛЕНО: Улучшенный промпт без указания "обратитесь в приёмную"
        if context:
            user_prompt = f"""Информация из базы знаний МУИВ:

{context}

---

Вопрос студента: {question}

Инструкции:
- Используй информацию из базы знаний для ответа
- Отвечай конкретно и по делу
- НЕ пиши общие фразы "свяжитесь с приёмной комиссией" если в базе есть конкретная информация
- Контакты университета указывай ТОЛЬКО если нужна дополнительная консультация ИЛИ информации нет в базе
- Будь естественным и дружелюбным
- Максимум 3-4 абзаца"""
        else:
            user_prompt = f"""{question}

(Информации в базе знаний не найдено - ответь что нужно уточнить у специалистов)"""
        
        messages.append({"role": "user", "content": user_prompt})
        
        try:
            # Запрос к AI модели
            logger.debug(f"Отправка запроса к {config.ai.model}")
            
            response = await self.client.chat.completions.create(
                model=config.ai.model,
                messages=messages,
                temperature=config.ai.temperature,
                max_tokens=config.ai.max_tokens
            )
            
            answer = response.choices[0].message.content.strip()
            
            logger.info(f"AI ответ получен. Токенов использовано: {response.usage.total_tokens}")
            
            return answer
            
        except Exception as e:
            logger.error(f"Ошибка при генерации ответа AI: {e}", exc_info=True)
            
            # Если есть контекст - вернем хотя бы его
            if context:
                # Берём первый ответ из контекста как запасной вариант
                first_answer = context.split('\n\n---\n\n')[0]
                # Убираем метку "📌 Тема:"
                if '📌 Тема:' in first_answer:
                    first_answer = '\n'.join(first_answer.split('\n')[1:])
                return first_answer
            
            # Иначе - стандартное сообщение об ошибке
            return """😔 Извините, произошла техническая ошибка.

Пожалуйста, попробуйте:
• Переформулировать вопрос
• Связаться с приёмной комиссией:

📞 8 (800) 550-03-63 (бесплатно)
✉️ pk@muiv.ru"""
    
    
    async def get_response(self, user_id: int, question: str) -> Dict:
        """
        Получить ответ на вопрос пользователя
        
        Args:
            user_id: ID пользователя
            question: Вопрос пользователя
            
        Returns:
            Dict с ключами: answer, found_in_db, sources
        """
        logger.info(f"Обработка вопроса от пользователя {user_id}")
        
        # Получить контекст из базы знаний
        context, found_in_db, sources = await self._get_context_from_kb(question)
        
        # Получить историю диалога
        history = self._get_user_history(user_id)
        
        # Генерация ответа
        answer = await self._generate_ai_response(question, context, history)
        
        # Добавить в историю
        self._add_to_history(user_id, "user", question)
        self._add_to_history(user_id, "assistant", answer)
        
        return {
            'answer': answer,
            'found_in_db': found_in_db,
            'sources': sources
        }
    
    
    async def get_direct_answer(self, question: str) -> str:
        """
        Получить прямой ответ без истории диалога
        Используется для разовых запросов
        
        Args:
            question: Вопрос
            
        Returns:
            Ответ
        """
        context, found_in_db, sources = await self._get_context_from_kb(question)
        answer = await self._generate_ai_response(question, context, [])
        
        return answer
