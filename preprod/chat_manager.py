"""
Менеджер диалогов с использованием AI
Управление контекстом, интеграция с базой знаний и LLM
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
        
        # Формирование контекста из результатов
        context_parts = []
        sources = []
        
        for result in results:
            # Создаем естественный контекст
            context_parts.append(
                f"Вопрос: {result['question']}\n"
                f"Ответ: {result['answer']}"
            )
            
            # Сохраняем информацию об источниках
            sources.append({
                'id': result['id'],
                'category': result['category'],
                'question': result['question']
            })
        
        context = "\n\n".join(context_parts)
        
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
        
        # Формируем промпт с контекстом
        if context:
            user_prompt = f"""Информация из базы данных университета:

{context}

Вопрос пользователя: {question}

Ответь на вопрос, используя эту информацию. Будь естественным, дружелюбным и кратким (2-3 абзаца)."""
        else:
            user_prompt = question
        
        messages.append({"role": "user", "content": user_prompt})
        
        try:
            # Запрос к AI модели
            response = await self.client.chat.completions.create(
                model=config.ai.model,
                messages=messages,
                temperature=config.ai.temperature,
                max_tokens=config.ai.max_tokens
            )
            
            answer = response.choices[0].message.content.strip()
            
            logger.info(f"AI ответ сгенерирован. Токенов: {response.usage.total_tokens if hasattr(response, 'usage') else 'N/A'}")
            
            return answer
            
        except Exception as e:
            logger.error(f"Ошибка при генерации AI ответа: {e}", exc_info=True)
            raise
    
    
    def _format_no_answer_response(self) -> str:
        """
        Форматированный ответ когда информация не найдена
        
        Returns:
            Текст ответа
        """
        return """К сожалению, я не нашел точной информации по вашему вопросу в базе знаний университета.

Рекомендую обратиться напрямую в приемную комиссию МУИВ:

📞 **Телефоны:**
8 (800) 550-03-63 (бесплатно по России)
+7 (495) 500-03-63

✉️ **Email:** pk@muiv.ru
🌐 **Сайт:** muiv.ru

Специалисты приемной комиссии с радостью ответят на все ваши вопросы! 😊"""
    
    
    async def get_response(self, user_id: int, question: str) -> Dict:
        """
        Получить ответ на вопрос пользователя
        Главный метод для обработки запросов
        
        Args:
            user_id: ID пользователя
            question: Вопрос пользователя
            
        Returns:
            Dict с полями: answer, found_in_db, sources
        """
        try:
            # 1. Получить контекст из базы знаний
            context, found_in_db, sources = await self._get_context_from_kb(question)
            
            # 2. Если ничего не найдено - вернуть стандартный ответ
            if not found_in_db:
                return {
                    'answer': self._format_no_answer_response(),
                    'found_in_db': False,
                    'sources': []
                }
            
            # 3. Получить историю диалога пользователя
            history = self._get_user_history(user_id)
            
            # 4. Сгенерировать ответ через AI
            answer = await self._generate_ai_response(question, context, history)
            
            # 5. Обновить историю диалога
            self._add_to_history(user_id, "user", question)
            self._add_to_history(user_id, "assistant", answer)
            
            return {
                'answer': answer,
                'found_in_db': True,
                'sources': sources
            }
            
        except Exception as e:
            logger.error(
                f"Ошибка при обработке вопроса от пользователя {user_id}: {e}",
                exc_info=True
            )
            
            # Возвращаем сообщение об ошибке
            return {
                'answer': """😔 Извините, произошла техническая ошибка при обработке вашего запроса.

Пожалуйста, попробуйте:
• Переформулировать вопрос
• Или свяжитесь с приемной комиссией: 8 (800) 550-03-63

Мы работаем над устранением проблемы.""",
                'found_in_db': False,
                'sources': [],
                'error': str(e)
            }
    
    
    def get_stats(self) -> Dict:
        """
        Получить статистику работы менеджера
        
        Returns:
            Словарь со статистикой
        """
        total_users = len(self.chat_history)
        total_messages = sum(len(msgs) for msgs in self.chat_history.values())
        
        return {
            'active_users': total_users,
            'total_messages': total_messages,
            'avg_messages_per_user': total_messages / total_users if total_users > 0 else 0,
            'model': config.ai.model
        }