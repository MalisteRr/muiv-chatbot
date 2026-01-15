"""
Менеджер диалогов с 4-слойной архитектурой классификации

СЛОЙ 0: Собственная LSTM модель (если уверенность >85%)
СЛОЙ 1: RuBERT классификатор (основной)
СЛОЙ 2: Keyword search (резервный)
СЛОЙ 3: DeepSeek API (fallback)

УЛУЧШЕНИЕ: Контекстный поиск в FAQ

Разработка: Синицин М.Д. (ВКР)
"""
import logging
import re
from typing import Dict, List, Optional
import torch
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class ChatManager:
    """Управление диалогами с пользователями"""
    
    def __init__(self):
        """Инициализация менеджера чата"""
        
        # История диалогов {user_id: [messages]}
        self.chat_history: Dict[int, List[Dict]] = {}
        
        # Контекст последнего ответа {user_id: {"category": str, "topic": str, "faq_question": str}}
        self.last_answer_context: Dict[int, Dict] = {}
        
        # Порог для LSTM (выше = используем LSTM, ниже = переходим к RuBERT)
        self.lstm_high_confidence_threshold = 0.85
        
        # ========== ПРОВЕРКА КЛАССИФИКАТОРОВ ==========
        try:
            from ml.custom_lstm_classifier import is_custom_classifier_available
            if is_custom_classifier_available():
                logger.info("✅ Собственная LSTM модель доступна (СЛОЙ 0)")
            else:
                logger.info("ℹ️ Собственная LSTM модель не загружена")
        except ImportError:
            logger.debug("ℹ️ Модуль custom_lstm_classifier не найден")
        
        try:
            from ml.intent_classifier import is_classifier_available
            if is_classifier_available():
                logger.info("✅ RuBERT классификатор доступен (СЛОЙ 1)")
            else:
                logger.warning("⚠️ RuBERT классификатор недоступен")
        except ImportError:
            logger.warning("⚠️ Модуль intent_classifier не найден")
    
    def _get_user_history(self, user_id: int) -> List[Dict]:
        """Получить историю диалога"""
        if user_id not in self.chat_history:
            self.chat_history[user_id] = []
        return self.chat_history[user_id]
    
    def _add_to_history(self, user_id: int, role: str, content: str):
        """Добавить сообщение в историю"""
        history = self._get_user_history(user_id)
        history.append({"role": role, "content": content})
        
        max_history = 10
        if len(history) > max_history:
            self.chat_history[user_id] = history[-max_history:]
    
    def _save_answer_context(self, user_id: int, category: str, faq_question: str, answer: str):
        """Сохранить контекст последнего ответа для follow-up вопросов"""
        # Извлекаем ключевую тему из ответа
        topic_keywords = []
        
        # Определяем тему по ключевым словам в ответе
        answer_lower = answer.lower()
        if 'документ' in answer_lower:
            topic_keywords.append('документы')
        if 'стоимость' in answer_lower or 'руб' in answer_lower or 'цена' in answer_lower:
            topic_keywords.append('стоимость')
        if 'общежит' in answer_lower or 'общага' in answer_lower:
            topic_keywords.append('общежитие')
        if 'егэ' in answer_lower:
            topic_keywords.append('егэ')
        if 'бюджет' in answer_lower:
            topic_keywords.append('бюджет')
        if 'контакт' in answer_lower or 'телефон' in answer_lower:
            topic_keywords.append('контакты')
        
        self.last_answer_context[user_id] = {
            "category": category,
            "faq_question": faq_question,
            "topics": topic_keywords,
            "answer_preview": answer[:200]  # Первые 200 символов для анализа
        }
        
        logger.debug(f"💾 Сохранён контекст: category={category}, topics={topic_keywords}")
    
    def _is_followup_question(self, question: str) -> bool:
        """Определить, является ли вопрос уточняющим (follow-up)"""
        followup_patterns = [
            r'^это\s+все',
            r'^а\s+ещ[её]',
            r'^что\s+ещ[её]',
            r'^какие\s+ещ[её]',
            r'^больше\s+ничего',
            r'^и\s+всё\??$',
            r'^только\s+это',
            r'^а\s+что\s+насч[её]т',
            r'^подробнее',
            r'^а\s+если',
            r'^а\s+как\s+насч[её]т',
            r'^уточните',
            r'^поподробнее',
            r'^расскажите\s+подробнее',
            r'^а\s+можно\s+подробнее',
        ]
        
        question_lower = question.lower().strip()
        
        for pattern in followup_patterns:
            if re.search(pattern, question_lower):
                return True
        
        # Короткие вопросы (1-4 слова) часто являются уточняющими
        if len(question_lower.split()) <= 4 and '?' in question:
            return True
        
        return False
    
    def _get_context_keywords(self, user_id: int) -> List[str]:
        """Получить ключевые слова из контекста для поиска в FAQ"""
        context = self.last_answer_context.get(user_id)
        if not context:
            return []
        
        keywords = []
        
        # Добавляем темы из контекста
        keywords.extend(context.get('topics', []))
        
        # Добавляем слова из исходного FAQ вопроса
        faq_question = context.get('faq_question', '')
        if faq_question:
            # Извлекаем ключевые слова из вопроса FAQ
            faq_words = re.findall(r'\b[а-яё]{4,}\b', faq_question.lower())
            keywords.extend(faq_words[:5])
        
        return list(set(keywords))
    
    def clear_history(self, user_id: int):
        """Очистить историю пользователя"""
        if user_id in self.chat_history:
            del self.chat_history[user_id]
        if user_id in self.last_answer_context:
            del self.last_answer_context[user_id]
        logger.info(f"История пользователя {user_id} очищена")
    
    def _correct_category_by_keywords(self, question: str, predicted_category: str, user_history: List[Dict]) -> Optional[str]:
        """
        Коррекция с учётом контекста
        """
        question_lower = question.lower().strip()
        
        # ========== КОНТЕКСТ (последний вопрос бота) ==========
        last_bot_message = ""
        if len(user_history) >= 2:
            for msg in reversed(user_history[:-1]):
                if msg['role'] == 'assistant':
                    last_bot_message = msg['content'].lower()
                    break
        
        # ========== КОНТЕКСТ (последний ответ бота) ==========
        last_bot_message = ""
        if len(user_history) >= 2:
            for msg in reversed(user_history[:-1]):
                if msg['role'] == 'assistant':
                    last_bot_message = msg['content'].lower()
                    break
        
        # Направления обучения
        directions = ['экономика', 'юриспруденция', 'программирование', 'it', 
                      'психология', 'менеджмент', 'информатика', 'право']
        
        # Если бот спрашивал про стоимость/направление, а пользователь назвал направление
        if ('стоимость' in last_bot_message or 'направление' in last_bot_message or 
            'какое направление' in last_bot_message):
            if any(word in question_lower for word in directions):
                logger.info(f"🔧 КОНТЕКСТ: Ответ на вопрос про стоимость/направление → 'Стоимость'")
                return 'Стоимость'
        
        # Если бот спрашивал про форму обучения, а пользователь ответил
        forms = ['очная', 'очное', 'заочная', 'заочное', 'дистанционн', 'вечерн', 'онлайн']
        if ('форм' in last_bot_message and 'обучен' in last_bot_message):
            if any(word in question_lower for word in forms):
                logger.info(f"🔧 КОНТЕКСТ: Ответ на вопрос про форму обучения → 'Формы обучения'")
                return 'Формы обучения'
        
        # Если бот спрашивал про документы
        if 'документ' in last_bot_message and 'какие' in last_bot_message:
            if any(word in question_lower for word in ['да', 'нет', 'все', 'это', 'ещё', 'еще']):
                logger.info(f"🔧 КОНТЕКСТ: Уточнение по документам → 'Документы'")
                return 'Документы'
        
        # ========== ПРИОРИТЕТНЫЕ КЛЮЧЕВЫЕ СЛОВА ==========
        priority_keywords = {
            'Общая информация': [
                'филиал', 'город', 'города', 'локация', 'находится', 'адрес',
                'где университет', 'в других городах', 'есть в', 'кампус',
                'отделение', 'представительство', 'офис', 'где расположен',
                'география', 'региональный', 'местонахождение'
            ],
            'Стоимость': [
                'сколько стоит', 'цена', 'стоимость', 'оплата', 'платить',
                'стоит ли', 'сколько надо', 'сколько нужно', 'прайс',
                'тариф', 'расценки', 'дорого', 'дешево', 'бабки', 'деньги',
                'сколько', 'стоит', 'плата', 'платно'
            ],
            'Бюджет': [
                'бюджет', 'бесплатно', 'без оплаты', 'бесплатное место',
                'бюджетное место', 'государственное финансирование',
                'грант', 'квота'
            ],
            'Общежитие': [
                'общежитие', 'общага', 'проживание', 'где жить',
                'комната', 'место в общежитии', 'общага есть',
                'жильё', 'жилье', 'поселение', 'комнату'
            ],
            'Без ЕГЭ': [
                'без егэ', 'без экзамена', 'егэ не нужен', 'поступить без егэ',
                'не сдавал егэ', 'можно без егэ', 'егэ', 'экзамен'
            ],
            'Документы': [
                'какие документы', 'список документов', 'что нужно принести',
                'документы для', 'нужны документы', 'документация',
                'справки', 'аттестат', 'диплом'
            ],
            'Поступление': [
                'как поступить', 'поступление', 'поступать', 'подать документы',
                'приёмная комиссия', 'прием', 'зачисление', 'абитуриент'
            ],
            'Контакты': [
                'телефон', 'позвонить', 'связаться', 'контакты', 'адрес',
                'где находится', 'email', 'почта', 'сайт', 'местоположение'
            ]
        }
        
        # ТОЧНОЕ СОВПАДЕНИЕ (короткие запросы)
        if len(question_lower.split()) == 1:
            exact_matches = {
                'общежитие': 'Общежитие',
                'общага': 'Общежитие',
                'стоимость': 'Стоимость',
                'цена': 'Стоимость',
                'бюджет': 'Бюджет',
                'контакты': 'Контакты',
                'документы': 'Документы',
                'егэ': 'Без ЕГЭ',
                'поступление': 'Поступление'
            }
            
            if question_lower in exact_matches:
                matched_category = exact_matches[question_lower]
                if matched_category != predicted_category:
                    logger.info(
                        f"🔧 ТОЧНОЕ СОВПАДЕНИЕ: '{predicted_category}' → '{matched_category}' "
                        f"(запрос: '{question_lower}')"
                    )
                    return matched_category
        
        # ========== ПРОВЕРКА КЛЮЧЕВЫХ СЛОВ ==========
        for category, keywords in priority_keywords.items():
            for keyword in keywords:
                if keyword in question_lower:
                    if category != predicted_category:
                        logger.info(
                            f"🔧 КОРРЕКЦИЯ: '{predicted_category}' → '{category}' "
                            f"(найдено: '{keyword}')"
                        )
                        return category
        
        return None
    
    async def _get_faq_with_context(self, category: str, question: str, user_id: int) -> Optional[Dict]:
        """
        Улучшенный поиск FAQ с учётом контекста диалога
        
        Если вопрос является уточняющим (follow-up), 
        ищем ответ, связанный с предыдущей темой
        """
        try:
            from database.crud import get_faq_answer_by_category, search_faq_by_keywords
            
            is_followup = self._is_followup_question(question)
            context = self.last_answer_context.get(user_id)
            
            if is_followup and context:
                logger.info(f"🔄 Обнаружен уточняющий вопрос, учитываем контекст: {context.get('topics', [])}")
                
                # Если спрашивают "это все?" про документы - ищем тот же FAQ
                context_topics = context.get('topics', [])
                
                # Комбинируем ключевые слова из вопроса и контекста
                question_lower = question.lower()
                context_keywords = self._get_context_keywords(user_id)
                
                # Если вопрос про "это все" или "что ещё" - ищем по контексту
                if any(word in question_lower for word in ['это все', 'ещё', 'еще', 'что-то еще', 'подробнее']):
                    # Ищем FAQ по ключевым словам контекста
                    if context_keywords:
                        logger.info(f"🔍 Поиск по контекстным ключевым словам: {context_keywords}")
                        faq_results = await search_faq_by_keywords(context_keywords, limit=1)
                        
                        if faq_results and len(faq_results) > 0:
                            return faq_results[0]
                
                # Пробуем найти FAQ по категории из контекста
                context_category = context.get('category')
                if context_category and context_category != category:
                    logger.info(f"🔄 Используем категорию из контекста: {context_category}")
                    faq_answer = await get_faq_answer_by_category(context_category, question)
                    if faq_answer:
                        return faq_answer
            
            # Стандартный поиск по категории
            return await get_faq_answer_by_category(category, question)
            
        except Exception as e:
            logger.error(f"Ошибка поиска FAQ с контекстом: {e}")
            return None
    
    async def _get_deepseek_answer(self, question: str, user_history: List[Dict] = None) -> str:
        """
        Получить ответ от DeepSeek API с учетом истории диалога
        """
        try:
            import httpx
            from config import config, SYSTEM_PROMPT
            
            url = f"{config.ai.base_url}/chat/completions"
            headers = {
                "Authorization": f"Bearer {config.ai.api_key}",
                "Content-Type": "application/json"
            }
            
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            
            if user_history:
                history_to_send = user_history[-5:] if len(user_history) > 5 else user_history
                messages.extend(history_to_send)
                logger.info(f"📜 Отправляю {len(history_to_send)} сообщений из истории в DeepSeek")
            
            if not user_history or user_history[-1].get("content") != question:
                messages.append({"role": "user", "content": question})
            
            payload = {
                "model": config.ai.model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1000
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                
                answer = data['choices'][0]['message']['content']
                logger.info("✅ Получен ответ от DeepSeek API (СЛОЙ 3)")
                return answer
            
        except Exception as e:
            logger.error(f"❌ Ошибка DeepSeek API: {e}")
            return (
                "К сожалению, я не могу дать точный ответ на этот вопрос. "
                "Рекомендую обратиться в приёмную комиссию МУИВ:\n\n"
                "📞 Телефон: 8 (800) 550-03-63\n"
                "📧 Email: pk@muiv.ru\n"
                "🌐 Сайт: muiv.ru"
            )
    
    async def get_response(
        self,
        user_id: int,
        question: str,
        use_context: bool = True
    ) -> dict:
        """
        Главная функция - Получить ответ на вопрос
        
        4-слойная архитектура:
        СЛОЙ 0: Собственная LSTM (если уверенность >85%) → FAQ
        СЛОЙ 1: RuBERT классификация → КОНТЕКСТНАЯ КОРРЕКЦИЯ → FAQ
        СЛОЙ 2: Keyword search в БД
        СЛОЙ 3: DeepSeek API
        """
        found_in_db = False
        source = "api"
        sources_used = []
        
        try:
            # Получаем историю ДО добавления текущего вопроса
            user_history = self._get_user_history(user_id)
            
            # Добавляем вопрос в историю
            self._add_to_history(user_id, "user", question)
            
            # ====================================================================
            # СЛОЙ 0: СОБСТВЕННАЯ LSTM МОДЕЛЬ (высокая уверенность)
            # ====================================================================
            try:
                from ml.custom_lstm_classifier import get_custom_classifier
                
                lstm_classifier = get_custom_classifier()
                if lstm_classifier and use_context:
                    logger.info("🧠 СЛОЙ 0: Запуск собственной LSTM модели...")
                    
                    lstm_prediction = lstm_classifier.predict(question)
                    lstm_category = lstm_prediction['category']
                    lstm_confidence = lstm_prediction['confidence']
                    
                    logger.info(f"📊 LSTM: {lstm_category} (уверенность: {lstm_confidence*100:.1f}%)")
                    
                    # Если LSTM очень уверена (>85%), используем её результат
                    if lstm_confidence >= self.lstm_high_confidence_threshold:
                        # Применяем коррекцию по ключевым словам
                        corrected_category = self._correct_category_by_keywords(
                            question, lstm_category, user_history
                        )
                        if corrected_category:
                            lstm_category = corrected_category
                        
                        # УЛУЧШЕНИЕ: Используем контекстный поиск FAQ
                        faq_answer = await self._get_faq_with_context(lstm_category, question, user_id)
                        
                        if faq_answer:
                            answer = faq_answer['answer']
                            faq_question = faq_answer.get('question', '')
                            
                            self._add_to_history(user_id, "assistant", answer)
                            
                            # Сохраняем контекст для follow-up вопросов
                            self._save_answer_context(user_id, lstm_category, faq_question, answer)
                            
                            logger.info(f"✅ СЛОЙ 0 (LSTM) → FAQ: Ответ найден по категории '{lstm_category}'")
                            
                            return {
                                "answer": answer,
                                "source": "lstm_faq",
                                "found_in_db": True,
                                "sources": [{"category": lstm_category, "confidence": lstm_confidence, "model": "lstm"}]
                            }
                        else:
                            logger.info(f"ℹ️ СЛОЙ 0: FAQ не найден для категории '{lstm_category}', переход к СЛОЙ 1")
                    else:
                        logger.info(f"ℹ️ СЛОЙ 0: Уверенность LSTM ({lstm_confidence*100:.1f}%) ниже порога ({self.lstm_high_confidence_threshold*100:.0f}%), переход к СЛОЙ 1")
                        
            except ImportError:
                logger.debug("ℹ️ Модуль custom_lstm_classifier не найден, пропускаем СЛОЙ 0")
            except Exception as e:
                logger.debug(f"ℹ️ СЛОЙ 0 недоступен: {e}")
            
            # ====================================================================
            # СЛОЙ 1: RUBERT + КОНТЕКСТНАЯ КОРРЕКЦИЯ
            # ====================================================================
            try:
                from ml.intent_classifier import get_classifier
                
                classifier = get_classifier()
                if classifier and use_context:
                    logger.info("🤖 СЛОЙ 1: Запуск RuBERT классификатора...")
                    
                    prediction = classifier.predict(question)
                    category = prediction['category']
                    confidence = prediction['confidence']
                    is_confident = prediction['is_confident']
                    
                    logger.info(f"📊 RuBERT: {category} (уверенность: {confidence*100:.1f}%)")
                    
                    # ========== КОРРЕКЦИЯ С УЧЁТОМ КОНТЕКСТА ==========
                    corrected_category = self._correct_category_by_keywords(
                        question, 
                        category, 
                        user_history
                    )
                    if corrected_category:
                        category = corrected_category
                        is_confident = True
                        logger.info(f"✅ Использую исправленную категорию: {category}")
                    # ==================================================
                    
                    if is_confident and category:
                        # УЛУЧШЕНИЕ: Используем контекстный поиск FAQ
                        faq_answer = await self._get_faq_with_context(category, question, user_id)
                        
                        if faq_answer:
                            answer = faq_answer['answer']
                            faq_question = faq_answer.get('question', '')
                            
                            self._add_to_history(user_id, "assistant", answer)
                            
                            # Сохраняем контекст для follow-up вопросов
                            self._save_answer_context(user_id, category, faq_question, answer)
                            
                            logger.info(f"✅ СЛОЙ 1 (RuBERT) → FAQ: Ответ найден по категории '{category}'")
                            
                            return {
                                "answer": answer,
                                "source": "rubert_faq",
                                "found_in_db": True,
                                "sources": [{"category": category, "confidence": confidence, "model": "rubert"}]
                            }
            except ImportError:
                logger.debug("RuBERT классификатор недоступен")
            
            # ====================================================================
            # СЛОЙ 2: KEYWORD SEARCH
            # ====================================================================
            if use_context:
                try:
                    from database.crud import search_faq_by_keywords
                    from utils.text_processing import extract_keywords
                    
                    logger.info("🔍 СЛОЙ 2: Запуск Keyword Search...")
                    
                    # Комбинируем ключевые слова из вопроса и контекста
                    keywords = extract_keywords(question)
                    
                    # Если это follow-up вопрос, добавляем контекстные ключевые слова
                    if self._is_followup_question(question):
                        context_keywords = self._get_context_keywords(user_id)
                        keywords.extend(context_keywords)
                        keywords = list(set(keywords))
                        logger.info(f"🔄 Добавлены контекстные ключевые слова: {context_keywords}")
                    
                    faq_results = await search_faq_by_keywords(keywords, limit=3)
                    
                    if faq_results and len(faq_results) > 0:
                        best_match = faq_results[0]
                        answer = best_match.get('answer', '')
                        faq_question = best_match.get('question', '')
                        category = best_match.get('category', 'unknown')
                        
                        question_lower = question.lower()
                        non_university_keywords = ['пицца', 'еда', 'кафе', 'ресторан', 'погода']
                        is_non_university = any(kw in question_lower for kw in non_university_keywords)
                        
                        if not is_non_university and answer:
                            self._add_to_history(user_id, "assistant", answer)
                            
                            # Сохраняем контекст
                            self._save_answer_context(user_id, category, faq_question, answer)
                            
                            logger.info(f"✅ СЛОЙ 2 (Keyword) → Ответ из БД")
                            
                            return {
                                "answer": answer,
                                "source": "keyword_search",
                                "found_in_db": True,
                                "sources": faq_results
                            }
                except Exception as e:
                    logger.debug(f"Keyword search недоступен: {e}")
            
            # ====================================================================
            # СЛОЙ 3: DeepSeek API (FALLBACK)
            # ====================================================================
            logger.info("🔄 СЛОЙ 3: Используем DeepSeek API...")
            answer = await self._get_deepseek_answer(question, user_history)
            source = "deepseek_api"
            
            self._add_to_history(user_id, "assistant", answer)
            
            # Очищаем контекст при переходе на API (новая тема)
            if user_id in self.last_answer_context:
                del self.last_answer_context[user_id]
            
            return {
                "answer": answer,
                "source": source,
                "found_in_db": False,
                "sources": []
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка генерации ответа: {e}", exc_info=True)
            return {
                "answer": "Извините, произошла ошибка при обработке вашего вопроса. Попробуйте переформулировать.",
                "source": "error",
                "found_in_db": False,
                "sources": []
            }
    
    async def get_answer(self, user_id: int, question: str, use_context: bool = True) -> str:
        """Alias для совместимости"""
        response = await self.get_response(user_id, question, use_context)
        return response['answer']
    
    async def get_streaming_answer(self, user_id: int, question: str, use_context: bool = True):
        """Alias для совместимости"""
        answer = await self.get_answer(user_id, question, use_context)
        yield answer
