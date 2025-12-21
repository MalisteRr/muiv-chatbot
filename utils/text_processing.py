"""
Утилиты для обработки естественного языка
Извлечение ключевых слов, нормализация текста
"""

import re
from typing import List, Set


# Русские стоп-слова
STOP_WORDS: Set[str] = {
    # Местоимения
    'я', 'ты', 'он', 'она', 'оно', 'мы', 'вы', 'они',
    'мой', 'твой', 'его', 'её', 'наш', 'ваш', 'их',
    'этот', 'тот', 'такой', 'таков', 'этакий',
    
    # Вопросительные слова
    'как', 'что', 'где', 'когда', 'почему', 'зачем', 'откуда',
    'куда', 'сколько', 'который', 'какой', 'какая', 'какое', 'какие',
    'чей', 'чья', 'чьё', 'чьи',
    
    # Союзы и частицы
    'и', 'а', 'но', 'или', 'да', 'ни', 'не', 'ли', 'же', 'бы', 'то',
    'если', 'чтобы', 'что', 'потому', 'поэтому', 'так',
    
    # Предлоги
    'в', 'на', 'с', 'у', 'к', 'по', 'о', 'об', 'от', 'до', 'из', 'за',
    'над', 'под', 'при', 'через', 'между', 'перед', 'для', 'без',
    
    # Глаголы-связки и вспомогательные
    'быть', 'есть', 'был', 'была', 'было', 'были', 'будет', 'будут',
    'можно', 'нужно', 'надо', 'должен', 'хочу', 'могу',
    
    # Прочие частые слова
    'это', 'то', 'все', 'весь', 'вся', 'всё', 'сам', 'самый',
    'один', 'два', 'три', 'первый', 'второй', 'другой',
    'еще', 'ещё', 'уже', 'только', 'очень', 'более', 'менее'
}


# Категории запросов (для классификации)
CATEGORY_KEYWORDS = {
    'документы': ['документ', 'справка', 'паспорт', 'аттестат', 'диплом', 'снилс', 'инн'],
    'стоимость': ['стоимость', 'цена', 'сколько', 'стоит', 'оплата', 'платно', 'деньги', 'рубл'],
    'бюджет': ['бюджет', 'бесплатно', 'бюджетн', 'место', 'квота'],
    'общежитие': ['общежити', 'проживан', 'жильё', 'жилье', 'комнат', 'койк'],
    'без_егэ': ['егэ', 'экзамен', 'вступительн', 'тест', 'испытан'],
    'формы': ['форма', 'очн', 'заочн', 'дистанц', 'вечерн', 'выходн'],
    'сроки': ['срок', 'когда', 'дата', 'подача', 'прием', 'начало', 'конец'],
    'направления': ['направлен', 'специальност', 'профиль', 'программ', 'факультет', 'кафедр']
}


def normalize_text(text: str) -> str:
    """
    Нормализация текста
    
    Args:
        text: Исходный текст
        
    Returns:
        Нормализованный текст
    """
    # Привести к нижнему регистру
    text = text.lower()
    
    # Убрать лишние пробелы
    text = re.sub(r'\s+', ' ', text)
    
    # Убрать знаки препинания в конце слов
    text = re.sub(r'([а-яёa-z]+)[.,!?;:]+', r'\1', text)
    
    # Убрать цифры (опционально)
    # text = re.sub(r'\d+', '', text)
    
    return text.strip()


def extract_keywords(text: str, max_keywords: int = 5) -> List[str]:
    """
    Извлечение ключевых слов из текста
    
    Args:
        text: Исходный текст
        max_keywords: Максимальное количество ключевых слов
        
    Returns:
        Список ключевых слов
    """
    # Нормализация
    text = normalize_text(text)
    
    # Разделение на слова
    words = text.split()
    
    # Фильтрация
    keywords = []
    for word in words:
        # Пропустить короткие слова (меньше 3 символов)
        if len(word) < 3:
            continue
        
        # Пропустить стоп-слова
        if word in STOP_WORDS:
            continue
        
        # Пропустить слова только из цифр
        if word.isdigit():
            continue
        
        keywords.append(word)
    
    # Убрать дубликаты, сохраняя порядок
    seen = set()
    unique_keywords = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique_keywords.append(kw)
    
    # Ограничить количество
    return unique_keywords[:max_keywords]


def classify_question(text: str) -> str:
    """
    Классификация вопроса по категории
    
    Args:
        text: Текст вопроса
        
    Returns:
        Название категории
    """
    text_lower = normalize_text(text)
    
    # Подсчет совпадений для каждой категории
    scores = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[category] = score
    
    # Вернуть категорию с максимальным score
    if scores:
        return max(scores, key=scores.get)
    
    return 'общий'


def extract_entities(text: str) -> dict:
    """
    Извлечение именованных сущностей
    (упрощенная версия без сторонних библиотек)
    
    Args:
        text: Исходный текст
        
    Returns:
        Словарь с найденными сущностями
    """
    entities = {
        'dates': [],
        'numbers': [],
        'emails': [],
        'phones': []
    }
    
    # Даты (простые паттерны)
    date_patterns = [
        r'\d{1,2}[./]\d{1,2}[./]\d{2,4}',  # 01.01.2024
        r'\d{1,2}\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)',
    ]
    for pattern in date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        entities['dates'].extend(matches)
    
    # Числа
    numbers = re.findall(r'\b\d+\b', text)
    entities['numbers'] = numbers
    
    # Email
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    entities['emails'] = emails
    
    # Телефоны
    phones = re.findall(r'[\+]?[7-8][\s-]?[\(]?\d{3}[\)]?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}', text)
    entities['phones'] = phones
    
    return entities


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Вычислить похожесть двух текстов (упрощенная версия)
    
    Args:
        text1: Первый текст
        text2: Второй текст
        
    Returns:
        Коэффициент похожести (0-1)
    """
    # Извлечь ключевые слова
    keywords1 = set(extract_keywords(text1, max_keywords=10))
    keywords2 = set(extract_keywords(text2, max_keywords=10))
    
    if not keywords1 or not keywords2:
        return 0.0
    
    # Коэффициент Жаккара
    intersection = len(keywords1 & keywords2)
    union = len(keywords1 | keywords2)
    
    return intersection / union if union > 0 else 0.0


def expand_abbreviations(text: str) -> str:
    """
    Расширить распространенные аббревиатуры
    
    Args:
        text: Текст с аббревиатурами
        
    Returns:
        Текст с расшифровками
    """
    abbreviations = {
        'муив': 'международный университет имени жириновского',
        'вкр': 'выпускная квалификационная работа',
        'егэ': 'единый государственный экзамен',
        'снилс': 'страховой номер индивидуального лицевого счета',
        'инн': 'идентификационный номер налогоплательщика',
        'рф': 'российская федерация',
        'мск': 'москва',
        'спб': 'санкт-петербург',
    }
    
    text_lower = text.lower()
    for abbr, full in abbreviations.items():
        text_lower = text_lower.replace(abbr, full)
    
    return text_lower


def clean_question(question: str) -> str:
    """
    Очистить вопрос от лишних элементов
    
    Args:
        question: Исходный вопрос
        
    Returns:
        Очищенный вопрос
    """
    # Убрать emoji
    question = re.sub(r'[\U00010000-\U0010ffff]', '', question)
    
    # Убрать множественные знаки препинания
    question = re.sub(r'([!?.]){2,}', r'\1', question)
    
    # Убрать лишние пробелы
    question = re.sub(r'\s+', ' ', question)
    
    # Убрать пробелы перед знаками препинания
    question = re.sub(r'\s+([,.!?;:])', r'\1', question)
    
    return question.strip()


def is_greeting(text: str) -> bool:
    """
    Проверить, является ли сообщение приветствием
    
    Args:
        text: Текст сообщения
        
    Returns:
        True если приветствие
    """
    greetings = {
        'привет', 'здравствуй', 'добрый день', 'доброе утро', 'добрый вечер',
        'здравствуйте', 'приветствую', 'салют', 'хай', 'hello', 'hi'
    }
    
    text_lower = normalize_text(text)
    words = text_lower.split()
    
    return any(greeting in words for greeting in greetings)


def is_farewell(text: str) -> bool:
    """
    Проверить, является ли сообщение прощанием
    
    Args:
        text: Текст сообщения
        
    Returns:
        True если прощание
    """
    farewells = {
        'пока', 'до свидания', 'прощай', 'спасибо', 'благодарю',
        'досвидания', 'пока-пока', 'bye', 'good bye', 'see you'
    }
    
    text_lower = normalize_text(text)
    
    return any(farewell in text_lower for farewell in farewells)