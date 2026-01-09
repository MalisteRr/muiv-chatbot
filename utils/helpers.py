"""
Вспомогательные функции
Утилиты общего назначения
"""

import re
from typing import Optional
from datetime import datetime
from config import config


def is_admin(user_id: int) -> bool:
    """
    Проверить, является ли пользователь администратором
    ОБНОВЛЕНО: Теперь поддерживает систему авторизации
    
    Args:
        user_id: ID пользователя
        
    Returns:
        True если администратор
    """
    # Проверяем по ID (старый способ для совместимости)
    if user_id in config.bot.admin_ids:
        return True
    
    # Проверяем по системе авторизации с паролями
    try:
        from utils.auth_system import has_role
        return has_role(user_id, 'admin')
    except ImportError:
        # Если auth_system не установлен, используем старый способ
        return user_id in config.bot.admin_ids


def format_user_info(user_data: dict) -> str:
    """
    Форматировать информацию о пользователе
    
    Args:
        user_data: Данные пользователя
        
    Returns:
        Отформатированная строка
    """
    user_id = user_data.get('user_id', 'N/A')
    username = user_data.get('username', 'N/A')
    first_name = user_data.get('first_name', '')
    last_name = user_data.get('last_name', '')
    
    full_name = f"{first_name} {last_name}".strip() or 'Unknown'
    
    return f"{full_name} (@{username}, ID: {user_id})"


def format_datetime(dt: Optional[datetime], format_str: str = '%d.%m.%Y %H:%M') -> str:
    """
    Форматировать дату и время
    
    Args:
        dt: Объект datetime
        format_str: Формат вывода
        
    Returns:
        Отформатированная строка или 'N/A'
    """
    if dt is None:
        return 'N/A'
    
    return dt.strftime(format_str)


def sanitize_text(text: str, max_length: int = 200) -> str:
    """
    Очистить и обрезать текст
    
    Args:
        text: Исходный текст
        max_length: Максимальная длина
        
    Returns:
        Очищенный текст
    """
    # Убрать лишние пробелы
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Обрезать если слишком длинный
    if len(text) > max_length:
        text = text[:max_length - 3] + '...'
    
    return text


def extract_command_args(text: str) -> tuple[str, str]:
    """
    Извлечь команду и аргументы из текста
    
    Args:
        text: Текст сообщения
        
    Returns:
        Tuple (команда, аргументы)
    """
    parts = text.split(maxsplit=1)
    command = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ''
    
    return command, args


def format_number(num: int) -> str:
    """
    Форматировать число с разделителями тысяч
    
    Args:
        num: Число
        
    Returns:
        Отформатированная строка
    """
    return f"{num:,}".replace(',', ' ')


def calculate_percentage(part: int, total: int, decimals: int = 1) -> float:
    """
    Вычислить процент
    
    Args:
        part: Часть
        total: Целое
        decimals: Количество знаков после запятой
        
    Returns:
        Процент
    """
    if total == 0:
        return 0.0
    
    result = (part / total) * 100
    return round(result, decimals)


def truncate_message(text: str, max_length: int = 4000) -> str:
    """
    Обрезать сообщение до допустимой длины Telegram
    
    Args:
        text: Текст сообщения
        max_length: Максимальная длина (Telegram limit = 4096)
        
    Returns:
        Обрезанное сообщение
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - 50] + "\n\n...\n_(Сообщение обрезано)_"


def escape_markdown(text: str) -> str:
    """
    Экранировать специальные символы Markdown
    
    Args:
        text: Исходный текст
        
    Returns:
        Текст с экранированными символами
    """
    # Символы которые нужно экранировать в Markdown
    special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    
    for char in special_chars:
        text = text.replace(char, f'\\{char}')
    
    return text


def validate_email(email: str) -> bool:
    """
    Проверить корректность email
    
    Args:
        email: Email адрес
        
    Returns:
        True если валидный
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_phone(phone: str) -> bool:
    """
    Проверить корректность телефона (российский формат)
    
    Args:
        phone: Номер телефона
        
    Returns:
        True если валидный
    """
    # Убрать все кроме цифр и +
    cleaned = re.sub(r'[^\d+]', '', phone)
    
    # Российский номер: +7 или 8, затем 10 цифр
    pattern = r'^(\+7|8)\d{10}$'
    return bool(re.match(pattern, cleaned))


def format_duration(seconds: float) -> str:
    """
    Форматировать длительность в читаемый вид
    
    Args:
        seconds: Количество секунд
        
    Returns:
        Отформатированная строка (например, "1ч 23м 45с")
    """
    if seconds < 60:
        return f"{seconds:.1f}с"
    
    minutes = int(seconds // 60)
    seconds = seconds % 60
    
    if minutes < 60:
        return f"{minutes}м {int(seconds)}с"
    
    hours = minutes // 60
    minutes = minutes % 60
    
    return f"{hours}ч {minutes}м"


def get_greeting_emoji() -> str:
    """
    Получить приветственный emoji в зависимости от времени суток
    
    Returns:
        Emoji строка
    """
    hour = datetime.now().hour
    
    if 5 <= hour < 12:
        return "🌅"  # Утро
    elif 12 <= hour < 17:
        return "☀️"  # День
    elif 17 <= hour < 22:
        return "🌆"  # Вечер
    else:
        return "🌙"  # Ночь


def create_progress_bar(current: int, total: int, length: int = 10) -> str:
    """
    Создать текстовый прогресс-бар
    
    Args:
        current: Текущее значение
        total: Максимальное значение
        length: Длина бара
        
    Returns:
        Строка с прогресс-баром
    """
    if total == 0:
        percentage = 0
    else:
        percentage = (current / total) * 100
    
    filled = int((current / total) * length) if total > 0 else 0
    bar = '█' * filled + '░' * (length - filled)
    
    return f"{bar} {percentage:.1f}%"
