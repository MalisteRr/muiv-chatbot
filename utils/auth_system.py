"""
Система авторизации с ролями
Роли: admin, moderator, user
"""

import logging
from typing import Optional, Dict, List
from datetime import datetime, timedelta
from functools import wraps

logger = logging.getLogger(__name__)

# Пароли для ролей
ROLE_PASSWORDS = {
    'admin': 'admin123',
    'moderator': 'moderator321',
    # 'user' - не требует пароля
}

# Хранилище сессий {user_id: {'role': 'admin', 'expires': datetime, 'password_entered': True}}
user_sessions: Dict[int, dict] = {}

# Хранилище ожидаемых паролей {user_id: 'admin'/'moderator'}
waiting_for_password: Dict[int, str] = {}


def require_role(allowed_roles: List[str]):
    """
    Декоратор для проверки роли пользователя
    Проверяет СНАЧАЛА сессию (auth_system), ПОТОМ базу данных
    
    ИЕРАРХИЯ РОЛЕЙ: admin > moderator > user
    Админ имеет доступ ко ВСЕМ функциям модератора!
    
    Args:
        allowed_roles: Список разрешенных ролей ['admin', 'moderator']
        
    Example:
        @require_role(["admin"])
        async def admin_only_function(message: Message):
            ...
        
        @require_role(["moderator"])  # Админ тоже может!
        async def moderator_function(message: Message):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(message, *args, **kwargs):
            user_id = message.from_user.id
            
            # Иерархия ролей: админ > модератор > пользователь
            role_hierarchy = {
                'admin': 3,
                'moderator': 2,
                'user': 1
            }
            
            # Находим минимально необходимый уровень из allowed_roles
            required_level = min([role_hierarchy.get(role, 0) for role in allowed_roles])
            
            # ========== ПРОВЕРКА 1: Временная сессия (auth_system) ==========
            session_role = get_user_role(user_id)
            session_level = role_hierarchy.get(session_role, 0)
            
            if session_level >= required_level:
                logger.debug(f"✅ Пользователь {user_id} доступ разрешен через сессию ({session_role}, уровень {session_level})")
                return await func(message, *args, **kwargs)
            
            # ========== ПРОВЕРКА 2: База данных (постоянная роль) ==========
            try:
                from database.crud import get_user_info
                
                user_info = await get_user_info(user_id)
                
                if user_info:
                    db_role = user_info.get('role', 'user')
                    db_level = role_hierarchy.get(db_role, 0)
                    
                    if db_level >= required_level:
                        logger.debug(f"✅ Пользователь {user_id} доступ разрешен через БД ({db_role}, уровень {db_level})")
                        return await func(message, *args, **kwargs)
            except Exception as e:
                logger.error(f"Ошибка проверки роли из БД: {e}")
            
            # ========== ДОСТУП ЗАПРЕЩЕН ==========
            logger.warning(
                f"❌ Пользователь {user_id} попытался получить доступ к {func.__name__} "
                f"(требуется: {allowed_roles}, есть: {session_role})"
            )
            
            await message.answer(
                "🚫 <b>Доступ запрещен</b>\n\n"
                f"Эта функция доступна только для: {', '.join(allowed_roles)}\n"
                "Ваша роль: " + session_role
            )
            return
        
        return wrapper
    return decorator


def start_password_prompt(user_id: int, role: str) -> bool:
    """
    Начать запрос пароля для роли
    
    Args:
        user_id: ID пользователя
        role: Роль (admin/moderator)
        
    Returns:
        True если пароль требуется
    """
    if role not in ROLE_PASSWORDS:
        return False
    
    waiting_for_password[user_id] = role
    logger.info(f"Пользователь {user_id} запросил доступ к роли {role}")
    return True


def check_password(user_id: int, password: str) -> Optional[str]:
    """
    Проверить пароль
    
    Args:
        user_id: ID пользователя
        password: Введенный пароль
        
    Returns:
        Роль если пароль верный, None если неверный
    """
    if user_id not in waiting_for_password:
        return None
    
    expected_role = waiting_for_password[user_id]
    expected_password = ROLE_PASSWORDS.get(expected_role)
    
    if password == expected_password:
        # Пароль верный - создаем сессию
        user_sessions[user_id] = {
            'role': expected_role,
            'expires': datetime.now() + timedelta(hours=24),  # Сессия на 24 часа
            'authorized_at': datetime.now()
        }
        del waiting_for_password[user_id]
        logger.info(f"✅ Пользователь {user_id} авторизован как {expected_role}")
        return expected_role
    else:
        logger.warning(f"❌ Пользователь {user_id} ввел неверный пароль для {expected_role}")
        return None


def is_waiting_for_password(user_id: int) -> bool:
    """
    Проверить ожидает ли пользователь ввода пароля
    
    Args:
        user_id: ID пользователя
        
    Returns:
        True если ожидает
    """
    return user_id in waiting_for_password


def cancel_password_prompt(user_id: int):
    """
    Отменить запрос пароля
    
    Args:
        user_id: ID пользователя
    """
    if user_id in waiting_for_password:
        del waiting_for_password[user_id]


def get_user_role(user_id: int) -> str:
    """
    Получить роль пользователя
    
    Args:
        user_id: ID пользователя
        
    Returns:
        Роль пользователя (admin/moderator/user)
    """
    # Проверяем сессию
    if user_id in user_sessions:
        session = user_sessions[user_id]
        
        # Проверяем не истекла ли сессия
        if session['expires'] > datetime.now():
            return session['role']
        else:
            # Сессия истекла
            del user_sessions[user_id]
            logger.info(f"Сессия пользователя {user_id} истекла")
    
    # По умолчанию - обычный пользователь
    return 'user'


def has_role(user_id: int, required_role: str) -> bool:
    """
    Проверить имеет ли пользователь требуемую роль
    
    Args:
        user_id: ID пользователя
        required_role: Требуемая роль
        
    Returns:
        True если имеет доступ
    """
    user_role = get_user_role(user_id)
    
    # Иерархия ролей: admin > moderator > user
    role_hierarchy = {
        'admin': 3,
        'moderator': 2,
        'user': 1
    }
    
    user_level = role_hierarchy.get(user_role, 0)
    required_level = role_hierarchy.get(required_role, 0)
    
    return user_level >= required_level


def logout(user_id: int):
    """
    Выйти из сессии
    
    Args:
        user_id: ID пользователя
    """
    if user_id in user_sessions:
        role = user_sessions[user_id]['role']
        del user_sessions[user_id]
        logger.info(f"Пользователь {user_id} вышел из роли {role}")
    
    if user_id in waiting_for_password:
        del waiting_for_password[user_id]


def get_session_info(user_id: int) -> Optional[dict]:
    """
    Получить информацию о сессии
    
    Args:
        user_id: ID пользователя
        
    Returns:
        Словарь с информацией о сессии или None
    """
    if user_id not in user_sessions:
        return None
    
    session = user_sessions[user_id]
    time_left = session['expires'] - datetime.now()
    
    return {
        'role': session['role'],
        'authorized_at': session['authorized_at'],
        'expires': session['expires'],
        'time_left_minutes': int(time_left.total_seconds() / 60)
    }


def extend_session(user_id: int, hours: int = 24):
    """
    Продлить сессию
    
    Args:
        user_id: ID пользователя
        hours: Количество часов
    """
    if user_id in user_sessions:
        user_sessions[user_id]['expires'] = datetime.now() + timedelta(hours=hours)
        logger.info(f"Сессия пользователя {user_id} продлена на {hours} часов")
