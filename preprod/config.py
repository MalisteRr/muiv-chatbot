"""
Конфигурация приложения
Все настройки загружаются из переменных окружения
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class BotConfig:
    """Конфигурация Telegram бота"""
    token: str
    admin_ids: list[int]
    
    @classmethod
    def from_env(cls):
        """Загрузить конфигурацию из переменных окружения"""
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not token:
            raise ValueError("TELEGRAM_BOT_TOKEN не установлен в .env")
        
        # Список ID администраторов (через запятую)
        admin_ids_str = os.getenv("ADMIN_IDS", "")
        admin_ids = [int(id.strip()) for id in admin_ids_str.split(",") if id.strip()]
        
        return cls(token=token, admin_ids=admin_ids)


@dataclass
class DatabaseConfig:
    """Конфигурация базы данных"""
    url: str
    min_pool_size: int = 2
    max_pool_size: int = 10
    command_timeout: int = 60
    
    @classmethod
    def from_env(cls):
        """Загрузить конфигурацию из переменных окружения"""
        url = os.getenv("DATABASE_URL")
        if not url:
            raise ValueError("DATABASE_URL не установлен в .env")
        
        return cls(
            url=url,
            min_pool_size=int(os.getenv("DB_MIN_POOL_SIZE", "2")),
            max_pool_size=int(os.getenv("DB_MAX_POOL_SIZE", "10")),
            command_timeout=int(os.getenv("DB_COMMAND_TIMEOUT", "60"))
        )


@dataclass
class AIConfig:
    """Конфигурация AI моделей"""
    api_key: str
    base_url: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 200
    
    @classmethod
    def from_env(cls):
        """Загрузить конфигурацию из переменных окружения"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY не установлен в .env")
        
        return cls(
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
            model=os.getenv("LLM_MODEL", "deepseek/deepseek-chat"),
            temperature=float(os.getenv("AI_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("AI_MAX_TOKENS", "200"))
        )


@dataclass
class AppConfig:
    """Общая конфигурация приложения"""
    bot: BotConfig
    database: DatabaseConfig
    ai: AIConfig
    debug: bool = False
    
    @classmethod
    def load(cls):
        """Загрузить всю конфигурацию"""
        return cls(
            bot=BotConfig.from_env(),
            database=DatabaseConfig.from_env(),
            ai=AIConfig.from_env(),
            debug=os.getenv("DEBUG", "false").lower() == "true"
        )


# Глобальный экземпляр конфигурации
config = AppConfig.load()


# System Prompt для AI
SYSTEM_PROMPT = """Ты - дружелюбный помощник приемной комиссии МУИВ (Московский Университет им С.Ю. Витте).

СТИЛЬ ОБЩЕНИЯ:
- Обращайся на "вы"
- Пиши кратко и по делу (2-3 абзаца максимум)
- Будь естественным и доброжелательным
- Используй emoji умеренно: 📚 🎓 💰 📞 ✉️ 🏠 📝

ВАЖНЫЕ ПРАВИЛА:
- Отвечай ТОЛЬКО на основе предоставленной информации из базы данных
- Если нет точного ответа - скажи честно и направь к специалистам
- НЕ придумывай факты, цифры и даты
- Всегда предлагай связаться с приемной комиссией для уточнений

КОНТАКТЫ МУИВ (всегда указывай в конце ответа):
📞 8 (800) 550-03-63 (бесплатно по России)
☎️ +7 (495) 500-03-63
✉️ pk@muiv.ru
🌐 muiv.ru

Твоя задача - помочь абитуриентам быстро найти нужную информацию и направить их к специалистам для детальной консультации."""


# Константы для работы с FAQ
FAQ_SEARCH_LIMIT = 3  # Количество результатов при поиске
CHAT_HISTORY_LIMIT = 10  # Максимальная длина истории диалога
CONTEXT_MESSAGES_LIMIT = 4  # Сколько сообщений из истории использовать для контекста