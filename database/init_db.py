"""
Универсальная инициализация БД
Поддержка PostgreSQL и SQLite
"""

import logging
import os
from typing import Optional
import aiosqlite
import asyncpg

from config import config

logger = logging.getLogger(__name__)

# Глобальные переменные для подключений
db_pool: Optional[asyncpg.Pool] = None
sqlite_conn: Optional[aiosqlite.Connection] = None
db_type: str = "unknown"  # "postgresql" или "sqlite"


async def init_db():
    """
    Инициализация подключения к БД
    Автоматически определяет тип БД по DATABASE_URL
    """
    global db_pool, sqlite_conn, db_type
    
    database_url = config.database.url
    
    try:
        # Определяем тип БД
        if database_url.startswith('sqlite'):
            db_type = "sqlite"
            await init_sqlite(database_url)
        elif database_url.startswith('postgresql'):
            db_type = "postgresql"
            await init_postgresql(database_url)
        else:
            raise ValueError(f"Неподдерживаемый тип БД в DATABASE_URL: {database_url}")
        
        logger.info(f"✅ База данных инициализирована ({db_type})")
        
    except Exception as e:
        logger.error(f"❌ Ошибка подключения к БД: {e}", exc_info=True)
        raise


async def init_sqlite(database_url: str):
    """Инициализация SQLite"""
    global sqlite_conn
    
    # Извлечь путь к файлу БД
    db_path = database_url.replace('sqlite:///', '').replace('sqlite://', '')
    
    # Создать директорию если не существует
    os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else '.', exist_ok=True)
    
    logger.info(f"Подключение к SQLite: {db_path}")
    
    # Создание подключения
    sqlite_conn = await aiosqlite.connect(db_path)
    sqlite_conn.row_factory = aiosqlite.Row
    
    # Включить поддержку внешних ключей
    await sqlite_conn.execute("PRAGMA foreign_keys = ON")
    
    logger.info("✅ SQLite подключен")
    
    # Создание таблиц
    await create_tables_sqlite()
    
    # Проверка данных
    async with sqlite_conn.execute("SELECT COUNT(*) FROM faq") as cursor:
        faq_count = (await cursor.fetchone())[0]
    async with sqlite_conn.execute("SELECT COUNT(*) FROM users") as cursor:
        users_count = (await cursor.fetchone())[0]
    
    logger.info(f"📊 Записей в FAQ: {faq_count}")
    logger.info(f"👥 Пользователей: {users_count}")


async def init_postgresql(database_url: str):
    """Инициализация PostgreSQL"""
    global db_pool
    
    logger.info("Подключение к PostgreSQL...")
    
    # Создание пула подключений
    db_pool = await asyncpg.create_pool(
        database_url,
        min_size=config.database.min_pool_size,
        max_size=config.database.max_pool_size,
        command_timeout=config.database.command_timeout
    )
    
    logger.info(f"✅ Пул подключений создан (размер: {config.database.min_pool_size}-{config.database.max_pool_size})")
    
    # Проверка подключения и создание расширений
    async with db_pool.acquire() as conn:
        # Проверка версии PostgreSQL
        version = await conn.fetchval("SELECT version()")
        logger.info(f"PostgreSQL версия: {version.split(',')[0]}")
        
        # Создание расширения pg_trgm для нечеткого поиска
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
            logger.info("✅ Расширение pg_trgm активировано")
        except Exception as e:
            logger.warning(f"Не удалось создать расширение pg_trgm: {e}")
            logger.warning("Нечеткий поиск будет недоступен")
        
        # Создание таблиц если их нет
        await create_tables_postgresql(conn)
        
        # Проверка количества записей
        faq_count = await conn.fetchval("SELECT COUNT(*) FROM faq")
        users_count = await conn.fetchval("SELECT COUNT(*) FROM users")
        
        logger.info(f"📊 Записей в FAQ: {faq_count}")
        logger.info(f"👥 Пользователей: {users_count}")


async def create_tables_sqlite():
    """Создание таблиц для SQLite"""
    global sqlite_conn
    
    logger.info("Создание таблиц SQLite...")
    
    # Таблица пользователей
    await sqlite_conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            username TEXT,
            first_name TEXT,
            last_name TEXT,
            role TEXT DEFAULT 'user',
            is_blocked INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Таблица FAQ
    await sqlite_conn.execute("""
        CREATE TABLE IF NOT EXISTS faq (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            category TEXT,
            keywords TEXT,
            priority INTEGER DEFAULT 0,
            is_active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Индексы для FAQ
    await sqlite_conn.execute("CREATE INDEX IF NOT EXISTS idx_faq_category ON faq(category)")
    await sqlite_conn.execute("CREATE INDEX IF NOT EXISTS idx_faq_question ON faq(question)")
    
    # Таблица истории чата
    await sqlite_conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            user_name TEXT,
            message TEXT NOT NULL,
            bot_response TEXT,
            source TEXT DEFAULT 'telegram',
            found_in_db INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        )
    """)
    
    await sqlite_conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_user_id ON chat_history(user_id)")
    await sqlite_conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_created ON chat_history(created_at DESC)")
    
    # Таблица аналитики
    await sqlite_conn.execute("""
        CREATE TABLE IF NOT EXISTS analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            question_text TEXT,
            found_answer INTEGER DEFAULT 0,
            sources_count INTEGER DEFAULT 0,
            response_time REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        )
    """)
    
    await sqlite_conn.execute("CREATE INDEX IF NOT EXISTS idx_analytics_user ON analytics(user_id)")
    await sqlite_conn.execute("CREATE INDEX IF NOT EXISTS idx_analytics_created ON analytics(created_at DESC)")
    
    # Таблица обратной связи
    await sqlite_conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            chat_message_id INTEGER,
            rating INTEGER CHECK (rating >= 1 AND rating <= 5),
            feedback_type TEXT,
            comment TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        )
    """)
    
    # Таблица статистики пользователей
    await sqlite_conn.execute("""
        CREATE TABLE IF NOT EXISTS user_stats (
            user_id INTEGER PRIMARY KEY,
            total_messages INTEGER DEFAULT 0,
            found_answers INTEGER DEFAULT 0,
            not_found INTEGER DEFAULT 0,
            avg_rating REAL DEFAULT 0,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        )
    """)
    
    # Таблица категорий FAQ
    await sqlite_conn.execute("""
        CREATE TABLE IF NOT EXISTS faq_categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            emoji TEXT,
            sort_order INTEGER DEFAULT 0,
            is_active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Таблица логов администратора
    await sqlite_conn.execute("""
        CREATE TABLE IF NOT EXISTS admin_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            admin_id INTEGER NOT NULL,
            action TEXT NOT NULL,
            details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (admin_id) REFERENCES users(user_id) ON DELETE CASCADE
        )
    """)
    
    # Таблица рассылок
    await sqlite_conn.execute("""
        CREATE TABLE IF NOT EXISTS broadcasts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            admin_id INTEGER NOT NULL,
            message_text TEXT NOT NULL,
            sent_count INTEGER DEFAULT 0,
            failed_count INTEGER DEFAULT 0,
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            FOREIGN KEY (admin_id) REFERENCES users(user_id) ON DELETE CASCADE
        )
    """)
    
    await sqlite_conn.commit()
    logger.info("✅ Все таблицы SQLite созданы/проверены")


async def create_tables_postgresql(conn: asyncpg.Connection):
    """Создание таблиц для PostgreSQL"""
    logger.info("Создание таблиц PostgreSQL...")
    
 # Таблица пользователей
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id BIGINT PRIMARY KEY,
            username VARCHAR(255),
            first_name VARCHAR(255),
            last_name VARCHAR(255),
            role VARCHAR(50) DEFAULT 'user',
            is_blocked BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Таблица FAQ (база знаний)
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS faq (
            id SERIAL PRIMARY KEY,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            category VARCHAR(255),
            keywords TEXT[],
            priority INTEGER DEFAULT 0,
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Индексы для быстрого поиска в FAQ
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_faq_category ON faq(category)
    """)
    
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_faq_keywords ON faq USING GIN(keywords)
    """)
    
    # Индексы для полнотекстового поиска
    try:
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_faq_question_trgm ON faq USING gin(question gin_trgm_ops)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_faq_answer_trgm ON faq USING gin(answer gin_trgm_ops)
        """)
    except:
        logger.warning("Не удалось создать GIN индексы (требуется pg_trgm)")
    
    # Таблица истории чата
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id SERIAL PRIMARY KEY,
            user_id BIGINT NOT NULL,
            user_name VARCHAR(255),
            message TEXT NOT NULL,
            bot_response TEXT,
            source VARCHAR(50) DEFAULT 'telegram',
            found_in_db BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        )
    """)
    
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_chat_history_user_id ON chat_history(user_id)
    """)
    
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_chat_history_created_at ON chat_history(created_at DESC)
    """)
    
    # Таблица аналитики
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS analytics (
            id SERIAL PRIMARY KEY,
            user_id BIGINT NOT NULL,
            question_text TEXT,
            found_answer BOOLEAN DEFAULT FALSE,
            sources_count INTEGER DEFAULT 0,
            response_time FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        )
    """)
    
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_analytics_user_id ON analytics(user_id)
    """)
    
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_analytics_created_at ON analytics(created_at DESC)
    """)
    
    # Таблица обратной связи
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id SERIAL PRIMARY KEY,
            user_id BIGINT NOT NULL,
            chat_message_id INTEGER,
            rating INTEGER CHECK (rating >= 1 AND rating <= 5),
            feedback_type VARCHAR(50),
            comment TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        )
    """)
    
    # Таблица статистики пользователей
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS user_stats (
            user_id BIGINT PRIMARY KEY,
            total_messages INTEGER DEFAULT 0,
            found_answers INTEGER DEFAULT 0,
            not_found INTEGER DEFAULT 0,
            avg_rating FLOAT DEFAULT 0,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        )
    """)
    
    # Таблица категорий FAQ
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS faq_categories (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) UNIQUE NOT NULL,
            description TEXT,
            emoji VARCHAR(10),
            sort_order INTEGER DEFAULT 0,
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Таблица логов администратора
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS admin_logs (
            id SERIAL PRIMARY KEY,
            admin_id BIGINT NOT NULL,
            action VARCHAR(255) NOT NULL,
            details JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (admin_id) REFERENCES users(user_id) ON DELETE CASCADE
        )
    """)
    
    # Таблица уведомлений/рассылок
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS broadcasts (
            id SERIAL PRIMARY KEY,
            admin_id BIGINT NOT NULL,
            message_text TEXT NOT NULL,
            sent_count INTEGER DEFAULT 0,
            failed_count INTEGER DEFAULT 0,
            status VARCHAR(50) DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            FOREIGN KEY (admin_id) REFERENCES users(user_id) ON DELETE CASCADE
        )
    """)
    
    logger.info("✅ Все таблицы созданы/проверены")


async def close_db():
    """Закрытие подключений к БД"""
    global db_pool, sqlite_conn, db_type
    
    logger.info("Закрытие подключений к БД...")
    
    if db_type == "postgresql" and db_pool:
        await db_pool.close()
        db_pool = None
    elif db_type == "sqlite" and sqlite_conn:
        await sqlite_conn.close()
        sqlite_conn = None
    
    logger.info("✅ Подключения к БД закрыты")


def get_pool() -> Optional[asyncpg.Pool]:
    """Получить пул PostgreSQL"""
    return db_pool


def get_sqlite() -> Optional[aiosqlite.Connection]:
    """Получить подключение SQLite"""
    return sqlite_conn


def get_db_type() -> str:
    """Получить тип БД"""
    return db_type