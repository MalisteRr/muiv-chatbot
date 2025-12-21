"""
Скрипт загрузки FAQ данных в базу данных
Поддерживает как SQLite так и PostgreSQL
"""

import asyncio
import json
import sys
from pathlib import Path

# Добавляем путь к модулям проекта
sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()
from database.init_db import init_db, get_pool, get_sqlite, get_db_type, close_db
from utils.logger import setup_logging

logger = setup_logging()


async def load_faq_postgresql(faq_data: list):
    """Загрузка FAQ в PostgreSQL"""
    pool = get_pool()
    if not pool:
        logger.error("PostgreSQL не подключен")
        return False
    
    success_count = 0
    
    async with pool.acquire() as conn:
        for item in faq_data:
            try:
                await conn.execute("""
                    INSERT INTO faq (question, answer, category, keywords, priority, is_active)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT DO NOTHING
                """, 
                    item['question'],
                    item['answer'],
                    item['category'],
                    item.get('keywords', []),
                    item.get('priority', 5),
                    item.get('is_active', True)
                )
                success_count += 1
                logger.info(f"✅ Загружен: {item['question'][:50]}...")
            except Exception as e:
                logger.error(f"❌ Ошибка загрузки записи: {e}")
    
    return success_count


async def load_faq_sqlite(faq_data: list):
    """Загрузка FAQ в SQLite"""
    conn = get_sqlite()
    if not conn:
        logger.error("SQLite не подключен")
        return False
    
    success_count = 0
    
    for item in faq_data:
        try:
            # В SQLite keywords хранится как JSON строка
            keywords_str = json.dumps(item.get('keywords', []))
            
            await conn.execute("""
                INSERT OR IGNORE INTO faq (question, answer, category, keywords, priority, is_active)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                item['question'],
                item['answer'],
                item['category'],
                keywords_str,
                item.get('priority', 5),
                1 if item.get('is_active', True) else 0
            ))
            await conn.commit()
            
            success_count += 1
            logger.info(f"✅ Загружен: {item['question'][:50]}...")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки записи: {e}")
    
    return success_count


async def main(json_file: str):
    """
    Главная функция загрузки FAQ
    
    Args:
        json_file: Путь к JSON файлу с данными
    """
    logger.info("=" * 60)
    logger.info("📚 Загрузка FAQ данных в базу знаний")
    logger.info("=" * 60)
    
    # Проверка файла
    file_path = Path(json_file)
    if not file_path.exists():
        logger.error(f"❌ Файл не найден: {json_file}")
        return
    
    # Чтение JSON
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            faq_data = json.load(f)
        
        logger.info(f"📖 Прочитано записей: {len(faq_data)}")
    except Exception as e:
        logger.error(f"❌ Ошибка чтения JSON: {e}")
        return
    
    # Инициализация БД
    try:
        await init_db()
    except Exception as e:
        logger.error(f"❌ Ошибка инициализации БД: {e}")
        return
    
    # Определяем тип БД и загружаем данные
    db_type = get_db_type()
    logger.info(f"🗄️  Тип БД: {db_type}")
    
    if db_type == "postgresql":
        success_count = await load_faq_postgresql(faq_data)
    elif db_type == "sqlite":
        success_count = await load_faq_sqlite(faq_data)
    else:
        logger.error(f"❌ Неизвестный тип БД: {db_type}")
        return
    
    # Закрытие подключений
    await close_db()
    
    # Итоги
    logger.info("=" * 60)
    logger.info(f"✅ Успешно загружено: {success_count} / {len(faq_data)} записей")
    logger.info("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Использование: python load_faq.py <path_to_json_file>")
        print("\nПример:")
        print("  python scripts/load_faq.py data/faq_sample.json")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    try:
        asyncio.run(main(json_file))
    except KeyboardInterrupt:
        print("\n⌨️  Прервано пользователем")
    except Exception as e:
        print(f"\n💥 Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()