import asyncio
import logging
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
import asyncpg

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import CommandStart, Command
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton
from aiogram.fsm.storage.memory import MemoryStorage

load_dotenv()

# Логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logs/bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Конфигурация
TOKEN = os.getenv("8198101919:AAFjUqrYN2bktrEtip01zV1e-fxd4UyN8tY")
OPENAI_KEY = os.getenv("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")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
MODEL = os.getenv("LLM_MODEL", "deepseek/deepseek-r1")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:muiv2024@localhost:5432/muiv_bot")

# Инициализация
bot = Bot(token=TOKEN)
dp = Dispatcher(storage=MemoryStorage())

# OpenRouter клиент
openai_client = AsyncOpenAI(
    api_key=OPENAI_KEY,
    base_url=BASE_URL
)

# Пул подключений к БД
db_pool = None

# История диалогов
chat_history = {}

# System Prompt
SYSTEM_PROMPT = """Ты - бот приемной комиссии МУИВ.

ПРАВИЛА:
- Обращайся на "вы", кратко (2-3 абзаца)
- Используй ТОЛЬКО информацию из контекста
- Если нет инфо - дай контакты: 8 (800) 550-03-63
- Emoji умеренно (📚 🎓 💰)

КОНТАКТЫ:
📞 8 (800) 550-03-63
✉️ pk@muiv.ru
🌐 muiv.ru"""


# ========== РАБОТА С БАЗОЙ ДАННЫХ ==========

async def init_db():
    """Инициализация пула подключений к БД"""
    global db_pool
    try:
        db_pool = await asyncpg.create_pool(
            DATABASE_URL,
            min_size=2,
            max_size=10,
            command_timeout=60
        )
        logger.info("✅ Подключение к PostgreSQL установлено")
    except Exception as e:
        logger.error(f"❌ Ошибка подключения к БД: {e}")
        raise


async def search_faq_by_keywords(keywords: list, limit: int = 3) -> list:
    """Поиск FAQ по ключевым словам"""
    if not db_pool:
        return []
    
    try:
        async with db_pool.acquire() as conn:
            # Полнотекстовый поиск
            query = """
            SELECT 
                id,
                question,
                answer,
                category,
                ts_rank(
                    to_tsvector('russian', question || ' ' || answer),
                    plainto_tsquery('russian', $1)
                ) as rank
            FROM faq
            WHERE to_tsvector('russian', question || ' ' || answer) @@ 
                  plainto_tsquery('russian', $1)
            ORDER BY rank DESC
            LIMIT $2
            """
            
            search_text = " ".join(keywords)
            rows = await conn.fetch(query, search_text, limit)
            
            return [
                {
                    "id": row["id"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "category": row["category"],
                    "rank": float(row["rank"])
                }
                for row in rows
            ]
    except Exception as e:
        logger.error(f"Ошибка поиска в БД: {e}")
        return []


async def get_faq_by_category(category: str) -> list:
    """Получить FAQ по категории"""
    if not db_pool:
        return []
    
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT question, answer 
                FROM faq 
                WHERE category ILIKE $1
                LIMIT 5
            """, f"%{category}%")
            
            return [
                {"question": row["question"], "answer": row["answer"]}
                for row in rows
            ]
    except Exception as e:
        logger.error(f"Ошибка получения категории: {e}")
        return []


async def save_chat_history(user_id: int, user_name: str, message: str, response: str):
    """Сохранить историю в БД"""
    if not db_pool:
        return
    
    try:
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO chat_history (user_id, user_name, message, bot_response, source)
                VALUES ($1, $2, $3, $4, $5)
            """, user_id, user_name, message, response, "telegram")
    except Exception as e:
        logger.error(f"Ошибка сохранения истории: {e}")


async def log_analytics(user_id: int, question: str, found: bool):
    """Логирование в аналитику"""
    if not db_pool:
        return
    
    try:
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO analytics (user_id, question_text, found_answer)
                VALUES ($1, $2, $3)
            """, user_id, question, found)
    except Exception as e:
        logger.error(f"Ошибка логирования: {e}")


# ========== КЛАВИАТУРЫ ==========

def get_main_keyboard():
    keyboard = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="📚 Документы"), KeyboardButton(text="💰 Стоимость")],
            [KeyboardButton(text="🎓 Бюджет"), KeyboardButton(text="🏠 Общежитие")],
            [KeyboardButton(text="📝 Без ЕГЭ"), KeyboardButton(text="🏫 Формы обучения")],
            [KeyboardButton(text="📞 Контакты"), KeyboardButton(text="❓ Помощь")]
        ],
        resize_keyboard=True,
        input_field_placeholder="Задайте ваш вопрос..."
    )
    return keyboard


# ========== ЛОГИКА ОТВЕТОВ ==========

def extract_keywords(text: str) -> list:
    """Извлечение ключевых слов"""
    text_lower = text.lower()
    
    # Стоп-слова
    stop_words = {"как", "что", "где", "когда", "почему", "какой", "есть", "ли", 
                  "можно", "нужно", "это", "то", "в", "на", "с", "у", "по", "для"}
    
    words = [w for w in text_lower.split() if len(w) > 3 and w not in stop_words]
    return words[:5]


async def get_context_from_db(question: str) -> str:
    """Получить контекст из БД"""
    keywords = extract_keywords(question)
    
    if not keywords:
        return "Релевантной информации не найдено."
    
    results = await search_faq_by_keywords(keywords, limit=3)
    
    if not results:
        return "Релевантной информации не найдено."
    
    context_parts = []
    for r in results:
        context_parts.append(f"Вопрос: {r['question']}\nОтвет: {r['answer']}\n---")
    
    return "\n\n".join(context_parts)


async def get_ai_response(user_id: int, question: str) -> str:
    """Получить ответ от AI с контекстом из БД"""
    try:
        # Получить контекст из БД
        context = await get_context_from_db(question)
        found_answer = context != "Релевантной информации не найдено."
        
        # Логирование
        await log_analytics(user_id, question, found_answer)
        
        # История
        history = chat_history.get(user_id, [])[-4:]
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        messages.extend(history)
        
        user_message = f"Контекст из базы знаний:\n{context}\n\nВопрос: {question}"
        messages.append({"role": "user", "content": user_message})
        
        # Запрос к AI
        response = await openai_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=600
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Обновить историю
        if user_id not in chat_history:
            chat_history[user_id] = []
        
        chat_history[user_id].append({"role": "user", "content": question})
        chat_history[user_id].append({"role": "assistant", "content": answer})
        
        if len(chat_history[user_id]) > 10:
            chat_history[user_id] = chat_history[user_id][-10:]
        
        return answer
        
    except Exception as e:
        logger.error(f"Ошибка генерации ответа: {e}")
        return "😔 Техническая ошибка. Свяжитесь: 8 (800) 550-03-63"


# ========== ОБРАБОТЧИКИ КОМАНД ==========

@dp.message(CommandStart())
async def cmd_start(message: Message):
    user_name = message.from_user.first_name
    
    welcome = f"""👋 Здравствуйте, {user_name}!

Я бот-помощник приемной комиссии **МУИВ**.

**Помогу узнать:**
📚 Документы для поступления
💰 Стоимость обучения
🎓 Бюджетные места
🏠 Общежитие
📝 Поступление без ЕГЭ

**Выберите тему или задайте вопрос!** 👇"""
    
    await message.answer(welcome, reply_markup=get_main_keyboard(), parse_mode="Markdown")
    logger.info(f"Новый пользователь: {message.from_user.id} - {user_name}")


@dp.message(Command("help"))
async def cmd_help(message: Message):
    help_text = """🤖 **Как пользоваться:**

1️⃣ Напишите вопрос текстом
2️⃣ Или выберите тему из меню

**Команды:**
/start - Начать заново
/help - Справка
/stats - Статистика бота

📞 8 (800) 550-03-63
✉️ pk@muiv.ru"""
    
    await message.answer(help_text, parse_mode="Markdown")


@dp.message(Command("stats"))
async def cmd_stats(message: Message):
    """Статистика из БД"""
    if not db_pool:
        await message.answer("БД недоступна")
        return
    
    try:
        async with db_pool.acquire() as conn:
            total_faq = await conn.fetchval("SELECT COUNT(*) FROM faq")
            total_chats = await conn.fetchval("SELECT COUNT(*) FROM chat_history")
            
            stats = f"""📊 **Статистика бота:**

📝 Вопросов в базе: {total_faq}
💬 Всего диалогов: {total_chats}

🗄️ База данных: PostgreSQL 18.0
🤖 Модель: {MODEL}"""
            
            await message.answer(stats, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Ошибка статистики: {e}")


# Обработчики кнопок
@dp.message(F.text.in_(["📚 Документы", "💰 Стоимость", "🎓 Бюджет", 
                        "🏠 Общежитие", "📝 Без ЕГЭ", "🏫 Формы обучения"]))
async def handle_category_button(message: Message):
    """Обработка кнопок категорий"""
    category_map = {
        "📚 Документы": "Поступление",
        "💰 Стоимость": "Стоимость",
        "🎓 Бюджет": "бюджет",
        "🏠 Общежитие": "Общежитие",
        "📝 Без ЕГЭ": "егэ",
        "🏫 Формы обучения": "Формы"
    }
    
    category = category_map.get(message.text, "")
    
    # Получить FAQ из категории
    faqs = await get_faq_by_category(category)
    
    if faqs:
        response = f"**{message.text}**\n\n{faqs[0]['answer']}"
    else:
        response = f"Информация по теме '{message.text}' не найдена.\n\n📞 8 (800) 550-03-63"
    
    await message.answer(response, parse_mode="Markdown", reply_markup=get_main_keyboard())


@dp.message(F.text == "📞 Контакты")
async def handle_contacts(message: Message):
    contacts = """📞 **Контакты МУИВ:**

☎️ 8 (800) 550-03-63 (бесплатно)
☎️ +7 (495) 500-03-63
✉️ pk@muiv.ru
📍 Москва, 2-й Кожуховский пр-д, 12, стр.1
🌐 muiv.ru

**Режим работы:**
Пн-Чт: 09:30-18:15
Пт: 09:30-17:00
Сб: 10:00-15:00"""
    
    await message.answer(contacts, parse_mode="Markdown")


# Обработка текстовых сообщений
@dp.message(F.text)
async def handle_text(message: Message):
    user_id = message.from_user.id
    user_name = message.from_user.full_name
    question = message.text
    
    # Приветствия
    if any(word in question.lower() for word in ["привет", "здравствуй", "добрый"]):
        await message.answer("👋 Здравствуйте! Задавайте вопросы о поступлении!", 
                           reply_markup=get_main_keyboard())
        return
    
    if any(word in question.lower() for word in ["спасибо", "благодарю"]):
        await message.answer("😊 Рад помочь! Обращайтесь!")
        return
    
    # Показать индикатор
    await bot.send_chat_action(message.chat.id, "typing")
    
    # Получить AI ответ
    logger.info(f"Вопрос от {user_id}: {question[:50]}...")
    answer = await get_ai_response(user_id, question)
    
    # Сохранить в БД
    await save_chat_history(user_id, user_name, question, answer)
    
    await message.answer(answer, parse_mode="Markdown", reply_markup=get_main_keyboard())


# ========== ЗАПУСК ==========

async def on_startup():
    logger.info("🤖 Бот запускается...")
    await init_db()
    logger.info(f"📡 Модель: {MODEL}")
    logger.info("✅ Бот готов!")


async def on_shutdown():
    logger.info("🛑 Бот останавливается...")
    if db_pool:
        await db_pool.close()
    logger.info("✅ Соединение с БД закрыто")


async def main():
    dp.startup.register(on_startup)
    dp.shutdown.register(on_shutdown)
    
    await bot.delete_webhook(drop_pending_updates=True)
    logger.info("🚀 Начинаю polling...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("⌨️ Бот остановлен")
    except Exception as e:
        logger.critical(f"💥 Ошибка: {e}", exc_info=True)