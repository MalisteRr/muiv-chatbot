from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import logging
from pathlib import Path

# Добавляем путь к backend для импорта
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

# Логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация FastAPI
app = FastAPI(title="MUIV ChatBot API")

# CORS для веб-интерфейса
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenRouter клиент
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
MODEL = os.getenv("LLM_MODEL", "deepseek/deepseek-r1")

openai_client = AsyncOpenAI(
    api_key=OPENAI_KEY,
    base_url=BASE_URL
)

# FAQ База (та же что в боте)
FAQ_BASE = {
    "документы": """📄 **Документы для поступления:**
• Паспорт
• Аттестат или диплом СПО
• Результаты ЕГЭ
• СНИЛС
• Фото 3x4 (6 шт)

Подача: лично, онлайн (Госуслуги, muiv.ru), по почте
📞 8 (800) 550-03-63""",

    "стоимость": """💰 **Стоимость обучения:**
• Очная: от 65 000 руб./сем
• Очно-заочная: от 55 000 руб./сем
• Заочная: от 35 000 руб./сем
• Дистанционная: от 30 000 руб./сем
📞 8 (800) 550-03-63""",

    "бюджет": """🎓 **Бюджетные места:**
Да! По направлениям: Экономика, Юриспруденция, Менеджмент, Управление персоналом, Прикладная информатика, Реклама и СО
📞 8 (800) 550-03-63""",

    "общежитие": """🏠 **Общежитие:**
Да! Для иногородних студентов. Места ограничены.
Условия: указать в заявлении, быть зачисленным.
📞 +7 (495) 500-03-63""",

    "егэ": """📝 **Без ЕГЭ:**
Можно! Выпускники СПО, второе высшее, иностранцы - сдают внутренние экзамены.
📞 8 (800) 550-03-63""",

    "контакты": """📞 **Контакты:**
☎️ 8 (800) 550-03-63, +7 (495) 500-03-63
✉️ pk@muiv.ru
📍 Москва, 2-й Кожуховский пр-д, 12, стр.1
🌐 muiv.ru"""
}

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


def find_context(question: str) -> str:
    """Поиск релевантного контекста"""
    question_lower = question.lower()
    contexts = []
    
    keywords = {
        "документы": ["документ", "нужн", "подать"],
        "стоимость": ["стоим", "цена", "сколько"],
        "бюджет": ["бюджет", "бесплатн", "кцп"],
        "общежитие": ["общежит", "жил"],
        "егэ": ["егэ", "без егэ"],
        "контакты": ["контакт", "телефон", "адрес"]
    }
    
    for key, words in keywords.items():
        if any(w in question_lower for w in words):
            if key in FAQ_BASE:
                contexts.append(FAQ_BASE[key])
    
    if not contexts:
        contexts = list(FAQ_BASE.values())[:2]
    
    return "\n\n".join(contexts[:3])


async def get_ai_answer(question: str) -> str:
    """Получить ответ от AI"""
    try:
        context = find_context(question)
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {question}"}
        ]
        
        response = await openai_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"Ошибка AI: {e}")
        return "😔 Техническая ошибка. Свяжитесь: 8 (800) 550-03-63"


# Монтирование статических файлов
frontend_path = Path(__file__).parent.parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")


@app.get("/")
async def root():
    """Главная страница"""
    index_path = frontend_path / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "MUIV ChatBot API", "status": "running"}


@app.get("/health")
async def health():
    """Проверка здоровья"""
    return {"status": "ok", "model": MODEL}


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket для реалтайм чата"""
    await websocket.accept()
    logger.info("WebSocket соединение установлено")
    
    try:
        while True:
            # Получить сообщение от клиента
            data = await websocket.receive_text()
            logger.info(f"Получен вопрос: {data[:50]}...")
            
            # Получить ответ от AI
            answer = await get_ai_answer(data)
            
            # Отправить ответ
            await websocket.send_text(answer)
            logger.info("Ответ отправлен")
            
    except WebSocketDisconnect:
        logger.info("WebSocket отключен")
    except Exception as e:
        logger.error(f"Ошибка WebSocket: {e}")
        await websocket.close()


@app.post("/api/chat")
async def chat_endpoint(data: dict):
    """REST API endpoint для чата"""
    question = data.get("message", "")
    
    if not question:
        return {"error": "Пустое сообщение"}
    
    answer = await get_ai_answer(question)
    
    return {
        "question": question,
        "answer": answer,
        "model": MODEL
    }


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("API_PORT", 8000))
    
    logger.info(f"🌐 Запуск FastAPI сервера на порту {port}")
    logger.info(f"📡 Модель: {MODEL}")
    logger.info(f"🔗 URL: http://localhost:{port}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )