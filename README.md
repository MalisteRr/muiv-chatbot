# 🤖 MUIV Chatbot

> Интеллектуальный чат-бот для автоматизации консультирования абитуриентов Московского университета имени С.Ю. Витте

## 📋 Описание

Telegram-бот на основе машинного обучения для автоматизации ответов на вопросы абитуриентов. Система использует **трёхуровневую каскадную архитектуру** с RuBERT классификатором для определения категории вопроса и DeepSeek R1 для генерации ответов на нестандартные запросы.

**Telegram:** 

---

## ✨ Основные возможности

- ✅ **Автоматическая классификация** вопросов с помощью RuBERT (97.49% точность)
- ✅ **База знаний** из 63 FAQ вопросов с категориями
- ✅ **Интеграция с DeepSeek R1** для сложных запросов
- ✅ **Система рейтинга** ответов (👍/👎 + детальная оценка)
- ✅ **Панели администратора и модератора** с авторизацией
- ✅ **История диалогов** с контекстом
- ✅ **Экспорт статистики** в CSV
- ✅ **Поддержка SQLite и PostgreSQL**

---

## 🏗️ Архитектура

Система использует **трёхуровневую каскадную архитектуру**:

```
┌─────────────────────────────────────────────────────────────┐
│                   Запрос пользователя                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ УРОВЕНЬ 1: RuBERT Классификатор                             │
│ ├─ Определение категории вопроса                            │
│ ├─ Уверенность: 70%+ → переход к FAQ                        │
│ └─ Уверенность: <70% → переход к DeepSeek                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ УРОВЕНЬ 2: FAQ Database Search                              │
│ ├─ Поиск по категории + ключевым словам                     │
│ ├─ База: 63 вопроса в 8 категориях                          │
│ └─ Если найдено → ОТВЕТ                                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ УРОВЕНЬ 3: DeepSeek R1 API                                  │
│ ├─ Генерация ответа для нестандартных вопросов              │
│ ├─ Использование контекста из FAQ                           │
│ └─ System Prompt с правилами МУИВ                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                     Ответ пользователю                       │
│                  + Кнопки рейтинга (👍/👎)                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧠 RuBERT Классификатор

**Модель:** [`cointegrated/rubert-tiny2`](https://huggingface.co/cointegrated/rubert-tiny2)

**Обучение:**
- **Датасет:** 4,776 примеров вопросов абитуриентов
- **Категории:** 8 (документы, стоимость, бюджет, общежитие, ЕГЭ, формы, сроки, направления)
- **Точность:** 97.49% на валидационной выборке
- **Inference:** ~50-100 мс на CPU

**Архитектура:**
- Encoder: 12-layer BERT
- Hidden size: 312
- Attention heads: 12
- Classification head: Linear(312 → 8)

**Обучение модели:**
```bash
# Notebook для обучения
jupyter notebook ml/train_rubert_classifier.ipynb

# Модель сохранена в:
ml/models/final_model/
```

---

## 📊 База данных

**Поддерживаемые СУБД:**
- **SQLite** (для разработки и тестирования)
- **PostgreSQL** (для продакшена)

**Схема БД (9 таблиц):**

1. **`users`** - Пользователи бота
   ```sql
   - user_id (PK)
   - username, first_name, last_name
   - role (admin/moderator/user)
   - is_active, created_at
   ```

2. **`chat_history`** - История диалогов
   ```sql
   - id (PK)
   - user_id (FK)
   - message, response, category
   - confidence, created_at
   ```

3. **`faq`** - База знаний FAQ
   ```sql
   - id (PK)
   - question, answer, category
   - keywords[], priority, is_active
   ```

4. **`ratings`** - Рейтинги ответов
   ```sql
   - id (PK)
   - user_id (FK), chat_message_id
   - rating (1-5), feedback_type
   - comment, created_at
   ```

5. **`analytics`** - Аналитика использования

---

## 🚀 Установка и запуск

### **Требования:**
- Python 3.10+
- pip
- 4 GB RAM (16 GB для продакшена с RuBERT)
- 2 GB свободного места (для PyTorch + RuBERT модели)

### **1. Клонирование репозитория**

```bash
git clone https://github.com/MalisteRr/muiv-chatbot.git
cd muiv-chatbot
```

### **2. Создание виртуального окружения**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### **3. Установка зависимостей**

```bash
pip install -r requirements.txt
```

**Минимальная установка (без RuBERT, только DeepSeek):**
```bash
pip install aiogram aiosqlite openai python-dotenv httpx
```
⚠️ Без RuBERT все запросы будут через DeepSeek API (медленнее, ~2-4 сек)

**Полная установка (с RuBERT - рекомендуется):**
```bash
pip install -r requirements.txt
```
✅ 87% запросов обрабатываются за <100 мс через RuBERT + FAQ

### **4. Настройка переменных окружения**

Создайте файл `.env` в корне проекта:

```bash
# Telegram Bot
TELEGRAM_BOT_TOKEN=your_bot_token_here
ADMIN_IDS=123456789,987654321

# Database (выберите один вариант)
# Вариант 1: SQLite (для разработки)
DATABASE_URL=sqlite:///data/muiv_bot.db

# Вариант 2: PostgreSQL (для продакшена)
# DATABASE_URL=postgresql://user:password@localhost:5432/muiv_bot

# AI Model (DeepSeek через OpenRouter)
OPENAI_API_KEY=your_openrouter_api_key
OPENAI_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=deepseek/deepseek-r1
AI_TEMPERATURE=0.7
AI_MAX_TOKENS=1000

# RuBERT Classifier (опционально)
RUBERT_MODEL_PATH=ml/models/rubert_final/final_model
RUBERT_THRESHOLD=0.7

# App Settings
DEBUG=false
LOG_LEVEL=INFO
```

**Получение токенов:**
- **Telegram Bot Token:** [@BotFather](https://t.me/BotFather)
- **OpenRouter API Key:** [openrouter.ai](https://openrouter.ai/keys)

### **5. Загрузка FAQ в базу данных**

```bash
# Инициализация БД и загрузка FAQ
python scripts/load_faq.py database/faq_61.json
```

### **6. Запуск бота**

```bash
python main.py
```

**Успешный запуск:**
```
✅ Loaded .env from /path/to/.env
🤖 Запуск чат-бота для абитуриентов МУИВ
🔧 Инициализация базы данных...
   📁 Используется SQLite: data/bot.db
✅ База данных инициализирована
🔄 Перезагрузка FAQ из database/faq_61.json...
✅ FAQ перезагружен: 63 вопросов
🤖 Загружаю RuBERT модель из ml/models/rubert_final/final_model...
✅ RuBERT классификатор успешно загружен!
   📊 Порог уверенности: 0.7
✅ Все системы готовы к работе!
🚀 Запуск polling режима...
```

---

## 🐳 Docker

### **Вариант 1: SQLite (простой, рекомендуется для начала)**

```bash
docker-compose --profile sqlite up -d
```

### **Вариант 2: PostgreSQL (для продакшена)**

```bash
# Добавить в .env
POSTGRES_PASSWORD=secure_password_here

# Запустить
docker-compose --profile postgres up -d
```

**Полезные команды:**
```bash
# Просмотр логов
docker-compose logs -f bot-sqlite

# Перезапуск с новым кодом
docker-compose --profile sqlite down
docker-compose build
docker-compose --profile sqlite up -d

# Подключение к PostgreSQL
docker exec -it muiv-postgres psql -U muiv_user -d muiv_bot
```

---

## 📂 Структура проекта

```
muiv-chatbot/
├── bot/                        # Telegram bot
│   ├── handlers/               # Обработчики команд
│   │   ├── admin.py            # Админ-панель
│   │   ├── moderator.py        # Модератор-панель
│   │   ├── user.py             # Обработка вопросов пользователей
│   │   ├── common.py           # /start, /help, /contacts
│   │   ├── auth_handlers.py    # /admin, /moderator, /logout
│   │   └── rating_handlers.py  # Обработка рейтингов
│   ├── dispatcher.py           # Инициализация бота
│   ├── keyboards.py            # Reply и Inline клавиатуры
│   └── rating_keyboards.py     # Клавиатуры для рейтингов
│
├── ml/                         # Machine Learning
│   ├── models/                 # Обученные модели
│   │   └── final_model/        # RuBERT классификатор
│   │       ├── config.json
│   │       ├── model.safetensors
│   │       ├── tokenizer_config.json
│   │       └── label_mapping.json
│   ├── intent_classifier.py    # Классификатор намерений
│   ├── chat_manager.py         # Менеджер диалогов
│   └── train_rubert_classifier.ipynb  # Notebook для обучения
│
├── database/                   # База данных
│   ├── init_db.py              # Инициализация схемы БД
│   ├── crud.py                 # CRUD операции
│   ├── faq_61.json             # База знаний FAQ
│   └── muiv_bot.db             # SQLite БД (создаётся автоматически)
│
├── utils/                      # Утилиты
│   ├── auth_system.py          # Система авторизации
│   ├── auth_states.py          # FSM состояния
│   ├── logger.py               # Логирование
│   ├── helpers.py              # Вспомогательные функции
│   └── text_processing.py      # Обработка текста
│
├── scripts/                    # Скрипты
│   └── load_faq.py             # Загрузка FAQ в БД
│
├── logs/                       # Логи (создаётся автоматически)
├── data/                       # Данные (создаётся автоматически)
│
├── main.py                     # Главный файл запуска
├── config.py                   # Конфигурация
├── requirements.txt            # Зависимости Python
├── .env                        # Переменные окружения (создать вручную)
├── .gitignore                  # Git ignore
├── Dockerfile                  # Docker образ
└── docker-compose.yml          # Docker Compose
```

---

## 📖 Использование

### **Команды для пользователей:**

- `/start` - Начать работу с ботом
- `/help` - Справка по использованию
- `/contacts` - Контакты приёмной комиссии
- `/clear` - Очистить историю диалога
- `/stats` - Моя статистика

### **Команды для модераторов:**

Вход: `/moderator` (требуется пароль)

- `/mod_stats` - Статистика за 7 дней
- `/mod_popular` - Топ-10 популярных вопросов
- `/mod_low_rated` - Вопросы с низкими оценками
- `/mod_export` - Экспорт данных в CSV
- `/logout` - Выход из панели модератора

### **Команды для администраторов:**

Вход: `/admin` (требуется пароль)

- `/stats_full` - Полная статистика
- `/analytics` - Детальная аналитика
- `/ratings` - Статистика рейтингов
- `/export_ratings` - Экспорт рейтингов в CSV
- `/reload_kb` - Перезагрузка базы знаний
- `/logout` - Выход из админ-панели

---

## 🎯 Метрики и результаты

### **RuBERT Классификатор:**
- **Точность:** 97.49%
- **F1-score:** 0.974
- **Precision:** 0.976
- **Recall:** 0.972

### **Производительность:**
- **Inference RuBERT:** 50-100 мс (CPU)
- **FAQ поиск:** <10 мс
- **DeepSeek API:** 2-4 сек
- **Средний ответ:** <500 мс (для 87% запросов через RuBERT+FAQ)

### **База знаний:**
- **FAQ вопросов:** 63
- **Категорий:** 8
- **Обучающих примеров:** 4,776
- **Покрытие запросов:** ~87% через FAQ

---

## 🔧 Разработка

### **Обучение новой модели RuBERT:**

1. Подготовить датасет в формате:
   ```python
   {
       "question": "Вопрос пользователя",
       "category": "категория"
   }
   ```

2. Запустить Jupyter Notebook:
   ```bash
   jupyter notebook ml/train_rubert_classifier.ipynb
   ```

3. Обучить модель (настройки в notebook)

4. Модель сохранится в `ml/models/final_model/`

### **Добавление новых FAQ:**

1. Отредактировать `database/faq_61.json`:
   ```json
   {
       "question": "Новый вопрос?",
       "answer": "Ответ на вопрос",
       "category": "категория",
       "keywords": ["ключ1", "ключ2"],
       "priority": 5,
       "is_active": true
   }
   ```

2. Перезагрузить FAQ:
   ```bash
   python scripts/load_faq.py database/faq_61.json
   ```

   Или через админ-панель бота: `/reload_kb`

### **Тестирование:**

```bash
# Запуск тестов (когда добавим)
pytest tests/

# Проверка типов
mypy bot/ ml/ database/

# Линтер
flake8 bot/ ml/ database/
```

---

## 📊 Аналитика

Бот собирает подробную аналитику:

- **Количество обращений** по дням/неделям
- **Популярные категории** вопросов
- **Рейтинги ответов** (средний балл, распределение)
- **Конверсия** (% найденных ответов)
- **Время обработки** запросов

Экспорт статистики:
```bash
# Через админ-панель
/analytics
/export_ratings

# Или прямо из БД
sqlite3 data/muiv_bot.db "SELECT * FROM analytics;"
```


## 📝 Лицензия

MIT License - см. [LICENSE](LICENSE)

---


## 🙏 Благодарности

- [Hugging Face](https://huggingface.co/) за модель RuBERT
- [Aiogram](https://docs.aiogram.dev/) за отличный фреймворк для Telegram ботов
- [DeepSeek](https://www.deepseek.com/) за мощную языковую модель
- [OpenRouter](https://openrouter.ai/) за удобный API-шлюз

---


<div align="center">

**Сделано с ❤️ для абитуриентов МУИВ**

</div>
