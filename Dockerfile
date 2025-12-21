# Dockerfile для чат-бота МУИВ
# Поддерживает как SQLite так и PostgreSQL

FROM python:3.10-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Рабочая директория
WORKDIR /app

# Копирование requirements.txt
COPY requirements.txt .

# Установка Python зависимостей
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копирование кода приложения
COPY . .

# Создание директорий для логов и данных
RUN mkdir -p /app/logs /app/data

# Переменные окружения по умолчанию
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

# Проверка здоровья контейнера
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Запуск бота
CMD ["python", "main.py"]
