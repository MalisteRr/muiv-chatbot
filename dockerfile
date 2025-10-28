# Используем официальный образ Python
FROM python:3.10

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем все файлы проекта
COPY . .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Запускаем и бота, и API одновременно
CMD bash -c "python backend/main_db.py & uvicorn backend.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"
