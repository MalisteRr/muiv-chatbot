"""
Что делает скрипт:
1. Загружает подготовленный датасет
2. Извлекает все вопросы и ответы
3. Строит словарь на основе частотности слов
4. Сохраняет токенизатор для дальнейшего использования
"""

import json
import os
import sys

# Добавляем корневую директорию в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ml.models.tokenizer import SimpleTokenizer
from ml.models.config import ModelConfig


def load_dataset():
    """
    Загрузка подготовленного датасета
    
    Returns:
        Список пар вопрос-ответ
    """
    try:
        with open(ModelConfig.DATA_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Загружен датасет: {len(data)} пар")
        return data
    except FileNotFoundError:
        print(f"❌ Файл датасета не найден: {ModelConfig.DATA_PATH}")
        print("   Сначала запустите: python scripts/prepare_dataset.py")
        return []


def extract_texts(dataset):
    """
    Извлечение всех текстов (вопросы + ответы) из датасета
    
    Args:
        dataset: Список пар вопрос-ответ
        
    Returns:
        Список всех текстов
    """
    all_texts = []
    
    for item in dataset:
        all_texts.append(item['question'])
        all_texts.append(item['answer'])
    
    print(f"📊 Извлечено {len(all_texts)} текстов:")
    print(f"   • Вопросов: {len(dataset)}")
    print(f"   • Ответов: {len(dataset)}")
    
    return all_texts


def print_vocabulary_stats(tokenizer):
    """
    Вывод статистики по построенному словарю
    
    Args:
        tokenizer: Токенизатор со словарём
    """
    print("\n" + "=" * 60)
    print("СТАТИСТИКА СЛОВАРЯ")
    print("=" * 60)
    
    print(f"📚 Размер словаря: {tokenizer.get_vocab_size()} слов")
    print(f"   • Специальные токены: 4")
    print(f"   • Обычные слова: {tokenizer.get_vocab_size() - 4}")
    
    # Топ-20 самых частотных слов
    print(f"\n🔝 Топ-20 самых частотных слов:")
    top_words = tokenizer.word_count.most_common(20)
    for idx, (word, count) in enumerate(top_words, 1):
        print(f"   {idx:2d}. {word:15s} - {count:4d} раз")
    
    # Примеры кодирования
    print(f"\n🧪 Примеры кодирования:")
    test_texts = [
        "Сколько стоит обучение?",
        "Есть ли бюджетные места?",
        "Какие документы нужны?"
    ]
    
    for text in test_texts:
        encoded = tokenizer.encode(text, max_length=20)
        decoded = tokenizer.decode(encoded)
        print(f"\n   Исходный: {text}")
        print(f"   Индексы: {encoded[:10]}... (первые 10)")
        print(f"   Декодированный: {decoded}")
    
    print("=" * 60)


def main():
    """
    Основная функция
    """
    print("\n" + "=" * 60)
    print("ПОСТРОЕНИЕ СЛОВАРЯ ДЛЯ МОДЕЛИ")
    print("=" * 60)
    
    # Загружаем датасет
    dataset = load_dataset()
    if not dataset:
        print("\n❌ Не удалось загрузить датасет. Завершение.")
        return
    
    # Извлекаем тексты
    all_texts = extract_texts(dataset)
    
    # Создаём токенизатор
    print(f"\n🔨 Создание токенизатора (vocab_size={ModelConfig.VOCAB_SIZE})...")
    tokenizer = SimpleTokenizer(vocab_size=ModelConfig.VOCAB_SIZE)
    
    # Строим словарь
    tokenizer.build_vocab(all_texts)
    
    # Создаём директорию для сохранения если не существует
    os.makedirs(os.path.dirname(ModelConfig.TOKENIZER_PATH), exist_ok=True)
    
    # Сохраняем токенизатор
    tokenizer.save(ModelConfig.TOKENIZER_PATH)
    
    # Статистика
    print_vocabulary_stats(tokenizer)
    
    print(f"\n✅ Словарь успешно построен и сохранён!")
    print(f"📁 Файл: {ModelConfig.TOKENIZER_PATH}")
    print(f"📚 Размер словаря: {tokenizer.get_vocab_size()} слов")
    print(f"\n💡 Теперь можно обучать модель!")


if __name__ == "__main__":
    main()
