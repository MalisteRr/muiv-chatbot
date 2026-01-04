"""
Что делает скрипт:
1. Загружает FAQ из файла faq_70_questions.json
2. Расширяет датасет через аугментацию (создание вариаций вопросов)
3. Сохраняет подготовленный датасет для обучения модели
"""

import json
import os
import sys
import random
from typing import List, Dict

# Добавляем корневую директорию в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ml.models.config import ModelConfig


def load_faq() -> List[Dict]:
    """
    Загрузка FAQ из файла
    
    Returns:
        Список словарей с вопросами и ответами
    """
    try:
        with open(ModelConfig.FAQ_SOURCE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Загружено {len(data)} вопросов из FAQ")
        return data
    except FileNotFoundError:
        print(f"❌ Файл не найден: {ModelConfig.FAQ_SOURCE_PATH}")
        return []


def augment_question(question: str) -> List[str]:
    """
    Создание вариаций вопроса для увеличения датасета
    
    Args:
        question: Исходный вопрос
        
    Returns:
        Список вариаций вопроса
    """
    variations = []
    q_lower = question.lower()
    
    # Вариации с разными формулировками "сколько стоит"
    if "сколько стоит" in q_lower:
        variations.append(question.replace("Сколько стоит", "Какая цена"))
        variations.append(question.replace("Сколько стоит", "Какая стоимость"))
        variations.append(question.replace("стоит", "будет стоить"))
    
    if "какая цена" in q_lower:
        variations.append(question.replace("Какая цена", "Сколько стоит"))
    
    # Вариации с "есть ли" / "имеется ли"
    if "есть ли" in q_lower:
        variations.append(question.replace("Есть ли", "Имеется ли"))
        variations.append(question.replace("Есть ли", "Доступно ли"))
    
    # Добавление вежливых форм
    if not q_lower.startswith(("подскажите", "скажите", "расскажите")):
        variations.append(f"Подскажите, {question.lower()}")
        variations.append(f"Скажите, {question.lower()}")
        variations.append(f"Не могли бы вы сказать, {question.lower()}")
    
    # Добавление "пожалуйста"
    if "пожалуйста" not in q_lower:
        variations.append(f"{question.rstrip('?')}, пожалуйста?")
    
    # Вариации начала вопроса
    if q_lower.startswith("какие"):
        variations.append(question.replace("Какие", "Что за"))
    
    if q_lower.startswith("как"):
        variations.append(question.replace("Как", "Каким образом"))
    
    # Краткие формы
    if len(question.split()) > 5:
        # Убираем вводные слова
        short = question
        for word in ["пожалуйста", "скажите", "подскажите"]:
            short = short.replace(word + ", ", "").replace(word + " ", "")
        if short != question:
            variations.append(short)
    
    return variations


def clean_duplicates(pairs: List[Dict]) -> List[Dict]:
    """
    Удаление дубликатов из датасета
    
    Args:
        pairs: Список пар вопрос-ответ
        
    Returns:
        Очищенный список без дубликатов
    """
    seen_questions = set()
    unique_pairs = []
    
    for pair in pairs:
        q_normalized = pair['question'].lower().strip()
        if q_normalized not in seen_questions:
            seen_questions.add(q_normalized)
            unique_pairs.append(pair)
    
    print(f"🧹 Удалено дубликатов: {len(pairs) - len(unique_pairs)}")
    return unique_pairs


def prepare_training_data(faq_data: List[Dict]) -> List[Dict]:
    """
    Подготовка обучающих данных с аугментацией
    
    Args:
        faq_data: Исходные данные FAQ
        
    Returns:
        Расширенный датасет для обучения
    """
    training_pairs = []
    
    print("\n🔄 Подготовка обучающих данных...")
    
    for idx, item in enumerate(faq_data, 1):
        question = item['question']
        answer = item['answer']
        category = item.get('category', 'Общее')
        
        # Добавляем оригинальную пару
        training_pairs.append({
            'question': question,
            'answer': answer,
            'category': category,
            'is_original': True
        })
        
        # Добавляем вариации
        variations = augment_question(question)
        for variation in variations:
            training_pairs.append({
                'question': variation,
                'answer': answer,
                'category': category,
                'is_original': False
            })
        
        if idx % 10 == 0:
            print(f"   Обработано {idx}/{len(faq_data)} вопросов...")
    
    # Очистка дубликатов
    training_pairs = clean_duplicates(training_pairs)
    
    # Перемешивание для лучшего обучения
    random.shuffle(training_pairs)
    
    return training_pairs


def save_dataset(data: List[Dict], filepath: str):
    """
    Сохранение датасета в JSON файл
    
    Args:
        data: Данные для сохранения
        filepath: Путь к файлу
    """
    # Создаём директорию если не существует
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Датасет сохранён: {filepath}")


def print_statistics(data: List[Dict]):
    """
    Вывод статистики по датасету
    
    Args:
        data: Датасет
    """
    print("\n" + "=" * 60)
    print("СТАТИСТИКА ДАТАСЕТА")
    print("=" * 60)
    
    total = len(data)
    originals = sum(1 for item in data if item.get('is_original', False))
    augmented = total - originals
    
    print(f"📊 Всего пар вопрос-ответ: {total}")
    print(f"   • Оригинальных: {originals}")
    print(f"   • Аугментированных: {augmented}")
    print(f"   • Коэффициент расширения: {total/originals:.1f}x")
    
    # Статистика по категориям
    categories = {}
    for item in data:
        cat = item.get('category', 'Без категории')
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"\n📚 Распределение по категориям:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        percentage = (count / total) * 100
        print(f"   • {cat}: {count} ({percentage:.1f}%)")
    
    # Средняя длина вопросов и ответов
    avg_q_len = sum(len(item['question'].split()) for item in data) / total
    avg_a_len = sum(len(item['answer'].split()) for item in data) / total
    
    print(f"\n📏 Средняя длина:")
    print(f"   • Вопрос: {avg_q_len:.1f} слов")
    print(f"   • Ответ: {avg_a_len:.1f} слов")
    
    print("=" * 60)


def main():
    """
    Основная функция
    """
    print("\n" + "=" * 60)
    print("ПОДГОТОВКА ДАТАСЕТА ДЛЯ ОБУЧЕНИЯ МОДЕЛИ")
    print("=" * 60)
    
    # Загружаем FAQ
    faq = load_faq()
    if not faq:
        print("❌ Не удалось загрузить FAQ. Проверьте путь к файлу.")
        return
    
    # Подготавливаем данные
    training_data = prepare_training_data(faq)
    
    # Статистика
    print_statistics(training_data)
    
    # Сохраняем
    save_dataset(training_data, ModelConfig.DATA_PATH)
    
    print(f"\n✅ Датасет успешно подготовлен!")
    print(f"📁 Файл: {ModelConfig.DATA_PATH}")
    print(f"📊 Размер: {len(training_data)} пар вопрос-ответ")


if __name__ == "__main__":
    main()
