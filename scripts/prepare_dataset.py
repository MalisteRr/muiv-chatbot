"""
Подготовка датасета для обучения модели

Загружает FAQ данные и применяет аугментацию для увеличения размера обучающей выборки.
"""

import sys
import os
import json
import random
from pathlib import Path

# Добавляем корневую директорию проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml.models.config import ModelConfig


def augment_question(question: str, templates: list) -> list:
    """
    Создаёт вариации вопроса используя шаблоны
    
    Args:
        question: Исходный вопрос
        templates: Список шаблонов для генерации вариаций
    
    Returns:
        Список вариаций вопроса
    """
    variations = [question]  # Исходный вопрос всегда включаем
    
    # Простые вариации (перефразирование)
    simple_variations = [
        f"{question}",
        f"Подскажите, {question.lower()}",
        f"Скажите пожалуйста, {question.lower()}",
        f"Хотел бы узнать, {question.lower()}",
        f"Можете рассказать, {question.lower()}"
    ]
    
    variations.extend(simple_variations[:3])  # Берём 3 вариации
    
    return variations


def augment_answer(answer: str) -> str:
    """
    Незначительно модифицирует ответ (сохраняя смысл)
    
    Args:
        answer: Исходный ответ
    
    Returns:
        Модифицированный ответ
    """
    # Пока возвращаем как есть (можно добавить синонимизацию позже)
    return answer


def prepare_dataset(
    input_path: str = None,
    output_path: str = None,
    augmentation_factor: int = 3
):
    """
    Подготовка и аугментация датасета
    
    Args:
        input_path: Путь к исходному FAQ файлу
        output_path: Путь для сохранения расширенного датасета
        augmentation_factor: Во сколько раз увеличить датасет
    """
    # Пути по умолчанию
    if input_path is None:
        input_path = os.path.join(project_root, 'data', 'faq_30.json')
    
    if output_path is None:
        output_path = ModelConfig.DATA_PATH
    
    print("\n" + "=" * 70)
    print("ПОДГОТОВКА ДАТАСЕТА ДЛЯ ОБУЧЕНИЯ")
    print("=" * 70)
    
    # 1. Загрузка исходных данных
    print(f"\n📂 Загрузка FAQ из: {input_path}")
    
    if not os.path.exists(input_path):
        print(f"❌ Файл не найден: {input_path}")
        print("\n💡 Создайте файл data/faq_data.json с вопросами и ответами")
        return
    
    with open(input_path, 'r', encoding='utf-8') as f:
        faq_data = json.load(f)
    
    print(f"✅ Загружено {len(faq_data)} исходных пар вопрос-ответ")
    
    # 2. Аугментация данных
    print(f"\n🔄 Аугментация данных (×{augmentation_factor})...")
    
    augmented_data = []
    templates = ["Подскажите", "Скажите", "Хотел бы узнать"]
    
    for item in faq_data:
        question = item['question']
        answer = item['answer']
        
        # Генерируем вариации вопроса
        question_variations = augment_question(question, templates)
        
        # Для каждой вариации создаём пару
        for q_var in question_variations[:augmentation_factor]:
            augmented_data.append({
                'question': q_var,
                'answer': answer,
                'category': item.get('category', 'Общее'),
                'original_question': question
            })
    
    print(f"✅ Создано {len(augmented_data)} обучающих примеров")
    
    # 3. Перемешивание данных
    print(f"\n🔀 Перемешивание данных...")
    random.shuffle(augmented_data)
    
    # 4. Сохранение
    print(f"\n💾 Сохранение расширенного датасета...")
    
    # Создаём директорию если не существует
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(augmented_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Датасет сохранён: {output_path}")
    
    # 5. Статистика
    print("\n" + "=" * 70)
    print("СТАТИСТИКА ДАТАСЕТА")
    print("=" * 70)
    print(f"📊 Исходных вопросов: {len(faq_data)}")
    print(f"📊 После аугментации: {len(augmented_data)}")
    print(f"📊 Увеличение: ×{len(augmented_data) / len(faq_data):.1f}")
    
    # Статистика по категориям
    categories = {}
    for item in augmented_data:
        cat = item.get('category', 'Общее')
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"\n📂 Распределение по категориям:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"   {cat}: {count}")
    
    # Примеры
    print(f"\n📝 Примеры обучающих данных:")
    for i, item in enumerate(augmented_data[:3], 1):
        print(f"\n{i}. Вопрос: {item['question'][:80]}...")
        print(f"   Ответ: {item['answer'][:80]}...")
        print(f"   Категория: {item['category']}")
    
    print("\n" + "=" * 70)
    print("✅ ПОДГОТОВКА ЗАВЕРШЕНА")
    print("=" * 70)
    print(f"\n📌 Следующий шаг:")
    print(f"   python scripts/build_vocabulary.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    prepare_dataset()
