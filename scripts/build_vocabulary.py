"""
Построение словаря (токенизатора) на основе подготовленных данных

Создаёт токенизатор, который будет использоваться для преобразования текста в последовательности чисел.
"""

import sys
import os
import json
from pathlib import Path

# Добавляем корневую директорию проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml.models.tokenizer import SimpleTokenizer
from ml.models.config import ModelConfig


def build_vocabulary():
    """
    Построение словаря на основе обучающих данных
    """
    print("\n" + "=" * 70)
    print("ПОСТРОЕНИЕ СЛОВАРЯ (ТОКЕНИЗАТОРА)")
    print("=" * 70)
    
    # 1. Проверка наличия данных
    print(f"\n📂 Проверка данных...")
    
    if not os.path.exists(ModelConfig.DATA_PATH):
        print(f"❌ Датасет не найден: {ModelConfig.DATA_PATH}")
        print(f"\n💡 Сначала запустите:")
        print(f"   python scripts/prepare_dataset.py")
        return
    
    print(f"✅ Датасет найден: {ModelConfig.DATA_PATH}")
    
    # 2. Загрузка данных
    print(f"\n📥 Загрузка данных...")
    
    with open(ModelConfig.DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ Загружено {len(data)} примеров")
    
    # 3. Извлечение текстов
    print(f"\n📝 Извлечение текстов...")
    
    texts = []
    for item in data:
        texts.append(item['question'])
        texts.append(item['answer'])
    
    print(f"✅ Извлечено {len(texts)} текстов")
    
    # 4. Построение словаря
    print(f"\n🔨 Построение словаря...")
    print(f"   Размер словаря: {ModelConfig.VOCAB_SIZE}")
    
    tokenizer = SimpleTokenizer(vocab_size=ModelConfig.VOCAB_SIZE)
    tokenizer.build_vocab(texts)
    
    actual_vocab_size = tokenizer.get_vocab_size()
    print(f"✅ Словарь построен: {actual_vocab_size} токенов")
    
    # 5. Сохранение токенизатора
    print(f"\n💾 Сохранение токенизатора...")
    
    # Создаём директорию если не существует
    os.makedirs(os.path.dirname(ModelConfig.TOKENIZER_PATH), exist_ok=True)
    
    tokenizer.save(ModelConfig.TOKENIZER_PATH)
    
    print(f"✅ Токенизатор сохранён: {ModelConfig.TOKENIZER_PATH}")
    
    # 6. Статистика
    print("\n" + "=" * 70)
    print("СТАТИСТИКА СЛОВАРЯ")
    print("=" * 70)
    print(f"📊 Размер словаря: {actual_vocab_size}")
    print(f"📊 Специальные токены:")
    print(f"   <PAD>: {tokenizer.word2idx.get('<PAD>', 'не найден')}")
    print(f"   <UNK>: {tokenizer.word2idx.get('<UNK>', 'не найден')}")
    print(f"   <SOS>: {tokenizer.word2idx.get('<SOS>', 'не найден')}")
    print(f"   <EOS>: {tokenizer.word2idx.get('<EOS>', 'не найден')}")
    
    # Топ слов
    print(f"\n📈 Топ-20 самых частых слов:")
    
    # Получаем топ слова (кроме специальных токенов)
    word_freq = {}
    for text in texts:
        words = tokenizer.tokenize(text)
        for word in words:
            if word not in ['<PAD>', '<UNK>', '<SOS>', '<EOS>']:
                word_freq[word] = word_freq.get(word, 0) + 1
    
    top_words = sorted(word_freq.items(), key=lambda x: -x[1])[:20]
    
    for i, (word, freq) in enumerate(top_words, 1):
        print(f"   {i:2d}. {word:20s} ({freq:4d} раз)")
    
    # Примеры токенизации
    print(f"\n📝 Примеры токенизации:")
    
    test_sentences = [
        "Сколько стоит обучение?",
        "Какие документы нужны для поступления?",
        "Есть ли бюджетные места?"
    ]
    
    for sent in test_sentences:
        tokens = tokenizer.encode(sent, add_sos=True, add_eos=True)
        decoded = tokenizer.decode(tokens, skip_special=True)
        
        print(f"\n   Исходный: {sent}")
        print(f"   Токены: {tokens[:15]}..." if len(tokens) > 15 else f"   Токены: {tokens}")
        print(f"   Длина: {len(tokens)}")
        print(f"   Декодированный: {decoded}")
    
    print("\n" + "=" * 70)
    print("✅ ПОСТРОЕНИЕ СЛОВАРЯ ЗАВЕРШЕНО")
    print("=" * 70)
    print(f"\n📌 Следующий шаг:")
    print(f"   python scripts/train_model.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    build_vocabulary()
