"""
DataLoader для подготовки батчей данных

Загрузка и подготовка данных для обучения Seq2Seq модели:
1. Загрузка датасета (пары вопрос-ответ)
2. Токенизация текстов
3. Создание батчей
4. Паддинг последовательностей
"""

import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple

from .tokenizer import SimpleTokenizer
from .config import ModelConfig


class QADataset(Dataset):
    """
    Dataset для пар вопрос-ответ
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: SimpleTokenizer,
        max_length: int = 100
    ):
        """
        Инициализация датасета
        
        Args:
            data_path: Путь к JSON файлу с данными
            tokenizer: Токенизатор
            max_length: Максимальная длина последовательности
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Загружаем данные
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"✅ Загружено {len(self.data)} пар вопрос-ответ")
    
    def __len__(self):
        """Размер датасета"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Получение одного примера
        
        Args:
            idx: Индекс примера
        
        Returns:
            question_indices: Закодированный вопрос
            answer_indices: Закодированный ответ
            question_length: Реальная длина вопроса
            answer_length: Реальная длина ответа
        """
        item = self.data[idx]
        question = item['question']
        answer = item['answer']
        
        # Кодируем вопрос и ответ
        question_indices = self.tokenizer.encode(
            question,
            max_length=self.max_length,
            add_sos=False,  # SOS не нужен для encoder
            add_eos=True    # EOS нужен
        )
        
        answer_indices = self.tokenizer.encode(
            answer,
            max_length=self.max_length,
            add_sos=True,   # SOS нужен для decoder
            add_eos=True    # EOS тоже нужен
        )
        
        # Вычисляем реальные длины (до паддинга)
        question_length = sum(1 for idx in question_indices if idx != 0)
        answer_length = sum(1 for idx in answer_indices if idx != 0)
        
        return (
            torch.LongTensor(question_indices),
            torch.LongTensor(answer_indices),
            question_length,
            answer_length
        )


if __name__ == "__main__":
    """
    Тестирование QADataset
    """
    print("\n" + "=" * 60)
    print("ТЕСТ QADATASET")
    print("=" * 60)
    
    # Создаём простой тестовый датасет
    import tempfile
    
    test_data = [
        {
            "question": "Сколько стоит обучение?",
            "answer": "Стоимость обучения составляет 150000 рублей в год.",
            "category": "Стоимость"
        },
        {
            "question": "Какие документы нужны?",
            "answer": "Необходимы паспорт, аттестат и фотографии.",
            "category": "Документы"
        },
        {
            "question": "Есть ли бюджетные места?",
            "answer": "Да, доступно 25 бюджетных мест.",
            "category": "Бюджет"
        }
    ] * 10
    
    # Сохраняем во временный файл
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False)
        temp_path = f.name
    
    # Создаём токенизатор
    tokenizer = SimpleTokenizer(vocab_size=1000)
    all_texts = []
    for item in test_data:
        all_texts.append(item['question'])
        all_texts.append(item['answer'])
    tokenizer.build_vocab(all_texts)
    
    # Создаём датасет
    dataset = QADataset(
        data_path=temp_path,
        tokenizer=tokenizer,
        max_length=50
    )
    
    print(f"\n📊 Dataset создан:")
    print(f"   Размер: {len(dataset)}")
    
    # Тестируем получение одного примера
    q, a, q_len, a_len = dataset[0]
    
    print(f"\n🧪 Первый пример:")
    print(f"   Question форма: {q.shape}")
    print(f"   Answer форма: {a.shape}")
    print(f"   Q length: {q_len}")
    print(f"   A length: {a_len}")
    
    # Декодируем
    decoded_q = tokenizer.decode(q.tolist())
    decoded_a = tokenizer.decode(a.tolist())
    print(f"\n📝 Декодированный пример:")
    print(f"   Вопрос: {decoded_q}")
    print(f"   Ответ: {decoded_a}")
    
    # Удаляем временный файл
    import os
    os.remove(temp_path)
    
    print("\n" + "=" * 60)
    print("✅ QADATASET РАБОТАЕТ")
    print("=" * 60)
