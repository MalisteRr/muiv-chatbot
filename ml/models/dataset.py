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
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple

from .tokenizer import SimpleTokenizer
from .config import ModelConfig


class QADataset(Dataset):
    """Dataset для пар вопрос-ответ"""
    
    def __init__(self, data_path: str, tokenizer: SimpleTokenizer, max_length: int = 100):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"✅ Загружено {len(self.data)} пар вопрос-ответ")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        answer = item['answer']
        
        question_indices = self.tokenizer.encode(
            question, max_length=self.max_length, add_sos=False, add_eos=True
        )
        answer_indices = self.tokenizer.encode(
            answer, max_length=self.max_length, add_sos=True, add_eos=True
        )
        
        question_length = sum(1 for idx in question_indices if idx != 0)
        answer_length = sum(1 for idx in answer_indices if idx != 0)
        
        return (
            torch.LongTensor(question_indices),
            torch.LongTensor(answer_indices),
            question_length,
            answer_length
        )


def collate_fn(batch):
    """Функция для объединения примеров в батч"""
    questions, answers, q_lengths, a_lengths = zip(*batch)
    
    questions = torch.stack(questions)
    answers = torch.stack(answers)
    question_lengths = torch.LongTensor(q_lengths)
    answer_lengths = torch.LongTensor(a_lengths)
    
    sorted_indices = question_lengths.argsort(descending=True)
    
    questions = questions[sorted_indices]
    answers = answers[sorted_indices]
    question_lengths = question_lengths[sorted_indices]
    answer_lengths = answer_lengths[sorted_indices]
    
    return questions, answers, question_lengths, answer_lengths


def create_dataloaders(
    train_path: str,
    val_path: str = None,
    tokenizer: SimpleTokenizer = None,
    batch_size: int = 32,
    max_length: int = 100,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """Создание DataLoader'ов для обучения и валидации"""
    if tokenizer is None:
        tokenizer = SimpleTokenizer.load(ModelConfig.TOKENIZER_PATH)
    
    train_dataset = QADataset(train_path, tokenizer, max_length)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True
    )
    
    val_loader = None
    if val_path:
        val_dataset = QADataset(val_path, tokenizer, max_length)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=num_workers, pin_memory=True
        )
    
    print(f"\n📊 DataLoaders созданы:")
    print(f"   Train батчей: {len(train_loader)}")
    print(f"   Val батчей: {len(val_loader) if val_loader else 0}")
    print(f"   Batch size: {batch_size}")
    
    return train_loader, val_loader


def split_dataset(
    data_path: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    save_splits: bool = True
) -> Tuple[str, str, str]:
    """
    Разделение датасета на train/val/test
    
    Args:
        data_path: Путь к полному датасету
        train_ratio: Доля обучающих данных
        val_ratio: Доля валидационных данных
        test_ratio: Доля тестовых данных
        save_splits: Сохранить ли разделённые данные
    
    Returns:
        train_path: Путь к обучающим данным
        val_path: Путь к валидационным данным
        test_path: Путь к тестовым данным
    """
    import os
    import random
    
    # Проверка соотношений
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Сумма соотношений должна быть 1.0"
    
    # Загружаем данные
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Перемешиваем
    random.shuffle(data)
    
    # Вычисляем размеры
    total = len(data)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    # Разделяем
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    print(f"\n✂️ Датасет разделён:")
    print(f"   Train: {len(train_data)} ({len(train_data)/total*100:.1f}%)")
    print(f"   Val: {len(val_data)} ({len(val_data)/total*100:.1f}%)")
    print(f"   Test: {len(test_data)} ({len(test_data)/total*100:.1f}%)")
    
    if save_splits:
        # Пути для сохранения
        base_dir = os.path.dirname(data_path)
        train_path = os.path.join(base_dir, 'train_data.json')
        val_path = os.path.join(base_dir, 'val_data.json')
        test_path = os.path.join(base_dir, 'test_data.json')
        
        # Сохраняем
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        
        with open(val_path, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)
        
        with open(test_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Файлы сохранены:")
        print(f"   {train_path}")
        print(f"   {val_path}")
        print(f"   {test_path}")
        
        return train_path, val_path, test_path
    
    return None, None, None


if __name__ == "__main__":
    """
    Тестирование split_dataset
    """
    print("\n" + "=" * 60)
    print("ТЕСТ SPLIT_DATASET")
    print("=" * 60)
    
    import tempfile
    
    # Создаём тестовый датасет
    test_data = [
        {"question": f"Вопрос {i}", "answer": f"Ответ {i}", "category": "Тест"}
        for i in range(100)
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False)
        temp_path = f.name
    
    # Разделяем датасет
    train_path, val_path, test_path = split_dataset(
        temp_path,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        save_splits=True
    )
    
    print(f"\n✅ Датасет успешно разделён!")
    
    # Удаляем временные файлы
    import os
    os.remove(temp_path)
    os.remove(train_path)
    os.remove(val_path)
    os.remove(test_path)
    
    print("\n" + "=" * 60)
    print("✅ DATASET ПОЛНОСТЬЮ ГОТОВ")
    print("=" * 60)
