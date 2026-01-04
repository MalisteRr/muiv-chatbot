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
    """
    Dataset для пар вопрос-ответ
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: SimpleTokenizer,
        max_length: int = 100
    ):
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
            question,
            max_length=self.max_length,
            add_sos=False,
            add_eos=True
        )
        
        answer_indices = self.tokenizer.encode(
            answer,
            max_length=self.max_length,
            add_sos=True,
            add_eos=True
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
    """
    Функция для объединения примеров в батч
    
    Args:
        batch: Список примеров (вопрос, ответ, длины)
    
    Returns:
        questions: Батч вопросов (batch_size, max_seq_len)
        answers: Батч ответов (batch_size, max_seq_len)
        question_lengths: Длины вопросов (batch_size,)
        answer_lengths: Длины ответов (batch_size,)
    """
    # Распаковываем батч
    questions, answers, q_lengths, a_lengths = zip(*batch)
    
    # Стекаем в тензоры
    questions = torch.stack(questions)
    answers = torch.stack(answers)
    question_lengths = torch.LongTensor(q_lengths)
    answer_lengths = torch.LongTensor(a_lengths)
    
    # Сортируем по убыванию длины вопросов (требование pack_padded_sequence)
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
    """
    Создание DataLoader'ов для обучения и валидации
    
    Args:
        train_path: Путь к обучающим данным
        val_path: Путь к валидационным данным (опционально)
        tokenizer: Токенизатор
        batch_size: Размер батча
        max_length: Максимальная длина последовательности
        num_workers: Количество процессов для загрузки данных
    
    Returns:
        train_loader: DataLoader для обучения
        val_loader: DataLoader для валидации (или None)
    """
    # Загружаем токенизатор если не передан
    if tokenizer is None:
        tokenizer = SimpleTokenizer.load(ModelConfig.TOKENIZER_PATH)
    
    # Создаём обучающий датасет
    train_dataset = QADataset(
        data_path=train_path,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True  # Ускоряет передачу на GPU
    )
    
    # Создаём валидационный датасет если есть
    val_loader = None
    if val_path:
        val_dataset = QADataset(
            data_path=val_path,
            tokenizer=tokenizer,
            max_length=max_length
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True
        )
    
    print(f"\n📊 DataLoaders созданы:")
    print(f"   Train батчей: {len(train_loader)}")
    print(f"   Val батчей: {len(val_loader) if val_loader else 0}")
    print(f"   Batch size: {batch_size}")
    
    return train_loader, val_loader


if __name__ == "__main__":
    """
    Тестирование DataLoader
    """
    print("\n" + "=" * 60)
    print("ТЕСТ DATALOADER")
    print("=" * 60)
    
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
        }
    ] * 10
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False)
        temp_path = f.name
    
    # Токенизатор
    tokenizer = SimpleTokenizer(vocab_size=1000)
    all_texts = []
    for item in test_data:
        all_texts.append(item['question'])
        all_texts.append(item['answer'])
    tokenizer.build_vocab(all_texts)
    
    # Датасет
    dataset = QADataset(temp_path, tokenizer, max_length=50)
    
    # DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    print(f"\n📊 DataLoader создан:")
    print(f"   Батчей: {len(dataloader)}")
    
    # Тест батча
    questions, answers, q_lengths, a_lengths = next(iter(dataloader))
    
    print(f"\n🧪 Тестовый батч:")
    print(f"   Questions: {questions.shape}")
    print(f"   Answers: {answers.shape}")
    print(f"   Q lengths: {q_lengths.tolist()}")
    print(f"   A lengths: {a_lengths.tolist()}")
    
    import os
    os.remove(temp_path)
    
    print("\n" + "=" * 60)
    print("✅ DATALOADER РАБОТАЕТ")
    print("=" * 60)
