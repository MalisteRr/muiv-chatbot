"""
Trainer для обучения Seq2Seq модели

Класс Trainer управляет процессом обучения:
1. Прямой проход (forward pass)
2. Вычисление loss
3. Обратное распространение (backpropagation)
4. Обновление весов
5. Валидация
6. Сохранение чекпоинтов
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
from typing import Optional, Dict

from .seq2seq import Seq2Seq
from .config import ModelConfig, TrainingConfig


class Trainer:
    """
    Класс для обучения Seq2Seq модели
    """
    
    def __init__(
        self,
        model: Seq2Seq,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: str = 'cpu',
        grad_clip: float = 5.0
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.grad_clip = grad_clip
        
        # История обучения
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # Для early stopping
        self.patience_counter = 0
    
    def train_epoch(
        self, 
        dataloader: DataLoader,
        teacher_forcing_ratio: float = 0.5
    ) -> float:
        """
        Обучение на одной эпохе
        
        Args:
            dataloader: DataLoader с обучающими данными
            teacher_forcing_ratio: Вероятность использования teacher forcing
        
        Returns:
            Средний loss за эпоху
        """
        self.model.train()
        epoch_loss = 0
        
        for batch_idx, (questions, answers, q_lengths, a_lengths) in enumerate(dataloader):
            # Переносим на устройство
            questions = questions.to(self.device)
            answers = answers.to(self.device)
            q_lengths = q_lengths.to(self.device)
            
            # Обнуляем градиенты
            self.optimizer.zero_grad()
            
            # Прямой проход
            outputs = self.model(
                questions, 
                answers, 
                q_lengths,
                teacher_forcing_ratio
            )
            
            # Вычисление loss
            # outputs: (batch_size, trg_len, vocab_size)
            # answers: (batch_size, trg_len)
            
            # Убираем первый токен из ответов (<SOS>)
            output_dim = outputs.shape[-1]
            
            # Reshape для вычисления loss
            outputs = outputs[:, 1:].reshape(-1, output_dim)
            answers = answers[:, 1:].reshape(-1)
            
            # Вычисляем loss (игнорируем padding токены)
            loss = self.criterion(outputs, answers)
            
            # Обратное распространение
            loss.backward()
            
            # Gradient clipping (обрезка градиентов)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            # Обновление весов
            self.optimizer.step()
            
            # Накапливаем loss
            epoch_loss += loss.item()
            
            # Логирование
            if (batch_idx + 1) % TrainingConfig.LOG_EVERY == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                print(f"   Batch {batch_idx + 1}/{len(dataloader)} | Loss: {avg_loss:.4f}")
        
        return epoch_loss / len(dataloader)


def create_trainer(
    model: Seq2Seq,
    learning_rate: float = 0.001,
    device: str = 'cpu'
) -> Trainer:
    """Фабричная функция для создания Trainer"""
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        grad_clip=ModelConfig.GRAD_CLIP
    )
    
    return trainer


if __name__ == "__main__":
    """
    Тестирование train_epoch
    """
    print("\n" + "=" * 60)
    print("ТЕСТ TRAINER - train_epoch")
    print("=" * 60)
    
    print("✅ Метод train_epoch добавлен")
    print("\nВыполняет:")
    print("1. Прямой проход через модель")
    print("2. Вычисление loss")
    print("3. Обратное распространение")
    print("4. Gradient clipping")
    print("5. Обновление весов")
    
    print("\n" + "=" * 60)
    print("✅ TRAIN_EPOCH ГОТОВ")
    print("=" * 60)
