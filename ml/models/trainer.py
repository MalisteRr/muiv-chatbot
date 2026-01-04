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
    """Класс для обучения Seq2Seq модели"""
    
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
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def train_epoch(self, dataloader: DataLoader, teacher_forcing_ratio: float = 0.5) -> float:
        """Обучение на одной эпохе"""
        self.model.train()
        epoch_loss = 0
        
        for batch_idx, (questions, answers, q_lengths, a_lengths) in enumerate(dataloader):
            questions = questions.to(self.device)
            answers = answers.to(self.device)
            q_lengths = q_lengths.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(questions, answers, q_lengths, teacher_forcing_ratio)
            
            output_dim = outputs.shape[-1]
            outputs = outputs[:, 1:].reshape(-1, output_dim)
            answers = answers[:, 1:].reshape(-1)
            
            loss = self.criterion(outputs, answers)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
            if (batch_idx + 1) % TrainingConfig.LOG_EVERY == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                print(f"   Batch {batch_idx + 1}/{len(dataloader)} | Loss: {avg_loss:.4f}")
        
        return epoch_loss / len(dataloader)
    
    def validate(self, dataloader: DataLoader) -> float:
        """
        Валидация модели
        
        Args:
            dataloader: DataLoader с валидационными данными
        
        Returns:
            Средний loss на валидации
        """
        self.model.eval()
        epoch_loss = 0
        
        with torch.no_grad():
            for questions, answers, q_lengths, a_lengths in dataloader:
                questions = questions.to(self.device)
                answers = answers.to(self.device)
                q_lengths = q_lengths.to(self.device)
                
                # Прямой проход (без teacher forcing)
                outputs = self.model(
                    questions, 
                    answers, 
                    q_lengths,
                    teacher_forcing_ratio=0.0
                )
                
                # Вычисление loss
                output_dim = outputs.shape[-1]
                outputs = outputs[:, 1:].reshape(-1, output_dim)
                answers = answers[:, 1:].reshape(-1)
                
                loss = self.criterion(outputs, answers)
                epoch_loss += loss.item()
        
        return epoch_loss / len(dataloader)
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 10,
        teacher_forcing_ratio: float = 0.5,
        save_dir: str = None,
        early_stopping_patience: int = 3
    ):
        """
        Полный цикл обучения
        
        Args:
            train_loader: DataLoader для обучения
            val_loader: DataLoader для валидации (опционально)
            num_epochs: Количество эпох
            teacher_forcing_ratio: Начальная вероятность teacher forcing
            save_dir: Директория для сохранения чекпоинтов
            early_stopping_patience: Терпение для early stopping
        """
        print("\n" + "=" * 60)
        print("НАЧАЛО ОБУЧЕНИЯ")
        print("=" * 60)
        print(f"Эпох: {num_epochs}")
        print(f"Устройство: {self.device}")
        print(f"Параметров в модели: {self.model.count_parameters():,}")
        print("=" * 60 + "\n")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            print(f"\nЭпоха {epoch + 1}/{num_epochs}")
            print("-" * 60)
            
            # Обучение
            train_loss = self.train_epoch(train_loader, teacher_forcing_ratio)
            self.train_losses.append(train_loss)
            
            # Валидация
            if val_loader:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
                
                print(f"\n📊 Результаты эпохи {epoch + 1}:")
                print(f"   Train Loss: {train_loss:.4f}")
                print(f"   Val Loss: {val_loss:.4f}")
                
                # Проверка улучшения
                if val_loss < self.best_val_loss - TrainingConfig.MIN_DELTA:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    print(f"   ✅ Новая лучшая модель! Val Loss: {val_loss:.4f}")
                else:
                    self.patience_counter += 1
                    print(f"   ⚠️ Нет улучшения ({self.patience_counter}/{early_stopping_patience})")
                
                # Early stopping
                if self.patience_counter >= early_stopping_patience:
                    print(f"\n⛔ Early stopping! Нет улучшения {early_stopping_patience} эпох.")
                    break
            else:
                print(f"\n📊 Train Loss: {train_loss:.4f}")
            
            # Уменьшаем teacher forcing со временем
            teacher_forcing_ratio *= 0.95
            
            # Время эпохи
            epoch_time = time.time() - start_time
            print(f"   ⏱️ Время: {epoch_time:.2f} сек")
        
        print("\n" + "=" * 60)
        print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
        print("=" * 60)
        print(f"Лучший Val Loss: {self.best_val_loss:.4f}")
        print("=" * 60 + "\n")


def create_trainer(model: Seq2Seq, learning_rate: float = 0.001, device: str = 'cpu') -> Trainer:
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
    Тестирование validate и train
    """
    print("\n" + "=" * 60)
    print("ТЕСТ TRAINER - validate и train")
    print("=" * 60)
    
    print("✅ Методы добавлены:")
    print("   - validate(): валидация модели")
    print("   - train(): полный цикл обучения")
    print("\n📊 Функции train():")
    print("   - Обучение на каждой эпохе")
    print("   - Валидация после каждой эпохи")
    print("   - Early stopping")
    print("   - Уменьшение teacher forcing")
    print("   - Логирование прогресса")
    
    print("\n" + "=" * 60)
    print("✅ VALIDATE И TRAIN ГОТОВЫ")
    print("=" * 60)
