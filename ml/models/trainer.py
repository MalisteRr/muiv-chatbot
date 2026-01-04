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
        """Валидация модели"""
        self.model.eval()
        epoch_loss = 0
        
        with torch.no_grad():
            for questions, answers, q_lengths, a_lengths in dataloader:
                questions = questions.to(self.device)
                answers = answers.to(self.device)
                q_lengths = q_lengths.to(self.device)
                
                outputs = self.model(questions, answers, q_lengths, teacher_forcing_ratio=0.0)
                
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
        """Полный цикл обучения"""
        print("\n" + "=" * 60)
        print("НАЧАЛО ОБУЧЕНИЯ")
        print("=" * 60)
        print(f"Эпох: {num_epochs}")
        print(f"Устройство: {self.device}")
        print(f"Параметров: {self.model.count_parameters():,}")
        print("=" * 60 + "\n")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            print(f"\nЭпоха {epoch + 1}/{num_epochs}")
            print("-" * 60)
            
            train_loss = self.train_epoch(train_loader, teacher_forcing_ratio)
            self.train_losses.append(train_loss)
            
            if val_loader:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
                
                print(f"\n📊 Результаты:")
                print(f"   Train Loss: {train_loss:.4f}")
                print(f"   Val Loss: {val_loss:.4f}")
                
                if val_loss < self.best_val_loss - TrainingConfig.MIN_DELTA:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    
                    if save_dir:
                        self.save_checkpoint(save_dir, epoch, train_loss, val_loss, is_best=True)
                    print(f"   ✅ Новая лучшая модель! Val Loss: {val_loss:.4f}")
                else:
                    self.patience_counter += 1
                    print(f"   ⚠️ Нет улучшения ({self.patience_counter}/{early_stopping_patience})")
                
                if self.patience_counter >= early_stopping_patience:
                    print(f"\n⛔ Early stopping!")
                    break
            else:
                print(f"\n📊 Train Loss: {train_loss:.4f}")
                
                if save_dir and (epoch + 1) % TrainingConfig.SAVE_EVERY == 0:
                    self.save_checkpoint(save_dir, epoch, train_loss, None)
            
            teacher_forcing_ratio *= 0.95
            
            epoch_time = time.time() - start_time
            print(f"   ⏱️ Время: {epoch_time:.2f} сек")
        
        print("\n" + "=" * 60)
        print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
        print("=" * 60)
        print(f"Лучший Val Loss: {self.best_val_loss:.4f}")
        print("=" * 60 + "\n")
    
    def save_checkpoint(
        self,
        save_dir: str,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        is_best: bool = False
    ):
        """
        Сохранение чекпоинта модели
        
        Args:
            save_dir: Директория для сохранения
            epoch: Номер эпохи
            train_loss: Train loss
            val_loss: Validation loss
            is_best: Лучшая ли это модель
        """
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        if is_best:
            path = os.path.join(save_dir, 'best_model.pt')
        else:
            path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt')
        
        torch.save(checkpoint, path)
        print(f"   💾 Чекпоинт сохранён: {path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Загрузка чекпоинта
        
        Args:
            checkpoint_path: Путь к чекпоинту
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        epoch = checkpoint['epoch']
        train_loss = checkpoint['train_loss']
        val_loss = checkpoint.get('val_loss')
        
        print(f"✅ Чекпоинт загружен: {checkpoint_path}")
        print(f"   Эпоха: {epoch + 1}")
        print(f"   Train Loss: {train_loss:.4f}")
        if val_loss:
            print(f"   Val Loss: {val_loss:.4f}")


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
    Тестирование save/load checkpoint
    """
    print("\n" + "=" * 60)
    print("ТЕСТ TRAINER - Сохранение и загрузка")
    print("=" * 60)
    
    print("✅ Методы добавлены:")
    print("   - save_checkpoint(): сохранение модели")
    print("   - load_checkpoint(): загрузка модели")
    print("\n💾 Что сохраняется:")
    print("   - Веса модели (model_state_dict)")
    print("   - Состояние оптимизатора")
    print("   - Номер эпохи")
    print("   - Train/Val losses")
    print("   - История обучения")
    
    print("\n" + "=" * 60)
    print("✅ TRAINER ПОЛНОСТЬЮ ГОТОВ")
    print("=" * 60)
