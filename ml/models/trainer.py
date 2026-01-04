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
        """
        Инициализация Trainer
        
        Args:
            model: Seq2Seq модель
            optimizer: Оптимизатор (Adam, SGD и т.д.)
            criterion: Функция потерь (CrossEntropyLoss)
            device: Устройство (cpu или cuda)
            grad_clip: Максимальное значение градиента (для стабильности)
        """
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


def create_trainer(
    model: Seq2Seq,
    learning_rate: float = 0.001,
    device: str = 'cpu'
) -> Trainer:
    """
    Фабричная функция для создания Trainer
    
    Args:
        model: Seq2Seq модель
        learning_rate: Скорость обучения
        device: Устройство
    
    Returns:
        Настроенный Trainer
    """
    # Оптимизатор (Adam)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Функция потерь (CrossEntropyLoss)
    # ignore_index=0 - игнорируем PAD токены
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Создаём Trainer
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
    Тестирование базовой структуры Trainer
    """
    print("\n" + "=" * 60)
    print("ТЕСТ TRAINER - Базовая структура")
    print("=" * 60)
    
    from .encoder import Encoder
    from .decoder import Decoder
    
    # Создаём модель
    encoder = Encoder(vocab_size=5000)
    decoder = Decoder(vocab_size=5000, use_attention=True)
    model = Seq2Seq(encoder, decoder, device='cpu')
    
    # Создаём trainer
    trainer = create_trainer(model, learning_rate=0.001, device='cpu')
    
    print(f"✅ Trainer создан:")
    print(f"   Оптимизатор: Adam")
    print(f"   Learning rate: 0.001")
    print(f"   Criterion: CrossEntropyLoss")
    print(f"   Grad clip: 5.0")
    
    print("\n" + "=" * 60)
    print("✅ БАЗОВАЯ СТРУКТУРА TRAINER ГОТОВА")
    print("=" * 60)
