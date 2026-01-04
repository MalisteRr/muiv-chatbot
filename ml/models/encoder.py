"""
Encoder (Кодировщик) на основе LSTM
Автор: Синицин Михаил
Тема ВКР: Разработка интеллектуального чат-бота для автоматизации консультаций абитуриентов

Encoder преобразует входную последовательность (вопрос) в контекстный вектор,
который затем используется Decoder'ом для генерации ответа.

Архитектура:
Input → Embedding → LSTM → Hidden State (контекстный вектор)
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Encoder на основе многослойного LSTM
    
    Принимает на вход последовательность индексов слов,
    возвращает скрытое состояние (hidden state) и состояние ячейки (cell state)
    """
    
    def __init__(
        self, 
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        """
        Инициализация Encoder
        
        Args:
            vocab_size: Размер словаря
            embedding_dim: Размерность эмбеддингов
            hidden_size: Размер скрытого слоя LSTM
            num_layers: Количество слоёв LSTM
            dropout: Вероятность dropout (регуляризация)
        """
        super(Encoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Слой эмбеддингов (преобразование индексов в векторы)
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0  # Индекс PAD токена
        )
        
        # LSTM слой (может быть многослойным)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,  # Dropout между слоями
            batch_first=True  # Формат: (batch, seq_len, features)
        )
        
        # Dropout для эмбеддингов
        self.dropout_layer = nn.Dropout(dropout)


if __name__ == "__main__":
    """
    Тестирование базовой структуры Encoder
    """
    print("\n" + "=" * 60)
    print("ТЕСТ ENCODER - Базовая структура")
    print("=" * 60)
    
    # Параметры
    vocab_size = 5000
    embedding_dim = 256
    hidden_size = 512
    num_layers = 2
    
    # Создаём Encoder
    encoder = Encoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=0.3
    )
    
    print(f"✅ Encoder создан:")
    print(f"   Vocab size: {vocab_size}")
    print(f"   Embedding dim: {embedding_dim}")
    print(f"   Hidden size: {hidden_size}")
    print(f"   Num layers: {num_layers}")
    print(f"\n📊 Параметров в модели: {sum(p.numel() for p in encoder.parameters()):,}")
    
    print("\n" + "=" * 60)
    print("✅ БАЗОВАЯ СТРУКТУРА ГОТОВА")
    print("=" * 60)
