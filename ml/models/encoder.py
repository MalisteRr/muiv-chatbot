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
    
    def forward(self, input_seq, input_lengths=None):
        """
        Прямой проход через Encoder
        
        Args:
            input_seq: Входная последовательность индексов
                       Форма: (batch_size, seq_length)
            input_lengths: Реальные длины последовательностей (опционально)
                          Форма: (batch_size,)
        
        Returns:
            outputs: Выходы LSTM для каждого временного шага
                    Форма: (batch_size, seq_length, hidden_size)
            hidden: Скрытое состояние последнего слоя
                   Форма: (num_layers, batch_size, hidden_size)
            cell: Состояние ячейки последнего слоя
                 Форма: (num_layers, batch_size, hidden_size)
        """
        # 1. Преобразуем индексы в эмбеддинги
        # input_seq: (batch_size, seq_length)
        # embedded: (batch_size, seq_length, embedding_dim)
        embedded = self.embedding(input_seq)
        
        # 2. Применяем dropout к эмбеддингам
        embedded = self.dropout_layer(embedded)
        
        # 3. Если есть реальные длины - используем pack_padded_sequence
        # Это позволяет LSTM игнорировать паддинг
        if input_lengths is not None:
            # Упаковываем последовательности
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, 
                input_lengths.cpu(), 
                batch_first=True, 
                enforce_sorted=False
            )
            
            # Пропускаем через LSTM
            packed_outputs, (hidden, cell) = self.lstm(packed)
            
            # Распаковываем обратно
            outputs, _ = nn.utils.rnn.pad_packed_sequence(
                packed_outputs, 
                batch_first=True
            )
        else:
            # Обычный проход без упаковки
            outputs, (hidden, cell) = self.lstm(embedded)
        
        # outputs: (batch_size, seq_length, hidden_size)
        # hidden: (num_layers, batch_size, hidden_size)
        # cell: (num_layers, batch_size, hidden_size)
        
        return outputs, hidden, cell


if __name__ == "__main__":
    """
    Тестирование Encoder с методом forward
    """
    print("\n" + "=" * 60)
    print("ТЕСТ ENCODER - Метод forward")
    print("=" * 60)
    
    # Параметры
    vocab_size = 5000
    embedding_dim = 256
    hidden_size = 512
    num_layers = 2
    batch_size = 4
    seq_length = 20
    
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
    
    # Тестовые данные
    test_input = torch.randint(0, vocab_size, (batch_size, seq_length))
    test_lengths = torch.tensor([20, 18, 15, 12])
    
    print(f"\n🧪 Тестовый вход:")
    print(f"   Форма: {test_input.shape}")
    print(f"   Длины: {test_lengths.tolist()}")
    
    # Прямой проход
    encoder.eval()
    with torch.no_grad():
        outputs, hidden, cell = encoder(test_input, test_lengths)
    
    print(f"\n📤 Выход Encoder:")
    print(f"   Outputs форма: {outputs.shape}")
    print(f"   Hidden форма: {hidden.shape}")
    print(f"   Cell форма: {cell.shape}")
    
    print("\n" + "=" * 60)
    print("✅ МЕТОД FORWARD РАБОТАЕТ")
    print("=" * 60)
