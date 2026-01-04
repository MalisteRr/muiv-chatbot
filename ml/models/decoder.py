"""
Decoder (Декодер) на основе LSTM с механизмом внимания (Attention)

Decoder генерирует ответ на основе контекстного вектора от Encoder'а.
Использует механизм внимания (Attention) для фокусировки на важных частях входа.

Архитектура:
Context Vector → LSTM → Attention → Linear → Output (следующее слово)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Механизм внимания (Bahdanau Attention)
    
    Позволяет decoder'у "смотреть" на разные части входной последовательности
    при генерации каждого слова ответа.
    """
    
    def __init__(self, hidden_size: int):
        """
        Args:
            hidden_size: Размер скрытого слоя
        """
        super(Attention, self).__init__()
        
        self.hidden_size = hidden_size
        
        # Линейные слои для вычисления весов внимания
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, hidden, encoder_outputs):
        """
        Вычисление весов внимания
        
        Args:
            hidden: Скрытое состояние decoder'а
                   Форма: (batch_size, hidden_size)
            encoder_outputs: Выходы encoder'а для всех временных шагов
                           Форма: (batch_size, seq_length, hidden_size)
        
        Returns:
            attention_weights: Веса внимания
                             Форма: (batch_size, seq_length)
        """
        batch_size = encoder_outputs.size(0)
        seq_length = encoder_outputs.size(1)
        
        # Повторяем hidden для каждого временного шага
        hidden = hidden.unsqueeze(1).repeat(1, seq_length, 1)
        
        # Конкатенируем hidden и encoder_outputs
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))
        
        # Вычисляем скоры внимания
        attention = self.v(energy).squeeze(2)
        
        # Применяем softmax для получения вероятностей
        attention_weights = F.softmax(attention, dim=1)
        
        return attention_weights


class Decoder(nn.Module):
    """
    Decoder на основе LSTM с механизмом внимания
    
    Генерирует ответ по одному слову за раз,
    используя контекст от encoder'а и предыдущие сгенерированные слова.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_attention: bool = True
    ):
        """
        Инициализация Decoder
        
        Args:
            vocab_size: Размер словаря
            embedding_dim: Размерность эмбеддингов
            hidden_size: Размер скрытого слоя
            num_layers: Количество слоёв LSTM
            dropout: Вероятность dropout
            use_attention: Использовать ли механизм внимания
        """
        super(Decoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_attention = use_attention
        
        # Слой эмбеддингов
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        
        # LSTM слой
        # Если есть attention, входной размер увеличивается
        lstm_input_size = embedding_dim + hidden_size if use_attention else embedding_dim
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Механизм внимания (опционально)
        if use_attention:
            self.attention = Attention(hidden_size)
        
        # Выходной слой (преобразование в распределение по словарю)
        fc_input_size = hidden_size * 2 if use_attention else hidden_size
        self.fc = nn.Linear(fc_input_size, vocab_size)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)


if __name__ == "__main__":
    """
    Тестирование базового Decoder
    """
    print("\n" + "=" * 60)
    print("ТЕСТ DECODER - Базовая структура")
    print("=" * 60)
    
    # Параметры
    vocab_size = 5000
    embedding_dim = 256
    hidden_size = 512
    num_layers = 2
    
    # Создаём Decoder с attention
    decoder = Decoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=0.3,
        use_attention=True
    )
    
    print(f"✅ Decoder создан:")
    print(f"   Vocab size: {vocab_size}")
    print(f"   Embedding dim: {embedding_dim}")
    print(f"   Hidden size: {hidden_size}")
    print(f"   Num layers: {num_layers}")
    print(f"   Attention: Да")
    print(f"\n📊 Параметров в модели: {sum(p.numel() for p in decoder.parameters()):,}")
    
    print("\n" + "=" * 60)
    print("✅ БАЗОВАЯ СТРУКТУРА DECODER ГОТОВА")
    print("=" * 60)
