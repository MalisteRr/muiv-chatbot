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
    """
    
    def __init__(self, hidden_size: int):
        super(Attention, self).__init__()
        
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.size(0)
        seq_length = encoder_outputs.size(1)
        
        hidden = hidden.unsqueeze(1).repeat(1, seq_length, 1)
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))
        attention = self.v(energy).squeeze(2)
        attention_weights = F.softmax(attention, dim=1)
        
        return attention_weights


class Decoder(nn.Module):
    """
    Decoder на основе LSTM с механизмом внимания
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
        super(Decoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_attention = use_attention
        
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        
        lstm_input_size = embedding_dim + hidden_size if use_attention else embedding_dim
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        if use_attention:
            self.attention = Attention(hidden_size)
        
        fc_input_size = hidden_size * 2 if use_attention else hidden_size
        self.fc = nn.Linear(fc_input_size, vocab_size)
        
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, input_token, hidden, cell, encoder_outputs=None):
        """
        Прямой проход через Decoder (генерация одного токена)
        """
        # Эмбеддинг
        embedded = self.embedding(input_token)
        embedded = self.dropout_layer(embedded)
        
        # Attention
        attention_weights = None
        if self.use_attention and encoder_outputs is not None:
            last_hidden = hidden[-1]
            attention_weights = self.attention(last_hidden, encoder_outputs)
            context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
            lstm_input = torch.cat([embedded, context], dim=2)
        else:
            lstm_input = embedded
        
        # LSTM
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        
        # Выходной слой
        if self.use_attention:
            fc_input = torch.cat([output, context], dim=2)
        else:
            fc_input = output
        
        fc_input = fc_input.squeeze(1)
        prediction = self.fc(fc_input)
        
        return prediction, hidden, cell, attention_weights


class SimpleDecoder(nn.Module):
    """
    Упрощённый Decoder без механизма внимания
    
    Используется для базовой модели или для сравнения производительности
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
        Инициализация SimpleDecoder
        
        Args:
            vocab_size: Размер словаря
            embedding_dim: Размерность эмбеддингов
            hidden_size: Размер скрытого слоя
            num_layers: Количество слоёв LSTM
            dropout: Вероятность dropout
        """
        super(SimpleDecoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Эмбеддинги
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM (без attention, поэтому входной размер = embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Выходной слой
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_token, hidden, cell, encoder_outputs=None):
        """
        Упрощённый forward без attention
        
        Args:
            input_token: Входной токен
            hidden: Скрытое состояние
            cell: Состояние ячейки
            encoder_outputs: Не используется (для совместимости с интерфейсом)
        
        Returns:
            prediction: Предсказание следующего слова
            hidden: Новое скрытое состояние
            cell: Новое состояние ячейки
            None: Нет весов внимания
        """
        # Эмбеддинг
        embedded = self.embedding(input_token)
        embedded = self.dropout(embedded)
        
        # LSTM
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        
        # Выходной слой
        output = output.squeeze(1)
        prediction = self.fc(output)
        
        return prediction, hidden, cell, None


if __name__ == "__main__":
    """
    Тестирование обоих Decoder'ов
    """
    print("\n" + "=" * 60)
    print("ТЕСТ DECODER - Обе версии")
    print("=" * 60)
    
    vocab_size = 5000
    batch_size = 4
    
    # Тест Decoder с Attention
    decoder_attn = Decoder(vocab_size=vocab_size, use_attention=True)
    print(f"✅ Decoder с Attention: {sum(p.numel() for p in decoder_attn.parameters()):,} параметров")
    
    # Тест SimpleDecoder
    decoder_simple = SimpleDecoder(vocab_size=vocab_size)
    print(f"✅ SimpleDecoder: {sum(p.numel() for p in decoder_simple.parameters()):,} параметров")
    
    print("\n" + "=" * 60)
    print("✅ DECODER ПОЛНОСТЬЮ ГОТОВ")
    print("=" * 60)
