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
        
        # Слой эмбеддингов
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        
        # LSTM слой
        lstm_input_size = embedding_dim + hidden_size if use_attention else embedding_dim
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Механизм внимания
        if use_attention:
            self.attention = Attention(hidden_size)
        
        # Выходной слой
        fc_input_size = hidden_size * 2 if use_attention else hidden_size
        self.fc = nn.Linear(fc_input_size, vocab_size)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, input_token, hidden, cell, encoder_outputs=None):
        """
        Прямой проход через Decoder (генерация одного токена)
        
        Args:
            input_token: Входной токен (предыдущее слово)
                        Форма: (batch_size, 1)
            hidden: Скрытое состояние
                   Форма: (num_layers, batch_size, hidden_size)
            cell: Состояние ячейки
                 Форма: (num_layers, batch_size, hidden_size)
            encoder_outputs: Выходы encoder'а (для attention)
                           Форма: (batch_size, seq_length, hidden_size)
        
        Returns:
            output: Распределение вероятностей для следующего слова
                   Форма: (batch_size, vocab_size)
            hidden: Новое скрытое состояние
            cell: Новое состояние ячейки
            attention_weights: Веса внимания (если используется)
        """
        # 1. Эмбеддинг входного токена
        embedded = self.embedding(input_token)
        embedded = self.dropout_layer(embedded)
        
        # 2. Вычисление контекстного вектора через attention
        attention_weights = None
        if self.use_attention and encoder_outputs is not None:
            # Берём последний слой hidden для attention
            last_hidden = hidden[-1]
            
            # Вычисляем веса внимания
            attention_weights = self.attention(last_hidden, encoder_outputs)
            
            # Вычисляем контекстный вектор (взвешенная сумма encoder outputs)
            context = torch.bmm(
                attention_weights.unsqueeze(1), 
                encoder_outputs
            )
            
            # Конкатенируем эмбеддинг и контекст
            lstm_input = torch.cat([embedded, context], dim=2)
        else:
            # Без attention
            lstm_input = embedded
        
        # 3. Пропускаем через LSTM
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        
        # 4. Подготавливаем вход для выходного слоя
        if self.use_attention:
            # Конкатенируем LSTM output и context
            fc_input = torch.cat([output, context], dim=2)
        else:
            fc_input = output
        
        # Убираем размерность seq_length (она = 1)
        fc_input = fc_input.squeeze(1)
        
        # 5. Выходной слой (распределение по словарю)
        prediction = self.fc(fc_input)
        
        return prediction, hidden, cell, attention_weights


if __name__ == "__main__":
    """
    Тестирование Decoder с forward
    """
    print("\n" + "=" * 60)
    print("ТЕСТ DECODER - Метод forward")
    print("=" * 60)
    
    # Параметры
    vocab_size = 5000
    embedding_dim = 256
    hidden_size = 512
    num_layers = 2
    batch_size = 4
    seq_length = 20
    
    # Создаём Decoder
    decoder = Decoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=0.3,
        use_attention=True
    )
    
    print(f"✅ Decoder создан с attention")
    
    # Тестовые данные
    test_input = torch.randint(0, vocab_size, (batch_size, 1))
    test_hidden = torch.randn(num_layers, batch_size, hidden_size)
    test_cell = torch.randn(num_layers, batch_size, hidden_size)
    test_encoder_outputs = torch.randn(batch_size, seq_length, hidden_size)
    
    print(f"\n🧪 Тестовый вход:")
    print(f"   Input форма: {test_input.shape}")
    print(f"   Encoder outputs форма: {test_encoder_outputs.shape}")
    
    # Прямой проход
    decoder.eval()
    with torch.no_grad():
        prediction, hidden, cell, attention_weights = decoder(
            test_input, 
            test_hidden, 
            test_cell, 
            test_encoder_outputs
        )
    
    print(f"\n📤 Выход Decoder:")
    print(f"   Prediction форма: {prediction.shape}")
    print(f"   Hidden форма: {hidden.shape}")
    print(f"   Attention weights форма: {attention_weights.shape}")
    
    # Проверяем распределение вероятностей
    probs = F.softmax(prediction, dim=1)
    print(f"\n📊 Вероятности:")
    print(f"   Сумма: {probs[0].sum().item():.4f} (должна быть ≈1.0)")
    
    print("\n" + "=" * 60)
    print("✅ DECODER С FORWARD РАБОТАЕТ")
    print("=" * 60)
