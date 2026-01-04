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
        # hidden: (batch_size, hidden_size) -> (batch_size, seq_length, hidden_size)
        hidden = hidden.unsqueeze(1).repeat(1, seq_length, 1)
        
        # Конкатенируем hidden и encoder_outputs
        # (batch_size, seq_length, hidden_size * 2)
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))
        
        # Вычисляем скоры внимания
        # (batch_size, seq_length, 1) -> (batch_size, seq_length)
        attention = self.v(energy).squeeze(2)
        
        # Применяем softmax для получения вероятностей
        attention_weights = F.softmax(attention, dim=1)
        
        return attention_weights


if __name__ == "__main__":
    """
    Тестирование Attention
    """
    print("\n" + "=" * 60)
    print("ТЕСТ ATTENTION")
    print("=" * 60)
    
    # Параметры
    hidden_size = 512
    batch_size = 4
    seq_length = 20
    
    # Создаём Attention
    attention = Attention(hidden_size)
    
    print(f"✅ Attention создан:")
    print(f"   Hidden size: {hidden_size}")
    
    # Тестовые данные
    test_hidden = torch.randn(batch_size, hidden_size)
    test_encoder_outputs = torch.randn(batch_size, seq_length, hidden_size)
    
    print(f"\n🧪 Тестовый вход:")
    print(f"   Hidden форма: {test_hidden.shape}")
    print(f"   Encoder outputs форма: {test_encoder_outputs.shape}")
    
    # Прямой проход
    with torch.no_grad():
        attention_weights = attention(test_hidden, test_encoder_outputs)
    
    print(f"\n📤 Выход Attention:")
    print(f"   Attention weights форма: {attention_weights.shape}")
    print(f"   Сумма весов (должна быть ~1.0): {attention_weights[0].sum().item():.4f}")
    print(f"   Макс вес: {attention_weights[0].max().item():.4f}")
    print(f"   Мин вес: {attention_weights[0].min().item():.4f}")
    
    print("\n" + "=" * 60)
    print("✅ ATTENTION РАБОТАЕТ")
    print("=" * 60)
