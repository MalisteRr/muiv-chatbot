"""
Seq2Seq модель (Encoder-Decoder)

Seq2Seq модель объединяет Encoder и Decoder для генерации ответов на вопросы.

Архитектура:
Вопрос → Encoder → Контекстный вектор → Decoder → Ответ
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from .encoder import Encoder
from .decoder import Decoder


class Seq2Seq(nn.Module):
    """
    Seq2Seq модель для генерации ответов
    
    Объединяет Encoder и Decoder в единую архитектуру
    """
    
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        device: str = 'cpu'
    ):
        """
        Инициализация Seq2Seq
        
        Args:
            encoder: Экземпляр Encoder
            decoder: Экземпляр Decoder
            device: Устройство (cpu или cuda)
        """
        super(Seq2Seq, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        # Проверка совместимости encoder и decoder
        assert encoder.hidden_size == decoder.hidden_size, \
            "Hidden size encoder'а и decoder'а должны совпадать!"
        assert encoder.num_layers == decoder.num_layers, \
            "Количество слоёв encoder'а и decoder'а должно совпадать!"
    
    def count_parameters(self):
        """
        Подсчёт количества обучаемых параметров
        
        Returns:
            Количество параметров
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    """
    Тестирование базовой структуры Seq2Seq
    """
    print("\n" + "=" * 60)
    print("ТЕСТ SEQ2SEQ - Базовая структура")
    print("=" * 60)
    
    vocab_size = 5000
    embedding_dim = 256
    hidden_size = 512
    num_layers = 2
    
    # Создаём encoder и decoder
    encoder = Encoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers
    )
    
    decoder = Decoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        use_attention=True
    )
    
    # Создаём seq2seq модель
    model = Seq2Seq(encoder, decoder, device='cpu')
    
    print(f"✅ Seq2Seq модель создана:")
    print(f"   Параметров: {model.count_parameters():,}")
    print(f"   Устройство: cpu")
    
    print("\n" + "=" * 60)
    print("✅ БАЗОВАЯ СТРУКТУРА SEQ2SEQ ГОТОВА")
    print("=" * 60)
