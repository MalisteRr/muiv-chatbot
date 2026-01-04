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
    """
    
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        device: str = 'cpu'
    ):
        super(Seq2Seq, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hidden_size == decoder.hidden_size, \
            "Hidden size encoder'а и decoder'а должны совпадать!"
        assert encoder.num_layers == decoder.num_layers, \
            "Количество слоёв encoder'а и decoder'а должно совпадать!"
    
    def forward(self, src, trg, src_lengths=None, teacher_forcing_ratio: float = 0.5):
        """Прямой проход через Seq2Seq (обучение)"""
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.vocab_size
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
        decoder_input = trg[:, 0].unsqueeze(1)
        
        for t in range(1, trg_len):
            prediction, hidden, cell, _ = self.decoder(
                decoder_input, hidden, cell,
                encoder_outputs if self.decoder.use_attention else None
            )
            
            outputs[:, t, :] = prediction
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            top_prediction = prediction.argmax(1)
            
            decoder_input = trg[:, t].unsqueeze(1) if use_teacher_forcing else top_prediction.unsqueeze(1)
        
        return outputs
    
    def generate(self, src, src_lengths=None, max_length: int = 100, sos_token: int = 2, eos_token: int = 3):
        """Генерация ответа (inference)"""
        self.eval()
        batch_size = src.size(0)
        
        with torch.no_grad():
            encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
            decoder_input = torch.full((batch_size, 1), sos_token, dtype=torch.long).to(self.device)
            generated_tokens = []
            finished = torch.zeros(batch_size, dtype=torch.bool).to(self.device)
            
            for _ in range(max_length):
                prediction, hidden, cell, _ = self.decoder(
                    decoder_input, hidden, cell,
                    encoder_outputs if self.decoder.use_attention else None
                )
                
                next_token = prediction.argmax(1)
                generated_tokens.append(next_token.unsqueeze(1))
                finished = finished | (next_token == eos_token)
                
                if finished.all():
                    break
                
                decoder_input = next_token.unsqueeze(1)
            
            generated_tokens = torch.cat(generated_tokens, dim=1)
        
        return generated_tokens
    
    def count_parameters(self):
        """Подсчёт количества обучаемых параметров"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def init_weights(model):
    """
    Инициализация весов модели
    
    Использует Xavier uniform инициализацию
    
    Args:
        model: Модель для инициализации
    """
    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.xavier_uniform_(param.data)
        elif 'bias' in name:
            nn.init.constant_(param.data, 0)


def create_seq2seq_model(
    vocab_size: int,
    embedding_dim: int = 256,
    hidden_size: int = 512,
    num_layers: int = 2,
    dropout: float = 0.3,
    use_attention: bool = True,
    device: str = 'cpu'
):
    """
    Фабричная функция для создания Seq2Seq модели
    
    Args:
        vocab_size: Размер словаря
        embedding_dim: Размерность эмбеддингов
        hidden_size: Размер скрытого слоя
        num_layers: Количество слоёв LSTM
        dropout: Вероятность dropout
        use_attention: Использовать ли механизм внимания
        device: Устройство (cpu или cuda)
    
    Returns:
        Инициализированная модель Seq2Seq
    """
    # Создаём encoder
    encoder = Encoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )
    
    # Создаём decoder
    decoder = Decoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        use_attention=use_attention
    )
    
    # Создаём seq2seq модель
    model = Seq2Seq(encoder, decoder, device)
    
    # Инициализируем веса
    model.apply(init_weights)
    
    # Переносим на устройство
    model = model.to(device)
    
    print(f"✅ Seq2Seq модель создана:")
    print(f"   Параметров: {model.count_parameters():,}")
    print(f"   Устройство: {device}")
    print(f"   Attention: {'Да' if use_attention else 'Нет'}")
    
    return model


if __name__ == "__main__":
    """
    Тестирование полной Seq2Seq модели
    """
    print("\n" + "=" * 60)
    print("ТЕСТ SEQ2SEQ - Полная модель")
    print("=" * 60)
    
    # Создаём модель через фабричную функцию
    model = create_seq2seq_model(
        vocab_size=5000,
        embedding_dim=256,
        hidden_size=512,
        num_layers=2,
        dropout=0.3,
        use_attention=True,
        device='cpu'
    )
    
    # Тестовые данные
    batch_size = 4
    src = torch.randint(0, 5000, (batch_size, 20))
    trg = torch.randint(0, 5000, (batch_size, 25))
    src_lengths = torch.tensor([20, 18, 15, 12])
    
    print(f"\n🧪 Тест обучения:")
    model.eval()
    with torch.no_grad():
        outputs = model(src, trg, src_lengths, teacher_forcing_ratio=1.0)
    print(f"   Выход: {outputs.shape}")
    
    print(f"\n🤖 Тест генерации:")
    with torch.no_grad():
        generated = model.generate(src, src_lengths, max_length=30)
    print(f"   Сгенерировано: {generated.shape}")
    
    print("\n" + "=" * 60)
    print("✅ SEQ2SEQ ПОЛНОСТЬЮ ГОТОВ")
    print("=" * 60)
