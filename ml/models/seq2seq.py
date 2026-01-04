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
        
        # Проверка совместимости
        assert encoder.hidden_size == decoder.hidden_size, \
            "Hidden size encoder'а и decoder'а должны совпадать!"
        assert encoder.num_layers == decoder.num_layers, \
            "Количество слоёв encoder'а и decoder'а должно совпадать!"
    
    def forward(
        self, 
        src, 
        trg, 
        src_lengths=None,
        teacher_forcing_ratio: float = 0.5
    ):
        """
        Прямой проход через Seq2Seq (обучение)
        
        Args:
            src: Входная последовательность (вопрос)
                Форма: (batch_size, src_seq_length)
            trg: Целевая последовательность (ответ)
                Форма: (batch_size, trg_seq_length)
            src_lengths: Реальные длины входных последовательностей
            teacher_forcing_ratio: Вероятность использования teacher forcing
                                  1.0 = всегда используем правильный токен
                                  0.0 = всегда используем предсказание модели
        
        Returns:
            outputs: Предсказания для каждого временного шага
                    Форма: (batch_size, trg_seq_length, vocab_size)
        """
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.vocab_size
        
        # Тензор для хранения выходов decoder'а
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # 1. ENCODER: обрабатываем входную последовательность
        encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
        
        # 2. DECODER: генерируем ответ токен за токеном
        
        # Первый токен decoder'а - это всегда <SOS> (Start Of Sequence)
        decoder_input = trg[:, 0].unsqueeze(1)  # (batch_size, 1)
        
        for t in range(1, trg_len):
            # Предсказываем следующий токен
            prediction, hidden, cell, attention_weights = self.decoder(
                decoder_input,
                hidden,
                cell,
                encoder_outputs if self.decoder.use_attention else None
            )
            
            # Сохраняем предсказание
            outputs[:, t, :] = prediction
            
            # Решаем использовать ли teacher forcing
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            
            # Получаем токен с максимальной вероятностью
            top_prediction = prediction.argmax(1)
            
            # Выбираем входной токен для следующего шага
            if use_teacher_forcing:
                # Teacher forcing: используем правильный токен
                decoder_input = trg[:, t].unsqueeze(1)
            else:
                # Используем предсказание модели
                decoder_input = top_prediction.unsqueeze(1)
        
        return outputs
    
    def count_parameters(self):
        """Подсчёт количества обучаемых параметров"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    """
    Тестирование Seq2Seq с forward
    """
    print("\n" + "=" * 60)
    print("ТЕСТ SEQ2SEQ - Метод forward")
    print("=" * 60)
    
    vocab_size = 5000
    batch_size = 4
    src_len = 20
    trg_len = 25
    
    # Создаём модель
    encoder = Encoder(vocab_size=vocab_size)
    decoder = Decoder(vocab_size=vocab_size, use_attention=True)
    model = Seq2Seq(encoder, decoder, device='cpu')
    
    # Тестовые данные
    src = torch.randint(0, vocab_size, (batch_size, src_len))
    trg = torch.randint(0, vocab_size, (batch_size, trg_len))
    src_lengths = torch.tensor([20, 18, 15, 12])
    
    print(f"✅ Модель создана: {model.count_parameters():,} параметров")
    print(f"\n🧪 Тестовый вход:")
    print(f"   Source: {src.shape}")
    print(f"   Target: {trg.shape}")
    
    # Прямой проход
    model.eval()
    with torch.no_grad():
        outputs = model(src, trg, src_lengths, teacher_forcing_ratio=1.0)
    
    print(f"\n📤 Выход:")
    print(f"   Форма: {outputs.shape}")
    print(f"   Ожидалось: ({batch_size}, {trg_len}, {vocab_size})")
    
    print("\n" + "=" * 60)
    print("✅ FORWARD РАБОТАЕТ")
    print("=" * 60)
