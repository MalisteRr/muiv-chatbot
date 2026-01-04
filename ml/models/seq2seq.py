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
    
    def forward(
        self, 
        src, 
        trg, 
        src_lengths=None,
        teacher_forcing_ratio: float = 0.5
    ):
        """
        Прямой проход через Seq2Seq (обучение)
        """
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.vocab_size
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # Encoder
        encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
        
        # Decoder
        decoder_input = trg[:, 0].unsqueeze(1)
        
        for t in range(1, trg_len):
            prediction, hidden, cell, attention_weights = self.decoder(
                decoder_input,
                hidden,
                cell,
                encoder_outputs if self.decoder.use_attention else None
            )
            
            outputs[:, t, :] = prediction
            
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            top_prediction = prediction.argmax(1)
            
            if use_teacher_forcing:
                decoder_input = trg[:, t].unsqueeze(1)
            else:
                decoder_input = top_prediction.unsqueeze(1)
        
        return outputs
    
    def generate(
        self, 
        src, 
        src_lengths=None,
        max_length: int = 100,
        sos_token: int = 2,
        eos_token: int = 3
    ):
        """
        Генерация ответа (inference/тестирование)
        
        Args:
            src: Входная последовательность (вопрос)
                Форма: (batch_size, src_seq_length)
            src_lengths: Реальные длины входных последовательностей
            max_length: Максимальная длина генерируемого ответа
            sos_token: Индекс токена <SOS>
            eos_token: Индекс токена <EOS>
        
        Returns:
            generated_tokens: Сгенерированные токены
                            Форма: (batch_size, generated_length)
        """
        self.eval()
        
        batch_size = src.size(0)
        
        with torch.no_grad():
            # 1. Encoder
            encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
            
            # 2. Decoder - генерация
            
            # Начинаем с <SOS> токена
            decoder_input = torch.full(
                (batch_size, 1), 
                sos_token, 
                dtype=torch.long
            ).to(self.device)
            
            # Список для хранения сгенерированных токенов
            generated_tokens = []
            
            # Флаг завершения генерации для каждого примера в батче
            finished = torch.zeros(batch_size, dtype=torch.bool).to(self.device)
            
            for _ in range(max_length):
                # Предсказываем следующий токен
                prediction, hidden, cell, _ = self.decoder(
                    decoder_input,
                    hidden,
                    cell,
                    encoder_outputs if self.decoder.use_attention else None
                )
                
                # Получаем токен с максимальной вероятностью
                next_token = prediction.argmax(1)
                
                # Сохраняем сгенерированный токен
                generated_tokens.append(next_token.unsqueeze(1))
                
                # Проверяем встретился ли <EOS> токен
                finished = finished | (next_token == eos_token)
                
                # Если все последовательности завершились - останавливаемся
                if finished.all():
                    break
                
                # Следующий вход для decoder'а
                decoder_input = next_token.unsqueeze(1)
            
            # Объединяем все токены
            generated_tokens = torch.cat(generated_tokens, dim=1)
        
        return generated_tokens
    
    def count_parameters(self):
        """Подсчёт количества обучаемых параметров"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    """
    Тестирование Seq2Seq с generate
    """
    print("\n" + "=" * 60)
    print("ТЕСТ SEQ2SEQ - Метод generate")
    print("=" * 60)
    
    vocab_size = 5000
    batch_size = 4
    src_len = 20
    
    encoder = Encoder(vocab_size=vocab_size)
    decoder = Decoder(vocab_size=vocab_size, use_attention=True)
    model = Seq2Seq(encoder, decoder, device='cpu')
    
    # Тестовые данные
    src = torch.randint(0, vocab_size, (batch_size, src_len))
    src_lengths = torch.tensor([20, 18, 15, 12])
    
    print(f"✅ Модель создана")
    print(f"\n🧪 Генерация ответа:")
    print(f"   Вход: {src.shape}")
    
    # Генерация
    with torch.no_grad():
        generated = model.generate(src, src_lengths, max_length=30)
    
    print(f"\n📤 Сгенерировано:")
    print(f"   Форма: {generated.shape}")
    print(f"   Первая последовательность (первые 10): {generated[0][:10].tolist()}")
    
    print("\n" + "=" * 60)
    print("✅ GENERATE РАБОТАЕТ")
    print("=" * 60)
