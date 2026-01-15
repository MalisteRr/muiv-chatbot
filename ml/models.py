"""
LSTM Seq2Seq модель для генерации ответов на русском языке
"""

import torch
import torch.nn as nn
import json
from typing import List, Dict
from pathlib import Path


class ModelConfig:
    """Конфигурация модели"""
    MAX_SEQ_LENGTH = 50
    VOCAB_SIZE = 10000
    EMBEDDING_DIM = 128
    HIDDEN_SIZE = 256
    NUM_LAYERS = 2
    DROPOUT = 0.3
    
    # Пути к файлам (адаптируй под свою структуру)
    MODEL_SAVE_PATH = "ml/models/lstm_model.pt"
    TOKENIZER_PATH = "ml/models/tokenizer.json"


class SimpleTokenizer:
    """
    Простой токенизатор для русского языка
    """
    
    def __init__(self):
        self.word2idx = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<SOS>': 2,
            '<EOS>': 3
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
    
    def fit(self, texts: List[str]):
        """Построить словарь на основе текстов"""
        words = set()
        for text in texts:
            words.update(text.lower().split())
        
        for word in sorted(words):
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        
        self.vocab_size = len(self.word2idx)
    
    def encode(self, text: str, max_length: int = 50, add_sos: bool = False, add_eos: bool = True) -> List[int]:
        """Закодировать текст в индексы"""
        words = text.lower().split()
        indices = []
        
        if add_sos:
            indices.append(self.word2idx['<SOS>'])
        
        for word in words[:max_length]:
            indices.append(self.word2idx.get(word, self.word2idx['<UNK>']))
        
        if add_eos:
            indices.append(self.word2idx['<EOS>'])
        
        # Padding
        while len(indices) < max_length:
            indices.append(self.word2idx['<PAD>'])
        
        return indices[:max_length]
    
    def decode(self, indices: List[int], skip_special: bool = True) -> str:
        """Декодировать индексы обратно в текст"""
        words = []
        special_tokens = {'<PAD>', '<UNK>', '<SOS>', '<EOS>'}
        
        for idx in indices:
            word = self.idx2word.get(idx, '<UNK>')
            if skip_special and word in special_tokens:
                continue
            words.append(word)
        
        return ' '.join(words)
    
    def get_vocab_size(self) -> int:
        """Размер словаря"""
        return self.vocab_size
    
    def save(self, path: str):
        """Сохранить токенизатор"""
        data = {
            'word2idx': self.word2idx,
            'idx2word': {str(k): v for k, v in self.idx2word.items()},
            'vocab_size': self.vocab_size
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """Загрузить токенизатор"""
        tokenizer = cls()
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            tokenizer.word2idx = data['word2idx']
            tokenizer.idx2word = {int(k): v for k, v in data['idx2word'].items()}
            tokenizer.vocab_size = data['vocab_size']
        except FileNotFoundError:
            # Если токенизатора нет, используем базовый
            pass
        
        return tokenizer


class Encoder(nn.Module):
    """
    Encoder для Seq2Seq модели
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, 
                 num_layers: int, dropout: float = 0.3):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, lengths):
        """
        Args:
            x: [batch_size, seq_len]
            lengths: [batch_size]
        """
        embedded = self.dropout(self.embedding(x))
        
        # Pack sequences
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        outputs, (hidden, cell) = self.lstm(packed)
        
        # Unpack
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        return outputs, hidden, cell


class Decoder(nn.Module):
    """
    Decoder для Seq2Seq модели
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int,
                 num_layers: int, dropout: float = 0.3):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, hidden, cell):
        """
        Args:
            x: [batch_size, 1]
            hidden: [num_layers, batch_size, hidden_size]
            cell: [num_layers, batch_size, hidden_size]
        """
        embedded = self.dropout(self.embedding(x))
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc(output.squeeze(1))
        
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    """
    Sequence-to-Sequence модель с LSTM
    """
    
    def __init__(self, encoder: Encoder, decoder: Decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src, src_len, trg, teacher_forcing_ratio: float = 0.5):
        """
        Args:
            src: [batch_size, src_len]
            src_len: [batch_size]
            trg: [batch_size, trg_len]
        """
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.vocab_size
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # Encoder
        _, hidden, cell = self.encoder(src, src_len)
        
        # Первый вход декодера - <SOS> токен
        input_token = trg[:, 0].unsqueeze(1)
        
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input_token, hidden, cell)
            outputs[:, t] = output
            
            # Teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_token = trg[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
        
        return outputs
    
    def generate(self, src, src_len, max_length: int = 50, sos_token: int = 2, eos_token: int = 3):
        """
        Генерация последовательности (inference)
        
        Args:
            src: [batch_size, src_len]
            src_len: [batch_size]
            max_length: Максимальная длина генерации
            sos_token: Индекс <SOS> токена
            eos_token: Индекс <EOS> токена
        
        Returns:
            generated: [batch_size, max_length]
        """
        self.eval()
        batch_size = src.shape[0]
        
        with torch.no_grad():
            # Encoder
            _, hidden, cell = self.encoder(src, src_len)
            
            # Начинаем с <SOS>
            input_token = torch.LongTensor([[sos_token]]).to(self.device)
            generated = [sos_token]
            
            for _ in range(max_length):
                output, hidden, cell = self.decoder(input_token, hidden, cell)
                top1 = output.argmax(1).item()
                
                generated.append(top1)
                
                if top1 == eos_token:
                    break
                
                input_token = torch.LongTensor([[top1]]).to(self.device)
        
        return torch.LongTensor([generated])


# Вспомогательные функции для сохранения/загрузки

def save_model(model, optimizer, epoch, path):
    """Сохранить модель"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'encoder_config': {
            'vocab_size': model.encoder.embedding.num_embeddings,
            'embedding_dim': model.encoder.embedding.embedding_dim,
            'hidden_size': model.encoder.lstm.hidden_size,
            'num_layers': model.encoder.lstm.num_layers
        },
        'decoder_config': {
            'vocab_size': model.decoder.vocab_size,
            'embedding_dim': model.decoder.embedding.embedding_dim,
            'hidden_size': model.decoder.lstm.hidden_size,
            'num_layers': model.decoder.lstm.num_layers
        }
    }, path)


def load_model(path, device):
    """Загрузить модель"""
    checkpoint = torch.load(path, map_location=device)
    
    encoder_config = checkpoint['encoder_config']
    decoder_config = checkpoint['decoder_config']
    
    encoder = Encoder(
        vocab_size=encoder_config['vocab_size'],
        embedding_dim=encoder_config['embedding_dim'],
        hidden_size=encoder_config['hidden_size'],
        num_layers=encoder_config['num_layers'],
        dropout=0.0
    )
    
    decoder = Decoder(
        vocab_size=decoder_config['vocab_size'],
        embedding_dim=decoder_config['embedding_dim'],
        hidden_size=decoder_config['hidden_size'],
        num_layers=decoder_config['num_layers'],
        dropout=0.0
    )
    
    model = Seq2Seq(encoder, decoder, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint['epoch']
