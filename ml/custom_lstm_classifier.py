"""
Собственный LSTM классификатор с Self-Attention
Разработано для ВКР: Синицин М.Д.

Интеграция с существующей системой muiv-chatbot
Может использоваться как альтернатива RuBERT
"""

import logging
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ================== КОНФИГУРАЦИЯ ==================

CATEGORY_MAPPING = {
    "Обучение": 0,
    "Поступление": 1,
    "Общежитие": 2,
    "Формы обучения": 3,
    "Стоимость": 4,
    "Без ЕГЭ": 5,
    "Контакты": 6,
    "Бюджет": 7
}
CATEGORY_NAMES = list(CATEGORY_MAPPING.keys())
ID2LABEL = {v: k for k, v in CATEGORY_MAPPING.items()}


# ================== MODEL CONFIG ==================

@dataclass
class ModelConfig:
    vocab_size: int = 9846
    embedding_dim: int = 100
    hidden_size: int = 256
    num_layers: int = 2
    num_classes: int = 8
    dropout: float = 0.4
    bidirectional: bool = True
    max_seq_length: int = 64
    attention_heads: int = 4
    
    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'ModelConfig':
        with open(path, 'r', encoding='utf-8') as f:
            return cls(**json.load(f))


# ================== TOKENIZER ==================

class TextTokenizer:
    """Токенизатор для LSTM классификатора"""
    
    def __init__(self, max_seq_length: int = 64):
        self.max_seq_length = max_seq_length
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
    
    def encode(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        words = text.lower().split()[:self.max_seq_length]
        ids = [self.word2idx.get(w, 1) for w in words]
        
        pad_len = self.max_seq_length - len(ids)
        attention_mask = [1.0] * len(ids) + [0.0] * pad_len
        ids = ids + [0] * pad_len
        
        return torch.tensor(ids), torch.tensor(attention_mask)
    
    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'word2idx': self.word2idx,
                'max_seq_length': self.max_seq_length
            }, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'TextTokenizer':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        tok = cls(max_seq_length=data['max_seq_length'])
        tok.word2idx = data['word2idx']
        tok.idx2word = {int(v): k for k, v in data['word2idx'].items()}
        return tok


# ================== MODEL ==================

class SelfAttention(nn.Module):
    """Multi-Head Self-Attention механизм"""
    
    def __init__(self, hidden_size: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        B, L, H = x.shape
        
        q = self.query(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, H)
        
        return self.out(out)


class LSTMClassifier(nn.Module):
    """LSTM классификатор с Self-Attention"""
    
    def __init__(self, config: ModelConfig, pretrained_embeddings=None):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        
        self.lstm = nn.LSTM(
            config.embedding_dim,
            config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional
        )
        
        lstm_output_size = config.hidden_size * (2 if config.bidirectional else 1)
        
        self.attention = SelfAttention(lstm_output_size, config.attention_heads, config.dropout)
        self.layer_norm = nn.LayerNorm(lstm_output_size)
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout / 2),
            nn.Linear(64, config.num_classes)
)

    
    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        lstm_out, _ = self.lstm(x)
        
        attn_out = self.attention(lstm_out, attention_mask)
        x = self.layer_norm(lstm_out + attn_out)
        
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)
        
        return self.classifier(x)


# ================== CLASSIFIER WRAPPER ==================

class CustomIntentClassifier:
    """
    Обёртка над LSTM классификатором для совместимости с RuBERT интерфейсом
    """
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.7):
        """
        Инициализация классификатора
        
        Args:
            model_path: Путь к папке с моделью (classifier.pt, tokenizer.json, config.json)
            confidence_threshold: Порог уверенности (0-1)
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.id2label = ID2LABEL
        self.label2id = CATEGORY_MAPPING
        
        self._load_model()
    
    def _load_model(self):
        """Загрузка модели и токенизатора"""
        try:
            logger.info(f"🧠 Загрузка собственной LSTM модели из {self.model_path}")
            
            # Загружаем токенизатор
            tokenizer_path = self.model_path / "tokenizer.json"
            self.tokenizer = TextTokenizer.load(str(tokenizer_path))
            
            # Загружаем конфиг
            config_path = self.model_path / "config.json"
            self.config = ModelConfig.load(str(config_path))
            
            # Загружаем модель
            model_file = self.model_path / "classifier.pt"
            checkpoint = torch.load(model_file, map_location=self.device)
            
            self.model = LSTMClassifier(self.config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"✅ Собственная LSTM модель загружена успешно!")
            logger.info(f"   📋 Категорий: {len(self.id2label)}")
            logger.info(f"   💻 Устройство: {self.device}")
            logger.info(f"   🎯 Порог уверенности: {self.confidence_threshold}")
            logger.info(f"   📊 Параметров: {sum(p.numel() for p in self.model.parameters()):,}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки LSTM модели: {e}", exc_info=True)
            raise
    
    def predict(self, text: str) -> Dict:
        """
        Предсказать категорию для текста
        
        Args:
            text: Текст вопроса пользователя
            
        Returns:
            dict: {
                'category': str,           # Предсказанная категория
                'confidence': float,       # Уверенность (0-1)
                'is_confident': bool,      # Выше ли порога
                'all_scores': dict         # Все категории с вероятностями
            }
        """
        try:
            # Токенизация
            input_ids, attention_mask = self.tokenizer.encode(text)
            input_ids = input_ids.unsqueeze(0).to(self.device)
            attention_mask = attention_mask.unsqueeze(0).to(self.device)
            
            # Предсказание
            with torch.no_grad():
                logits = self.model(input_ids, attention_mask)
                probabilities = torch.softmax(logits, dim=1)[0]
            
            # Получаем предсказанный класс
            predicted_id = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_id].item()
            predicted_category = self.id2label[predicted_id]
            
            # Все вероятности
            all_scores = {
                self.id2label[i]: probabilities[i].item()
                for i in range(len(probabilities))
            }
            
            result = {
                'category': predicted_category,
                'confidence': confidence,
                'is_confident': confidence >= self.confidence_threshold,
                'all_scores': all_scores
            }
            
            logger.debug(
                f"LSTM классификация: '{text[:50]}...' → {predicted_category} "
                f"({confidence*100:.1f}%)"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка предсказания LSTM: {e}", exc_info=True)
            return {
                'category': None,
                'confidence': 0.0,
                'is_confident': False,
                'all_scores': {}
            }
    
    def get_top_categories(self, text: str, top_k: int = 3) -> list:
        """
        Получить топ-K наиболее вероятных категорий
        """
        prediction = self.predict(text)
        sorted_scores = sorted(
            prediction['all_scores'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_scores[:top_k]


# ================== ГЛОБАЛЬНЫЕ ФУНКЦИИ ==================

_custom_classifier_instance: Optional[CustomIntentClassifier] = None


def init_custom_classifier(model_path: str, confidence_threshold: float = 0.7):
    """
    Инициализировать глобальный экземпляр собственного классификатора
    
    Args:
        model_path: Путь к модели
        confidence_threshold: Порог уверенности
    """
    global _custom_classifier_instance
    
    try:
        _custom_classifier_instance = CustomIntentClassifier(model_path, confidence_threshold)
        logger.info("✅ Собственный LSTM классификатор инициализирован")
    except Exception as e:
        logger.error(f"❌ Не удалось инициализировать LSTM классификатор: {e}")
        _custom_classifier_instance = None


def get_custom_classifier() -> Optional[CustomIntentClassifier]:
    """
    Получить глобальный экземпляр собственного классификатора
    """
    return _custom_classifier_instance


def is_custom_classifier_available() -> bool:
    """
    Проверить доступен ли собственный классификатор
    """
    return _custom_classifier_instance is not None