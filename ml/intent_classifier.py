"""
Классификатор намерений на основе RuBERT
Используется для определения категории вопроса перед поиском в FAQ
"""

import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Optional
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class IntentClassifier:
    """
    Классификатор намерений пользователя на базе обученного RuBERT
    """
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.7):
        """
        Инициализация классификатора
        
        Args:
            model_path: Путь к обученной модели RuBERT
            confidence_threshold: Порог уверенности (0-1)
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self._load_model()
        
    def _load_model(self):
        """Загрузка модели и токенизатора"""
        try:
            logger.info(f"🤖 Загрузка RuBERT модели из {self.model_path}")
            
            # Загружаем токенизатор и модель
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()  # Режим инференса
            
            # Загружаем маппинг категорий
            label_mapping_path = self.model_path.parent / "label_mapping.json"
            if label_mapping_path.exists():
                with open(label_mapping_path, 'r', encoding='utf-8') as f:
                    mapping = json.load(f)
                    self.id2label = {int(k): v for k, v in mapping['id2label'].items()}
                    self.label2id = mapping['label2id']
            else:
                # Берём из конфига модели
                self.id2label = self.model.config.id2label
                self.label2id = self.model.config.label2id
            
            logger.info(f"✅ RuBERT модель загружена успешно!")
            logger.info(f"   📋 Категорий: {len(self.id2label)}")
            logger.info(f"   💻 Устройство: {self.device}")
            logger.info(f"   🎯 Порог уверенности: {self.confidence_threshold}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели RuBERT: {e}", exc_info=True)
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
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=128
            ).to(self.device)
            
            # Предсказание
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)[0]
            
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
                f"Классификация: '{text[:50]}...' → {predicted_category} "
                f"({confidence*100:.1f}%)"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка предсказания: {e}", exc_info=True)
            return {
                'category': None,
                'confidence': 0.0,
                'is_confident': False,
                'all_scores': {}
            }
    
    def get_top_categories(self, text: str, top_k: int = 3) -> list:
        """
        Получить топ-K наиболее вероятных категорий
        
        Args:
            text: Текст вопроса
            top_k: Количество топ категорий
            
        Returns:
            list: [(category, confidence), ...]
        """
        prediction = self.predict(text)
        sorted_scores = sorted(
            prediction['all_scores'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_scores[:top_k]


# Глобальный экземпляр (будет инициализирован при старте бота)
_classifier_instance: Optional[IntentClassifier] = None


def init_classifier(model_path: str, confidence_threshold: float = 0.7):
    """
    Инициализировать глобальный экземпляр классификатора
    
    Args:
        model_path: Путь к модели
        confidence_threshold: Порог уверенности
    """
    global _classifier_instance
    
    try:
        _classifier_instance = IntentClassifier(model_path, confidence_threshold)
        logger.info("✅ RuBERT классификатор инициализирован")
    except Exception as e:
        logger.error(f"❌ Не удалось инициализировать RuBERT: {e}")
        logger.warning("⚠️ Бот будет работать без RuBERT классификатора (только DeepSeek)")
        _classifier_instance = None


def get_classifier() -> Optional[IntentClassifier]:
    """
    Получить глобальный экземпляр классификатора
    
    Returns:
        Классификатор или None если не инициализирован
    """
    return _classifier_instance


def is_classifier_available() -> bool:
    """
    Проверить доступен ли классификатор
    
    Returns:
        True если классификатор загружен и доступен
    """
    return _classifier_instance is not None
