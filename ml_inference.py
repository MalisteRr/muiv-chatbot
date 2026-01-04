"""
Inference модуль для генерации ответов с помощью обученной ML модели

Использует обученную Seq2Seq модель для генерации ответов.
Если модель не уверена - fallback на DeepSeek API.
"""

import torch
import os
from typing import Optional, Tuple

from ml.models import (
    Seq2Seq,
    Encoder,
    Decoder,
    SimpleTokenizer,
    ModelConfig
)


class MLModelInference:
    """
    Класс для inference обученной модели
    """
    
    def __init__(
        self,
        model_path: str = None,
        tokenizer_path: str = None,
        device: str = None,
        confidence_threshold: float = 0.6
    ):
        """
        Инициализация inference модуля
        
        Args:
            model_path: Путь к обученной модели
            tokenizer_path: Путь к токенизатору
            device: Устройство (cpu/cuda)
            confidence_threshold: Порог уверенности (0-1)
        """
        self.model_path = model_path or ModelConfig.MODEL_SAVE_PATH
        self.tokenizer_path = tokenizer_path or ModelConfig.TOKENIZER_PATH
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
    
    def load_model(self) -> bool:
        """
        Загрузка модели и токенизатора
        
        Returns:
            True если загрузка успешна
        """
        try:
            # Проверка файлов
            if not os.path.exists(self.model_path):
                print(f"⚠️ Модель не найдена: {self.model_path}")
                return False
            
            if not os.path.exists(self.tokenizer_path):
                print(f"⚠️ Токенизатор не найден: {self.tokenizer_path}")
                return False
            
            # Загружаем токенизатор
            self.tokenizer = SimpleTokenizer.load(self.tokenizer_path)
            
            # Загружаем checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Создаём модель
            encoder = Encoder(
                vocab_size=checkpoint['vocab_size'],
                embedding_dim=checkpoint['embedding_dim'],
                hidden_size=checkpoint['hidden_size'],
                num_layers=checkpoint['num_layers'],
                dropout=0.0
            )
            
            decoder = Decoder(
                vocab_size=checkpoint['vocab_size'],
                embedding_dim=checkpoint['embedding_dim'],
                hidden_size=checkpoint['hidden_size'],
                num_layers=checkpoint['num_layers'],
                dropout=0.0,
                use_attention=True
            )
            
            self.model = Seq2Seq(encoder, decoder, self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.is_loaded = True
            
            print(f"✅ ML модель загружена")
            print(f"   Устройство: {self.device}")
            print(f"   Параметров: {self.model.count_parameters():,}")
            
            return True
            
        except Exception as e:
            print(f"❌ Ошибка загрузки: {e}")
            return False


if __name__ == "__main__":
    """Тест загрузки модели"""
    print("\n" + "=" * 60)
    print("ТЕСТ ML INFERENCE - Загрузка модели")
    print("=" * 60)
    
    inference = MLModelInference(confidence_threshold=0.6)
    success = inference.load_model()
    
    if success:
        print("\n✅ Модель успешно загружена")
    else:
        print("\n❌ Ошибка загрузки")
        print("   Обучите модель: python scripts/train_model.py")
    
    print("\n" + "=" * 60)
