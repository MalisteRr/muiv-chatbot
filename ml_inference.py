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
    """Класс для inference обученной модели"""
    
    def __init__(
        self,
        model_path: str = None,
        tokenizer_path: str = None,
        device: str = None,
        confidence_threshold: float = 0.6
    ):
        self.model_path = model_path or ModelConfig.MODEL_SAVE_PATH
        self.tokenizer_path = tokenizer_path or ModelConfig.TOKENIZER_PATH
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
    
    def load_model(self) -> bool:
        """Загрузка модели и токенизатора"""
        try:
            if not os.path.exists(self.model_path):
                print(f"⚠️ Модель не найдена: {self.model_path}")
                return False
            
            if not os.path.exists(self.tokenizer_path):
                print(f"⚠️ Токенизатор не найден: {self.tokenizer_path}")
                return False
            
            self.tokenizer = SimpleTokenizer.load(self.tokenizer_path)
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
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
            print(f"✅ ML модель загружена ({self.device})")
            
            return True
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            return False
    
    def generate_answer(self, question: str, max_length: int = 100) -> Tuple[Optional[str], float]:
        """
        Генерация ответа на вопрос
        
        Args:
            question: Вопрос пользователя
            max_length: Максимальная длина ответа
        
        Returns:
            (answer, confidence): Ответ и уверенность (0-1)
        """
        if not self.is_loaded:
            return None, 0.0
        
        try:
            # Токенизация
            question_indices = self.tokenizer.encode(
                question,
                max_length=ModelConfig.MAX_SEQ_LENGTH,
                add_sos=False,
                add_eos=True
            )
            
            # Тензоры
            question_tensor = torch.LongTensor(question_indices).unsqueeze(0).to(self.device)
            question_length = torch.LongTensor([sum(1 for idx in question_indices if idx != 0)])
            
            # Генерация
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    question_tensor,
                    question_length,
                    max_length=max_length,
                    sos_token=2,
                    eos_token=3
                )
            
            # Декодирование
            answer = self.tokenizer.decode(
                generated_tokens[0].cpu().tolist(),
                skip_special=True
            )
            
            # Упрощённая уверенность (по длине)
            answer_length = len(answer.split())
            confidence = min(answer_length / 10.0, 1.0) if answer_length > 3 else 0.3
            
            return answer, confidence
            
        except Exception as e:
            print(f"❌ Ошибка генерации: {e}")
            return None, 0.0
    
    def should_use_ml(self, confidence: float) -> bool:
        """Решение использовать ли ML ответ"""
        return confidence >= self.confidence_threshold


# Глобальный экземпляр
ml_inference = MLModelInference(confidence_threshold=0.6)


def initialize_ml_model() -> bool:
    """
    Инициализация ML модели при старте бота
    
    Returns:
        True если модель загружена
    """
    return ml_inference.load_model()


def get_ml_answer(question: str) -> Tuple[Optional[str], bool]:
    """
    Получение ответа от ML модели
    
    Args:
        question: Вопрос пользователя
    
    Returns:
        (answer, use_ml): Ответ и флаг использовать ли его
                         (None, False) если fallback на API
    """
    if not ml_inference.is_loaded:
        return None, False
    
    answer, confidence = ml_inference.generate_answer(question)
    
    if answer and ml_inference.should_use_ml(confidence):
        return answer, True
    
    return None, False


if __name__ == "__main__":
    """Тест генерации"""
    print("\n" + "=" * 60)
    print("ТЕСТ ML INFERENCE")
    print("=" * 60)
    
    success = initialize_ml_model()
    
    if success:
        test_questions = [
            "Сколько стоит обучение?",
            "Какие документы нужны?",
            "Есть ли бюджетные места?"
        ]
        
        print("\n🧪 Тестирование:")
        for q in test_questions:
            print(f"\n❓ {q}")
            answer, use_ml = get_ml_answer(q)
            
            if use_ml:
                print(f"✅ ML: {answer}")
            else:
                print(f"⚠️ Fallback на API")
    else:
        print("\n❌ Модель не загружена")
    
    print("\n" + "=" * 60)
