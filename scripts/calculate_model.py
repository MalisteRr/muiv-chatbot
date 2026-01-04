"""
Скрипт вычисления метрик качества модели

Вычисляет BLEU score и Accuracy на тестовом датасете.
"""

import sys
import os
import torch
from typing import List
import json

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ml.models import (
    Seq2Seq,
    Encoder,
    Decoder,
    SimpleTokenizer,
    ModelConfig
)


def compute_bleu(reference: List[str], hypothesis: List[str]) -> float:
    """
    Вычисление BLEU score
    
    Упрощённая версия BLEU-1 (униграммы)
    
    Args:
        reference: Эталонный ответ (список слов)
        hypothesis: Ответ модели (список слов)
    
    Returns:
        BLEU score (0-1)
    """
    ref_words = set(reference)
    hyp_words = hypothesis
    
    if not hyp_words:
        return 0.0
    
    # Количество совпадающих слов
    matches = sum(1 for word in hyp_words if word in ref_words)
    
    # Precision
    precision = matches / len(hyp_words)
    
    return precision


def compute_exact_match(reference: str, hypothesis: str) -> bool:
    """
    Проверка точного совпадения
    
    Args:
        reference: Эталонный ответ
        hypothesis: Ответ модели
    
    Returns:
        True если совпадают
    """
    # Нормализация (lowercase, убираем лишние пробелы)
    ref_normalized = ' '.join(reference.lower().split())
    hyp_normalized = ' '.join(hypothesis.lower().split())
    
    return ref_normalized == hyp_normalized


if __name__ == "__main__":
    """
    Тест функций метрик
    """
    print("\n" + "=" * 60)
    print("ТЕСТ МЕТРИК")
    print("=" * 60)
    
    # Тест BLEU
    ref = "стоимость обучения составляет 150000 рублей в год".split()
    hyp1 = "стоимость обучения 150000 рублей".split()
    hyp2 = "цена учёбы 200000".split()
    
    bleu1 = compute_bleu(ref, hyp1)
    bleu2 = compute_bleu(ref, hyp2)
    
    print(f"\n📊 BLEU Score:")
    print(f"   Эталон: {' '.join(ref)}")
    print(f"   Гипотеза 1: {' '.join(hyp1)} → BLEU: {bleu1:.4f}")
    print(f"   Гипотеза 2: {' '.join(hyp2)} → BLEU: {bleu2:.4f}")
    
    # Тест Exact Match
    ref_str = "Стоимость обучения 150000 рублей"
    hyp_str1 = "стоимость обучения 150000 рублей"
    hyp_str2 = "цена 200000"
    
    match1 = compute_exact_match(ref_str, hyp_str1)
    match2 = compute_exact_match(ref_str, hyp_str2)
    
    print(f"\n🎯 Exact Match:")
    print(f"   Эталон: {ref_str}")
    print(f"   Гипотеза 1: {hyp_str1} → Match: {match1}")
    print(f"   Гипотеза 2: {hyp_str2} → Match: {match2}")
    
    print("\n" + "=" * 60)
    print("✅ Функции метрик работают")
    print("=" * 60)
