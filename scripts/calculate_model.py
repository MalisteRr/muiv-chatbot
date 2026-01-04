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
    """Вычисление BLEU score (упрощённая версия BLEU-1)"""
    ref_words = set(reference)
    hyp_words = hypothesis
    
    if not hyp_words:
        return 0.0
    
    matches = sum(1 for word in hyp_words if word in ref_words)
    precision = matches / len(hyp_words)
    
    return precision


def compute_exact_match(reference: str, hypothesis: str) -> bool:
    """Проверка точного совпадения"""
    ref_normalized = ' '.join(reference.lower().split())
    hyp_normalized = ' '.join(hypothesis.lower().split())
    
    return ref_normalized == hyp_normalized


def evaluate_model(
    model_path: str,
    tokenizer_path: str,
    test_data_path: str,
    device: str = 'cpu'
):
    """
    Оценка модели на тестовых данных
    
    Args:
        model_path: Путь к модели
        tokenizer_path: Путь к токенизатору
        test_data_path: Путь к тестовым данным
        device: Устройство
    
    Returns:
        Словарь с результатами
    """
    print("\n" + "=" * 70)
    print("ОЦЕНКА КАЧЕСТВА МОДЕЛИ")
    print("=" * 70)
    
    # 1. Загружаем токенизатор
    print(f"\n📚 Загрузка токенизатора...")
    tokenizer = SimpleTokenizer.load(tokenizer_path)
    
    # 2. Загружаем модель
    print(f"🏗️ Загрузка модели...")
    checkpoint = torch.load(model_path, map_location=device)
    
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
    
    model = Seq2Seq(encoder, decoder, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Модель загружена")
    
    # 3. Загружаем тестовые данные
    print(f"\n📦 Загрузка тестовых данных...")
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"   Тестовых примеров: {len(test_data)}")
    
    # 4. Оценка
    print(f"\n🧪 Оценка модели...")
    
    bleu_scores = []
    exact_matches = 0
    total = 0
    
    for idx, item in enumerate(test_data):
        question = item['question']
        reference_answer = item['answer']
        
        # Токенизация
        question_indices = tokenizer.encode(
            question,
            max_length=ModelConfig.MAX_SEQ_LENGTH,
            add_sos=False,
            add_eos=True
        )
        
        question_tensor = torch.LongTensor(question_indices).unsqueeze(0).to(device)
        question_length = torch.LongTensor([sum(1 for idx in question_indices if idx != 0)])
        
        # Генерация
        with torch.no_grad():
            generated_tokens = model.generate(
                question_tensor,
                question_length,
                max_length=100,
                sos_token=2,
                eos_token=3
            )
        
        # Декодирование
        hypothesis_answer = tokenizer.decode(
            generated_tokens[0].cpu().tolist(),
            skip_special=True
        )
        
        # Метрики
        ref_words = reference_answer.lower().split()
        hyp_words = hypothesis_answer.lower().split()
        
        bleu = compute_bleu(ref_words, hyp_words)
        bleu_scores.append(bleu)
        
        if compute_exact_match(reference_answer, hypothesis_answer):
            exact_matches += 1
        
        total += 1
        
        # Прогресс
        if (idx + 1) % 10 == 0:
            print(f"   Обработано {idx + 1}/{len(test_data)}...")
    
    # 5. Результаты
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    accuracy = exact_matches / total if total > 0 else 0.0
    
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ")
    print("=" * 70)
    print(f"📊 Метрики на {total} примерах:")
    print(f"   BLEU Score: {avg_bleu:.4f} ({avg_bleu * 100:.2f}%)")
    print(f"   Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"   Exact Matches: {exact_matches}/{total}")
    
    # Целевые значения
    target_bleu = ModelConfig.TARGET_BLEU
    target_acc = ModelConfig.TARGET_ACCURACY
    
    print(f"\n🎯 Сравнение с целевыми:")
    print(f"   BLEU: {avg_bleu:.4f} vs {target_bleu:.4f} {'✅' if avg_bleu >= target_bleu else '⚠️'}")
    print(f"   Accuracy: {accuracy:.4f} vs {target_acc:.4f} {'✅' if accuracy >= target_acc else '⚠️'}")
    
    # Примеры
    print(f"\n📝 Примеры генерации:")
    for i in range(min(3, len(test_data))):
        item = test_data[i]
        question = item['question']
        reference = item['answer']
        
        question_indices = tokenizer.encode(question, max_length=ModelConfig.MAX_SEQ_LENGTH, add_sos=False, add_eos=True)
        question_tensor = torch.LongTensor(question_indices).unsqueeze(0).to(device)
        question_length = torch.LongTensor([sum(1 for idx in question_indices if idx != 0)])
        
        with torch.no_grad():
            generated_tokens = model.generate(question_tensor, question_length, max_length=100, sos_token=2, eos_token=3)
        
        hypothesis = tokenizer.decode(generated_tokens[0].cpu().tolist(), skip_special=True)
        
        print(f"\n   {i+1}. Вопрос: {question}")
        print(f"      Эталон: {reference}")
        print(f"      Модель: {hypothesis}")
    
    print("\n" + "=" * 70 + "\n")
    
    return {
        'bleu': avg_bleu,
        'accuracy': accuracy,
        'total': total,
        'exact_matches': exact_matches
    }


def main():
    """Основная функция"""
    
    # Пути
    model_path = ModelConfig.MODEL_SAVE_PATH
    tokenizer_path = ModelConfig.TOKENIZER_PATH
    test_data_path = os.path.join(
        os.path.dirname(ModelConfig.DATA_PATH),
        'test_data.json'
    )
    
    # Проверки
    if not os.path.exists(model_path):
        print(f"❌ Модель не найдена: {model_path}")
        print("   Обучите модель: python scripts/train_model.py")
        return
    
    if not os.path.exists(test_data_path):
        print(f"❌ Тестовый датасет не найден: {test_data_path}")
        print("   Запустите: python scripts/prepare_dataset.py")
        return
    
    # Оценка
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = evaluate_model(model_path, tokenizer_path, test_data_path, device)
    
    # Сохранение
    results_path = os.path.join(ModelConfig.CHECKPOINT_DIR, 'evaluation_results.json')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Результаты сохранены: {results_path}")


if __name__ == "__main__":
    main()
