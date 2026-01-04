"""
Скрипт обучения Seq2Seq модели

Запуск обучения модели для генерации ответов на вопросы абитуриентов.

Использование:
    python scripts/train_model.py
"""

import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ml.models import (
    create_seq2seq_model,
    create_dataloaders,
    create_trainer,
    SimpleTokenizer,
    ModelConfig,
    split_dataset
)


def check_prerequisites():
    """Проверка наличия необходимых файлов"""
    print(f"\n📁 Проверка данных...")
    
    if not os.path.exists(ModelConfig.DATA_PATH):
        print(f"❌ Датасет не найден: {ModelConfig.DATA_PATH}")
        print("   Запустите сначала:")
        print("   1. python scripts/prepare_dataset.py")
        print("   2. python scripts/build_vocabulary.py")
        return False
    
    if not os.path.exists(ModelConfig.TOKENIZER_PATH):
        print(f"❌ Токенизатор не найден: {ModelConfig.TOKENIZER_PATH}")
        print("   Запустите: python scripts/build_vocabulary.py")
        return False
    
    print(f"✅ Все необходимые файлы найдены")
    return True


def prepare_data_splits():
    """Разделение датасета на train/val/test"""
    print(f"\n✂️ Подготовка разделения данных...")
    
    train_path = os.path.join(os.path.dirname(ModelConfig.DATA_PATH), 'train_data.json')
    val_path = os.path.join(os.path.dirname(ModelConfig.DATA_PATH), 'val_data.json')
    test_path = os.path.join(os.path.dirname(ModelConfig.DATA_PATH), 'test_data.json')
    
    if os.path.exists(train_path) and os.path.exists(val_path):
        print(f"✅ Датасет уже разделён")
        return train_path, val_path, test_path
    
    print(f"   Разделение датасета...")
    train_path, val_path, test_path = split_dataset(
        ModelConfig.DATA_PATH,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        save_splits=True
    )
    
    return train_path, val_path, test_path


def load_tokenizer_and_data(train_path, val_path):
    """
    Загрузка токенизатора и создание DataLoader'ов
    
    Returns:
        (tokenizer, train_loader, val_loader, vocab_size)
    """
    # Загружаем токенизатор
    print(f"\n📚 Загрузка токенизатора...")
    tokenizer = SimpleTokenizer.load(ModelConfig.TOKENIZER_PATH)
    vocab_size = tokenizer.get_vocab_size()
    print(f"   Размер словаря: {vocab_size}")
    
    # Создаём DataLoader'ы
    print(f"\n📦 Создание DataLoader'ов...")
    train_loader, val_loader = create_dataloaders(
        train_path=train_path,
        val_path=val_path,
        tokenizer=tokenizer,
        batch_size=ModelConfig.BATCH_SIZE,
        max_length=ModelConfig.MAX_SEQ_LENGTH,
        num_workers=0
    )
    
    return tokenizer, train_loader, val_loader, vocab_size


def create_model_and_trainer(vocab_size, device):
    """
    Создание модели и trainer'а
    
    Returns:
        (model, trainer)
    """
    # Создаём модель
    print(f"\n🏗️ Создание модели...")
    model = create_seq2seq_model(
        vocab_size=vocab_size,
        embedding_dim=ModelConfig.EMBEDDING_DIM,
        hidden_size=ModelConfig.HIDDEN_SIZE,
        num_layers=ModelConfig.NUM_LAYERS,
        dropout=ModelConfig.DROPOUT,
        use_attention=True,
        device=device
    )
    
    # Создаём Trainer
    print(f"\n🎓 Создание Trainer...")
    trainer = create_trainer(
        model=model,
        learning_rate=ModelConfig.LEARNING_RATE,
        device=device
    )
    
    return model, trainer


def main():
    """Основная функция обучения"""
    print("\n" + "=" * 70)
    print("ОБУЧЕНИЕ SEQ2SEQ МОДЕЛИ ДЛЯ ЧАТ-БОТА МУИВ")
    print("=" * 70)
    
    # 1. Устройство
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️ Устройство: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # 2. Проверки
    if not check_prerequisites():
        return
    
    # 3. Данные
    train_path, val_path, test_path = prepare_data_splits()
    
    # 4. Загрузка
    tokenizer, train_loader, val_loader, vocab_size = load_tokenizer_and_data(
        train_path, val_path
    )
    
    # 5. Модель
    model, trainer = create_model_and_trainer(vocab_size, device)
    
    print("\n✅ Подготовка завершена")
    print("   Готово к обучению!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Прервано пользователем")
    except Exception as e:
        print(f"\n\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
