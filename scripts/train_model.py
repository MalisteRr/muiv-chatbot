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
        print("   Запустите: python scripts/prepare_dataset.py")
        return False
    
    if not os.path.exists(ModelConfig.TOKENIZER_PATH):
        print(f"❌ Токенизатор не найден: {ModelConfig.TOKENIZER_PATH}")
        print("   Запустите: python scripts/build_vocabulary.py")
        return False
    
    print(f"✅ Все файлы найдены")
    return True


def prepare_data_splits():
    """Разделение датасета на train/val/test"""
    train_path = os.path.join(os.path.dirname(ModelConfig.DATA_PATH), 'train_data.json')
    val_path = os.path.join(os.path.dirname(ModelConfig.DATA_PATH), 'val_data.json')
    
    if not os.path.exists(train_path):
        print(f"\n✂️ Разделение датасета...")
        train_path, val_path, _ = split_dataset(
            ModelConfig.DATA_PATH,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            save_splits=True
        )
    else:
        print(f"\n✅ Датасет уже разделён")
    
    return train_path, val_path


def load_tokenizer_and_data(train_path, val_path):
    """Загрузка токенизатора и DataLoader'ов"""
    print(f"\n📚 Загрузка токенизатора...")
    tokenizer = SimpleTokenizer.load(ModelConfig.TOKENIZER_PATH)
    vocab_size = tokenizer.get_vocab_size()
    print(f"   Размер словаря: {vocab_size}")
    
    print(f"\n📦 Создание DataLoader'ов...")
    train_loader, val_loader = create_dataloaders(
        train_path=train_path,
        val_path=val_path,
        tokenizer=tokenizer,
        batch_size=ModelConfig.BATCH_SIZE,
        max_length=ModelConfig.MAX_SEQ_LENGTH,
        num_workers=0
    )
    
    return train_loader, val_loader, vocab_size


def create_model_and_trainer(vocab_size, device):
    """Создание модели и trainer'а"""
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
    
    print(f"\n🎓 Создание Trainer...")
    trainer = create_trainer(
        model=model,
        learning_rate=ModelConfig.LEARNING_RATE,
        device=device
    )
    
    return model, trainer


def train_model(model, trainer, train_loader, val_loader):
    """
    Запуск процесса обучения
    """
    print(f"\n🚀 Начало обучения...")
    print(f"   Эпох: {ModelConfig.NUM_EPOCHS}")
    print(f"   Batch size: {ModelConfig.BATCH_SIZE}")
    print(f"   Learning rate: {ModelConfig.LEARNING_RATE}")
    
    # Создаём директорию для чекпоинтов
    os.makedirs(ModelConfig.CHECKPOINT_DIR, exist_ok=True)
    
    # Обучение
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=ModelConfig.NUM_EPOCHS,
        teacher_forcing_ratio=ModelConfig.TEACHER_FORCING_RATIO,
        save_dir=ModelConfig.CHECKPOINT_DIR,
        early_stopping_patience=3
    )
    
    return trainer


def save_final_model(model, trainer, vocab_size):
    """
    Сохранение финальной модели
    """
    print(f"\n💾 Сохранение финальной модели...")
    final_model_path = ModelConfig.MODEL_SAVE_PATH
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'embedding_dim': ModelConfig.EMBEDDING_DIM,
        'hidden_size': ModelConfig.HIDDEN_SIZE,
        'num_layers': ModelConfig.NUM_LAYERS,
        'dropout': ModelConfig.DROPOUT,
        'train_losses': trainer.train_losses,
        'val_losses': trainer.val_losses
    }, final_model_path)
    
    print(f"✅ Модель сохранена: {final_model_path}")
    
    return final_model_path


def main():
    """Основная функция"""
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
    train_path, val_path = prepare_data_splits()
    train_loader, val_loader, vocab_size = load_tokenizer_and_data(train_path, val_path)
    
    # 4. Модель
    model, trainer = create_model_and_trainer(vocab_size, device)
    
    # 5. Обучение
    trainer = train_model(model, trainer, train_loader, val_loader)
    
    # 6. Сохранение
    final_path = save_final_model(model, trainer, vocab_size)
    
    # 7. Итоги
    print("\n" + "=" * 70)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("=" * 70)
    print(f"✅ Финальная модель: {final_path}")
    print(f"✅ Лучшая модель: {os.path.join(ModelConfig.CHECKPOINT_DIR, 'best_model.pt')}")
    print(f"📊 Лучший Val Loss: {trainer.best_val_loss:.4f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Обучение прервано")
    except Exception as e:
        print(f"\n\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
