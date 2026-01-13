#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ОБУЧЕНИЕ RuBERT ДЛЯ КЛАССИФИКАЦИИ НАМЕРЕНИЙ
Для дипломной работы - Чат-бот для абитуриентов
"""

import json
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

class RuBERTTrainer:
    """
    Класс для обучения RuBERT на классификацию намерений
    """
    
    def __init__(self, data_file, model_name='DeepPavlov/rubert-base-cased'):
        print("="*70)
        print("🎓 ОБУЧЕНИЕ RuBERT ДЛЯ КЛАССИФИКАЦИИ НАМЕРЕНИЙ")
        print("="*70)
        
        self.data_file = data_file
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"\n🔧 Настройки:")
        print(f"   - Модель: {model_name}")
        print(f"   - Устройство: {self.device}")
        print(f"   - Данные: {data_file}")
        
        # Создаём папку для результатов
        self.output_dir = f"rubert_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/plots", exist_ok=True)
        
        print(f"   - Выходная папка: {self.output_dir}")
    
    def load_data(self):
        """
        Загружает и подготавливает данные
        """
        print("\n📂 Загружаю данные...")
        
        with open(self.data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ Загружено {len(data)} примеров")
        
        # Создаём DataFrame
        df = pd.DataFrame(data)
        
        # Статистика
        print(f"\n📊 Статистика данных:")
        print(f"   - Всего примеров: {len(df)}")
        print(f"   - Уникальных категорий: {df['category'].nunique()}")
        print(f"   - Оригинальных FAQ: {len(df[df['source'] == 'original'])}")
        print(f"   - Сгенерированных: {len(df[df['source'] != 'original'])}")
        
        print(f"\n📋 Распределение по категориям:")
        category_counts = df['category'].value_counts()
        for cat, count in category_counts.items():
            print(f"   - {cat}: {count} примеров ({count/len(df)*100:.1f}%)")
        
        # Проверка на дисбаланс
        max_count = category_counts.max()
        min_count = category_counts.min()
        imbalance_ratio = max_count / min_count
        
        if imbalance_ratio > 5:
            print(f"\n⚠️ ВНИМАНИЕ: Дисбаланс классов {imbalance_ratio:.1f}x")
            print("   Рекомендую добавить больше примеров в маленькие категории")
        else:
            print(f"\n✅ Баланс классов хороший ({imbalance_ratio:.1f}x)")
        
        return df
    
    def prepare_labels(self, df):
        """
        Подготавливает метки классов
        """
        print("\n🏷️ Подготавливаю метки...")
        
        # Создаём маппинг категорий в числа
        categories = sorted(df['category'].unique())
        self.label2id = {cat: i for i, cat in enumerate(categories)}
        self.id2label = {i: cat for cat, i in self.label2id.items()}
        
        # Добавляем числовые метки
        df['label'] = df['category'].map(self.label2id)
        
        print(f"✅ Создано {len(categories)} классов:")
        for cat, idx in self.label2id.items():
            print(f"   {idx}: {cat}")
        
        # Сохраняем маппинг
        with open(f"{self.output_dir}/label_mapping.json", 'w', encoding='utf-8') as f:
            json.dump({
                'label2id': self.label2id,
                'id2label': self.id2label
            }, f, ensure_ascii=False, indent=2)
        
        return df
    
    def split_data(self, df, test_size=0.2, val_size=0.1):
        """
        Разбивает данные на train/val/test
        """
        print(f"\n✂️ Разбиваю данные (train/val/test)...")
        
        # Сначала отделяем test
        train_val, test = train_test_split(
            df,
            test_size=test_size,
            stratify=df['label'],
            random_state=42
        )
        
        # Потом отделяем validation от train
        val_size_adjusted = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val,
            test_size=val_size_adjusted,
            stratify=train_val['label'],
            random_state=42
        )
        
        print(f"✅ Разбивка:")
        print(f"   - Train: {len(train)} примеров ({len(train)/len(df)*100:.1f}%)")
        print(f"   - Val: {len(val)} примеров ({len(val)/len(df)*100:.1f}%)")
        print(f"   - Test: {len(test)} примеров ({len(test)/len(df)*100:.1f}%)")
        
        return train, val, test
    
    def tokenize_data(self, train, val, test):
        """
        Токенизирует данные
        """
        print(f"\n🔤 Токенизирую данные...")
        
        # Загружаем токенизатор
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Токенизируем
        def tokenize_function(examples):
            return self.tokenizer(
                examples['question'].tolist(),
                padding='max_length',
                truncation=True,
                max_length=128
            )
        
        print("   - Train...")
        train_encodings = tokenize_function(train)
        train_dataset = self.create_dataset(train_encodings, train['label'].tolist())
        
        print("   - Val...")
        val_encodings = tokenize_function(val)
        val_dataset = self.create_dataset(val_encodings, val['label'].tolist())
        
        print("   - Test...")
        test_encodings = tokenize_function(test)
        test_dataset = self.create_dataset(test_encodings, test['label'].tolist())
        
        print("✅ Токенизация завершена!")
        
        return train_dataset, val_dataset, test_dataset
    
    def create_dataset(self, encodings, labels):
        """
        Создаёт PyTorch Dataset
        """
        class IntentDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels
            
            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item
            
            def __len__(self):
                return len(self.labels)
        
        return IntentDataset(encodings, labels)
    
    def load_model(self):
        """
        Загружает модель RuBERT
        """
        print(f"\n🤖 Загружаю модель {self.model_name}...")
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id
        )
        
        self.model.to(self.device)
        
        # Считаем параметры
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"✅ Модель загружена!")
        print(f"   - Всего параметров: {total_params:,}")
        print(f"   - Обучаемых: {trainable_params:,}")
    
    def compute_metrics(self, eval_pred):
        """
        Вычисляет метрики для валидации
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        
        return {
            'accuracy': accuracy,
            'f1': f1
        }
    
    def train_model(self, train_dataset, val_dataset, epochs=5, batch_size=16, learning_rate=2e-5):
        """
        Обучает модель
        """
        print(f"\n🎯 Начинаю обучение...")
        print(f"   - Эпох: {epochs}")
        print(f"   - Batch size: {batch_size}")
        print(f"   - Learning rate: {learning_rate}")
        
        # Настройки обучения
        training_args = TrainingArguments(
            output_dir=f"{self.output_dir}/checkpoints",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            eval_strategy="epoch",  # Изменено с evaluation_strategy
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=10,
            save_total_limit=2,
            fp16=self.device == 'cuda',  # Быстрее на GPU
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
        )
        
        # Обучение
        print("\n" + "="*70)
        print("🚀 ТРЕНИРОВКА НАЧАЛАСЬ!")
        print("="*70)
        
        train_result = trainer.train()
        
        print("\n" + "="*70)
        print("✅ ТРЕНИРОВКА ЗАВЕРШЕНА!")
        print("="*70)
        
        # Сохраняем модель
        trainer.save_model(f"{self.output_dir}/final_model")
        self.tokenizer.save_pretrained(f"{self.output_dir}/final_model")
        
        print(f"\n💾 Модель сохранена в: {self.output_dir}/final_model")
        
        return trainer, train_result
    
    def evaluate_model(self, trainer, test_dataset, test_df):
        """
        Оценивает модель на тестовых данных
        """
        print(f"\n📊 Оцениваю модель на тестовых данных...")
        
        # Предсказания
        predictions = trainer.predict(test_dataset)
        pred_labels = np.argmax(predictions.predictions, axis=1)
        true_labels = predictions.label_ids
        
        # Метрики
        accuracy = accuracy_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels, average='weighted')
        
        print(f"\n🎯 ФИНАЛЬНЫЕ МЕТРИКИ:")
        print(f"   - Accuracy: {accuracy*100:.2f}%")
        print(f"   - F1-score: {f1*100:.2f}%")
        
        # Детальный отчёт
        print(f"\n📋 Детальный отчёт по классам:")
        report = classification_report(
            true_labels,
            pred_labels,
            target_names=[self.id2label[i] for i in range(len(self.id2label))],
            digits=3
        )
        print(report)
        
        # Сохраняем отчёт
        with open(f"{self.output_dir}/evaluation_report.txt", 'w', encoding='utf-8') as f:
            f.write(f"ФИНАЛЬНЫЕ МЕТРИКИ\n")
            f.write(f"="*50 + "\n")
            f.write(f"Accuracy: {accuracy*100:.2f}%\n")
            f.write(f"F1-score: {f1*100:.2f}%\n\n")
            f.write(f"ДЕТАЛЬНЫЙ ОТЧЁТ\n")
            f.write(f"="*50 + "\n")
            f.write(report)
        
        # Confusion Matrix
        self.plot_confusion_matrix(true_labels, pred_labels)
        
        # Примеры ошибок
        self.show_errors(test_df, true_labels, pred_labels)
        
        return accuracy, f1
    
    def plot_confusion_matrix(self, true_labels, pred_labels):
        """
        Строит confusion matrix
        """
        print(f"\n📊 Строю confusion matrix...")
        
        cm = confusion_matrix(true_labels, pred_labels)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=[self.id2label[i] for i in range(len(self.id2label))],
            yticklabels=[self.id2label[i] for i in range(len(self.id2label))]
        )
        plt.title('Confusion Matrix', fontsize=16)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/plots/confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Сохранено: {self.output_dir}/plots/confusion_matrix.png")
    
    def show_errors(self, test_df, true_labels, pred_labels, n=10):
        """
        Показывает примеры ошибок
        """
        print(f"\n❌ Примеры ошибок модели (первые {n}):")
        
        test_df = test_df.reset_index(drop=True)
        errors = []
        
        for i, (true, pred) in enumerate(zip(true_labels, pred_labels)):
            if true != pred:
                errors.append({
                    'question': test_df.iloc[i]['question'],
                    'true_category': self.id2label[true],
                    'predicted_category': self.id2label[pred]
                })
        
        print(f"\n   Всего ошибок: {len(errors)} из {len(true_labels)} ({len(errors)/len(true_labels)*100:.1f}%)")
        
        for i, error in enumerate(errors[:n], 1):
            print(f"\n   {i}. Вопрос: {error['question'][:80]}...")
            print(f"      Истинная: {error['true_category']}")
            print(f"      Предсказана: {error['predicted_category']}")
        
        # Сохраняем все ошибки
        with open(f"{self.output_dir}/errors.json", 'w', encoding='utf-8') as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Все ошибки сохранены: {self.output_dir}/errors.json")
    
    def test_predictions(self):
        """
        Тестирует модель на новых примерах
        """
        print(f"\n🧪 ТЕСТИРУЕМ МОДЕЛЬ НА НОВЫХ ПРИМЕРАХ:")
        print("="*70)
        
        test_questions = [
            "Сколько стоит учёба?",
            "Какие документы нужны?",
            "Есть ли общежитие?",
            "Можно без ЕГЭ поступить?",
            "Когда начинается приём документов?",
            "Где находится университет?",
            "Есть ли бюджетные места?",
            "Можно заочно учиться?",
        ]
        
        for question in test_questions:
            inputs = self.tokenizer(
                question,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=128
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.softmax(outputs.logits, dim=1)
                predicted_class = torch.argmax(predictions, dim=1).item()
                confidence = predictions[0][predicted_class].item()
            
            print(f"\n❓ Вопрос: {question}")
            print(f"✅ Категория: {self.id2label[predicted_class]}")
            print(f"📊 Уверенность: {confidence*100:.1f}%")
    
    def run(self, epochs=5, batch_size=16, learning_rate=2e-5):
        """
        Запускает полный pipeline обучения
        """
        start_time = datetime.now()
        
        # 1. Загрузка данных
        df = self.load_data()
        
        # 2. Подготовка меток
        df = self.prepare_labels(df)
        
        # 3. Разбивка данных
        train, val, test = self.split_data(df)
        
        # 4. Токенизация
        train_dataset, val_dataset, test_dataset = self.tokenize_data(train, val, test)
        
        # 5. Загрузка модели
        self.load_model()
        
        # 6. Обучение
        trainer, train_result = self.train_model(
            train_dataset,
            val_dataset,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        # 7. Оценка
        accuracy, f1 = self.evaluate_model(trainer, test_dataset, test)
        
        # 8. Тестирование
        self.test_predictions()
        
        # Итоги
        elapsed = datetime.now() - start_time
        
        print("\n" + "="*70)
        print("🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО!")
        print("="*70)
        print(f"⏱️ Время обучения: {elapsed}")
        print(f"🎯 Accuracy: {accuracy*100:.2f}%")
        print(f"📊 F1-score: {f1*100:.2f}%")
        print(f"💾 Модель сохранена: {self.output_dir}/final_model")
        print(f"📁 Все результаты: {self.output_dir}/")
        print("="*70)
        
        # Сохраняем summary
        summary = {
            'model': self.model_name,
            'data_file': self.data_file,
            'total_examples': len(df),
            'num_classes': len(self.label2id),
            'train_examples': len(train),
            'val_examples': len(val),
            'test_examples': len(test),
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'accuracy': float(accuracy),
            'f1_score': float(f1),
            'training_time': str(elapsed),
            'device': self.device
        }
        
        with open(f"{self.output_dir}/training_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n📊 Summary сохранён: {self.output_dir}/training_summary.json")


def main():
    print("\n" + "="*70)
    print("   ОБУЧЕНИЕ RuBERT ДЛЯ ДИПЛОМНОЙ РАБОТЫ")
    print("="*70)
    
    # Параметры
    data_file = input("\n📂 Файл с данными (например: final_dataset.json): ").strip()
    if not data_file:
        data_file = "final_dataset.json"
    
    print(f"\n⚙️ Настройки обучения:")
    print("   (нажми Enter для значений по умолчанию)")
    
    epochs = input("   Количество эпох (по умолчанию: 5): ").strip()
    epochs = int(epochs) if epochs else 5
    
    batch_size = input("   Batch size (по умолчанию: 16): ").strip()
    batch_size = int(batch_size) if batch_size else 16
    
    learning_rate = input("   Learning rate (по умолчанию: 2e-5): ").strip()
    learning_rate = float(learning_rate) if learning_rate else 2e-5
    
    print(f"\n✅ Настройки:")
    print(f"   - Данные: {data_file}")
    print(f"   - Эпохи: {epochs}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Learning rate: {learning_rate}")
    
    input("\n👉 Нажми Enter чтобы начать обучение...")
    
    # Запуск
    trainer = RuBERTTrainer(data_file)
    trainer.run(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Прервано пользователем!")
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
