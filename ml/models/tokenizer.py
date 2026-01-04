"""
Токенизатор выполняет:
1. Построение словаря из обучающих текстов
2. Кодирование текста в последовательность индексов
3. Декодирование индексов обратно в текст
4. Сохранение/загрузка из файла
"""

import pickle
import re
from collections import Counter
from typing import List, Dict, Tuple


class SimpleTokenizer:
    """
    Простой токенизатор для русского текста
    """
    
    # Специальные токены
    PAD_TOKEN = '<PAD>'    # Паддинг (заполнение до нужной длины)
    UNK_TOKEN = '<UNK>'    # Неизвестное слово
    SOS_TOKEN = '<SOS>'    # Start of sequence (начало последовательности)
    EOS_TOKEN = '<EOS>'    # End of sequence (конец последовательности)
    
    def __init__(self, vocab_size: int = 5000):
        """
        Инициализация токенизатора
        
        Args:
            vocab_size: Максимальный размер словаря
        """
        self.vocab_size = vocab_size
        
        # Словари для преобразования слов в индексы и обратно
        self.word2idx = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.SOS_TOKEN: 2,
            self.EOS_TOKEN: 3
        }
        self.idx2word = {
            0: self.PAD_TOKEN,
            1: self.UNK_TOKEN,
            2: self.SOS_TOKEN,
            3: self.EOS_TOKEN
        }
        
        # Счётчик частоты слов
        self.word_count = Counter()
        
        # Статистика
        self.num_words = 4  # Начинаем с 4 спец токенов
    
    def preprocess(self, text: str) -> List[str]:
        """
        Предобработка текста
        
        Args:
            text: Исходный текст
            
        Returns:
            Список токенов (слов)
        """
        # Приводим к нижнему регистру
        text = text.lower()
        
        # Убираем лишние пробелы
        text = ' '.join(text.split())
        
        # Разделяем знаки препинания
        text = re.sub(r'([.,!?;:])', r' \1 ', text)
        
        # Убираем специальные символы кроме букв, цифр и основной пунктуации
        text = re.sub(r'[^а-яёa-z0-9\s.,!?;:\-]', '', text)
        
        # Разбиваем на токены
        tokens = text.split()
        
        return tokens
    
    def build_vocab(self, texts: List[str]):
        """
        Построение словаря на основе корпуса текстов
        
        Args:
            texts: Список текстов для анализа
        """
        print(f"\n🔨 Построение словаря из {len(texts)} текстов...")
        
        # Подсчёт частоты слов
        for idx, text in enumerate(texts, 1):
            tokens = self.preprocess(text)
            self.word_count.update(tokens)
            
            if idx % 100 == 0:
                print(f"   Обработано {idx}/{len(texts)} текстов...")
        
        print(f"   Уникальных слов найдено: {len(self.word_count)}")
        
        # Берём топ N самых частотных слов
        most_common = self.word_count.most_common(self.vocab_size - 4)
        
        # Добавляем в словарь
        for word, freq in most_common:
            if word not in self.word2idx:
                idx = self.num_words
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                self.num_words += 1
        
        print(f"✅ Словарь построен: {self.num_words} слов")
        print(f"   Покрытие: {len(most_common)}/{len(self.word_count)} "
              f"({len(most_common)/len(self.word_count)*100:.1f}%)")
    
    def encode(self, text: str, max_length: int = 100, 
               add_sos: bool = False, add_eos: bool = False) -> List[int]:
        """
        Кодирование текста в последовательность индексов
        
        Args:
            text: Текст для кодирования
            max_length: Максимальная длина последовательности
            add_sos: Добавить токен начала последовательности
            add_eos: Добавить токен конца последовательности
            
        Returns:
            Список индексов
        """
        # Токенизация
        tokens = self.preprocess(text)
        
        # Преобразование в индексы (неизвестные слова -> UNK)
        indices = [self.word2idx.get(token, 1) for token in tokens]
        
        # Добавление специальных токенов
        if add_sos:
            indices = [2] + indices  # 2 = <SOS>
        if add_eos:
            indices = indices + [3]  # 3 = <EOS>
        
        # Обрезка или паддинг до max_length
        if len(indices) < max_length:
            # Дополняем паддингом
            indices += [0] * (max_length - len(indices))
        else:
            # Обрезаем (оставляем место для EOS если нужно)
            if add_eos and len(indices) > max_length:
                indices = indices[:max_length-1] + [3]
            else:
                indices = indices[:max_length]
        
        return indices
    
    def encode_batch(self, texts: List[str], max_length: int = 100,
                     add_sos: bool = False, add_eos: bool = False) -> List[List[int]]:
        """
        Кодирование батча текстов
        
        Args:
            texts: Список текстов
            max_length: Максимальная длина
            add_sos: Добавить SOS токен
            add_eos: Добавить EOS токен
            
        Returns:
            Список закодированных последовательностей
        """
        return [self.encode(text, max_length, add_sos, add_eos) for text in texts]
    
    def decode(self, indices: List[int], skip_special: bool = True) -> str:
        """
        Декодирование последовательности индексов в текст
        
        Args:
            indices: Список индексов
            skip_special: Пропускать ли специальные токены
            
        Returns:
            Декодированный текст
        """
        # Специальные токены которые нужно пропустить
        special_tokens = {self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN}
        
        words = []
        for idx in indices:
            # Получаем слово по индексу
            word = self.idx2word.get(idx, self.UNK_TOKEN)
            
            # Пропускаем специальные токены если нужно
            if skip_special and word in special_tokens:
                continue
            
            # Останавливаемся на EOS если встретили
            if word == self.EOS_TOKEN:
                break
            
            words.append(word)
        
        # Собираем текст
        text = ' '.join(words)
        
        # Убираем пробелы перед знаками препинания
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        
        return text
    
    def decode_batch(self, indices_batch: List[List[int]], 
                     skip_special: bool = True) -> List[str]:
        """
        Декодирование батча последовательностей
        
        Args:
            indices_batch: Список последовательностей индексов
            skip_special: Пропускать специальные токены
            
        Returns:
            Список декодированных текстов
        """
        return [self.decode(indices, skip_special) for indices in indices_batch]
    
    def get_vocab_size(self) -> int:
        """Возвращает размер словаря"""
        return self.num_words
    
    def save(self, filepath: str):
        """
        Сохранение токенизатора в файл
        
        Args:
            filepath: Путь к файлу
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"💾 Токенизатор сохранён: {filepath}")
        print(f"   Размер словаря: {self.num_words} слов")
    
    @staticmethod
    def load(filepath: str) -> 'SimpleTokenizer':
        """
        Загрузка токенизатора из файла
        
        Args:
            filepath: Путь к файлу
            
        Returns:
            Загруженный токенизатор
        """
        with open(filepath, 'rb') as f:
            tokenizer = pickle.load(f)
        print(f"✅ Токенизатор загружен: {filepath}")
        print(f"   Размер словаря: {tokenizer.num_words} слов")
        return tokenizer
    
    def __len__(self) -> int:
        """Возвращает размер словаря"""
        return self.num_words
    
    def __repr__(self) -> str:
        """Строковое представление токенизатора"""
        return f"SimpleTokenizer(vocab_size={self.num_words})"


if __name__ == "__main__":
    # Тестирование токенизатора
    print("\n" + "=" * 60)
    print("ТЕСТ ТОКЕНИЗАТОРА")
    print("=" * 60)
    
    # Создаём токенизатор
    tokenizer = SimpleTokenizer(vocab_size=100)
    
    # Пример текстов
    texts = [
        "Сколько стоит обучение в МУИВ?",
        "Какие документы нужны для поступления?",
        "Есть ли бюджетные места на IT направлениях?",
        "Как подать документы онлайн?"
    ]
    
    # Строим словарь
    tokenizer.build_vocab(texts * 10)  # Повторяем для частоты
    
    # Тестируем кодирование
    test_text = "Сколько стоит обучение?"
    print(f"\n📝 Исходный текст: {test_text}")
    
    encoded = tokenizer.encode(test_text, max_length=20, add_sos=True, add_eos=True)
    print(f"🔢 Закодировано: {encoded}")
    
    decoded = tokenizer.decode(encoded)
    print(f"📝 Декодировано: {decoded}")
    
    print("\n" + "=" * 60)
