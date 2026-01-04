"""
ML Models Package

Содержит все компоненты для обучения и использования Seq2Seq модели
"""

# Конфигурация
from .config import ModelConfig, TrainingConfig

# Токенизатор
from .tokenizer import SimpleTokenizer

# Компоненты модели
from .encoder import Encoder
from .decoder import Decoder, Attention, SimpleDecoder
from .seq2seq import Seq2Seq, init_weights, create_seq2seq_model

# Dataset и DataLoader
from .dataset import QADataset, collate_fn, create_dataloaders, split_dataset

# Trainer
from .trainer import Trainer, create_trainer


__all__ = [
    # Конфигурация
    'ModelConfig',
    'TrainingConfig',
    
    # Токенизатор
    'SimpleTokenizer',
    
    # Модель
    'Encoder',
    'Decoder',
    'Attention',
    'SimpleDecoder',
    'Seq2Seq',
    'init_weights',
    'create_seq2seq_model',
    
    # Dataset
    'QADataset',
    'collate_fn',
    'create_dataloaders',
    'split_dataset',
    
    # Trainer
    'Trainer',
    'create_trainer',
]
