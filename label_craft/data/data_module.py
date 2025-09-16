"""
Data module for text classification using PyTorch Lightning
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from transformers import AutoTokenizer
from omegaconf import DictConfig
from loguru import logger


class TextClassificationDataset(Dataset):
    """Dataset for text classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class TextDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for text classification"""
    
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.data_path = cfg.data_path
        self.batch_size = cfg.data.batch_size
        self.max_length = cfg.data.max_length
        self.test_size = cfg.data.test_size
        self.val_size = cfg.data.val_size
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
        self.label_encoder = LabelEncoder()
        
    def setup(self, stage=None):
        # Load data
        df = pd.read_parquet(self.data_path)
        
        # Prepare texts and labels
        texts = df['source_name']
        labels = self.label_encoder.fit_transform(df['cat_id'])
        
        # Filter out categories with only 1 sample for stratification
        label_counts = pd.Series(labels).value_counts()
        valid_labels = label_counts[label_counts >= 2].index
        
        # Filter data to only include valid labels
        mask = pd.Series(labels).isin(valid_labels)
        texts_filtered = texts[mask]
        labels_filtered = labels[mask]
        
        logger.info(f"Original samples: {len(texts)}")
        logger.info(f"After filtering rare categories: {len(texts_filtered)}")
        logger.info(f"Number of categories: {len(valid_labels)}")
        
        # Split data with stratification
        X_temp, X_test, y_temp, y_test = train_test_split(
            texts_filtered, labels_filtered, test_size=self.test_size, 
            random_state=42, stratify=labels_filtered
        )
        
        val_ratio = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, 
            random_state=42, stratify=y_temp
        )
        
        # Create datasets
        self.train_dataset = TextClassificationDataset(
            X_train, y_train, self.tokenizer, self.max_length
        )
        self.val_dataset = TextClassificationDataset(
            X_val, y_val, self.tokenizer, self.max_length
        )
        self.test_dataset = TextClassificationDataset(
            X_test, y_test, self.tokenizer, self.max_length
        )
        
        self.num_classes = len(self.label_encoder.classes_)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
