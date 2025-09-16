"""
Data module for text classification using PyTorch Lightning
"""

import pandas as pd
import numpy as np
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
        self.batch_size = cfg.data.data.batch_size
        self.max_length = cfg.data.data.max_length
        self.test_size = cfg.data.data.test_size
        self.val_size = cfg.data.data.val_size
        self.training_size = cfg.data.data.get('training_size', 1.0)
        self.max_samples_per_category = cfg.data.data.get('max_samples_per_category', 5000)
        self.num_workers = cfg.data.data.get('num_workers', 0)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.models.model.name)
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
        
        # Sample training data if training_size < 1.0
        if self.training_size < 1.0:
            # Use stratified sampling to maintain class distribution
            from sklearn.model_selection import train_test_split
            sample_size = int(len(texts_filtered) * self.training_size)
            if sample_size < len(texts_filtered):
                texts_filtered, _, labels_filtered, _ = train_test_split(
                    texts_filtered, labels_filtered, 
                    train_size=sample_size, 
                    random_state=42, 
                    stratify=labels_filtered
                )
        
        # Limit samples per category to prevent imbalance
        texts_filtered, labels_filtered = self._limit_samples_per_category(
            texts_filtered, labels_filtered
        )
        
        logger.info(f"Original samples: {len(texts)}")
        logger.info(f"After filtering rare categories: {len(texts_filtered)}")
        logger.info(f"Training size: {self.training_size:.1%}")
        logger.info(f"After category limiting: {len(texts_filtered)}")
        logger.info(f"Number of categories: {len(valid_labels)}")
        logger.info(f"DataLoader workers: {self.num_workers}")
        
        # Split data with stratification (if possible)
        try:
            X_temp, X_test, y_temp, y_test = train_test_split(
                texts_filtered, labels_filtered, test_size=self.test_size, 
                random_state=42, stratify=labels_filtered
            )
            
            val_ratio = self.val_size / (1 - self.test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_ratio, 
                random_state=42, stratify=y_temp
            )
        except ValueError:
            # Fallback to non-stratified split if stratification fails
            logger.warning("Stratification failed, using random split")
            X_temp, X_test, y_temp, y_test = train_test_split(
                texts_filtered, labels_filtered, test_size=self.test_size, 
                random_state=42
            )
            
            val_ratio = self.val_size / (1 - self.test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_ratio, 
                random_state=42
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
    
    def _limit_samples_per_category(self, texts, labels):
        """Limit the number of samples per category to prevent imbalance"""
        if self.max_samples_per_category is None or self.max_samples_per_category <= 0:
            return texts, labels
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame({'text': texts, 'label': labels})
        
        # Group by label and sample up to max_samples_per_category
        sampled_dfs = []
        category_stats = {}
        
        for label in df['label'].unique():
            label_df = df[df['label'] == label]
            category_count = len(label_df)
            
            if category_count > self.max_samples_per_category:
                # Sample max_samples_per_category samples
                sampled_df = label_df.sample(
                    n=self.max_samples_per_category, 
                    random_state=42
                )
                category_stats[label] = f"{self.max_samples_per_category}/{category_count}"
            else:
                # Keep all samples
                sampled_df = label_df
                category_stats[label] = f"{category_count}/{category_count}"
            
            sampled_dfs.append(sampled_df)
        
        # Combine all sampled data
        result_df = pd.concat(sampled_dfs, ignore_index=True)
        
        # Log category statistics
        limited_categories = sum(1 for v in category_stats.values() if '/' in v and v.split('/')[0] != v.split('/')[1])
        total_categories = len(category_stats)
        
        logger.info(f"Category sampling limit: {self.max_samples_per_category}")
        logger.info(f"Categories with samples limited: {limited_categories}/{total_categories}")
        
        # Show some examples of limited categories
        if limited_categories > 0:
            limited_examples = [f"{k}: {v}" for k, v in category_stats.items() if '/' in v and v.split('/')[0] != v.split('/')[1]][:5]
            logger.info(f"Examples of limited categories: {limited_examples}")
        
        return result_df['text'], result_df['label'].values
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True if self.num_workers > 0 else False
        )
