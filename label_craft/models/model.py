"""
PyTorch Lightning model for text classification using LaBSE
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModel
from torchmetrics import Accuracy, F1Score
import torch.nn.functional as F
from omegaconf import DictConfig


class TextClassifier(pl.LightningModule):
    """Text classification model using LaBSE embeddings"""
    
    def __init__(self, num_classes: int, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        
        # Load pre-trained model
        self.encoder = AutoModel.from_pretrained(cfg.model.name)
        
        # Classification head
        self.dropout = nn.Dropout(cfg.model.dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_classes)
        
        # Metrics
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=num_classes)
        
        self.train_f1 = F1Score(task='multiclass', num_classes=num_classes, average='macro')
        self.val_f1 = F1Score(task='multiclass', num_classes=num_classes, average='macro')
        self.test_f1 = F1Score(task='multiclass', num_classes=num_classes, average='macro')
        
    def forward(self, input_ids, attention_mask):
        # Get embeddings from LaBSE
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        
        # Apply dropout and classification
        output = self.dropout(pooled_output)
        logits = self.classifier(output)
        
        return logits
    
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        logits = self(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)
        
        # Metrics
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, labels)
        self.train_f1(preds, labels)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc, prog_bar=True)
        self.log('train_f1', self.train_f1, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        logits = self(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)
        
        # Metrics
        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, labels)
        self.val_f1(preds, labels)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=True)
        self.log('val_f1', self.val_f1, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        logits = self(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)
        
        # Metrics
        preds = torch.argmax(logits, dim=1)
        self.test_acc(preds, labels)
        self.test_f1(preds, labels)
        
        self.log('test_loss', loss)
        self.log('test_acc', self.test_acc)
        self.log('test_f1', self.test_f1)
        
        return loss
    
    def configure_optimizers(self):
        # Get optimizer and scheduler configs
        opt_cfg = self.hparams.cfg.optimizer
        sched_cfg = self.hparams.cfg.scheduler
        
        # Create optimizer
        if opt_cfg.name == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=opt_cfg.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {opt_cfg.name}")
        
        # Create scheduler
        if sched_cfg.name == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode=sched_cfg.mode, factor=sched_cfg.factor, patience=sched_cfg.patience
            )
        else:
            raise ValueError(f"Unsupported scheduler: {sched_cfg.name}")
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': sched_cfg.monitor
            }
        }
