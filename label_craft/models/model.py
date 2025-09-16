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
from loguru import logger
from label_craft.metrics import HierarchicalDistanceAccuracy


class TextClassifier(pl.LightningModule):
    """Text classification model using LaBSE embeddings"""
    
    def __init__(self, num_classes: int, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        
        # Load pre-trained model
        self.encoder = AutoModel.from_pretrained(cfg.models.model.name)
        
        # Freeze encoder layers if configured
        self._freeze_encoder_layers(cfg.models.model)
        
        # Classification head
        self.dropout = nn.Dropout(cfg.models.model.dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_classes)
        
        # Metrics
        acc_kwargs = {'task': 'multiclass', 'num_classes': num_classes}
        f1_kwargs = {**acc_kwargs, 'average': 'macro'}
        
        self.train_acc = Accuracy(**acc_kwargs)
        self.val_acc = Accuracy(**acc_kwargs)
        self.test_acc = Accuracy(**acc_kwargs)
        
        self.train_f1 = F1Score(**f1_kwargs)
        self.val_f1 = F1Score(**f1_kwargs)
        self.test_f1 = F1Score(**f1_kwargs)
        
        # Initialize HDA metric
        self.hda_metric = HierarchicalDistanceAccuracy('data/category_tree.csv')
    
    def _freeze_encoder_layers(self, model_cfg):
        """Freeze encoder layers based on configuration"""
        if not model_cfg.get('freeze_encoder', False):
            return
        
        freeze_layers = model_cfg.get('freeze_layers', 0)
        total_layers = len(self.encoder.encoder.layer)
        
        if freeze_layers == -1:
            # Freeze all encoder layers
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info(f"Frozen all {total_layers} encoder layers")
        elif freeze_layers > 0:
            # Freeze first N layers
            layers_to_freeze = min(freeze_layers, total_layers)
            for i in range(layers_to_freeze):
                for param in self.encoder.encoder.layer[i].parameters():
                    param.requires_grad = False
            logger.info(f"Frozen first {layers_to_freeze} of {total_layers} encoder layers")
        else:
            logger.info("No encoder layers frozen")
        
        # Log trainable parameters info
        self._log_trainable_params()
    
    def _log_trainable_params(self):
        """Log information about trainable parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Frozen parameters: {frozen_params:,}")
        logger.info(f"Trainable ratio: {trainable_params/total_params:.2%}")
    
    def unfreeze_encoder(self):
        """Unfreeze all encoder parameters"""
        for param in self.encoder.parameters():
            param.requires_grad = True
        logger.info("Unfrozen all encoder parameters")
        self._log_trainable_params()
    
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
        
        # Calculate HDA metric
        hda_score = self.hda_metric(labels, preds)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc, prog_bar=True)
        self.log('train_f1', self.train_f1, prog_bar=True)
        self.log('train_hda', hda_score, prog_bar=True)
        
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
        
        # Calculate HDA metric
        hda_score = self.hda_metric(labels, preds)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=True)
        self.log('val_f1', self.val_f1, prog_bar=True)
        self.log('val_hda', hda_score, prog_bar=True)
        
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
        
        # Calculate HDA metric
        hda_score = self.hda_metric(labels, preds)
        
        # Get detailed HDA statistics
        hda_details = self.hda_metric.get_detailed_scores(labels, preds)
        
        self.log('test_loss', loss)
        self.log('test_acc', self.test_acc)
        self.log('test_f1', self.test_f1)
        self.log('test_hda', hda_score)
        self.log('test_exact_match_rate', hda_details['exact_match_rate'])
        self.log('test_partial_match_rate', hda_details['partial_match_rate'])
        self.log('test_mean_level_diff', hda_details['mean_level_diff'])
        
        return loss
    
    def configure_optimizers(self):
        # Get optimizer and scheduler configs
        opt_cfg = self.hparams.cfg.models.optimizer
        sched_cfg = self.hparams.cfg.models.scheduler
        
        # Create optimizer
        if opt_cfg.name == "AdamW":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=opt_cfg.learning_rate
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_cfg.name}")
        
        # Create scheduler
        if sched_cfg.name == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode=sched_cfg.mode, 
                factor=sched_cfg.factor, patience=sched_cfg.patience
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
