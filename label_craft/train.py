#!/usr/bin/env python3
"""
Training script for text classification using PyTorch Lightning
"""

from datetime import datetime

import pytorch_lightning as pl
import torch
from hydra import compose, initialize
from loguru import logger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from label_craft.data.data_module import TextDataModule
from label_craft.logging import MLflowLogger
from label_craft.models.model import TextClassifier


def main() -> None:
    """Main training function"""

    # Initialize Hydra and compose config without creating outputs folder
    with initialize(config_path="../config", version_base=None):
        cfg = compose(config_name="train")

    # Data module
    data_module = TextDataModule(cfg)

    # Setup data to get number of classes
    data_module.setup()
    num_classes = data_module.num_classes

    logger.info(f"Classes: {num_classes}")
    logger.info(f"Train samples: {len(data_module.train_dataset)}")
    logger.info(f"Val samples: {len(data_module.val_dataset)}")
    logger.info(f"Test samples: {len(data_module.test_dataset)}")

    # Model
    model = TextClassifier(num_classes=num_classes, cfg=cfg)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.callbacks.model_checkpoint.monitor,
        mode=cfg.callbacks.model_checkpoint.mode,
        save_top_k=cfg.callbacks.model_checkpoint.save_top_k,
        filename=cfg.callbacks.model_checkpoint.filename,
        save_last=cfg.callbacks.model_checkpoint.save_last,
    )

    early_stopping = EarlyStopping(
        monitor=cfg.callbacks.early_stopping.monitor,
        mode=cfg.callbacks.early_stopping.mode,
        patience=cfg.callbacks.early_stopping.patience,
        verbose=True,
    )

    # Loggers
    tb_logger = TensorBoardLogger(cfg.logging.log_dir, name=cfg.logging.name)

    # MLflow logger
    run_name = cfg.mlflow.run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow_logger = MLflowLogger(
        experiment_name=cfg.mlflow.experiment_name,
        run_name=run_name,
        tracking_uri=cfg.mlflow.tracking_uri,
        tags=cfg.mlflow.tags,
    )

    # Combine loggers
    loggers = [tb_logger, mlflow_logger]

    # Trainer
    precision = cfg.training.precision if torch.cuda.is_available() else 32
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        callbacks=[checkpoint_callback, early_stopping],
        logger=loggers,
        accelerator="auto",
        devices="auto",
        precision=precision,
        gradient_clip_val=cfg.training.gradient_clip_val,
        log_every_n_steps=cfg.training.log_every_n_steps,
    )

    # Train
    logger.info("Starting training...")
    try:
        trainer.fit(model, data_module)

        # Test
        logger.info("Testing...")
        trainer.test(model, data_module)

        # Log model to MLflow
        logger.info("Logging model to MLflow...")
        mlflow_logger.log_model(model, "model")

        # Log additional artifacts
        mlflow_logger.log_artifacts("logs", "logs")

        # Finalize MLflow run
        mlflow_logger.finalize("FINISHED")

        logger.info("Training completed!")
        logger.info(f"MLflow run ID: {mlflow_logger.run_id}")
        logger.info(f"MLflow experiment: {mlflow_logger.experiment_name}")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        mlflow_logger.finalize("KILLED")
        logger.info(f"MLflow run ID: {mlflow_logger.run_id}")
        raise
    except Exception as e:
        logger.error(f"Training failed: {e}")
        mlflow_logger.finalize("FAILED")
        logger.info(f"MLflow run ID: {mlflow_logger.run_id}")
        raise


if __name__ == "__main__":
    main()
