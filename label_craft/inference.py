#!/usr/bin/env python3
"""
Inference script for text classification using trained model
"""

import pandas as pd
import torch
from hydra import compose, initialize
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from label_craft.data.data_module import TextDataModule
from label_craft.metrics import HierarchicalDistanceAccuracy
from label_craft.models.model import TextClassifier


def load_model(
    checkpoint_path: str, cfg: DictConfig, num_classes: int
) -> TextClassifier:
    """Load trained model from checkpoint"""
    logger.info(f"Loading model from {checkpoint_path}")

    # Load model from checkpoint
    model = TextClassifier.load_from_checkpoint(
        checkpoint_path, num_classes=num_classes, cfg=cfg, strict=False
    )

    # Set to evaluation mode
    model.eval()

    return model


def predict_batch(model: TextClassifier, batch: dict, device: str) -> tuple:
    """Make predictions on a batch of data"""
    with torch.no_grad():
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Get predictions
        logits = model(input_ids, attention_mask)
        predictions = torch.argmax(logits, dim=1)
        probabilities = torch.softmax(logits, dim=1)

        return predictions.cpu(), probabilities.cpu()


def evaluate_predictions(
    predictions: torch.Tensor, labels: torch.Tensor, category_tree_path: str
) -> dict:
    """Evaluate predictions against ground truth labels"""
    logger.info("Computing evaluation metrics...")

    # Convert to numpy for metrics computation
    preds_np = predictions.numpy()
    labels_np = labels.numpy()

    # Initialize HDA metric
    hda_metric = HierarchicalDistanceAccuracy(category_tree_path)

    # Calculate HDA score
    hda_score = hda_metric(labels_np, preds_np)
    hda_details = hda_metric.get_detailed_scores(labels_np, preds_np)

    # Calculate accuracy
    accuracy = (preds_np == labels_np).mean()

    # Calculate F1 score (macro average)
    from sklearn.metrics import f1_score

    f1 = f1_score(labels_np, preds_np, average="macro")

    metrics = {
        "accuracy": accuracy,
        "f1_score": f1,
        "hda_score": hda_score,
        "exact_match_rate": hda_details["exact_match_rate"],
        "partial_match_rate": hda_details["partial_match_rate"],
        "mean_level_diff": hda_details["mean_level_diff"],
    }

    return metrics


def main() -> None:
    """Main inference function"""

    # Initialize Hydra and compose config
    with initialize(config_path="../config", version_base=None):
        cfg = compose(config_name="inference")

    # Set device
    if cfg.inference.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = cfg.inference.device

    logger.info(f"Using device: {device}")

    # Load test data
    logger.info(f"Loading test data from {cfg.data_path}")
    test_df = pd.read_parquet(cfg.data_path)
    logger.info(f"Loaded {len(test_df)} test samples")

    # Check if we have ground truth labels
    has_labels = "label" in test_df.columns
    logger.info(f"Ground truth labels available: {has_labels}")

    # Create data module for preprocessing
    data_module = TextDataModule(cfg)
    data_module.setup()
    num_classes = data_module.num_classes

    # Create test dataset using the same preprocessing
    test_dataset = data_module.create_dataset(test_df, is_training=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.inference.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )

    # Load model
    model = load_model(cfg.model_checkpoint, cfg, num_classes)
    model = model.to(device)

    # Make predictions
    logger.info("Making predictions...")
    all_predictions = []
    all_probabilities = []
    all_labels = [] if has_labels else None

    for batch in test_loader:
        if has_labels:
            all_labels.append(batch["labels"])

        predictions, probabilities = predict_batch(model, batch, device)
        all_predictions.append(predictions)
        all_probabilities.append(probabilities)

    # Concatenate all predictions
    all_predictions = torch.cat(all_predictions, dim=0)
    all_probabilities = torch.cat(all_probabilities, dim=0)

    if has_labels:
        all_labels = torch.cat(all_labels, dim=0)

    logger.info(f"Completed predictions for {len(all_predictions)} samples")

    # Evaluate if we have ground truth labels
    if has_labels and cfg.evaluation.compute_metrics:
        logger.info("Evaluating predictions...")
        metrics = evaluate_predictions(
            all_predictions, all_labels, "data/category_tree.csv"
        )

        # Print metrics
        logger.info("Evaluation Results:")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")

    # Save predictions if requested
    if cfg.inference.output_predictions:
        logger.info(f"Saving predictions to {cfg.inference.predictions_file}")

        # Create predictions dataframe
        pred_df = test_df.copy()
        pred_df["predicted_label"] = all_predictions.numpy()
        pred_df["prediction_confidence"] = torch.max(all_probabilities, dim=1)[
            0
        ].numpy()

        # Add top-3 predictions if needed
        top3_probs, top3_indices = torch.topk(all_probabilities, k=3, dim=1)
        for i in range(3):
            pred_df[f"top_{i+1}_prediction"] = top3_indices[:, i].numpy()
            pred_df[f"top_{i+1}_confidence"] = top3_probs[:, i].numpy()

        pred_df.to_csv(cfg.inference.predictions_file, index=False)
        logger.info(f"Predictions saved to {cfg.inference.predictions_file}")

    logger.info("Inference completed successfully!")


if __name__ == "__main__":
    main()
