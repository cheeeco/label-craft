"""
Hierarchical Distance Accuracy (HDA) metric implementation
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger


class HierarchicalDistanceAccuracy:
    """
    Hierarchical Distance Accuracy (HDA) metric that considers category tree
    structure.

    The metric gives partial credit for predictions that are close in the
    category hierarchy.
    """

    def __init__(self, category_tree_path: str):
        """
        Initialize HDA metric with category tree.

        Args:
            category_tree_path: Path to category tree CSV file
        """
        self.category_tree_path = category_tree_path
        self.category_tree = None
        self.category_to_level = {}
        self.category_to_parent = {}
        self._load_category_tree()

    def _load_category_tree(self):
        """Load and process category tree from CSV file."""
        try:
            # Load category tree
            df = pd.read_csv(self.category_tree_path)
            logger.info(f"Loaded category tree with {len(df)} categories")

            # Build category hierarchy
            self.category_tree = df
            self._build_hierarchy()

        except Exception as e:
            logger.error(f"Failed to load category tree: {e}")
            raise

    def _build_hierarchy(self):
        """Build category hierarchy mappings."""
        # First, create category to parent mapping
        for _, row in self.category_tree.iterrows():
            cat_id = row["cat_id"]
            parent_id = self._find_parent_id(row)
            if parent_id is not None:
                self.category_to_parent[cat_id] = parent_id

        # Then, calculate levels for each category
        for _, row in self.category_tree.iterrows():
            cat_id = row["cat_id"]
            level = self._calculate_level(cat_id)
            self.category_to_level[cat_id] = level

        logger.info(f"Built hierarchy with {len(self.category_to_level)} categories")
        logger.info(f"Level distribution: {self._get_level_distribution()}")

    def _calculate_level(self, cat_id: int) -> int:
        """Calculate the level of a category by traversing up the tree."""
        level = 0
        current = cat_id

        while current in self.category_to_parent:
            current = self.category_to_parent[current]
            level += 1

        return level

    def _find_parent_id(self, row: pd.Series) -> Optional[int]:
        """Find the parent category ID for a given category."""
        parent_id = row.get("parent_id")
        if pd.notna(parent_id):
            return int(parent_id)
        return None

    def _get_level_distribution(self) -> Dict[int, int]:
        """Get distribution of categories by level."""
        level_counts = {}
        for level in self.category_to_level.values():
            level_counts[level] = level_counts.get(level, 0) + 1
        return level_counts

    def _find_lca(self, cat1: int, cat2: int) -> Optional[int]:
        """
        Find Lowest Common Ancestor (LCA) between two categories.

        Args:
            cat1: First category ID
            cat2: Second category ID

        Returns:
            LCA category ID or None if no common ancestor
        """
        if cat1 == cat2:
            return cat1

        # Get paths to root for both categories
        path1 = self._get_path_to_root(cat1)
        path2 = self._get_path_to_root(cat2)

        # Find common ancestor
        for cat in path1:
            if cat in path2:
                return cat

        return None

    def _get_path_to_root(self, cat_id: int) -> List[int]:
        """Get path from category to root."""
        path = [cat_id]
        current = cat_id

        while current in self.category_to_parent:
            parent = self.category_to_parent[current]
            path.append(parent)
            current = parent

        return path

    def _calculate_level_diff(self, true_cat: int, pred_cat: int) -> int:
        """
        Calculate level difference between true and predicted categories.

        Args:
            true_cat: True category ID
            pred_cat: Predicted category ID

        Returns:
            Level difference (max(0, Level(true) - Level(LCA)))
        """
        lca = self._find_lca(true_cat, pred_cat)
        if lca is None:
            return 0

        true_level = self.category_to_level.get(true_cat, 0)
        lca_level = self.category_to_level.get(lca, 0)

        return max(0, true_level - lca_level)

    def _calculate_distance(self, true_cat: int, pred_cat: int) -> float:
        """
        Calculate distance between true and predicted categories.

        Args:
            true_cat: True category ID
            pred_cat: Predicted category ID

        Returns:
            Distance score (0 if no LCA, exp(-level_diff) otherwise)
        """
        lca = self._find_lca(true_cat, pred_cat)
        if lca is None:
            return 0.0

        level_diff = self._calculate_level_diff(true_cat, pred_cat)
        return np.exp(-level_diff)

    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """
        Calculate HDA metric.

        Args:
            y_true: True category IDs (tensor)
            y_pred: Predicted category IDs (tensor)

        Returns:
            HDA score (0-1, higher is better)
        """
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()

        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")

        if len(y_true) == 0:
            return 0.0

        # Calculate distances for each sample
        distances = []
        for true_cat, pred_cat in zip(y_true, y_pred):
            distance = self._calculate_distance(int(true_cat), int(pred_cat))
            distances.append(distance)

        # Calculate HDA as mean of distances
        hda = np.mean(distances)
        return float(hda)

    def get_detailed_scores(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict:
        """
        Get detailed HDA scores and statistics.

        Args:
            y_true: True category IDs (tensor)
            y_pred: Predicted category IDs (tensor)

        Returns:
            Dictionary with detailed scores and statistics
        """
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()

        distances = []
        level_diffs = []
        exact_matches = 0
        partial_matches = 0
        no_matches = 0

        for true_cat, pred_cat in zip(y_true, y_pred):
            true_cat, pred_cat = int(true_cat), int(pred_cat)

            if true_cat == pred_cat:
                exact_matches += 1
                distances.append(1.0)
                level_diffs.append(0)
            else:
                lca = self._find_lca(true_cat, pred_cat)
                if lca is None:
                    no_matches += 1
                    distances.append(0.0)
                    level_diffs.append(0)
                else:
                    partial_matches += 1
                    level_diff = self._calculate_level_diff(true_cat, pred_cat)
                    level_diffs.append(level_diff)
                    distances.append(np.exp(-level_diff))

        return {
            "hda_score": np.mean(distances),
            "exact_matches": exact_matches,
            "partial_matches": partial_matches,
            "no_matches": no_matches,
            "exact_match_rate": exact_matches / len(y_true),
            "partial_match_rate": partial_matches / len(y_true),
            "no_match_rate": no_matches / len(y_true),
            "mean_level_diff": np.mean(level_diffs),
            "max_level_diff": np.max(level_diffs) if level_diffs else 0,
            "distances": distances,
            "level_diffs": level_diffs,
        }
