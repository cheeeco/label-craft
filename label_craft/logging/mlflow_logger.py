"""
MLflow logger for PyTorch Lightning
"""

from typing import Any, Dict, Optional

import mlflow
import mlflow.pytorch
from loguru import logger
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities import rank_zero_only


class MLflowLogger(Logger):
    """
    MLflow logger for PyTorch Lightning that tracks experiments and metrics.
    """

    def __init__(
        self,
        experiment_name: str = "label-craft",
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize MLflow logger.

        Args:
            experiment_name: Name of the MLflow experiment
            run_name: Name of the current run
            tracking_uri: MLflow tracking URI
            tags: Additional tags for the run
        """
        super().__init__()

        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tracking_uri = tracking_uri
        self.tags = tags or {}

        # Set tracking URI if provided
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # Set or create experiment
        self._setup_experiment()

        # Start run
        self._start_run()

    def _setup_experiment(self):
        """Set up MLflow experiment."""
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(self.experiment_name)
                logger.info(f"Created new MLflow experiment: {self.experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing MLflow experiment: {self.experiment_name}")

            self.experiment_id = experiment_id
        except Exception as e:
            logger.error(f"Failed to setup MLflow experiment: {e}")
            raise

    def _start_run(self):
        """Start MLflow run."""
        try:
            self.run = mlflow.start_run(
                experiment_id=self.experiment_id, run_name=self.run_name
            )
            self.run_id = self.run.info.run_id
            logger.info(f"Started MLflow run: {self.run_id}")

            # Log tags
            if self.tags:
                mlflow.set_tags(self.tags)

        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")
            raise

    @rank_zero_only
    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters to MLflow."""
        try:
            # Filter out non-serializable parameters
            serializable_params = {}
            for key, value in params.items():
                try:
                    # Test if value is serializable
                    import json

                    json.dumps(value)
                    serializable_params[key] = value
                except (TypeError, ValueError):
                    # Convert to string if not serializable
                    serializable_params[key] = str(value)

            mlflow.log_params(serializable_params)
            logger.info(f"Logged {len(serializable_params)} hyperparameters to MLflow")

        except Exception as e:
            logger.error(f"Failed to log hyperparameters: {e}")

    @rank_zero_only
    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Log metrics to MLflow."""
        try:
            # Filter out non-numeric metrics
            numeric_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not (
                    isinstance(value, float) and (value != value)
                ):  # Check for NaN
                    numeric_metrics[key] = value

            if step is not None:
                # Log metrics with step
                for key, value in numeric_metrics.items():
                    mlflow.log_metric(key, value, step=step)
            else:
                # Log metrics without step
                mlflow.log_metrics(numeric_metrics)

            logger.debug(f"Logged {len(numeric_metrics)} metrics to MLflow")

        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")

    @rank_zero_only
    def log_model(self, model, artifact_path: str = "model") -> None:
        """Log model to MLflow."""
        try:
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path=artifact_path,
                registered_model_name=f"{self.experiment_name}_model",
            )
            logger.info(f"Logged model to MLflow: {artifact_path}")

        except Exception as e:
            logger.error(f"Failed to log model: {e}")

    @rank_zero_only
    def log_artifacts(
        self, local_dir: str, artifact_path: Optional[str] = None
    ) -> None:
        """Log artifacts to MLflow."""
        try:
            mlflow.log_artifacts(local_dir, artifact_path)
            logger.info(f"Logged artifacts from {local_dir} to MLflow")

        except Exception as e:
            logger.error(f"Failed to log artifacts: {e}")

    @rank_zero_only
    def log_figure(self, figure, artifact_path: str) -> None:
        """Log figure to MLflow."""
        try:
            mlflow.log_figure(figure, artifact_path)
            logger.info(f"Logged figure to MLflow: {artifact_path}")

        except Exception as e:
            logger.error(f"Failed to log figure: {e}")

    @rank_zero_only
    def finalize(self, status: str = "FINISHED") -> None:
        """Finalize the MLflow run."""
        try:
            # Ensure status is uppercase and valid
            valid_statuses = [
                "RUNNING",
                "SCHEDULED",
                "FINISHED",
                "FAILED",
                "KILLED",
            ]
            status_upper = status.upper()

            if status_upper not in valid_statuses:
                logger.warning(f"Invalid status '{status}', using 'FINISHED' instead")
                status_upper = "FINISHED"

            mlflow.end_run(status=status_upper)
            logger.info(f"Finalized MLflow run with status: {status_upper}")

        except Exception as e:
            logger.error(f"Failed to finalize MLflow run: {e}")

    @property
    def name(self) -> str:
        """Return the logger name."""
        return "MLflowLogger"

    @property
    def version(self) -> str:
        """Return the run ID as version."""
        return self.run_id

    @property
    def experiment(self) -> Any:
        """Return the MLflow experiment."""
        return mlflow.get_experiment(self.experiment_id)
