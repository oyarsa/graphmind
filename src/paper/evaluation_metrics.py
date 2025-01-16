"""Metric calculation (precision, recall, F1 and accuracy) for ratings 1-5."""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from paper.util import metrics


class Metrics(BaseModel):
    """Classification and regression metrics."""

    model_config = ConfigDict(frozen=True)

    precision: float
    recall: float
    f1: float
    accuracy: float
    mae: float
    mse: float
    correlation: float | None
    confusion: list[list[int]]

    def __str__(self) -> str:
        """Display metrics, one per line."""
        corr = f"{self.correlation:.4f}" if self.correlation is not None else "N/A"
        return "\n".join(
            (
                f"Precision  : {self.precision:.4f}",
                f"Recall     : {self.recall:.4f}",
                f"F1         : {self.f1:.4f}",
                f"Accuracy   : {self.accuracy:.4f}",
                f"MAE        : {self.mae:.4f}",
                f"MSE        : {self.mse:.4f}",
                f"Correlation: {corr}",
                "Confusion Matrix:",
                self._format_confusion(),
            )
        )

    def _format_confusion(self) -> str:
        """Format confusion matrix as a string."""
        return "\n".join(" ".join(f"{x:3d}" for x in row) for row in self.confusion)


def calculate_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Metrics:
    """Calculate classification metrics for multi-class classification (labels 1-5).

    Args:
        y_true: Ground truth labels (values 1-5)
        y_pred: Predicted labels (values 1-5)

    Returns:
        Metrics object containing macro-averaged precision, recall, F1 score, accuracy,
        Mean Absolute Error, Mean Squared Error, Pearson Correlation and the
        classification confusion matrix.

    Raises:
        ValueError: If input sequences are empty or of different lengths
    """
    if not y_true or not y_pred:
        raise ValueError("Input sequences cannot be empty")

    if len(y_true) != len(y_pred):
        raise ValueError("Input sequences must have the same length")

    return Metrics(
        precision=metrics.precision(y_true, y_pred),
        recall=metrics.recall(y_true, y_pred),
        f1=metrics.f1_score(y_true, y_pred),
        accuracy=metrics.accuracy(y_true, y_pred),
        mae=metrics.mean_absolute_error(y_true, y_pred),
        mse=metrics.mean_squared_error(y_true, y_pred),
        correlation=metrics.pearson_correlation(y_true, y_pred),
        confusion=metrics.confusion_matrix(y_true, y_pred, labels=range(1, 6)),
    )
