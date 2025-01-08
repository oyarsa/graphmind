"""Metric calculation (precision, recall, F1 and accuracy) for ratings 1-5."""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict


class Metrics(BaseModel):
    """Classification metrics."""

    model_config = ConfigDict(frozen=True)

    precision: float
    recall: float
    f1: float
    accuracy: float

    def __str__(self) -> str:
        """Display metrics, one per line."""
        return "\n".join(
            (
                f"P   : {self.precision:.4f}",
                f"R   : {self.recall:.4f}",
                f"F1  : {self.f1:.4f}",
                f"Acc : {self.accuracy:.4f}",
            )
        )


def calculate_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Metrics:
    """Calculate classification metrics for multi-class classification (labels 1-5).

    Args:
        y_true: Ground truth (correct) labels (values 1-5)
        y_pred: Predicted labels (values 1-5)

    Returns:
        Metrics object containing macro-averaged precision, recall, F1 score and accuracy

    Raises:
        ValueError: If input sequences are empty or of different lengths
    """
    if not y_true or not y_pred:
        raise ValueError("Input sequences cannot be empty")

    if len(y_true) != len(y_pred):
        raise ValueError("Input sequences must have the same length")

    classes = range(1, 6)  # Labels 1-5
    precisions: list[float] = []
    recalls: list[float] = []

    for cls in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)

        precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)

    # Macro averaging (equal weight to each class)
    macro_precision = sum(precisions) / len(classes)
    macro_recall = sum(recalls) / len(classes)
    macro_f1 = (
        2 * (macro_precision * macro_recall) / (macro_precision + macro_recall)
        if (macro_precision + macro_recall) > 0
        else 0.0
    )
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)

    return Metrics(
        precision=macro_precision, recall=macro_recall, f1=macro_f1, accuracy=accuracy
    )
