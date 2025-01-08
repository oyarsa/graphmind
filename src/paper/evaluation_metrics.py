"""Metric calculation (precision, recall, F1 and accuracy) for binary classification."""

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


# TODO: Fix this to use integer comparison, not bool
def calculate_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Metrics:
    """Calculate classification metrics from true and predicted ratings.

    Computes precision, recall, F1 score, and accuracy for discrete classification
    results.

    Args:
        y_true: A sequence of true ratings (ground truth).
        y_pred: A sequence of predicted ratings.

    Returns:
        A Metrics object containing the calculated metrics.

    Raises:
        ValueError: If the input sequences have different lengths or are empty.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Input sequences must have the same length")
    if not y_true:
        raise ValueError("Input sequences cannot be empty")

    true_positives = sum(t and p for t, p in zip(y_true, y_pred))
    true_negatives = sum(not t and not p for t, p in zip(y_true, y_pred))
    false_positives = sum(not t and p for t, p in zip(y_true, y_pred))
    false_negatives = sum(t and not p for t, p in zip(y_true, y_pred))

    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives > 0
        else 0
    )
    f1_score = (
        2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    )
    accuracy = (true_positives + true_negatives) / len(y_true)

    return Metrics(precision=precision, recall=recall, f1=f1_score, accuracy=accuracy)
