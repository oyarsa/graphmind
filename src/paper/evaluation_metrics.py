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

    # For each rating (1-5), calculate metrics treating it as the positive class
    metrics_per_rating = {rating: {"tp": 0, "fp": 0, "fn": 0} for rating in range(1, 6)}

    for t, p in zip(y_true, y_pred):
        if t == p:  # Correct prediction
            metrics_per_rating[t]["tp"] += 1
        else:  # Wrong prediction
            metrics_per_rating[t]["fn"] += 1  # Missed the true rating
            metrics_per_rating[p]["fp"] += 1  # Falsely predicted this rating

    # Calculate macro-averaged metrics across all ratings
    total_tp = sum(m["tp"] for m in metrics_per_rating.values())
    total_fp = sum(m["fp"] for m in metrics_per_rating.values())
    total_fn = sum(m["fn"] for m in metrics_per_rating.values())

    # Macro-averaged P/R/F1
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0
    f1_score = (
        2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    )
    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)

    return Metrics(precision=precision, recall=recall, f1=f1_score, accuracy=accuracy)
