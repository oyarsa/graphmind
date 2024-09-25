from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict


class Metrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    precision: float
    recall: float
    f1: float
    accuracy: float


def calculate_metrics(y_true: Sequence[bool], y_pred: Sequence[bool]) -> Metrics:
    """Calculate classification metrics from true and predicted binary labels.

    Computes precision, recall, F1 score, and accuracy for binary classification results.

    Args:
        y_true: A sequence of true labels (ground truth).
        y_pred: A sequence of predicted labels.

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
