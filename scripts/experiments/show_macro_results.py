"""Show classification results with standard, negative label and macro averages."""
# pyright: basic

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer
import zstandard as zstd

from paper.evaluation_metrics import safediv


@dataclass(frozen=True)
class ClassificationMetrics:
    """Dataclass to hold classification metrics results."""

    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class_precision: list[float]
    per_class_recall: list[float]
    per_class_f1: list[float]
    confidence: float | None


def calculate_metrics(confusion_matrix: list[list[int]]) -> ClassificationMetrics:
    """Calculate macro-average precision, recall, and F1 score from a confusion matrix.

    Args:
        confusion_matrix: 2D array where element [i][j] represents instances of class i
            predicted as class j.

    Returns:
        dict: Dictionary containing macro-average precision, recall, and F1.
    """
    n_classes = len(confusion_matrix)

    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []

    for i in range(n_classes):
        # True Positives: diagonal element
        tp = confusion_matrix[i][i]

        # False Positives: sum of column i excluding diagonal
        fp = sum(confusion_matrix[j][i] for j in range(n_classes) if j != i)

        # False Negatives: sum of row i excluding diagonal
        fn = sum(confusion_matrix[i][j] for j in range(n_classes) if j != i)

        # Calculate precision for class i
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        precisions.append(precision)

        # Calculate recall for class i
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recalls.append(recall)

        # Calculate F1 for class i
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        f1_scores.append(f1)

    # Macro-average: simple average across all classes
    macro_precision = sum(precisions) / n_classes
    macro_recall = sum(recalls) / n_classes
    macro_f1 = sum(f1_scores) / n_classes

    return ClassificationMetrics(
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        per_class_precision=precisions,
        per_class_recall=recalls,
        per_class_f1=f1_scores,
        confidence=None,
    )


def find_metrics_files(directory: Path) -> list[Path]:
    """Recursively find all metrics.json and metrics.json.zst files in a directory.

    Args:
        directory: Directory to search recursively.

    Returns:
        List of paths to metrics files found.
    """
    metrics_files: list[Path] = []

    for ext in ["json", "json.zst"]:
        metrics_files.extend(directory.rglob(f"metrics.{ext}"))

    return sorted(metrics_files)


def read_json_file(file_path: Path) -> dict:
    """Read JSON data from a file, supporting both regular JSON and compressed .zst files.

    Args:
        file_path: Path to the JSON file.

    Returns:
        Parsed JSON data as dictionary>

    Raises:
        FileNotFoundError: If file doesn't exist.
        json.JSONDecodeError: If JSON is invalid.
    """
    if file_path.suffix == ".zst":
        with open(file_path, "rb") as f:
            dctx = zstd.ZstdDecompressor()
            decompressed_data = dctx.decompress(f.read())
            return json.loads(decompressed_data.decode("utf-8"))
    else:
        return json.loads(file_path.read_text())


def eprint(*args: Any, **kwargs: Any) -> None:
    """Print to stderr."""
    print(*args, **kwargs, file=sys.stderr)


def die(*args: Any) -> NoReturn:
    """Print error message to stderr and exit."""
    eprint(*args)
    sys.exit(1)


def calculate_accuracy(confusion_matrix: list[list[int]]) -> float:
    """Calculate accuracy from a confusion matrix.

    Args:
        confusion_matrix: 2D array where element [i][j] represents instances of class i
            predicted as class j.

    Returns:
        Accuracy score.
    """
    total_correct = sum(confusion_matrix[i][i] for i in range(len(confusion_matrix)))
    total_samples = sum(sum(row) for row in confusion_matrix)
    return safediv(total_correct, total_samples)


def process_metrics_file(
    json_file: Path,
) -> tuple[str, float, ClassificationMetrics] | None:
    """Process a single metrics JSON file and extract results.

    Args:
        json_file: Path to the metrics JSON file.

    Returns:
        Tuple of (model_name, accuracy, metrics) or None if processing failed.
    """
    try:
        data = read_json_file(json_file)
        if "confusion" not in data:
            eprint(f"Error: 'confusion' key not found in {json_file}")
            return None

        confusion_matrix = data["confusion"]

        if not confusion_matrix or not isinstance(confusion_matrix, list):
            eprint(
                f"Error: 'confusion' value must be a 2D array in {json_file}",
            )
            return None

        model_name = json_file.parent.name
        base_metrics = calculate_metrics(confusion_matrix)
        accuracy = calculate_accuracy(confusion_matrix)
        confidence = data.get("confidence")

        metrics = ClassificationMetrics(
            macro_precision=base_metrics.macro_precision,
            macro_recall=base_metrics.macro_recall,
            macro_f1=base_metrics.macro_f1,
            per_class_precision=base_metrics.per_class_precision,
            per_class_recall=base_metrics.per_class_recall,
            per_class_f1=base_metrics.per_class_f1,
            confidence=confidence,
        )

    except FileNotFoundError:
        eprint(f"Error: File '{json_file}' not found")
        return None
    except json.JSONDecodeError as e:
        eprint(f"Error: Invalid JSON format in {json_file} - {e}")
        return None
    except Exception as e:
        eprint(f"Error processing {json_file}: {e}")
        return None
    else:
        return (model_name, accuracy, metrics)


def collect_results(
    metrics_files: list[Path],
) -> list[tuple[str, float, ClassificationMetrics]]:
    """Collect and process results from all metrics files.

    Args:
        metrics_files: List of paths to metrics JSON files.

    Returns:
        List of tuples containing (model_name, accuracy, metrics), sorted by accuracy descending.
    """
    results: list[tuple[str, float, ClassificationMetrics]] = []

    for json_file in metrics_files:
        result = process_metrics_file(json_file)
        if result is not None:
            results.append(result)

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def print_results_table(
    results: list[tuple[str, float, ClassificationMetrics]],
) -> None:
    """Print results as a formatted table.

    Args:
        results: List of tuples containing (model_name, accuracy, metrics).
    """
    headers = [
        "Model",
        "Acc",
        "Conf",
        "P-P",
        "P-R",
        "P-F1",
        "N-P",
        "N-R",
        "N-F1",
        "M-P",
        "M-R",
        "M-F1",
    ]

    max_model_width = max(len(name) for name, _, _ in results)
    model_width = max(max_model_width, len("Model"))

    widths = [model_width, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]

    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    separator_line = "| " + " | ".join("-" * w for w in widths) + " |"

    print(header_line)
    print(separator_line)

    for model_name, accuracy, metrics in results:
        confidence_str = (
            f"{metrics.confidence:.4f}" if metrics.confidence is not None else "N/A"
        )
        values = [
            model_name,
            f"{accuracy:.4f}",
            confidence_str,
            f"{metrics.per_class_precision[0]:.4f}",
            f"{metrics.per_class_recall[0]:.4f}",
            f"{metrics.per_class_f1[0]:.4f}",
            f"{metrics.per_class_precision[1]:.4f}",
            f"{metrics.per_class_recall[1]:.4f}",
            f"{metrics.per_class_f1[1]:.4f}",
            f"{metrics.macro_precision:.4f}",
            f"{metrics.macro_recall:.4f}",
            f"{metrics.macro_f1:.4f}",
        ]
        data_line = "| " + " | ".join(v.ljust(w) for v, w in zip(values, widths)) + " |"
        print(data_line)


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command()
def main(
    directory: Annotated[
        Path,
        typer.Argument(
            help="Directory to recursively search for metrics.json and metrics.json.zst files",
            file_okay=False,
            exists=True,
        ),
    ],
) -> None:
    """Calculate macro-average precision, recall, and F1 from confusion matrices in metrics files."""
    metrics_files = find_metrics_files(directory)

    if not metrics_files:
        die(
            f"Error: No metrics.json or metrics.json.zst files found in '{directory}'",
        )

    results = collect_results(metrics_files)

    if not results:
        die("Error: No valid results to display")

    print_results_table(results)


if __name__ == "__main__":
    app()
