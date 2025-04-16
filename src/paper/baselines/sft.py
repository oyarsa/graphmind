"""Fine-tune a model (e.g. Llama) using LoRA for a rating prediction task.

Can use two types of input:
- basic: paper title and abstract
- graph: in addition to basic, includes the hierarchical graph and related papers
"""
# pyright: basic

from __future__ import annotations

import json
import platform
import random
import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Annotated, Literal, cast

import numpy as np
import numpy.typing as npt
import toml
import torch
import typer
from datasets import Dataset
from peft.mapping_func import get_peft_model
from peft.tuners.lora import LoraConfig as PeftLoraConfig
from peft.utils.other import prepare_model_for_kbit_training
from peft.utils.peft_types import TaskType
from pydantic import BaseModel, Field
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BatchEncoding,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)

from paper import gpt
from paper import peerread as pr
from paper import semantic_scholar as s2
from paper.evaluation_metrics import Metrics, calculate_metrics
from paper.util import metrics, sample
from paper.util.serde import load_data, save_data


class LoraConfig(BaseModel):
    """Configuration for LoRA fine-tuning."""

    r: Annotated[int, Field(description="LoRA attention dimension")]
    alpha: Annotated[int, Field(description="LoRA alpha parameter")]
    dropout: Annotated[float, Field(description="Dropout probability for LoRA layers")]
    target_modules: Annotated[list[str], Field(description="Target modules for LoRA")]


# Using Literal instead of Enum because the latter bugs out with tomlw when writing
# the configuration file.
type LabelMode = Literal["original", "binary"]
type InputMode = Literal["basic", "graph"]


class ModelConfig(BaseModel):
    """Configuration for model settings."""

    name: Annotated[str, Field(description="Pretrained model name")]
    num_labels: Annotated[int, Field(description="Number of classification labels")]
    quantisation_enabled: Annotated[
        bool, Field(description="Whether to use quantisation. Ignored on macOS")
    ]
    label_mode: Annotated[
        LabelMode, Field(description="What label mode to use (binary/original).")
    ] = "original"
    input_mode: Annotated[
        InputMode,
        Field(description="What input to use for the model (minimal vs full graph)"),
    ] = "basic"


class TrainingConfig(BaseModel):
    """Configuration for training settings."""

    batch_size: int
    learning_rate: float
    num_epochs: int
    warmup_steps: int
    weight_decay: float
    max_length: Annotated[int, Field(description="Maximum number of tokens for input")]
    logging_steps: Annotated[
        int, Field(description="How many steps between stats logging")
    ]
    eval_steps: Annotated[
        int | float | None,
        Field(
            description="Run an evaluation every X steps. Should be an integer or a"
            " float in range `[0,1)`. If smaller than 1, will be interpreted as ratio"
            " of total training steps."
        ),
    ] = None

    def eval_strategy(self) -> str:
        """Evaluate on "steps" if `eval_steps` is set, else use "epoch"."""
        return "steps" if self.eval_steps else "epoch"


class AppConfig(BaseModel):
    """Main application configuration."""

    model: ModelConfig
    lora: LoraConfig
    training: TrainingConfig


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command(no_args_is_help=True)
def train(
    train_file: Annotated[
        Path, typer.Option("--train", help="Path to the train dataset file.")
    ],
    dev_file: Annotated[
        Path,
        typer.Option(
            "--dev",
            help="Path to the development dataset file for validation during training.",
        ),
    ],
    test_file: Annotated[
        Path,
        typer.Option(
            "--test", help="Path to the test dataset file for final evaluation."
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Directory to save the fine-tuned model and other training artefacts.",
        ),
    ],
    config_path: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to the TOML configuration file."),
    ],
    num_train: Annotated[
        int | None,
        typer.Option(
            help="Number of training examples to use. If None, use all items."
        ),
    ] = None,
    num_dev: Annotated[
        int | None,
        typer.Option(help="Number of dev examples to use. If None, use all items."),
    ] = None,
    num_test: Annotated[
        int | None,
        typer.Option(help="Number of test examples to use. If None, use all items."),
    ] = None,
    seed: Annotated[int, typer.Option(help="Seed used to sample the dataset.")] = 0,
) -> None:
    """Fine-tune a Llama 3 model with LoRA for novelty rating prediction.

    Uses the paper title and abstract as input features and predicts the rating (1-4) or
    novelty label (0/1).

    Uses train data for training, dev data for validation during training, and test data
    for final evaluation.

    The dataset format is the output of `paper gpt petersum` from `paper-hypergraph`.
    """
    random.seed(seed)
    suppress_hf_warnings()
    gpu_mem = GPUMemoryTracker()

    config = read_config(config_path)

    train_dataset = load_dataset(
        train_file, num_train, config.model.label_mode, config.model.input_mode
    )
    dev_dataset = load_dataset(
        dev_file, num_dev, config.model.label_mode, config.model.input_mode
    )
    test_dataset = load_dataset(
        test_file, num_test, config.model.label_mode, config.model.input_mode
    )

    model, tokeniser = setup_model_and_tokeniser(config)
    lora_model = configure_lora(model, config)

    output_dir.mkdir(exist_ok=True, parents=True)

    train_dataset_tokenised = tokenise_dataset(train_dataset, tokeniser, config)
    dev_dataset_tokenised = tokenise_dataset(dev_dataset, tokeniser, config)
    test_dataset_tokenised = tokenise_dataset(test_dataset, tokeniser, config)

    trainer = train_model(
        lora_model,
        tokeniser,
        train_dataset_tokenised,
        dev_dataset_tokenised,
        output_dir,
        config,
    )
    save_model(lora_model, tokeniser, output_dir, config)

    print(
        evaluate_model(
            trainer, test_dataset_tokenised, output_dir, config.model.label_mode
        )
    )
    print()
    print(gpu_mem)


def read_config(file: Path) -> AppConfig:
    """Read configuration from TOML `file` path."""
    return AppConfig.model_validate(toml.loads(file.read_text()))


@app.command(no_args_is_help=True)
def infer(
    model_path: Annotated[
        Path, typer.Option("--model", "-m", help="Path to the trained model directory.")
    ],
    input_file: Annotated[
        Path,
        typer.Option("--input", "-i", help="Path to the input file for inference."),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Directory to save the inference results.",
        ),
    ],
    config_path: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Path to the TOML configuration file. If not provided, "
            "the configuration will be loaded from the model directory.",
        ),
    ] = None,
    num_examples: Annotated[
        int | None,
        typer.Option(
            help="Number of examples to use for inference. If None, use all items."
        ),
    ] = None,
    seed: Annotated[int, typer.Option(help="Seed used to sample the dataset.")] = 0,
) -> None:
    """Run inference using a trained model on a dataset.

    Loads a fine-tuned model and runs inference/evaluation on the provided data file.
    Outputs evaluation metrics and prediction details.

    The configuration is loaded from the model directory if available, or can be
    explicitly provided via the --config option.

    The dataset format is the same as used for training: output of `paper gpt petersum`
    from `paper-hypergraph`.
    """
    random.seed(seed)
    suppress_hf_warnings()
    gpu_mem = GPUMemoryTracker()

    model_config_path = model_path / "config.toml"
    if config_path is None and model_config_path.exists():
        print(f"Using configuration from model directory: {model_config_path}")
        config_path = model_config_path
    elif config_path is None:
        raise typer.BadParameter(
            "No configuration file found in model directory."
            " Please provide a configuration file using the --config option."
        )

    config = read_config(config_path)
    output_dir.mkdir(exist_ok=True, parents=True)

    model, tokeniser = setup_model_and_tokeniser(config, model_path)

    dataset = load_dataset(
        input_file, num_examples, config.model.label_mode, config.model.input_mode
    )
    dataset_tokenised = tokenise_dataset(dataset, tokeniser, config)

    data_collator = DataCollatorWithPadding(tokenizer=tokeniser)
    inference_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_eval_batch_size=config.training.batch_size,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=inference_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print(f"\nRunning inference on {len(dataset)} examples...")
    print(
        evaluate_model(trainer, dataset_tokenised, output_dir, config.model.label_mode)
    )
    print()
    print(gpu_mem)


def suppress_hf_warnings() -> None:
    """Suppress some annoyihng and unnecessary huggingface warnings."""
    warnings.filterwarnings(
        "ignore", message=".*use_reentrant parameter should be passed explicitly.*"
    )


def load_dataset(
    file: Path, n: int | None, label_mode: LabelMode, model_input: InputMode
) -> Dataset:
    """Load JSON file and prepare input dataset.

    See `preprocess_dataset` for how the data is prepared.
    """
    if model_input == "basic":
        data = sample(
            gpt.PromptResult.unwrap(
                load_data(file, gpt.PromptResult[gpt.PaperWithRelatedSummary])
            ),
            n,
        )
        return preprocess_dataset_basic(data, label_mode)

    data = sample(
        gpt.PromptResult.unwrap(load_data(file, gpt.PromptResult[gpt.ExtractedGraph])),
        n,
    )
    return preprocess_dataset_graph(data, label_mode)


def _fix_rating(rating: int, label_mode: LabelMode) -> int:
    """Fix rating for model compatibility.

    If `label_mode` is `original`, convert 1-4 to 0-3.
    Else (binary), ratings >= 3 become 1, and the rest 0.
    """
    if label_mode == "original":
        return rating - 1

    return int(rating >= 3)


def preprocess_dataset_basic(
    dataset: list[gpt.PaperWithRelatedSummary], label_mode: LabelMode
) -> Dataset:
    """Convert raw dataset into HuggingFace dataset format and prepare for training.

    Converts the label to the appropriate mode and builds the input prompt from the
    paper title and abstract.
    """
    texts = [_format_basic_template(item.paper.paper) for item in dataset]
    labels = [_fix_rating(item.paper.paper.rating, label_mode) for item in dataset]

    return Dataset.from_dict({"text": texts, "label": labels})


def _format_basic_template(paper: s2.PaperWithS2Refs) -> str:
    """Format basic template using paper title and abstract."""
    return f"Title: {paper.title}\nAbstract: {paper.abstract}"


def preprocess_dataset_graph(
    dataset: list[gpt.ExtractedGraph], label_mode: LabelMode
) -> Dataset:
    """Convert raw dataset into HuggingFace dataset format and prepare for training.

    Converts the label to the appropriate mode and builds the input prompt from the
    paper graph and related papers.
    """
    texts = [_format_graph_template(item) for item in dataset]
    labels = [_fix_rating(item.paper.rating, label_mode) for item in dataset]

    return Dataset.from_dict({"text": texts, "label": labels})


GRAPH_PROMPT = """\
Title: {title}
Abstract: {abstract}

Paper summary:
{graph}

Supporting papers:
{positive}

Contrasting papers:
{negative}
"""


def _format_graph_template(item: gpt.ExtractedGraph) -> str:
    """Format graph template using the paper graph and PETER-queried related papers."""
    paper = item.paper
    graph = item.graph
    return GRAPH_PROMPT.format(
        title=paper.title,
        abstract=paper.abstract,
        positive=_format_related(
            p for p in paper.related if p.polarity is pr.ContextPolarity.POSITIVE
        ),
        negative=_format_related(
            p for p in paper.related if p.polarity is pr.ContextPolarity.NEGATIVE
        ),
        graph=graph.to_text(),
    )


def _format_related(related: Iterable[gpt.PaperRelatedSummarised]) -> str:
    """Build prompt from related papers titles and summaries."""
    return "\n\n".join(
        f"Title: {paper.title}\nSummary: {paper.summary}\n" for paper in related
    )


def setup_model_and_tokeniser(
    config: AppConfig, path: Path | None = None
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Set up the Llama model and tokeniser for fine-tuning.

    Loads the model with appropriate quantisation settings for efficient fine-tuning.

    If `path` is provided and exists, load the model and tokeniser from it.
    """
    if path is not None and path.exists():
        model_name = path
    else:
        model_name = config.model.name

    quantisation_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        if config.model.quantisation_enabled and cuda_available()
        else None
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=config.model.num_labels,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quantisation_config,
    )
    model.config.use_cache = False

    tokeniser = AutoTokenizer.from_pretrained(model_name)
    if tokeniser.pad_token is None:
        tokeniser.pad_token = tokeniser.eos_token
    model.config.pad_token_id = tokeniser.pad_token_id

    return model, tokeniser


def cuda_available() -> bool:
    """Check if we're on Linux and CUDA is available."""
    if platform.system() == "Darwin":
        return False

    return torch.cuda.is_available()


class GPUMemoryTracker:
    """Track memory usage from the moment the object is constructed. CUDA-only."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = cuda_available() and enabled
        if self.enabled:
            torch.cuda.reset_peak_memory_stats()

    def usage(self) -> dict[str, float] | None:
        """Get current memory usage and peak so far."""
        if not self.enabled:
            return None

        return {
            "current": torch.cuda.memory_allocated() / (1024**3),  # GB
            "peak": torch.cuda.max_memory_allocated() / (1024**3),  # GB
        }

    def __str__(self) -> str:
        """Show memory usage."""
        if not self.enabled:
            return "GPU memory tracker is disabled."

        output: list[str] = []

        if usage := self.usage():
            output.append("GPU memory usage:")
            for key, val in usage.items():
                output.append(f"{key}: {val:.4f} GB")

        return "\n".join(output)


def configure_lora[T: PreTrainedModel](model: T, config: AppConfig) -> T:
    """Configure LoRA (Low-Rank Adaptation) for efficient fine-tuning.

    LoRA reduces the number of trainable parameters by using low-rank matrices for
    adaptation instead of fine-tuning the entire model.
    """
    # Prepare the model for k-bit training if quantisation is enabled
    if config.model.quantisation_enabled:
        model = prepare_model_for_kbit_training(model)

    model.config.label2id = {str(i + 1): i for i in range(config.model.num_labels)}
    model.config.id2label = {id: label for label, id in model.config.label2id.items()}

    lora_config = PeftLoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=config.lora.target_modules,
    )

    return cast(T, get_peft_model(model, lora_config))


def tokenise_function(
    examples: dict[str, list], tokeniser: PreTrainedTokenizer, max_length: int
) -> BatchEncoding:
    """Tokenise the input texts for model training."""
    return tokeniser(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )


def train_model(
    model: PreTrainedModel,
    tokeniser: PreTrainedTokenizer,
    train_dataset: Dataset,
    dev_dataset: Dataset,
    output_dir: Path,
    config: AppConfig,
) -> Trainer:
    """Train the model using HuggingFace Trainer.

    Handles the complete training process including dataset preparation, setting up
    training configuration, and executing the training loop. Uses development dataset
    for validation during training with accuracy as the metric.

    Args:
        model: The model to train
        tokeniser: Tokeniser for data processing
        train_dataset: Dataset for training
        dev_dataset: Development dataset for validation during training
        output_dir: Directory to save training outputs
        config: Application configuration

    Returns:
        The trained Trainer object for evaluation.
    """

    data_collator = DataCollatorWithPadding(tokenizer=tokeniser)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.training.num_epochs,
        per_device_train_batch_size=config.training.batch_size,
        per_device_eval_batch_size=config.training.batch_size,
        warmup_steps=config.training.warmup_steps,
        weight_decay=config.training.weight_decay,
        logging_dir=str(output_dir / "logs"),
        logging_steps=config.training.logging_steps,
        save_strategy="epoch",
        eval_strategy=config.training.eval_strategy(),
        eval_steps=config.training.eval_steps,
        load_best_model_at_end=True,
        learning_rate=config.training.learning_rate,
        fp16=True,
        report_to="none",
        metric_for_best_model="accuracy",
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    return trainer


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    """Computes accuracy, MSE and MAE from evaluation prediction.

    Used during training to report evaluation metrics.
    """
    logits = cast(np.ndarray, eval_pred.predictions)
    labels: np.ndarray = cast(np.ndarray, eval_pred.label_ids)

    # Classification metrics (accuracy)
    predicted_classes: np.ndarray = np.argmax(logits, axis=1)
    accuracy = (predicted_classes == labels).mean()

    # Regression metrics (MSE, MAE)
    # Convert predicted classes back to original scale (0-3 to 1-4)
    pred_values = list(predicted_classes + 1)
    true_values = list(labels + 1)

    mse = metrics.mean_absolute_error(true_values, pred_values)
    mae = metrics.mean_absolute_error(true_values, pred_values)

    return {
        "accuracy": accuracy,
        "mse": mse,
        "mae": mae,
    }


def tokenise_dataset(
    dataset: Dataset, tokeniser: PreTrainedTokenizer, config: AppConfig
) -> Dataset:
    """Apply tokeniser to the whole dataset."""
    return dataset.map(
        lambda examples: tokenise_function(
            examples, tokeniser, config.training.max_length
        ),
        batched=True,
    )


def save_model(
    model: PreTrainedModel,
    tokeniser: PreTrainedTokenizer,
    output_dir: Path,
    config: AppConfig,
) -> None:
    """Save the fine-tuned model and tokeniser.

    Persists the model to disk for later use in inference.
    If config_path is provided, the configuration is also saved with the model.
    """
    final_dir = output_dir / "final_model"
    model.save_pretrained(final_dir)
    tokeniser.save_pretrained(final_dir)
    (final_dir / "config.toml").write_text(toml.dumps(config.model_dump()))

    print(f"Model, tokeniser, and configuration saved to {final_dir}")


def evaluate_model(
    trainer: Trainer,
    test_dataset_tokenised: Dataset,
    output_dir: Path,
    label_mode: LabelMode,
) -> Metrics:
    """Run comprehensive evaluation on the test dataset.

    Computes classification and regression metrics using the trained model. The output
    files are saved to `output_dir`.

    Args:
        trainer: The trained Trainer object.
        test_dataset_tokenised: Tokenised dataset for model prediction.
        output_dir: Directory to save evaluation results.
        label_mode: Type of label from the data.
    """
    print("\nRunning evaluation on test set...")

    predictions = trainer.predict(test_dataset_tokenised)  # pyright: ignore
    true_labels = predictions.label_ids

    logits = predictions.predictions
    predicted_classes = np.argmax(logits, axis=1)

    return evaluate_model_predictions(
        true_labels=true_labels,  # pyright: ignore
        pred_labels=predicted_classes,
        logits=logits,  # pyright: ignore
        output_dir=output_dir,
        label_mode=label_mode,
    )


def evaluate_model_predictions(
    true_labels: Sequence[int],
    pred_labels: Sequence[int],
    logits: Sequence[Sequence[float] | npt.NDArray[np.float64]],
    output_dir: Path,
    label_mode: LabelMode,
) -> Metrics:
    """Evaluate model predictions and save results to `output_dir`.

    Saved files:
    - output_dir / evaluation_metrics.txt: Report of evaluated metrics.
    - output_dir / predictions.json: The model predictions.

    Args:
        true_labels: Ground truth labels.
        pred_labels: Predicted class indices.
        logits: Raw logits from the model.
        output_dir: Directory to save evaluation results.
        label_mode: The kind of label (original or binary). If it's original, we'll
            convert the 0-4 labels to 1-5 to match the original data.

    Returns:
        Evaluation results.
    """
    if label_mode == "original":
        true_labels_adjusted = [int(label) + 1 for label in true_labels]
        pred_labels_adjusted = [int(label) + 1 for label in pred_labels]
    else:
        true_labels_adjusted = [int(label) for label in true_labels]
        pred_labels_adjusted = [int(label) for label in pred_labels]

    logits_list = [[float(x) for x in logit] for logit in logits]
    metrics_result = calculate_metrics(true_labels_adjusted, pred_labels_adjusted)

    metrics_path = output_dir / "evaluation_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(f"Evaluation Results\n{'-' * 20}\n")
        f.write(str(metrics_result))

    prediction_path = output_dir / "predictions.json"
    prediction_data = {
        "true_labels": true_labels_adjusted,
        "predicted_labels": pred_labels_adjusted,
        "logits": logits_list,
    }

    prediction_path.write_text(json.dumps(prediction_data, indent=2))

    print(f"Evaluation metrics saved to {metrics_path}")
    print(f"Raw predictions saved to {prediction_path}")

    return metrics_result


class FormattedData(BaseModel):
    """Container for original data and its formatted representation for the model."""

    original_data: gpt.PaperWithRelatedSummary | gpt.ExtractedGraph
    input: str


@app.command(no_args_is_help=True)
def format(
    input_file: Annotated[
        Path,
        typer.Option("--input", "-i", help="Path to the input file for formatting."),
    ],
    output_file: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Path to save the formatted data.",
        ),
    ],
    input_mode: Annotated[
        str,
        typer.Option(help="Input mode to use (basic or graph)."),
    ],
    num_examples: Annotated[
        int | None,
        typer.Option(help="Number of examples to format. If None, use all items."),
    ] = None,
    seed: Annotated[int, typer.Option(help="Seed used to sample the dataset.")] = 0,
) -> None:
    """Format input data according to the specified mode and save it.

    Reads data from the input file, formats the text representation based on the
    input_mode ('basic' or 'graph'), and saves the original data along with the
    formatted text to the output file.
    """
    random.seed(seed)

    if input_mode == "basic":
        data = sample(
            gpt.PromptResult.unwrap(
                load_data(input_file, gpt.PromptResult[gpt.PaperWithRelatedSummary])
            ),
            num_examples,
        )
        formatted_items = [
            FormattedData(
                original_data=item,
                input=_format_basic_template(item.paper.paper),
            )
            for item in data
        ]
    elif input_mode == "graph":
        data = sample(
            gpt.PromptResult.unwrap(
                load_data(input_file, gpt.PromptResult[gpt.ExtractedGraph])
            ),
            num_examples,
        )
        formatted_items = [
            FormattedData(original_data=item, input=_format_graph_template(item))
            for item in data
        ]

    save_data(output_file, formatted_items)


if __name__ == "__main__":
    app()
