"""Fine-tune a causal LM (e.g. Llama) using LoRA for rationale + rating generation.

Unlike sft.py which does classification, this module trains the model to generate
text that includes a rationale (from peer review) followed by a rating prediction.

Input: Paper title and abstract
Output: Review rationale + Rating (1-5)
"""
# pyright: basic

from __future__ import annotations

import logging
import random
import re
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, cast

import toml
import torch
import typer
from datasets import Dataset
from peft.mapping_func import get_peft_model
from peft.tuners.lora import LoraConfig as PeftLoraConfig
from peft.utils.other import prepare_model_for_kbit_training
from peft.utils.peft_types import TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.trainer import Trainer
from transformers.trainer_utils import set_seed
from transformers.training_args import TrainingArguments
from transformers.utils.quantization_config import BitsAndBytesConfig

from paper import gpt
from paper.baselines.sft_utils import LoraConfig, cuda_available, suppress_hf_warnings
from paper.evaluation_metrics import (
    calculate_metrics,
    display_metrics,
    display_regular_negative_macro_metrics,
)
from paper.types import Immutable
from paper.util import sample, setup_logging
from paper.util.serde import load_data, save_data

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.tokenization_utils import PreTrainedTokenizer

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


class ModelConfig(Immutable):
    """Configuration for model settings."""

    name: Annotated[str, Annotated[str, "Pretrained model name"]]
    quantisation_enabled: Annotated[
        bool, Annotated[str, "Whether to use quantisation. Ignored on macOS"]
    ]


class TrainingConfig(Immutable):
    """Configuration for training settings."""

    batch_size: int
    learning_rate: float
    num_epochs: int
    warmup_steps: int
    weight_decay: float
    max_length: Annotated[int, Annotated[str, "Maximum number of tokens for input"]]
    logging_steps: Annotated[
        int, Annotated[str, "How many steps between stats logging"]
    ]
    eval_steps: Annotated[
        int | float | None,
        Annotated[
            str,
            "Run an evaluation every X steps. Should be an integer or a"
            " float in range `[0,1)`. If smaller than 1, will be interpreted as ratio"
            " of total training steps.",
        ],
    ] = None
    max_new_tokens: Annotated[
        int, Annotated[str, "Maximum number of tokens to generate during inference"]
    ] = 256

    def eval_strategy(self) -> str:
        """Evaluate on "steps" if `eval_steps` is set, else use "epoch"."""
        return "steps" if self.eval_steps else "epoch"


class GenerationConfig(Immutable):
    """Configuration for text generation during inference."""

    max_new_tokens: int = 256
    temperature: float = 0.1
    do_sample: bool = True
    top_p: float = 0.9


class AppConfig(Immutable):
    """Main application configuration."""

    model: ModelConfig
    lora: LoraConfig | None = None
    training: TrainingConfig
    generation: GenerationConfig = GenerationConfig()


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


# =============================================================================
# Data preprocessing
# =============================================================================

INPUT_TEMPLATE = """\
Title: {title}
Abstract: {abstract}

Based on the paper above, provide a review focusing on novelty and originality, \
then give a rating from 1-5 (1=not novel, 5=highly novel).

Review:"""

OUTPUT_TEMPLATE = """\
{rationale}

Rating: {rating}"""

FULL_TEMPLATE = """\
{input}{output}"""


def format_input(title: str, abstract: str) -> str:
    """Format the input prompt for the model."""
    return INPUT_TEMPLATE.format(title=title, abstract=abstract)


def format_output(rationale: str, rating: int) -> str:
    """Format the expected output (rationale + rating)."""
    return OUTPUT_TEMPLATE.format(rationale=rationale.strip(), rating=rating)


def format_training_example(
    title: str, abstract: str, rationale: str, rating: int
) -> str:
    """Format a complete training example (input + output)."""
    input_text = format_input(title, abstract)
    output_text = format_output(rationale, rating)
    return FULL_TEMPLATE.format(input=input_text, output=output_text)


def preprocess_dataset(dataset: list[gpt.PaperWithRelatedSummary]) -> Dataset:
    """Convert raw dataset into HuggingFace dataset format for generative training.

    Each example contains the full text (input + output) for causal LM training.
    """
    texts = []
    labels = []  # Store labels separately for evaluation

    for item in dataset:
        paper = item.paper.paper
        rating = paper.originality_rating
        rationale = paper.rationale

        # Skip items without rationale
        if not rationale or not rationale.strip():
            logger.warning("Skipping paper without rationale: %s", paper.title[:50])
            continue

        text = format_training_example(
            title=paper.title,
            abstract=paper.abstract,
            rationale=rationale,
            rating=rating,
        )
        texts.append(text)
        labels.append(rating)

    return Dataset.from_dict({"text": texts, "label": labels})


# =============================================================================
# Model setup
# =============================================================================


def bf16_supported() -> bool:
    """Check if bf16 is supported on the current GPU.

    BF16 requires Ampere (sm_80+) or newer architecture.
    V100 (sm_70) only supports fp16.
    """
    if not cuda_available():
        return False
    # Check compute capability - bf16 needs 8.0+ (Ampere)
    major, minor = torch.cuda.get_device_capability()
    device_name = torch.cuda.get_device_name()
    supported = major >= 8
    logger.info(
        "GPU: %s, compute capability: %d.%d, bf16 supported: %s",
        device_name,
        major,
        minor,
        supported,
    )
    return supported


def setup_model_and_tokeniser(
    config: AppConfig, path: Path | None = None
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Set up the causal LM model and tokeniser for fine-tuning.

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

    # Try f16/multi-device configuration. If it fails, use the simplest possible.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16 if cuda_available() else None,
            quantization_config=quantisation_config,
        )
    except ValueError:
        logger.info("Using basic configuration (no mixed precision, single device).")
        model = AutoModelForCausalLM.from_pretrained(model_name)

    tokeniser = AutoTokenizer.from_pretrained(
        config.model.name if path is None else path
    )
    if tokeniser.pad_token is None:
        tokeniser.pad_token = tokeniser.eos_token

    return model, tokeniser


def configure_lora(model: PreTrainedModel, config: AppConfig) -> PreTrainedModel:
    """Configure LoRA (Low-Rank Adaptation) for efficient fine-tuning."""
    if config.model.quantisation_enabled:
        model = prepare_model_for_kbit_training(model)

    if config.lora is None:
        raise ValueError("LoRA configuration must be present")

    lora_config = PeftLoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=config.lora.target_modules,
    )

    return cast("PreTrainedModel", get_peft_model(model, lora_config))


# =============================================================================
# Training
# =============================================================================


def tokenise_dataset(
    dataset: Dataset, tokeniser: PreTrainedTokenizer, max_length: int
) -> Dataset:
    """Tokenise dataset for causal LM training.

    For causal LM, labels are the same as input_ids (shifted internally by the model).
    """

    def tokenise(examples: dict[str, list]) -> dict[str, Any]:
        tokenised = tokeniser(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        # For causal LM, labels = input_ids (model handles shifting)
        input_ids = tokenised["input_ids"]
        return {
            "input_ids": input_ids,
            "attention_mask": tokenised["attention_mask"],
            "labels": input_ids,  # Same as input_ids, model shifts internally
        }

    return dataset.map(tokenise, batched=True, remove_columns=["text"])


def train_model(
    model: PreTrainedModel,
    tokeniser: PreTrainedTokenizer,
    train_dataset: Dataset,
    dev_dataset: Dataset,
    output_dir: Path,
    config: AppConfig,
    seed: int,
) -> Trainer:
    """Train the model using standard HuggingFace Trainer.

    Args:
        model: The model to train
        tokeniser: Tokeniser for data processing
        train_dataset: Dataset for training (already tokenised)
        dev_dataset: Development dataset for validation during training
        output_dir: Directory to save training outputs
        config: Application configuration
        seed: Random seed for reproducibility

    Returns:
        The trained Trainer object.
    """
    # Use same mixed precision settings as sft.py
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
        learning_rate=config.training.learning_rate,
        fp16=cuda_available(),
        report_to="none",
        seed=seed,
        data_seed=seed,
    )

    # Data collator handles padding for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokeniser,
        mlm=False,  # Causal LM, not masked LM
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
    )
    trainer.train()

    return trainer


def save_model(
    model: PreTrainedModel,
    tokeniser: PreTrainedTokenizer,
    output_dir: Path,
    config: AppConfig,
) -> None:
    """Save the fine-tuned model and tokeniser."""
    final_dir = output_dir / "final_model"
    model.save_pretrained(final_dir)
    tokeniser.save_pretrained(final_dir)
    (final_dir / "config.toml").write_text(toml.dumps(config.model_dump()))

    logger.info("Model, tokeniser, and configuration saved to %s", final_dir)


# =============================================================================
# Evaluation
# =============================================================================

# Pattern to extract rating from generated text
RATING_PATTERN = re.compile(r"Rating:\s*(\d)", re.IGNORECASE)


def parse_rating(text: str) -> int | None:
    """Extract rating from generated text.

    Returns None if no valid rating found.
    """
    match = RATING_PATTERN.search(text)
    if match:
        rating = int(match.group(1))
        # Clamp to valid range (1-5)
        return max(1, min(5, rating))
    return None


class PaperEvaluated(Immutable):
    """Paper evaluated using a generative SFT model."""

    y_true: int
    y_pred: int
    generated_text: str


def evaluate_model(
    model: PreTrainedModel,
    tokeniser: PreTrainedTokenizer,
    test_dataset: Dataset,
    original_data: list[gpt.PaperWithRelatedSummary],
    output_dir: Path,
    config: AppConfig,
) -> list[PaperEvaluated]:
    """Run evaluation on the test dataset by generating text and parsing ratings.

    Args:
        model: The trained model
        tokeniser: Tokeniser for encoding/decoding
        test_dataset: Dataset with text and labels
        original_data: Original data for input formatting
        output_dir: Directory to save evaluation results
        config: Application configuration

    Returns:
        List of evaluation results
    """
    logger.info("Running evaluation on test set...")

    model.eval()
    device = next(model.parameters()).device

    results: list[PaperEvaluated] = []
    true_labels: list[int] = []
    pred_labels: list[int] = []

    # Generate for each example
    for i, item in enumerate(original_data):
        paper = item.paper.paper
        true_rating = paper.originality_rating

        # Format input (without the output part)
        input_text = format_input(paper.title, paper.abstract)

        # Tokenise and generate
        inputs = tokeniser(input_text, return_tensors="pt").to(device)

        with torch.no_grad():
            # Cast to Any because PreTrainedModel.generate isn't properly typed
            outputs = cast(Any, model).generate(
                **inputs,
                max_new_tokens=config.generation.max_new_tokens,
                temperature=config.generation.temperature,
                do_sample=config.generation.do_sample,
                top_p=config.generation.top_p,
                pad_token_id=tokeniser.pad_token_id,
                eos_token_id=tokeniser.eos_token_id,
            )

        generated = tokeniser.decode(outputs[0], skip_special_tokens=True)
        # Extract only the generated part (after input)
        generated_output = generated[len(input_text) :].strip()

        # Parse rating from generated text
        pred_rating = parse_rating(generated_output)
        if pred_rating is None:
            logger.warning(
                "Could not parse rating from generated text for paper %d: %s",
                i,
                generated_output[:100],
            )
            # Default to middle rating if parsing fails
            pred_rating = 3

        true_labels.append(true_rating)
        pred_labels.append(pred_rating)
        results.append(
            PaperEvaluated(
                y_true=true_rating,
                y_pred=pred_rating,
                generated_text=generated_output,
            )
        )

        if (i + 1) % 10 == 0:
            logger.info("Evaluated %d/%d examples", i + 1, len(original_data))

    # Calculate and save metrics
    metrics_result = calculate_metrics(true_labels, pred_labels)

    metrics_path = output_dir / "evaluation_metrics.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"Evaluation Results\n{'-' * 20}\n")
        f.write(
            display_metrics(
                metrics_result,
                [
                    PaperEvaluated(y_true=t, y_pred=p, generated_text="")
                    for t, p in zip(true_labels, pred_labels)
                ],
            )
        )

    # Save predictions with generated text
    predictions_path = output_dir / "predictions.json.zst"
    save_data(predictions_path, results)

    logger.info("Evaluation metrics saved to %s", metrics_path)
    logger.info("Predictions saved to %s", predictions_path)

    return results


# =============================================================================
# CLI commands
# =============================================================================


def read_config(file: Path) -> AppConfig:
    """Read configuration from TOML `file` path."""
    return AppConfig.model_validate(toml.loads(file.read_text(encoding="utf-8")))


def load_dataset_from_file(
    file: Path,
    n: int | None,
    rng: random.Random,
) -> tuple[Dataset, list[gpt.PaperWithRelatedSummary]]:
    """Load JSON file and prepare dataset for generative training.

    Returns both the HuggingFace Dataset and the original data (for evaluation).
    """
    data = gpt.PromptResult.unwrap(
        load_data(file, gpt.PromptResult[gpt.PaperWithRelatedSummary])
    )
    sampled = sample(data, n, rng)
    dataset = preprocess_dataset(sampled)
    return dataset, sampled


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
    num_examples: Annotated[
        int | None,
        typer.Option(
            help="Number of examples to use for all splits. If given, overrides"
            " --num-dev, --num-train and --num-test"
        ),
    ] = None,
    seed: Annotated[int, typer.Option(help="Seed used to sample the dataset.")] = 0,
) -> None:
    """Fine-tune a causal LM with LoRA for rationale + rating generation.

    Uses the paper title and abstract as input and trains the model to generate
    a review rationale followed by a rating (1-5).

    Uses train data for training, dev data for validation during training, and test
    data for final evaluation.
    """
    rng = random.Random(seed)
    suppress_hf_warnings()

    config = read_config(config_path)

    # Override number of items with `--num-examples` shortcut.
    if num_examples is not None:
        num_train = num_examples
        num_dev = num_examples
        num_test = num_examples

    logger.debug("Loading datasets: start")
    train_dataset, _ = load_dataset_from_file(train_file, num_train, rng)
    dev_dataset, _ = load_dataset_from_file(dev_file, num_dev, rng)
    test_dataset, test_data = load_dataset_from_file(test_file, num_test, rng)
    logger.debug("Loading datasets: done")

    logger.debug("Setting seed for reproducibility")
    set_seed(seed)

    logger.debug("Setting up model: start")
    model, tokeniser = setup_model_and_tokeniser(config)
    logger.debug("Setting up model: done")

    if config.lora:
        logger.debug("Setting up LoRA: start")
        model = configure_lora(model, config)
        logger.debug("Setting up LoRA: done")

    output_dir.mkdir(exist_ok=True, parents=True)

    logger.debug("Tokenising datasets: start")
    train_dataset_tok = tokenise_dataset(
        train_dataset, tokeniser, config.training.max_length
    )
    dev_dataset_tok = tokenise_dataset(
        dev_dataset, tokeniser, config.training.max_length
    )
    logger.debug("Tokenising datasets: done")

    logger.debug("Training model: start")
    trainer = train_model(
        model,
        tokeniser,
        train_dataset_tok,
        dev_dataset_tok,
        output_dir,
        config,
        seed,
    )
    logger.debug("Training model: end")

    logger.debug("Saving model: start")
    trained_model = cast("PreTrainedModel", trainer.model)
    save_model(trained_model, tokeniser, output_dir, config)
    logger.debug("Saving model: end")

    logger.debug("Evaluating: start")
    evaluated = evaluate_model(
        trained_model,
        tokeniser,
        test_dataset,
        test_data,
        output_dir,
        config,
    )
    logger.info("\n%s", display_regular_negative_macro_metrics(evaluated))
    logger.debug("Evaluating: end")


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
    """Run inference using a trained generative model on a dataset.

    Loads a fine-tuned model and generates rationales + ratings for the provided data.
    Outputs evaluation metrics and generated text.
    """
    rng = random.Random(seed)
    suppress_hf_warnings()

    model_config_path = model_path / "config.toml"
    if config_path is None and model_config_path.exists():
        logger.info("Using configuration from model directory: %s", model_config_path)
        config_path = model_config_path
    elif config_path is None:
        raise typer.BadParameter(
            "No configuration file found in model directory."
            " Please provide a configuration file using the --config option."
        )

    config = read_config(config_path)
    output_dir.mkdir(exist_ok=True, parents=True)

    model, tokeniser = setup_model_and_tokeniser(config, model_path)

    dataset, original_data = load_dataset_from_file(input_file, num_examples, rng)

    logger.info("Running inference on %d examples...", len(original_data))
    evaluated = evaluate_model(
        model,
        tokeniser,
        dataset,
        original_data,
        output_dir,
        config,
    )
    logger.info("\n%s", display_regular_negative_macro_metrics(evaluated))


@app.callback()
def main() -> None:
    """Set up logging."""
    setup_logging()


if __name__ == "__main__":
    app()
