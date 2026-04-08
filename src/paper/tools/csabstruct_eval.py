"""Evaluate abstract background/target classification against CSAbstruct gold labels.

Runs our LLM-based abstract classifier on CSAbstruct data and computes:
- Text-overlap metrics: ROUGE-1/2/L, BERTScore
- Sentence-level metrics: exact match accuracy, precision, recall, F1
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import typer
from tqdm import tqdm

from paper.gpt.annotate_paper import (
    ABS_SYSTEM_PROMPT,
    AbstractDemonstration,
    GPTAbstractClassify,
    format_abstract_demonstrations,
)
from paper.gpt.prompts.abstract_classification import ABS_USER_PROMPTS
from paper.gpt.prompts.abstract_demonstrations import ABS_DEMO_PROMPTS
from paper.gpt.run_gpt import LLMClient
from paper.types import Immutable
from paper.util import dotenv, setup_logging
from paper.util.serde import save_data

logger = logging.getLogger(__name__)

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
    help=__doc__,
)


class ClassificationResult(Immutable):
    """Result of classifying a single abstract."""

    abstract: str
    gold_background: str
    gold_target: str
    pred_background: str
    pred_target: str


class EvalMetrics(Immutable):
    """All evaluation metrics for the CSAbstruct evaluation."""

    n_abstracts: int
    # Text overlap (background)
    rouge1_bg: float
    rouge2_bg: float
    rouge_l_bg: float
    bertscore_f1_bg: float
    # Text overlap (target)
    rouge1_tgt: float
    rouge2_tgt: float
    rouge_l_tgt: float
    bertscore_f1_tgt: float
    # Sentence-level
    sentence_accuracy: float
    sentence_macro_f1: float
    sentence_f1_bg: float
    sentence_f1_tgt: float
    sentence_precision_bg: float
    sentence_recall_bg: float
    sentence_precision_tgt: float
    sentence_recall_tgt: float


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using NLTK."""
    from paper.tools._text_metrics import split_sentences

    return split_sentences(text)


def _compute_rouge(
    predictions: Sequence[str], references: Sequence[str]
) -> dict[str, float]:
    """Compute ROUGE-1/2/L F1 scores averaged over pairs."""
    from paper.tools._text_metrics import compute_rouge

    return compute_rouge(predictions, references)


def _compute_bertscore(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Compute BERTScore F1 averaged over pairs."""
    from paper.tools._text_metrics import compute_bertscore

    return compute_bertscore(predictions, references)


def _compute_sentence_metrics(
    results: Sequence[ClassificationResult],
) -> dict[str, float]:
    """Compute sentence-level classification metrics.

    For each abstract, split gold and predicted text blocks into sentences.
    Match each gold sentence to its predicted bucket (background or target)
    based on which predicted block contains it.
    """
    tp_bg = 0  # gold=bg, pred=bg
    fp_bg = 0  # gold=tgt, pred=bg
    fn_bg = 0  # gold=bg, pred=tgt
    tp_tgt = 0
    fp_tgt = 0
    fn_tgt = 0
    correct = 0
    total = 0

    for r in results:
        gold_bg_sents = set(_split_sentences(r.gold_background))
        gold_tgt_sents = set(_split_sentences(r.gold_target))
        pred_bg_text = r.pred_background.lower()
        pred_tgt_text = r.pred_target.lower()

        for sent in gold_bg_sents:
            sent_lower = sent.lower()
            in_bg = sent_lower in pred_bg_text
            in_tgt = sent_lower in pred_tgt_text

            # If sentence appears in both or neither, use background as default
            # since the model might rephrase
            if in_bg or (not in_bg and not in_tgt):
                predicted_bg = in_bg or not in_tgt
            else:
                predicted_bg = False

            if predicted_bg:
                tp_bg += 1
                correct += 1
            else:
                fn_bg += 1
                fp_tgt += 1
            total += 1

        for sent in gold_tgt_sents:
            sent_lower = sent.lower()
            in_bg = sent_lower in pred_bg_text
            in_tgt = sent_lower in pred_tgt_text

            if in_tgt or (not in_bg and not in_tgt):
                predicted_tgt = in_tgt or not in_bg
            else:
                predicted_tgt = False

            if predicted_tgt:
                tp_tgt += 1
                correct += 1
            else:
                fn_tgt += 1
                fp_bg += 1
            total += 1

    accuracy = correct / total if total else 0.0

    prec_bg = tp_bg / (tp_bg + fp_bg) if (tp_bg + fp_bg) else 0.0
    rec_bg = tp_bg / (tp_bg + fn_bg) if (tp_bg + fn_bg) else 0.0
    f1_bg = 2 * prec_bg * rec_bg / (prec_bg + rec_bg) if (prec_bg + rec_bg) else 0.0

    prec_tgt = tp_tgt / (tp_tgt + fp_tgt) if (tp_tgt + fp_tgt) else 0.0
    rec_tgt = tp_tgt / (tp_tgt + fn_tgt) if (tp_tgt + fn_tgt) else 0.0
    f1_tgt = (
        2 * prec_tgt * rec_tgt / (prec_tgt + rec_tgt) if (prec_tgt + rec_tgt) else 0.0
    )

    macro_f1 = (f1_bg + f1_tgt) / 2

    # Micro F1: compute from total TP/FP/FN across both classes
    micro_tp = tp_bg + tp_tgt
    micro_fp = fp_bg + fp_tgt
    micro_fn = fn_bg + fn_tgt
    micro_prec = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 0.0
    micro_rec = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else 0.0
    micro_f1 = (
        2 * micro_prec * micro_rec / (micro_prec + micro_rec)
        if (micro_prec + micro_rec)
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "f1_bg": f1_bg,
        "f1_tgt": f1_tgt,
        "precision_bg": prec_bg,
        "recall_bg": rec_bg,
        "precision_tgt": prec_tgt,
        "recall_tgt": rec_tgt,
    }


async def classify_abstracts(
    client: LLMClient,
    gold_data: Sequence[AbstractDemonstration],
    demos: Sequence[AbstractDemonstration],
    prompt_key: str,
    demo_prompt_key: str,
    batch_size: int,
) -> list[ClassificationResult]:
    """Run LLM classification on all abstracts."""
    user_prompt = ABS_USER_PROMPTS[prompt_key]
    demo_prompt = ABS_DEMO_PROMPTS[demo_prompt_key]
    demonstrations = format_abstract_demonstrations(list(demos), demo_prompt)

    results: list[ClassificationResult] = []

    async def classify_one(
        entry: AbstractDemonstration,
    ) -> ClassificationResult:
        prompt_text = user_prompt.template.format(
            demonstrations=demonstrations, abstract=entry.abstract
        )
        result = await client.run(GPTAbstractClassify, ABS_SYSTEM_PROMPT, prompt_text)
        pred = result.result or GPTAbstractClassify.empty()
        return ClassificationResult(
            abstract=entry.abstract,
            gold_background=entry.background,
            gold_target=entry.target,
            pred_background=pred.background,
            pred_target=pred.target,
        )

    import itertools

    from paper.util import progress

    batches = list(itertools.batched(gold_data, batch_size))
    for batch_idx, batch in enumerate(tqdm(batches, desc="Processing batches"), 1):
        tasks = [classify_one(entry) for entry in batch]
        for task in progress.as_completed(tasks, desc=f"Classifying batch {batch_idx}"):
            results.append(await task)

    return results


def compute_all_metrics(results: Sequence[ClassificationResult]) -> EvalMetrics:
    """Compute all metrics from classification results."""
    # Text overlap
    pred_bg = [r.pred_background for r in results]
    gold_bg = [r.gold_background for r in results]
    pred_tgt = [r.pred_target for r in results]
    gold_tgt = [r.gold_target for r in results]

    rouge_bg = _compute_rouge(pred_bg, gold_bg)
    rouge_tgt = _compute_rouge(pred_tgt, gold_tgt)

    logger.info("Computing BERTScore (background)...")
    bertscore_bg = _compute_bertscore(pred_bg, gold_bg)
    logger.info("Computing BERTScore (target)...")
    bertscore_tgt = _compute_bertscore(pred_tgt, gold_tgt)

    # Sentence-level
    sent_metrics = _compute_sentence_metrics(results)

    return EvalMetrics(
        n_abstracts=len(results),
        rouge1_bg=rouge_bg["rouge1"],
        rouge2_bg=rouge_bg["rouge2"],
        rouge_l_bg=rouge_bg["rougeL"],
        bertscore_f1_bg=bertscore_bg,
        rouge1_tgt=rouge_tgt["rouge1"],
        rouge2_tgt=rouge_tgt["rouge2"],
        rouge_l_tgt=rouge_tgt["rougeL"],
        bertscore_f1_tgt=bertscore_tgt,
        sentence_accuracy=sent_metrics["accuracy"],
        sentence_macro_f1=sent_metrics["macro_f1"],
        sentence_f1_bg=sent_metrics["f1_bg"],
        sentence_f1_tgt=sent_metrics["f1_tgt"],
        sentence_precision_bg=sent_metrics["precision_bg"],
        sentence_recall_bg=sent_metrics["recall_bg"],
        sentence_precision_tgt=sent_metrics["precision_tgt"],
        sentence_recall_tgt=sent_metrics["recall_tgt"],
    )


@app.command(no_args_is_help=True)
def evaluate(
    input_file: Annotated[
        Path,
        typer.Argument(
            help="CSAbstruct JSON file (output of demonstrations abstract)."
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output", "-o", help="Directory to save results."),
    ],
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="LLM model for classification."),
    ] = "gpt-4o-mini",
    demo_file: Annotated[
        Path | None,
        typer.Option("--demos", help="JSON file with few-shot demonstrations."),
    ] = None,
    n_demos: Annotated[
        int,
        typer.Option("--n-demos", help="Number of demonstrations to use."),
    ] = 4,
    prompt_key: Annotated[
        str,
        typer.Option("--prompt", help="Abstract classification prompt."),
    ] = "simple",
    demo_prompt_key: Annotated[
        str,
        typer.Option("--demo-prompt", help="Demonstration formatting prompt."),
    ] = "simple",
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="Number of concurrent requests."),
    ] = 50,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Random seed."),
    ] = 42,
) -> None:
    """Evaluate abstract classification against CSAbstruct gold labels."""
    setup_logging()
    dotenv.load_dotenv()

    # Load evaluation data
    with open(input_file) as f:
        all_entries = [AbstractDemonstration(**e) for e in json.load(f)]

    logger.info(f"Loaded {len(all_entries)} CSAbstruct entries from {input_file}")

    # Split: use first n_demos as demonstrations, rest as evaluation
    if demo_file:
        with open(demo_file) as f:
            demos = [AbstractDemonstration(**e) for e in json.load(f)][:n_demos]
        eval_entries = all_entries
    else:
        demos = all_entries[:n_demos]
        eval_entries = all_entries[n_demos:]

    logger.info(
        f"Using {len(demos)} demonstrations, evaluating {len(eval_entries)} abstracts"
    )

    client = LLMClient.new_env(model=model, seed=seed, temperature=0.0)

    # Classify
    results = asyncio.run(
        classify_abstracts(
            client, eval_entries, demos, prompt_key, demo_prompt_key, batch_size
        )
    )

    # Compute metrics
    metrics = compute_all_metrics(results)

    # Print results
    print(f"\n{'=' * 60}")
    print(f"CSAbstruct Evaluation Results (n={metrics.n_abstracts})")
    print(f"{'=' * 60}")
    print("\nText Overlap — Background:")
    print(f"  ROUGE-1: {metrics.rouge1_bg:.4f}")
    print(f"  ROUGE-2: {metrics.rouge2_bg:.4f}")
    print(f"  ROUGE-L: {metrics.rouge_l_bg:.4f}")
    print(f"  BERTScore F1: {metrics.bertscore_f1_bg:.4f}")
    print("\nText Overlap — Target:")
    print(f"  ROUGE-1: {metrics.rouge1_tgt:.4f}")
    print(f"  ROUGE-2: {metrics.rouge2_tgt:.4f}")
    print(f"  ROUGE-L: {metrics.rouge_l_tgt:.4f}")
    print(f"  BERTScore F1: {metrics.bertscore_f1_tgt:.4f}")
    print("\nSentence-Level:")
    print(f"  Accuracy: {metrics.sentence_accuracy:.4f}")
    print(f"  Macro F1: {metrics.sentence_macro_f1:.4f}")
    print(f"  Background F1: {metrics.sentence_f1_bg:.4f}")
    print(f"    Precision: {metrics.sentence_precision_bg:.4f}")
    print(f"    Recall: {metrics.sentence_recall_bg:.4f}")
    print(f"  Target F1: {metrics.sentence_f1_tgt:.4f}")
    print(f"    Precision: {metrics.sentence_precision_tgt:.4f}")
    print(f"    Recall: {metrics.sentence_recall_tgt:.4f}")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    save_data(output_dir / "metrics.json", metrics)
    save_data(output_dir / "results.json", results)
    logger.info(f"Saved results to {output_dir}")
