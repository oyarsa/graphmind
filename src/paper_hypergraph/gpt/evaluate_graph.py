import logging
from collections.abc import Sequence
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, TypeAdapter
from tqdm import tqdm

from paper_hypergraph import evaluation_metrics
from paper_hypergraph.gpt.model import Graph, Paper
from paper_hypergraph.gpt.run_gpt import GPTResult, run_gpt
from paper_hypergraph.util import BlockTimer

logger = logging.getLogger("gpt.evaluate_graph")


CLASSIFY_SYSTEM_PROMPT = (
    "Approve or reject the scientific paper based on the extracted entities."
)
CLASSIFY_USER_PROMPTS = {
    "simple": """\
The following data contains information about a scientific paper. It includes the \
paper's title, abstract, and a graph representation of the paper and its network.

Based on the extracted entities graph, approve or reject the paper. First, generate the \
rationale for your decision, then give the final decision.
"""
}


def evaluate_graphs(
    client: OpenAI,
    model: str,
    papers: Sequence[Paper],
    graphs: Sequence[Graph],
    user_prompt_key: str,
    output_dir: Path,
) -> None:
    """Evaluate papers acceptance based on their structured graphs.

    The output - input dataset information, predicted values and metrics - is written
    to {output_dir}/classification.
    """

    classify_user_prompt = CLASSIFY_USER_PROMPTS[user_prompt_key]

    with BlockTimer() as timer_class:
        results = _classify_papers(client, model, classify_user_prompt, papers, graphs)

    metrics = _calculate_metrics(results.result)
    logger.info(f"Metrics:\n{metrics.model_dump_json(indent=2)}")

    logger.info(f"Classification time elapsed: {timer_class.human}")
    logger.info(f"Total classification cost: ${results.cost:.10f}")

    classification_dir = output_dir / "classification"
    classification_dir.mkdir(parents=True, exist_ok=True)

    (classification_dir / "metrics.json").write_text(metrics.model_dump_json(indent=2))
    (classification_dir / "result.json").write_bytes(
        TypeAdapter(list[PaperResult]).dump_json(results.result, indent=2)
    )


class PaperResult(Paper):
    """ASAP-Review dataset paper with added approval ground truth and GPT prediction."""

    y_true: bool
    y_pred: bool


def _calculate_metrics(papers: Sequence[PaperResult]) -> evaluation_metrics.Metrics:
    return evaluation_metrics.calculate_metrics(
        [p.y_true for p in papers], [p.y_pred for p in papers]
    )


class GPTClassify(BaseModel):
    model_config = ConfigDict(frozen=True)

    rationale: str
    approved: bool


def _classify_papers(
    client: OpenAI,
    model: str,
    user_prompt_template: str,
    papers: Sequence[Paper],
    graphs: Sequence[Graph],
) -> GPTResult[list[PaperResult]]:
    """Classify Papers into approved/not approved using the generated graphs.

    Args:
        client: OpenAI client to use GPT
        model: GPT model code to use (must support Structured Outputs)
        user_prompt_template: User prompt template to use for classification to be filled
        papers: Papers from the ASAP-Review dataset to classify
        graphs: Graphs generated from the papers
        output_dir: Directory to save the classification results

    Returns:
        List of classified papers wrapped in a GPTResult.
    """
    results: list[PaperResult] = []
    total_cost = 0

    for paper, graph in tqdm(
        zip(papers, graphs), desc="Classifying papers", total=len(papers)
    ):
        user_prompt = user_prompt_template.format(
            title=paper.title,
            abstract=paper.abstract,
            graph=graph.model_dump_json(),
        )
        result = run_gpt(
            GPTClassify, client, CLASSIFY_SYSTEM_PROMPT, user_prompt, model
        )
        total_cost += result.cost
        classified = result.result

        results.append(
            PaperResult(
                title=paper.title,
                abstract=paper.abstract,
                ratings=paper.ratings,
                sections=paper.sections,
                y_true=paper.is_approved(),
                y_pred=classified.approved if classified else False,
            )
        )

    return GPTResult(results, total_cost)
