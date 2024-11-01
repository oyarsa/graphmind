"""Evaluate a paper's approval based on its structured graph."""

import logging
from collections.abc import Sequence
from pathlib import Path

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, TypeAdapter
from tqdm import tqdm

from paper.gpt.model import PaperGraph, PaperResult, calculate_paper_metrics
from paper.gpt.prompts import PromptTemplate, load_prompts
from paper.gpt.run_gpt import GPTResult, run_gpt
from paper.util import Timer

logger = logging.getLogger("paper.gpt.evaluate_graph")


CLASSIFY_SYSTEM_PROMPT = (
    "Approve or reject the scientific paper based on the extracted entities."
)
CLASSIFY_USER_PROMPTS = load_prompts("evaluate_graph")


async def evaluate_graphs(
    client: AsyncOpenAI,
    model: str,
    paper_graphs: Sequence[PaperGraph],
    user_prompt_key: str,
    output_dir: Path,
) -> None:
    """Evaluate papers acceptance based on their structured graphs.

    The output - input dataset information, predicted values and metrics - is written
    to {output_dir}/classification.
    """

    classify_user_prompt = CLASSIFY_USER_PROMPTS[user_prompt_key]

    with Timer() as timer_class:
        results = await _classify_papers(
            client, model, classify_user_prompt, paper_graphs
        )

    metrics = calculate_paper_metrics(results.result)
    logger.info(f"Metrics:\n{metrics.model_dump_json(indent=2)}")

    logger.info(f"Classification time elapsed: {timer_class.human}")
    logger.info(f"Total classification cost: ${results.cost:.10f}")

    classification_dir = output_dir / "classification"
    classification_dir.mkdir(parents=True, exist_ok=True)

    (classification_dir / "metrics.json").write_text(metrics.model_dump_json(indent=2))
    (classification_dir / "result.json").write_bytes(
        TypeAdapter(list[PaperResult]).dump_json(results.result, indent=2)
    )


class GPTClassify(BaseModel):
    model_config = ConfigDict(frozen=True)

    rationale: str
    approved: bool


_CLASSIFY_TYPES = {
    "classify": GPTClassify,
}


async def _classify_papers(
    client: AsyncOpenAI,
    model: str,
    user_prompt: PromptTemplate,
    paper_graphs: Sequence[PaperGraph],
) -> GPTResult[list[PaperResult]]:
    """Classify Papers into approved/not approved using the generated graphs.

    Args:
        client: OpenAI client to use GPT
        model: GPT model code to use (must support Structured Outputs)
        user_prompt: User prompt template to use for classification to be filled
        papers: Papers from the ASAP-Review dataset to classify
        graphs: Graphs generated from the papers
        output_dir: Directory to save the classification results

    Returns:
        List of classified papers wrapped in a GPTResult.
    """
    results: list[PaperResult] = []
    total_cost = 0

    for pg in tqdm(paper_graphs, desc="Classifying papers"):
        user_prompt_text = user_prompt.template.format(
            title=pg.paper.title,
            abstract=pg.paper.abstract,
            graph=pg.graph.model_dump_json(),
        )
        result = await run_gpt(
            _CLASSIFY_TYPES[user_prompt.type_name],
            client,
            CLASSIFY_SYSTEM_PROMPT,
            user_prompt_text,
            model,
        )
        total_cost += result.cost
        classified = result.result

        results.append(
            PaperResult(
                title=pg.paper.title,
                abstract=pg.paper.abstract,
                ratings=pg.paper.ratings,
                sections=pg.paper.sections,
                y_true=pg.paper.is_approved(),
                y_pred=classified.approved if classified else False,
                approval=pg.paper.approval,
            )
        )

    return GPTResult(results, total_cost)
