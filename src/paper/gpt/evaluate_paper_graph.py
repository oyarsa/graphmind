"""Evaluate a paper's approval based on its structured graph."""

import logging
from collections.abc import Sequence
from pathlib import Path

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, TypeAdapter

from paper.gpt.evaluate_paper import (
    PaperResult,
    calculate_paper_metrics,
    display_metrics,
)
from paper.gpt.model import PaperGraph, Prompt, PromptResult
from paper.gpt.prompts import PromptTemplate, load_prompts
from paper.gpt.run_gpt import (
    GPTResult,
    append_intermediate_result,
    get_id,
    get_remaining_items,
    run_gpt,
)
from paper.progress import as_completed
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
    continue_papers_file: Path | None,
    clean_run: bool,
) -> None:
    """Evaluate papers acceptance based on their structured graphs.

    The output - input dataset information, predicted values and metrics - is written
    to {output_dir}/classification.
    """

    classify_user_prompt = CLASSIFY_USER_PROMPTS[user_prompt_key]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_intermediate_file = output_dir / "results.tmp.json"
    papers_remaining = get_remaining_items(
        PaperGraph,
        output_intermediate_file,
        continue_papers_file,
        paper_graphs,
        clean_run=clean_run,
        continue_key=get_id,
        original_key=get_id,
    )
    if not papers_remaining.remaining:
        logger.info(
            "No items left to process. They're all on the `continues` file. Exiting."
        )
        return

    logger.info(
        "Skipping %d items from the `continue` file.", len(papers_remaining.done)
    )

    with Timer() as timer_class:
        results = await _classify_papers(
            client, model, classify_user_prompt, paper_graphs, output_intermediate_file
        )

    results_items = [result.item for result in results.result]
    metrics = calculate_paper_metrics(results_items)
    logger.info(display_metrics(metrics, results_items))

    logger.info(f"Classification time elapsed: {timer_class.human}")
    logger.info(f"Total classification cost: ${results.cost:.10f}")

    classification_dir = output_dir / "classification"
    classification_dir.mkdir(parents=True, exist_ok=True)

    (classification_dir / "metrics.json").write_text(metrics.model_dump_json(indent=2))
    (classification_dir / "result.json").write_bytes(
        TypeAdapter(list[PromptResult[PaperResult]]).dump_json(results.result, indent=2)
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
    output_intermediate_file: Path,
) -> GPTResult[list[PromptResult[PaperResult]]]:
    """Classify Papers into approved/not approved using the generated graphs.

    Args:
        client: OpenAI client to use GPT
        model: GPT model code to use (must support Structured Outputs)
        user_prompt: User prompt template to use for classification to be filled
        paper_graphs: Graphs generated from the papers
        output_intermediate_file: File to write new results after each task is completed

    Returns:
        List of classified papers wrapped in a GPTResult.
    """

    results: list[PromptResult[PaperResult]] = []
    total_cost = 0

    tasks = [_classify_paper(client, model, pg, user_prompt) for pg in paper_graphs]

    for task in as_completed(tasks, desc="Classifying papers"):
        result = await task
        total_cost += result.cost

        results.append(result.result)
        append_intermediate_result(PaperResult, output_intermediate_file, result.result)

    return GPTResult(results, total_cost)


async def _classify_paper(
    client: AsyncOpenAI, model: str, pg: PaperGraph, user_prompt: PromptTemplate
) -> GPTResult[PromptResult[PaperResult]]:
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
    classified = result.result

    return GPTResult(
        result=PromptResult(
            item=PaperResult(
                title=pg.paper.title,
                abstract=pg.paper.abstract,
                reviews=pg.paper.reviews,
                sections=pg.paper.sections,
                approval=pg.paper.approval,
                y_true=pg.paper.is_approved(),
                y_pred=classified.approved if classified else False,
                rationale=classified.rationale if classified else "<error>",
            ),
            prompt=Prompt(system=CLASSIFY_SYSTEM_PROMPT, user=user_prompt_text),
        ),
        cost=result.cost,
    )
