"""Single paper processing pipeline for ORC dataset.

This module provides functionality to process a single paper through the complete PETER
pipeline, including S2 reference enhancement, GPT annotations, context classification,
related paper discovery, and summarisation.

Main entry point:
    main: CLI command for processing papers

Core pipeline functions:
    annotate_paper_pipeline: Complete PETER pipeline processing
    process_paper_from_query: End-to-end processing from title/ID
    process_paper_from_selection: Processing from pre-selected paper
"""

from paper.embedding import DEFAULT_SENTENCE_MODEL
from paper.gpt.extract_graph import EvaluationResult
from paper.single_paper.abstract_evaluation import abstract_evaluation
from paper.single_paper.cli import main
from paper.single_paper.graph_evaluation import ProgressCallback
from paper.single_paper.graph_evaluation_multi import (
    EvaluationResultMulti,
    process_paper_from_selection_multi,
)
from paper.single_paper.paper_retrieval import (
    ArxivRateLimitError,
    arxiv_id_from_url,
    fetch_s2_paper_info,
    search_arxiv_papers,
    search_arxiv_papers_filtered,
)
from paper.single_paper.pipeline import process_paper_from_selection

__all__ = (
    "DEFAULT_SENTENCE_MODEL",
    "ArxivRateLimitError",
    "EvaluationResult",
    "EvaluationResultMulti",
    "ProgressCallback",
    "abstract_evaluation",
    "arxiv_id_from_url",
    "fetch_s2_paper_info",
    "main",
    "process_paper_from_selection",
    "process_paper_from_selection_multi",
    "search_arxiv_papers",
    "search_arxiv_papers_filtered",
)
