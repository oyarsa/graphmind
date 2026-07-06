"""Regression tests for the single_paper batch CLI wrapper.

These mock `evaluate_papers` (the actual library function, tested elsewhere)
to prove `run_batch`'s own on-disk contract — filenames, which results get
written vs collected as failures — survived being refactored into a thin
wrapper around it, without needing real network/API access.
"""

from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, patch

import pytest

from paper.gpt.extract_graph import EvaluationResult
from paper.single_paper.api import (
    EvaluationError,
    PaperEvaluation,
    PaperEvaluationConfig,
)
from paper.single_paper.batch import run_batch


def _fake_result() -> EvaluationResult:
    """Sentinel result for tests.

    `run_batch` never inspects a successful result's contents, only routes it
    to `save_data`, so a plain object stands in for a real `EvaluationResult`.
    """
    return cast(EvaluationResult, object())


@pytest.mark.asyncio
async def test_run_batch_writes_results_and_collects_failures(tmp_path: Path) -> None:
    """Successful papers get a report file; failed ones land in failures.json."""
    titles_file = tmp_path / "titles.json"
    titles_file.write_text('["Good Paper", "Bad Paper"]')
    output_dir = tmp_path / "out"

    fake_result = _fake_result()
    canned = [
        PaperEvaluation(title="Good Paper", result=fake_result),
        PaperEvaluation(
            title="Bad Paper",
            result=EvaluationError(title="Bad Paper", message="not found"),
        ),
    ]

    with (
        patch(
            "paper.single_paper.batch.evaluate_papers",
            AsyncMock(return_value=canned),
        ) as mock_evaluate,
        patch("paper.single_paper.batch.save_data") as mock_save,
    ):
        await run_batch(
            titles_file,
            output_dir,
            output_dir / "s2_cache.json",
            4,
            PaperEvaluationConfig(),
        )

    mock_evaluate.assert_awaited_once()

    assert mock_save.call_count == 2
    result_call, failures_call = mock_save.call_args_list

    result_path, result_value = result_call.args
    assert result_path == output_dir / "000_good-paper.json"
    assert result_value is fake_result

    failures_path, failures_value = failures_call.args
    assert failures_path == output_dir / "failures.json"
    assert failures_value == {"Bad Paper": "not found"}


@pytest.mark.asyncio
async def test_run_batch_skips_failures_file_when_all_succeed(tmp_path: Path) -> None:
    """No failures.json is written when every paper succeeds."""
    titles_file = tmp_path / "titles.json"
    titles_file.write_text('["Paper One", "Paper Two"]')
    output_dir = tmp_path / "out"

    canned = [
        PaperEvaluation(title="Paper One", result=_fake_result()),
        PaperEvaluation(title="Paper Two", result=_fake_result()),
    ]

    with (
        patch(
            "paper.single_paper.batch.evaluate_papers",
            AsyncMock(return_value=canned),
        ),
        patch("paper.single_paper.batch.save_data") as mock_save,
    ):
        await run_batch(
            titles_file,
            output_dir,
            output_dir / "s2_cache.json",
            4,
            PaperEvaluationConfig(),
        )

    assert mock_save.call_count == 2
    saved_paths = {call.args[0] for call in mock_save.call_args_list}
    assert saved_paths == {
        output_dir / "000_paper-one.json",
        output_dir / "001_paper-two.json",
    }
