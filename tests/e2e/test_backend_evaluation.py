"""End-to-end tests for the backend evaluation API.

These tests use FastAPI's TestClient to run the app in-process and verify the response
structure and data quality. They are marked as slow and require --runslow to run.

Each test makes one request and runs multiple checks, collecting all failures
before reporting them together.
"""

import json
import os
import re
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi.testclient import TestClient

from paper.backend.api import app

# Skip if API key not set
if not os.getenv("OPENAI_API_KEY"):
    pytest.skip("OPENAI_API_KEY environment variable not set", allow_module_level=True)

EVALUATE_ENDPOINT = "/mind/evaluate"


@dataclass
class CheckResult:
    """Result of a single check."""

    name: str
    passed: bool
    message: str = ""


def _empty_check_list() -> list[CheckResult]:
    return []


@dataclass
class EvaluationChecker:
    """Collects checks on an evaluation response and reports all failures."""

    paper_id: str
    response_data: dict[str, Any]
    checks: list[CheckResult] = field(default_factory=_empty_check_list)

    def check(self, name: str, condition: bool, message: str = "") -> None:
        """Add a check result."""
        self.checks.append(CheckResult(name=name, passed=condition, message=message))

    def check_equal(self, name: str, actual: Any, expected: Any) -> None:
        """Check that actual equals expected."""
        passed = actual == expected
        message = f"expected {expected!r}, got {actual!r}" if not passed else ""
        self.checks.append(CheckResult(name=name, passed=passed, message=message))

    def check_in_range(
        self, name: str, value: float, min_val: float, max_val: float
    ) -> None:
        """Check that value is in range [min_val, max_val]."""
        passed = min_val <= value <= max_val
        message = f"expected {min_val}-{max_val}, got {value}" if not passed else ""
        self.checks.append(CheckResult(name=name, passed=passed, message=message))

    def check_not_empty(self, name: str, value: Any) -> None:
        """Check that value is not empty/None."""
        passed = bool(value)
        message = "value is empty or None" if not passed else ""
        self.checks.append(CheckResult(name=name, passed=passed, message=message))

    def check_max_length(self, name: str, collection: list[Any], max_len: int) -> None:
        """Check that collection has at most max_len items."""
        actual_len = len(collection)
        passed = actual_len <= max_len
        message = f"expected at most {max_len}, got {actual_len}" if not passed else ""
        self.checks.append(CheckResult(name=name, passed=passed, message=message))

    def check_all_years_before(
        self, name: str, items: list[dict[str, Any]], max_year: int
    ) -> None:
        """Check that all items have year <= max_year."""
        violations = [
            (item.get("title", "?")[:40], item.get("year"))
            for item in items
            if item.get("year") and item["year"] > max_year
        ]
        passed = len(violations) == 0
        message = f"papers from future: {violations}" if not passed else ""
        self.checks.append(CheckResult(name=name, passed=passed, message=message))

    def check_no_latex(self, name: str, text: str) -> None:
        """Check that text doesn't contain raw LaTeX commands."""
        latex_patterns = [
            r"\\cite[pt]?\{",
            r"\\footnote\{",
            r"\\textbf\{",
            r"\\emph\{",
            r"~\\",
        ]
        found = [p for p in latex_patterns if re.search(p, text)]
        passed = len(found) == 0
        message = f"found LaTeX patterns: {found}" if not passed else ""
        self.checks.append(CheckResult(name=name, passed=passed, message=message))

    def check_no_journal_titles(self, name: str, items: list[dict[str, Any]]) -> None:
        """Check that no item has a journal/venue name as title."""
        journal_patterns = [
            r"^arXiv preprint",
            r"^CoRR$",
            r"^Proc\.",
            r"^In Proceedings",
            r"^IEEE Trans",
            r"^ACM ",
        ]
        violations = [
            item.get("title", "")
            for item in items
            if any(
                re.match(p, item.get("title", ""), re.IGNORECASE)
                for p in journal_patterns
            )
        ]
        passed = len(violations) == 0
        message = f"journal names as titles: {violations}" if not passed else ""
        self.checks.append(CheckResult(name=name, passed=passed, message=message))

    def get_failures(self) -> list[CheckResult]:
        """Return list of failed checks."""
        return [c for c in self.checks if not c.passed]

    def assert_all_passed(self) -> None:
        """Assert all checks passed, reporting all failures."""
        failures = self.get_failures()
        if failures:
            failure_msgs = [f"  - {f.name}: {f.message}" for f in failures]
            msg = f"Paper {self.paper_id}: {len(failures)} check(s) failed:\n"
            msg += "\n".join(failure_msgs)
            pytest.fail(msg)


def fetch_evaluation(
    client: TestClient,
    arxiv_id: str,
    title: str,
    *,
    filter_by_date: bool = True,
    llm_model: str = "gpt-4o-mini",
) -> dict[str, Any]:
    """Fetch evaluation for a paper via SSE endpoint."""
    params = {
        "id": arxiv_id,
        "title": title,
        "llm_model": llm_model,
        "filter_by_date": str(filter_by_date).lower(),
    }

    # Parse SSE events properly
    result_data: dict[str, Any] = {}
    current_event: str | None = None
    current_data: list[str] = []

    with client.stream("GET", EVALUATE_ENDPOINT, params=params) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            # Empty line signals end of event
            if not line:
                if current_data and current_event == "complete":
                    data_str = "\n".join(current_data)
                    result_data = json.loads(data_str)
                    break
                current_event = None
                current_data = []
                continue

            # Skip comments/keep-alive
            if line.startswith(":"):
                continue

            # Parse field
            if line.startswith("event:"):
                current_event = line[6:].strip()
            elif line.startswith("data:"):
                current_data.append(line[5:].strip())

    return result_data


@pytest.fixture(scope="module")
def test_client() -> Generator[TestClient, None, None]:
    """Create a TestClient for the FastAPI app."""
    with TestClient(app) as client:
        yield client


@pytest.mark.slow
def test_attention_paper_evaluation(test_client: TestClient) -> None:
    """Test evaluation of 'Attention Is All You Need' (2017).

    This is a well-known older paper, good for testing:
    - Date filtering (should not get 2020+ papers as related)
    - Title extraction from BibTeX references
    - Evidence distribution
    """
    arxiv_id = "1706.03762"
    title = "Attention Is All You Need"

    response = fetch_evaluation(test_client, arxiv_id, title)
    checker = EvaluationChecker(paper_id=arxiv_id, response_data=response)

    # Get nested data - structure is response["result"]["result"]["paper"]
    outer_result = response.get("result", {})
    result = outer_result.get("result", {})
    paper = result.get("paper", {})
    related = result.get("related", [])
    structured_eval = paper.get("structured_evaluation", {})
    supporting = structured_eval.get("supporting_evidence", [])
    contradictory = structured_eval.get("contradictory_evidence", [])

    # Basic structure checks
    checker.check_not_empty("result exists", result)
    checker.check_not_empty("paper exists", paper)
    checker.check_not_empty("related exists", related)
    checker.check_not_empty("structured_evaluation exists", structured_eval)

    # Paper metadata
    paper_year = paper.get("year")
    checker.check_equal("paper year", paper_year, 2017)

    # Structured evaluation fields
    checker.check_in_range("label in range", structured_eval.get("label", 0), 1, 5)
    checker.check_not_empty(
        "paper_summary exists", structured_eval.get("paper_summary")
    )
    checker.check_not_empty("conclusion exists", structured_eval.get("conclusion"))

    # Evidence distribution: max 5 per type
    checker.check_max_length("supporting evidence max 5", supporting, 5)
    checker.check_max_length("contradictory evidence max 5", contradictory, 5)

    # Date filtering: related papers should be from 2017 or earlier
    checker.check_all_years_before("related papers date filter", related, 2017)
    checker.check_all_years_before("supporting evidence date filter", supporting, 2017)
    checker.check_all_years_before(
        "contradictory evidence date filter", contradictory, 2017
    )

    # Title quality: no journal names as titles
    checker.check_no_journal_titles("related paper titles", related)

    # LaTeX cleanup in summaries
    for i, ev in enumerate(supporting[:3]):  # Check first 3
        text = ev.get("text", "")
        checker.check_no_latex(f"supporting[{i}] no LaTeX", text)

    for i, ev in enumerate(contradictory[:3]):
        text = ev.get("text", "")
        checker.check_no_latex(f"contradictory[{i}] no LaTeX", text)

    checker.assert_all_passed()


@pytest.mark.slow
def test_recent_paper_evaluation(test_client: TestClient) -> None:
    """Test evaluation of a more recent paper.

    Tests that recent papers also work and have proper structure.
    Using a 2024 paper to test that date filtering allows recent related papers.
    """
    # A 2024 paper about LLMs
    arxiv_id = "2401.02954"
    title = "Sleeper Agents: Training Deceptive LLMs"

    response = fetch_evaluation(test_client, arxiv_id, title)
    checker = EvaluationChecker(paper_id=arxiv_id, response_data=response)

    outer_result = response.get("result", {})
    result = outer_result.get("result", {})
    paper = result.get("paper", {})
    related = result.get("related", [])
    structured_eval = paper.get("structured_evaluation", {})
    supporting = structured_eval.get("supporting_evidence", [])
    contradictory = structured_eval.get("contradictory_evidence", [])

    # Basic structure
    checker.check_not_empty("result exists", result)
    checker.check_not_empty("paper exists", paper)
    checker.check_not_empty("structured_evaluation exists", structured_eval)

    # Paper year should be 2024
    paper_year = paper.get("year")
    checker.check_equal("paper year", paper_year, 2024)

    # Structured evaluation
    checker.check_in_range("label in range", structured_eval.get("label", 0), 1, 5)
    checker.check_not_empty(
        "paper_summary exists", structured_eval.get("paper_summary")
    )

    # Evidence limits
    checker.check_max_length("supporting evidence max 5", supporting, 5)
    checker.check_max_length("contradictory evidence max 5", contradictory, 5)

    # Date filtering: related should be <= 2024
    checker.check_all_years_before("related papers date filter", related, 2024)

    # Title quality
    checker.check_no_journal_titles("related paper titles", related)

    checker.assert_all_passed()


@pytest.mark.slow
def test_evidence_source_distribution(test_client: TestClient) -> None:
    """Test that evidence comes from both semantic and citation sources.

    The expected distribution is up to 3 semantic + up to 2 citations per type,
    with backfill if one source has fewer results.
    """
    arxiv_id = "1706.03762"
    title = "Attention Is All You Need"

    response = fetch_evaluation(test_client, arxiv_id, title)
    checker = EvaluationChecker(paper_id=arxiv_id, response_data=response)

    outer_result = response.get("result", {})
    result = outer_result.get("result", {})
    structured_eval = result.get("paper", {}).get("structured_evaluation", {})
    supporting = structured_eval.get("supporting_evidence", [])
    contradictory = structured_eval.get("contradictory_evidence", [])

    # Count sources
    supporting_semantic = sum(1 for e in supporting if e.get("source") == "semantic")
    supporting_citations = sum(1 for e in supporting if e.get("source") == "citations")
    contradictory_semantic = sum(
        1 for e in contradictory if e.get("source") == "semantic"
    )
    contradictory_citations = sum(
        1 for e in contradictory if e.get("source") == "citations"
    )

    # At least some evidence should exist
    total_evidence = len(supporting) + len(contradictory)
    checker.check(
        "has some evidence",
        total_evidence > 0,
        f"total evidence: {total_evidence}",
    )

    # Semantic should be capped at 3 per type
    checker.check(
        "supporting semantic <= 3",
        supporting_semantic <= 3,
        f"got {supporting_semantic}",
    )
    checker.check(
        "contradictory semantic <= 3",
        contradictory_semantic <= 3,
        f"got {contradictory_semantic}",
    )

    # Citations should be capped at 2 per type (before backfill)
    # Note: with backfill, could be more if semantic has fewer
    checker.check(
        "supporting citations reasonable",
        supporting_citations <= 5,
        f"got {supporting_citations}",
    )
    checker.check(
        "contradictory citations reasonable",
        contradictory_citations <= 5,
        f"got {contradictory_citations}",
    )

    checker.assert_all_passed()


@pytest.mark.slow
def test_summary_quality(test_client: TestClient) -> None:
    """Test that summaries don't start with 'The Related Paper...' pattern.

    This was a bug where summaries would echo the prompt terminology.
    """
    arxiv_id = "1706.03762"
    title = "Attention Is All You Need"

    response = fetch_evaluation(test_client, arxiv_id, title)
    checker = EvaluationChecker(paper_id=arxiv_id, response_data=response)

    outer_result = response.get("result", {})
    result = outer_result.get("result", {})
    structured_eval = result.get("paper", {}).get("structured_evaluation", {})
    supporting = structured_eval.get("supporting_evidence", [])
    contradictory = structured_eval.get("contradictory_evidence", [])

    bad_patterns = [
        r"^The [Rr]elated [Pp]aper",
        r"^The [Mm]ain [Pp]aper",
        r"^This related paper",
    ]

    for i, ev in enumerate(supporting):
        text = ev.get("text", "")
        for pattern in bad_patterns:
            has_bad_pattern = bool(re.match(pattern, text))
            checker.check(
                f"supporting[{i}] no bad pattern '{pattern}'",
                not has_bad_pattern,
                f"text starts with: {text[:50]}..." if has_bad_pattern else "",
            )

    for i, ev in enumerate(contradictory):
        text = ev.get("text", "")
        for pattern in bad_patterns:
            has_bad_pattern = bool(re.match(pattern, text))
            checker.check(
                f"contradictory[{i}] no bad pattern '{pattern}'",
                not has_bad_pattern,
                f"text starts with: {text[:50]}..." if has_bad_pattern else "",
            )

    checker.assert_all_passed()
