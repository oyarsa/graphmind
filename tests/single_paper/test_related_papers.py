"""Tests for related paper date filtering logic."""

from datetime import date

import pytest

from paper.single_paper.related_papers import (
    _arxiv_year_month,
    _is_published_not_after_main,
    _resolve_main_year_month,
)


@pytest.mark.parametrize(
    ("arxiv_id", "expected"),
    [
        ("2502.12345", (2025, 2)),
        ("2502.12345v3", (2025, 2)),
        ("cs/0501001", None),
        ("not-an-arxiv-id", None),
        (None, None),
    ],
)
def test_arxiv_year_month(
    arxiv_id: str | None, expected: tuple[int, int] | None
) -> None:
    """Extract year/month from modern arXiv IDs only."""
    assert _arxiv_year_month(arxiv_id) == expected


@pytest.mark.parametrize(
    (
        "main_year",
        "main_year_month",
        "candidate_year",
        "candidate_publication_date",
        "expected",
    ),
    [
        # Earlier year always allowed
        (2025, (2025, 2), 2024, None, True),
        # Later year always rejected
        (2025, (2025, 2), 2026, date(2026, 1, 1), False),
        # Same year with month precision on both sides
        (2025, (2025, 2), 2025, date(2025, 1, 10), True),
        (2025, (2025, 2), 2025, date(2025, 2, 28), True),
        (2025, (2025, 2), 2025, date(2025, 10, 1), False),
        # Same year without month precision on either side is rejected
        (2025, (2025, 2), 2025, None, False),
        (2025, None, 2025, date(2025, 1, 1), False),
        # Same year with inconsistent year in candidate date is rejected
        (2025, (2025, 2), 2025, date(2024, 12, 31), False),
        # Missing candidate year is rejected
        (2025, (2025, 2), None, date(2025, 1, 1), False),
    ],
)
def test_is_published_not_after_main(
    main_year: int,
    main_year_month: tuple[int, int] | None,
    candidate_year: int | None,
    candidate_publication_date: date | None,
    expected: bool,
) -> None:
    """Apply strict date filtering for novelty comparisons."""
    assert (
        _is_published_not_after_main(
            main_year=main_year,
            main_year_month=main_year_month,
            candidate_year=candidate_year,
            candidate_publication_date=candidate_publication_date,
        )
        == expected
    )


def test_resolve_main_year_month_prefers_arxiv_month_over_s2() -> None:
    """Use arXiv month first when both arXiv and S2 month precision are available."""
    assert _resolve_main_year_month(
        main_year=2025,
        publication_date=date(2025, 10, 1),
        arxiv_id="2502.12345v1",
    ) == (2025, 2)


def test_resolve_main_year_month_falls_back_to_s2_when_arxiv_mismatch() -> None:
    """Use S2 month when arXiv month exists but does not match the declared main year."""
    assert _resolve_main_year_month(
        main_year=2025,
        publication_date=date(2025, 10, 1),
        arxiv_id="2402.12345v1",
    ) == (2025, 10)
