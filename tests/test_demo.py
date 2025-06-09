"""Test demo data conversion."""

from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from paper.demo_data import _clean_conference, _get_year
from paper.types import PaperSectionProtocol


@dataclass
class MockPaper:
    """Mock paper for testing _get_year function."""

    year: int | None
    conference: str
    id: str = "mock-id"
    title: str = "Mock Title"
    abstract: str = "Mock abstract"
    label: int = 1
    rating: int = 1
    rationale: str = "Mock rationale"
    approval: bool | None = True
    sections: Sequence[PaperSectionProtocol] = ()


@pytest.mark.parametrize(
    ("conference", "expected"),
    [
        # Standard cases with year suffix
        ("ICML2023", "ICML"),
        ("NeurIPS2022", "NeurIPS"),
        ("ICLR2024", "ICLR"),
        ("AAAI2021", "AAAI"),
        # Cases without year suffix
        ("ICML", "ICML"),
        ("NeurIPS", "NeurIPS"),
        ("ICLR", "ICLR"),
        ("AAAI", "AAAI"),
        # Edge cases
        ("", ""),
        ("ABC", "ABC"),
        ("A123", "A123"),  # Only 3 chars, no change
        # Cases with non-digit suffix
        ("ICMLabc", "ICMLabc"),
        ("NeurIPSxyz", "NeurIPSxyz"),
        ("ICLR202a", "ICLR202a"),
        # Cases with mixed content
        ("Workshop2023", "Workshop"),
        ("Symposium1999", "Symposium"),
        ("Meeting2000", "Meeting"),
    ],
)
def test_clean_conference(conference: str, expected: str) -> None:
    """Test conference name cleaning."""
    assert _clean_conference(conference) == expected


@pytest.mark.parametrize(
    ("year", "conference", "expected"),
    [
        # Cases where year is provided
        (2023, "ICML2022", 2023),
        (2021, "NeurIPS", 2021),
        (2020, "", 2020),
        (1999, "Conference1998", 1999),
        # Cases where year is None but conference has year
        (None, "ICML2023", 2023),
        (None, "NeurIPS2022", 2022),
        (None, "ICLR2024", 2024),
        (None, "AAAI2021", 2021),
        (None, "Workshop2020", 2020),
        # Cases where year is None and conference has no valid year
        (None, "ICML", None),
        (None, "NeurIPS", None),
        (None, "", None),
        (None, "ABC", None),
        (None, "Conferenceabc", None),
        (None, "Workshop202a", None),
        (None, "Sym", None),
        # Edge cases
        (None, "A123", None),  # Only 3 digits, invalid
        (None, "Conference0000", 0),
        (None, "Meeting9999", 9999),
    ],
)
def test_get_year(year: int | None, conference: str, expected: int | None) -> None:
    """Test year extraction from paper."""
    paper = MockPaper(year=year, conference=conference)
    assert _get_year(paper) == expected
