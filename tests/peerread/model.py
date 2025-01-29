"""Test PeerRead module, including paper content transformation."""

from paper.peerread.model import (
    clean_maintext,
    compress_whitespace,
    normalise_paragraphs,
    remove_line_numbers,
    remove_page_numbers,
)


def test_remove_line_numbers() -> None:
    assert remove_line_numbers("123 Hello\n456 World") == "Hello\nWorld"
    assert remove_line_numbers("No numbers here") == "No numbers here"


def test_compress_whitespace() -> None:
    assert compress_whitespace("Too    many    spaces") == "Too many spaces"
    assert compress_whitespace("Multiple\n\n\n\nlines") == "Multiple\n\nlines"


def test_remove_page_numbers() -> None:
    assert remove_page_numbers("Content\n123\nMore") == "Content\n\nMore"
    assert remove_page_numbers("Not a page 123 number") == "Not a page 123 number"


def test_normalize_paragraphs() -> None:
    assert normalise_paragraphs("P1\n\nP2  \n\n\nP3") == "P1\n\nP2\n\nP3"
    assert normalise_paragraphs("Single paragraph") == "Single paragraph"


def test_clean_maintext() -> None:
    """Test full paper content cleaning pipeline."""
    input_text = """123 First line
456   Multiple   spaces
Broken
sentence
1 Section with numbers
789

(Citation  2020)"""

    expected = """First line
Multiple spaces
Broken
sentence
Section with numbers

(Citation 2020)"""

    assert clean_maintext(input_text) == expected
