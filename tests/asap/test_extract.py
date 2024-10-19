import pytest

from paper_hypergraph.asap.extract import (
    _CONTEXT_MIN_FUZZY,
    _SPACY_MODEL,
    SpacyModel,
    _expand_citation_context,
    _split_paragraph,
)
from paper_hypergraph.util import load_spacy_model


@pytest.fixture(scope="session")
def spacy_model() -> SpacyModel:
    return load_spacy_model(_SPACY_MODEL)


def test_split_paragraph(spacy_model: SpacyModel) -> None:
    paragraph = """\
This is a sentence before the citation. \
Smith et al. (2020) found that AI can be very useful. \
This is a sentence after the citation. \
Another study by Johnson (2019) showed different results. \
The final sentence is here.\
"""
    sentences = _split_paragraph(spacy_model, paragraph)

    expected = [
        "This is a sentence before the citation.",
        "Smith et al. (2020) found that AI can be very useful.",
        "This is a sentence after the citation.",
        "Another study by Johnson (2019) showed different results.",
        "The final sentence is here.",
    ]

    assert sentences == expected


def test_expand_citation_context(spacy_model: SpacyModel) -> None:
    paragraph = """\
This is a sentence before the citation. \
Smith et al. (2020) found that AI can be very useful. \
This is a sentence after the citation. \
Another study by Johnson (2019) showed different results. \
The final sentence is here.\
"""
    citation_sentence = "Smith et al. (2020) found that AI can be very useful."

    result = _expand_citation_context(
        spacy_model, paragraph, citation_sentence, _CONTEXT_MIN_FUZZY, n=1
    )
    expected = """\
This is a sentence before the citation.
Smith et al. (2020) found that AI can be very useful.
This is a sentence after the citation.\
"""

    assert result == expected
