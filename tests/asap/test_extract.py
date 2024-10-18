from paper_hypergraph.asap.extract import _expand_citation_context


def test_extract_citation_context() -> None:
    paragraph = """\
This is a sentence before the citation. \
Smith et al. (2020) found that AI can be very useful. \
This is a sentence after the citation. \
Another study by Johnson (2019) showed different results. \
The final sentence is here.\
"""
    citation_sentence = "Smith et al. (2020) found that AI can be very useful."

    result = _expand_citation_context(paragraph, citation_sentence, n=1)
    expected = """\
Smith et al. (2020) found that AI can be very useful.
This is a sentence after the citation.\
"""

    assert result == expected
