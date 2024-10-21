import pytest
import spacy

from paper_hypergraph.asap.extract import (
    _CONTEXT_MIN_FUZZY,
    _SPACY_MODEL,
    SpacyModel,
    _expand_citation_context,
    _split_paragraph,
)


@pytest.fixture(scope="session")
def spacy_model() -> SpacyModel:
    return spacy.load(_SPACY_MODEL)


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


@pytest.mark.parametrize(
    "paragraph, sentence, expected",
    [
        (
            """\
This is a sentence before the citation. \
Smith et al. (2020) found that AI can be very useful. \
This is a sentence after the citation. \
Another study by Johnson (2019) showed different results. \
The final sentence is here.\
""",
            "Smith et al. (2020) found that AI can be very useful.",
            """\
This is a sentence before the citation.
Smith et al. (2020) found that AI can be very useful.
This is a sentence after the citation.\
""",
        ),
        (
            """\
The vanilla SGD does not incorporate any curvature information about the objective \
function, resulting in slow convergence in certain cases. Momentum (Qian, 1999; \
Nesterov, 2013; Sutskever et al., 2013) or adaptive gradient-based methods \
(Duchi et al., 2011; Kingma and Ba, 2014) are sometimes used to rectify these issues. \
These adaptive methods can be seen as implicitly computing finite-difference \
approximations to the diagonal entries of the Hessian matrix (LeCun et al., 1998).\
""",
            # Regression test to make sure the citation sentence is being expanded to
            # the same value as before (i.e. the "paragraph" is the result of the
            # expansion). Note that the output isn't exactly the same: the expansion
            # process gives each sentence in its own line.
            """\
Momentum (Qian, 1999; Nesterov, 2013; Sutskever et al., 2013) or adaptive gradient-based \
methods (Duchi et al., 2011; Kingma and Ba, 2014) are sometimes used to rectify these \
issues.\
""",
            """\
The vanilla SGD does not incorporate any curvature information about the objective \
function, resulting in slow convergence in certain cases.\

Momentum (Qian, 1999; Nesterov, 2013; Sutskever et al., 2013) or adaptive gradient-based \
methods (Duchi et al., 2011; Kingma and Ba, 2014) are sometimes used to rectify these \
issues.\
""",
        ),
    ],
    ids=["smith", "momentum"],
)
def test_expand_citation_context(
    paragraph: str, sentence: str, expected: str, spacy_model: SpacyModel
) -> None:
    result = _expand_citation_context(
        spacy_model, paragraph, sentence, _CONTEXT_MIN_FUZZY, n=1
    )
    assert result == expected
