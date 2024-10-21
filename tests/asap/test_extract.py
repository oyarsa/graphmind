# pyright: basic
# ruff: noqa: RUF001 - allow ambiguous unicode characters in string
import pytest
import spacy

from paper_hypergraph.asap.extract import (
    _CONTEXT_MIN_FUZZY,
    _SPACY_MODEL,
    SpacyModel,
    _contains_citation,
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
        # smith
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
        # momentum
        (
            """\
The vanilla SGD does not incorporate any curvature information about the objective \
function, resulting in slow convergence in certain cases. Momentum (Qian, 1999; \
Nesterov, 2013; Sutskever et al., 2013) or adaptive gradient-based methods \
(Duchi et al., 2011; Kingma and Ba, 2014) are sometimes used to rectify these issues. \
These adaptive methods can be seen as implicitly computing finite-difference \
approximations to the diagonal entries of the Hessian matrix (LeCun et al., 1998).\
""",
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


@pytest.mark.parametrize(
    "sentence,is_citation",
    [
        # sgd
        (
            """The vanilla SGD does not incorporate any curvature information about the objective function, resulting in slow convergence in certain cases.""",
            False,
        ),
        # momentum
        (
            """Momentum (Qian, 1999; Nesterov, 2013; Sutskever et al., 2013) or adaptive gradient-based methods (Duchi et al., 2011; Kingma and Ba, 2014) are sometimes used to rectify these issues.""",
            True,
        ),
        # lecun
        (
            """These adaptive methods can be seen as implicitly computing finite-difference approximations to the diagonal entries of the Hessian matrix (LeCun et al., 1998).""",
            True,
        ),
        # lstm
        (
            "The neural network has three LSTM (Hochreiter and Schmidhuber, 1997; Gers et al., 2002) layers followed by a fully-connected layer on the final layer’s last hidden state.",
            True,
        ),
        # mnist
        (
            "The MNIST data (28 × 28) is downsampled to (7 × 7) by average pooling.",
            False,
        ),
        # cifar10
        (
            "We evaluate the performance of the block-diagonal HF method on three deep architectures: a deep autoencoder on the MNIST dataset, a 3-layer LSTM for downsampled sequential MNIST classification, and a deep CNN based on the ResNet architecture for CIFAR10 classification.",
            False,
        ),
        # hessian-free
        (
            "We then demonstrate the advantage of the block-diagonal method over ordinary Hessian-free by comparing their performance at various curvature mini-batch sizes.",
            False,
        ),
        # cpo
        ("(1) Constrained Policy Optimization (CPO) (Achiam et al., 2017).", True),
        # latent
        (
            "(1) Note that we distinguish between hidden parameters x representing unobservable real-world properties and latent variables z carrying information intrinsic to our model.",
            False,
        ),
        # mil
        (
            "“healthy”, “cancer present”) can be used within the framework of multiple instance learning (MIL) (Dietterich et al., 1997; Amores, 2013; Xu et al., 2014).",
            True,
        ),
        # q-prop
        (
            "“c-” and “v-” denote conservative and aggressive Q-Prop variants as described in Section 3.2.",
            False,
        ),
        # vlae
        (
            "“Free bits” (Kingma et al., 2016) is used to improve optimization stability of VLAE (not for MAE).",
            True,
        ),
        # delta2
        ("∆2 is the common weight-decay frequently used in deep-learning.", False),
        # polyd
        ("≤ with probability at least (1− 1/poly(d)).", False),
        # gxy
        ("≈ x and GXY(GYX(y))", False),
        # papyan
        (
            "α ∈ R. This is inspired by the fact that the top eigenvalues of H can be well approximated using G (non-centered K) (Papyan, 2019; Sagun et al., 2017).",
            True,
        ),
    ],
    ids=[
        "sgd",
        "momentum",
        "lecun",
        "lstm",
        "mnist",
        "cifar10",
        "hessian-free",
        "cpo",
        "latent",
        "mil",
        "q-prop",
        "vlae",
        "delta2",
        "polyd",
        "gxy",
        "papyan",
    ],
)
def test_contains_citation(sentence: str, is_citation: bool) -> None:
    assert _contains_citation(sentence) == is_citation
