"""Test `_get_full_type_name` function from `util.serde`."""

from paper.util.serde import _get_full_type_name


def test_get_full_type_name() -> None:
    """Test that `_get_full_type_name` gets the correct name for type."""
    from paper import peerread as pr
    from paper.peerread import Paper as PRPaper
    from paper.semantic_scholar import Paper as S2Paper

    assert _get_full_type_name(S2Paper) == "paper.semantic_scholar.model.Paper"
    assert _get_full_type_name(PRPaper) == "paper.peerread.model.Paper"
    assert (
        _get_full_type_name(pr.CitationContext)
        == "paper.peerread.model.CitationContext"
    )
