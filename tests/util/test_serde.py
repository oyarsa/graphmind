"""Test (de)serialisation utilities."""

from pydantic import BaseModel

from paper.util.serde import PydanticProtocol, get_full_type_name


def test_pydantic_protocol() -> None:
    """Test if PydanticProtocol is compatible with BaseModel."""

    class TestModel(BaseModel):
        pass

    b = TestModel()
    assert isinstance(b, PydanticProtocol)


def test_get_full_type_name() -> None:
    """Test that `_get_full_type_name` gets the correct name for type."""
    from paper import peerread as pr
    from paper.peerread import Paper as PRPaper
    from paper.semantic_scholar import Paper as S2Paper

    assert get_full_type_name(S2Paper) == "paper.semantic_scholar.model.Paper"
    assert get_full_type_name(PRPaper) == "paper.peerread.model.Paper"
    assert (
        get_full_type_name(pr.CitationContext) == "paper.peerread.model.CitationContext"
    )
