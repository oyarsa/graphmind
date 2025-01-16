"""Test if PydanticProtocol is valid for BaseModel."""

from pydantic import BaseModel

from paper.util.serde import PydanticProtocol


def test_pydantic_protocol() -> None:
    """Test if PydanticProtocol is compatible with BaseModel."""

    class TestModel(BaseModel):
        pass

    b = TestModel()
    assert isinstance(b, PydanticProtocol)
