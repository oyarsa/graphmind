"""Test (de)serialisation utilities."""

from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import BaseModel

import pytest

from paper.util.serde import (
    Compress,
    PydanticProtocol,
    get_full_type_name,
    load_data_single,
    save_data,
)


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


@pytest.mark.parametrize(
    ("compress_type", "expected_ext"),
    [
        (Compress.ZSTD, ".zst"),
        (Compress.GZIP, ".gz"),
        (Compress.NONE, ""),
    ],
)
def test_compression_round_trip(compress_type: Compress, expected_ext: str) -> None:
    """Test that compression and decompression work correctly for all formats."""

    class TestModel(BaseModel):
        name: str
        value: int

    test_data = TestModel(name="test", value=42)

    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        file_path = tmp_path / f"test_{compress_type.value}.json"
        save_data(file_path, test_data, compress=compress_type)

        # Check file extension was added correctly
        actual_file = (
            file_path.with_suffix(file_path.suffix + expected_ext)
            if expected_ext
            else file_path
        )
        assert actual_file.exists(), f"File not created: {actual_file}"

        # Test loading
        loaded_data = load_data_single(actual_file, TestModel)
        assert loaded_data == test_data
