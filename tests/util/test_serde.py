"""Test (de)serialisation utilities."""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from pydantic import BaseModel

import pytest

from paper.util.serde import (
    Compress,
    PydanticProtocol,
    SerdeError,
    get_compressed_file_path,
    get_full_type_name,
    load_data_jsonl,
    load_data_single,
    save_data,
    save_data_jsonl,
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
    ("input_path", "compress", "expected_path"),
    [
        # Explicit compression tests
        ("/tmp/test.json", Compress.NONE, "/tmp/test.json"),
        ("/tmp/test.json", Compress.GZIP, "/tmp/test.json.gz"),
        ("/tmp/test.json", Compress.ZSTD, "/tmp/test.json.zst"),
        # Already has correct extension
        ("/tmp/test.json.gz", Compress.GZIP, "/tmp/test.json.gz"),
        ("/tmp/test.json.zst", Compress.ZSTD, "/tmp/test.json.zst"),
        # Inferred compression tests (compress=AUTO)
        ("/tmp/test.json", Compress.AUTO, "/tmp/test.json"),
        ("/tmp/test.json.gz", Compress.AUTO, "/tmp/test.json.gz"),
        ("/tmp/test.json.zst", Compress.AUTO, "/tmp/test.json.zst"),
        # Case insensitive inference
        ("/tmp/test.json.GZ", Compress.AUTO, "/tmp/test.json.GZ"),
        ("/tmp/test.json.ZST", Compress.AUTO, "/tmp/test.json.ZST"),
        ("/tmp/test.json.Gz", Compress.AUTO, "/tmp/test.json.Gz"),
        ("/tmp/test.json.ZsT", Compress.AUTO, "/tmp/test.json.ZsT"),
        # Multiple extensions
        ("/tmp/test.tar", Compress.GZIP, "/tmp/test.tar.gz"),
        ("/tmp/test.tar", Compress.ZSTD, "/tmp/test.tar.zst"),
        # Edge cases with existing compression wanting different compression
        ("/tmp/test.json.gz", Compress.ZSTD, "/tmp/test.json.gz.zst"),
        ("/tmp/test.json.zst", Compress.GZIP, "/tmp/test.json.zst.gz"),
    ],
)
def test_get_compressed_file_path(
    input_path: str, compress: Compress, expected_path: str
) -> None:
    """Test get_compressed_file_path with various inputs."""
    result = get_compressed_file_path(Path(input_path), compress)
    assert result == Path(expected_path)


class SerdeTestModel(BaseModel):
    """Model for testing serialization functions."""

    id: str
    name: str
    value: int


@pytest.mark.parametrize(
    ("compress_type", "expected_ext"),
    [
        (Compress.ZSTD, ".zst"),
        (Compress.GZIP, ".gz"),
        (Compress.NONE, ""),
    ],
)
def test_save_data_round_trip(compress_type: Compress, expected_ext: str) -> None:
    """Test that compression and decompression work correctly for all formats."""
    test_data = SerdeTestModel(id="test_id", name="test", value=42)

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
        loaded_data = load_data_single(actual_file, SerdeTestModel)
        assert loaded_data == test_data


class TestJsonlSerialization:
    """Test class for JSONL save and load operations."""

    @pytest.mark.parametrize("mode", ["w", "a"])
    @pytest.mark.parametrize("extension", ["", ".gz", ".zst"])
    def test_save_single(self, mode: Literal["w", "a"], extension: str) -> None:
        """Test save_data_jsonl with single data items."""
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            file_path = tmp_path / f"test.json{extension}"

            test_data = SerdeTestModel(id="1", name="test", value=42)

            save_data_jsonl(file_path, test_data, mode=mode)

            assert file_path.exists()

            loaded_data = load_data_jsonl(file_path, SerdeTestModel)
            assert len(loaded_data) == 1
            assert loaded_data[0] == test_data

    @pytest.mark.parametrize("mode", ["w", "a"])
    @pytest.mark.parametrize("extension", ["", ".gz", ".zst"])
    def test_save_sequence(self, mode: Literal["w", "a"], extension: str) -> None:
        """Test save_data_jsonl with sequence data."""
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            file_path = tmp_path / f"test.json{extension}"

            test_data = [
                SerdeTestModel(id="1", name="test1", value=42),
                SerdeTestModel(id="2", name="test2", value=84),
            ]

            save_data_jsonl(file_path, test_data, mode=mode)

            assert file_path.exists()

            loaded_data = load_data_jsonl(file_path, SerdeTestModel)
            assert len(loaded_data) == 2
            assert loaded_data == test_data

    def test_append_mode(self) -> None:
        """Test that append mode correctly adds to existing files."""
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            file_path = tmp_path / "test.json"

            # Save initial data
            initial_data = SerdeTestModel(id="1", name="test1", value=42)
            save_data_jsonl(file_path, initial_data, mode="w")

            # Append more data
            additional_data = SerdeTestModel(id="2", name="test2", value=84)
            save_data_jsonl(file_path, additional_data, mode="a")

            # Load and verify all data is present
            loaded_data = load_data_jsonl(file_path, SerdeTestModel)
            assert len(loaded_data) == 2
            assert loaded_data[0] == initial_data
            assert loaded_data[1] == additional_data

    def test_write_mode_overwrites(self) -> None:
        """Test that write mode overwrites existing files."""
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            file_path = tmp_path / "test.json"

            # Save initial data
            initial_data = SerdeTestModel(id="1", name="test1", value=42)
            save_data_jsonl(file_path, initial_data, mode="w")

            # Overwrite with new data
            new_data = SerdeTestModel(id="2", name="test2", value=84)
            save_data_jsonl(file_path, new_data, mode="w")

            # Load and verify only new data is present
            loaded_data = load_data_jsonl(file_path, SerdeTestModel)
            assert len(loaded_data) == 1
            assert loaded_data[0] == new_data

    @pytest.mark.parametrize("extension", ["", ".gz", ".zst"])
    def test_load_compression(self, extension: str) -> None:
        """Test load_data_jsonl with different compression formats."""
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            file_path = tmp_path / f"test.json{extension}"

            # Create test data
            test_data = [
                SerdeTestModel(id="1", name="test1", value=42),
                SerdeTestModel(id="2", name="test2", value=84),
            ]

            # Save and load data
            save_data_jsonl(file_path, test_data)
            loaded_data = load_data_jsonl(file_path, SerdeTestModel)

            assert loaded_data == test_data

    def test_load_empty_lines(self) -> None:
        """Test that load_data_jsonl skips empty lines."""
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            file_path = tmp_path / "test.json"

            # Create file with empty lines
            content = (
                '{"id": "1", "name": "test1", "value": 42}\n'
                "\n"
                '{"id": "2", "name": "test2", "value": 84}\n'
                "   \n"
                '{"id": "3", "name": "test3", "value": 126}\n'
            )
            file_path.write_text(content)

            # Load data
            loaded_data = load_data_jsonl(file_path, SerdeTestModel)

            # Should only load valid lines
            assert len(loaded_data) == 3
            assert loaded_data[0].id == "1"
            assert loaded_data[1].id == "2"
            assert loaded_data[2].id == "3"

    def test_load_partial_validation_errors(self) -> None:
        """Test that load_data_jsonl handles partial validation errors gracefully."""
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            file_path = tmp_path / "test.json"

            # Create file with invalid JSON line
            content = (
                '{"id": "1", "name": "test1", "value": 42}\n'
                '{"id": "2", "name": "test2", "value": "not_a_number"}\n'
            )
            file_path.write_text(content)

            # Should not raise an error and should return the valid entry
            loaded_data = load_data_jsonl(file_path, SerdeTestModel)
            assert len(loaded_data) == 1
            assert loaded_data[0].id == "1"

    def test_load_all_invalid_validation_raises_error(self) -> None:
        """Test that load_data_jsonl raises SerdeError when all lines are invalid."""
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            file_path = tmp_path / "test.json"

            # Create file with all invalid lines
            content = (
                '{"id": "1", "name": "test1", "value": "not_a_number"}\n'
                '{"id": "2", "name": "test2", "value": "also_not_a_number"}\n'
            )
            file_path.write_text(content)

            # Should raise SerdeError
            with pytest.raises(SerdeError, match="All lines in .* failed validation"):
                load_data_jsonl(file_path, SerdeTestModel)

    def test_load_mixed_valid_invalid_json(self) -> None:
        """Test that load_data_jsonl handles mixed valid/invalid lines gracefully."""
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            file_path = tmp_path / "test.json"

            # Create file with mix of valid and invalid JSON lines
            content = '{"id": "1", "name": "test1", "value": 42}\nnot valid json\n'
            file_path.write_text(content)

            # Should return only the valid entries
            loaded_data = load_data_jsonl(file_path, SerdeTestModel)
            assert len(loaded_data) == 1
            assert loaded_data[0].id == "1"

    def test_load_all_invalid_json_raises_error(self) -> None:
        """Test that load_data_jsonl raises SerdeError when all lines are invalid JSON."""
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            file_path = tmp_path / "test.json"

            # Create file with all invalid JSON lines
            content = "not valid json\nalso not valid json\n"
            file_path.write_text(content)

            # Should raise SerdeError since no valid lines exist
            with pytest.raises(SerdeError, match="All lines in .* failed validation"):
                load_data_jsonl(file_path, SerdeTestModel)
