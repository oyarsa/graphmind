"""Unit tests for the S2ORC module."""

import gzip
import tempfile
from pathlib import Path

import orjson
import pytest

from paper.s2orc import (
    acl,
    build_index,
    datasets,
    extract,
    filesizes,
    filter,
    search_papers,
)
from paper.util.serde import read_file_bytes


class TestExtract:
    """Test the extract module functions."""

    def test_extract_annotation_valid(self) -> None:
        """Test extraction of valid annotation from text."""
        text = "This is the abstract. This is the title. This is more text."
        annotations = {
            "abstract": '[{"start": 0, "end": 20}]',
            "title": '[{"start": 22, "end": 39}]',
        }

        result = extract._extract_annotation(text, annotations, "abstract")
        assert result == "This is the abstract"

        result = extract._extract_annotation(text, annotations, "title")
        assert result == "This is the title"

    def test_extract_annotation_multiple_segments(self) -> None:
        """Test extraction with multiple annotation segments."""
        text = "First part. Some other text. Second part."
        annotations = {
            "abstract": '[{"start": 0, "end": 10}, {"start": 29, "end": 41}]',
        }

        result = extract._extract_annotation(text, annotations, "abstract")
        assert result == "First part\nSecond part."

    def test_extract_annotation_missing_key(self) -> None:
        """Test extraction when annotation key is missing."""
        text = "Some text"
        annotations = {"abstract": '[{"start": 0, "end": 9}]'}

        result = extract._extract_annotation(text, annotations, "title")
        assert result is None

    def test_extract_annotation_invalid_json(self) -> None:
        """Test extraction with invalid JSON in annotation."""
        text = "Some text"
        annotations = {"abstract": "invalid json"}

        result = extract._extract_annotation(text, annotations, "abstract")
        assert result is None

    def test_extract_annotation_index_error(self) -> None:
        """Test extraction when indices are out of bounds."""
        text = "Short"
        annotations = {"abstract": '[{"start": 0, "end": 100}]'}

        result = extract._extract_annotation(text, annotations, "abstract")
        assert result == "Short"  # Returns full string when end index out of bounds

    def test_process_file_valid_data(self) -> None:
        """Test processing a file with valid data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.gz"

            # Create test data with proper structure
            test_data = {
                "content": {
                    "text": "Abstract content. Title content. Full paper text.",
                    "annotations": {
                        "abstract": '[{"start": 0, "end": 16}]',
                        "title": '[{"start": 18, "end": 32}]',
                        "venue": '[{"start": 34, "end": 48}]',
                    },
                }
            }

            # Write test data as JSON lines to gzipped file
            with gzip.open(test_file, "wt") as f:
                f.write(orjson.dumps(test_data).decode() + "\n")

            results = extract._process_file(test_file)

            assert len(results) == 1
            assert results[0]["abstract"] == "Abstract content"
            assert results[0]["title"] == "Title content."
            assert results[0]["venue"] == "ull paper text"
            assert (
                results[0]["text"]
                == "Abstract content. Title content. Full paper text."
            )

    def test_process_file_missing_content(self) -> None:
        """Test processing file with missing content fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.gz"

            # Create test data without content field
            test_data = {"other_field": "value"}

            with gzip.open(test_file, "wt") as f:
                f.write(orjson.dumps(test_data).decode() + "\n")

            results = extract._process_file(test_file)
            assert len(results) == 0

    def test_process_file_missing_annotations(self) -> None:
        """Test processing file with missing required annotations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.gz"

            # Create test data with missing venue annotation
            test_data = {
                "content": {
                    "text": "Some text",
                    "annotations": {
                        "abstract": '[{"start": 0, "end": 4}]',
                        "title": '[{"start": 5, "end": 9}]',
                        # Missing venue
                    },
                }
            }

            with gzip.open(test_file, "wt") as f:
                f.write(orjson.dumps(test_data).decode() + "\n")

            results = extract._process_file(test_file)
            assert len(results) == 0


class TestFilter:
    """Test the filter module functions."""

    def test_normalise_text(self) -> None:
        """Test text normalisation."""
        assert filter._normalise_text("Hello, World!") == "hello world"
        assert filter._normalise_text("Test-123") == "test123"
        assert filter._normalise_text("Multiple   Spaces") == "multiple   spaces"
        assert filter._normalise_text("Special@#$%Characters") == "specialcharacters"

    def test_get_acl_venues_exact_match(self) -> None:
        """Test ACL venue matching with exact matches."""
        venues = {
            "acl 2023",
            "empirical methods in natural language processing",
            "other conference",
        }

        acl_venues = filter._get_acl_venues(venues)

        assert "acl 2023" in acl_venues
        assert "empirical methods in natural language processing" in acl_venues
        assert "other conference" not in acl_venues

    def test_get_acl_venues_partial_match(self) -> None:
        """Test ACL venue matching with partial matches."""
        venues = {
            "proceedings of acl",
            "workshop on machine translation 2023",
            "unrelated venue",
        }

        acl_venues = filter._get_acl_venues(venues)

        assert "proceedings of acl" in acl_venues
        assert "workshop on machine translation 2023" in acl_venues
        assert "unrelated venue" not in acl_venues

    def test_get_unique_venues(self) -> None:
        """Test extraction of unique venues from files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            file1 = Path(tmpdir) / "file1.json.gz"
            file2 = Path(tmpdir) / "file2.json.gz"

            data1 = [
                {"venue": "ACL 2023"},
                {"venue": "EMNLP"},
                {"venue": ""},  # Empty venue
            ]
            data2 = [
                {"venue": "ACL 2023"},  # Duplicate
                {"venue": "COLING"},
                {"title": "No venue field"},  # Missing venue
            ]

            with gzip.open(file1, "wt") as f:
                f.write(orjson.dumps(data1).decode())

            with gzip.open(file2, "wt") as f:
                f.write(orjson.dumps(data2).decode())

            venues = filter._get_unique_venues([file1, file2])

            assert len(venues) == 3
            assert "acl 2023" in venues
            assert "emnlp" in venues
            assert "coling" in venues

    def test_process_acl_papers(self) -> None:
        """Test processing ACL papers from files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test file
            file1 = Path(tmpdir) / "file1.json.gz"

            data = [
                {"venue": "ACL 2023", "title": "Paper 1"},
                {"venue": "EMNLP", "title": "Paper 2"},
                {"venue": "Other Conference", "title": "Paper 3"},
            ]

            with gzip.open(file1, "wt") as f:
                f.write(orjson.dumps(data).decode())

            papers = filter.process_acl_papers([file1])

            assert len(papers) == 2
            assert any(p["title"] == "Paper 1" for p in papers)
            assert any(p["title"] == "Paper 2" for p in papers)
            assert not any(p["title"] == "Paper 3" for p in papers)

    def test_save_acl_papers(self) -> None:
        """Test saving ACL papers to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "output.json.gz"

            papers = [
                {"title": "Paper 1", "venue": "ACL"},
                {"title": "Paper 2", "venue": "EMNLP"},
            ]

            filter.save_acl_papers(papers, output_file)

            assert output_file.exists()

            # Read and verify
            saved_papers = orjson.loads(read_file_bytes(output_file))

            assert len(saved_papers) == 2
            assert saved_papers[0]["title"] == "Paper 1"
            assert saved_papers[1]["title"] == "Paper 2"


class TestBuildIndex:
    """Test the build_index module."""

    def test_build_index_from_files(self) -> None:
        """Test building index from multiple files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            input_dir.mkdir()
            output_file = Path(tmpdir) / "index.json"

            # Create test files
            file1 = input_dir / "file1.json.gz"
            file2 = input_dir / "file2.json.gz"

            data1 = [
                {"title": "Paper One", "abstract": "Abstract 1"},
                {"title": "Paper Two", "abstract": "Abstract 2"},
            ]
            data2 = [
                {"title": "Paper Three", "abstract": "Abstract 3"},
                {"abstract": "No title"},  # Missing title
            ]

            with gzip.open(file1, "wt") as f:
                f.write(orjson.dumps(data1).decode())

            with gzip.open(file2, "wt") as f:
                f.write(orjson.dumps(data2).decode())

            # Run build_index.main through the function
            build_index.main(input_dir, output_file)

            # Check output
            index = orjson.loads(read_file_bytes(output_file))

            assert len(index) == 3
            assert index["Paper One"] == "file1.json.gz"
            assert index["Paper Two"] == "file1.json.gz"
            assert index["Paper Three"] == "file2.json.gz"


class TestSearchPapers:
    """Test the search_papers module."""

    def test_preprocess_title(self) -> None:
        """Test title preprocessing."""
        assert (
            search_papers._preprocess_title("The Quick Brown Fox") == "quick brown fox"
        )
        assert search_papers._preprocess_title("A Study on NLP") == "study nlp"
        assert (
            search_papers._preprocess_title("Machine-Learning: An Overview")
            == "machine learning overview"
        )
        assert search_papers._preprocess_title("BERT: A New Model") == "bert new model"

    def test_paper_match_comparison(self) -> None:
        """Test PaperMatch comparison."""
        match1 = search_papers.PaperMatch(
            title_query="query", title_s2orc="match1", score=90
        )
        match2 = search_papers.PaperMatch(
            title_query="query", title_s2orc="match2", score=85
        )
        match3 = search_papers.PaperMatch(
            title_query="query", title_s2orc="match3", score=90
        )

        assert match2 < match1  # Lower score
        assert match1 < match3  # Same score, lexicographic comparison

    def test_topk_set(self) -> None:
        """Test TopKSet functionality."""
        topk = search_papers.TopKSet(k=3)

        # Add items
        match1 = search_papers.PaperMatch(title_query="q", title_s2orc="m1", score=80)
        match2 = search_papers.PaperMatch(title_query="q", title_s2orc="m2", score=90)
        match3 = search_papers.PaperMatch(title_query="q", title_s2orc="m3", score=85)
        match4 = search_papers.PaperMatch(title_query="q", title_s2orc="m4", score=95)
        match5 = search_papers.PaperMatch(
            title_query="q", title_s2orc="m5", score=70
        )  # Lower than all

        topk.add(match1)
        topk.add(match2)
        topk.add(match3)
        assert len(topk.items) == 3

        # Adding higher score should replace lowest
        topk.add(match4)
        items = topk.items
        assert len(items) == 3
        assert items[0].score == 95
        assert items[1].score == 90
        assert items[2].score == 85

        # Adding lower score should not change
        topk.add(match5)
        assert len(topk.items) == 3
        assert all(item.score >= 85 for item in topk.items)

    def test_topk_set_duplicates(self) -> None:
        """Test that TopKSet handles duplicates correctly."""
        topk = search_papers.TopKSet(k=3)

        match1 = search_papers.PaperMatch(title_query="q", title_s2orc="m1", score=90)
        match2 = search_papers.PaperMatch(
            title_query="q", title_s2orc="m1", score=90
        )  # Duplicate

        topk.add(match1)
        topk.add(match2)

        assert len(topk.items) == 1

    def test_search_paper_fuzzy(self) -> None:
        """Test fuzzy paper search."""
        papers_s2orc = {
            "machine learning for nlp",
            "deep learning models",
            "nlp research",
        }

        with tempfile.NamedTemporaryFile() as tmp:
            result = search_papers._search_paper_fuzzy(
                "machine learning nlp applications",
                papers_s2orc,
                min_fuzzy=50,
                output_intermediate_file=Path(tmp.name),
            )

            assert result is not None
            assert result.query == "machine learning nlp applications"
            assert len(result.matches) > 0
            assert any("machine learning" in m.title_s2orc for m in result.matches)

    def test_search_papers_fuzzy_single_threaded(self) -> None:
        """Test fuzzy papers search without multiprocessing."""
        papers_search = {"machine learning", "deep learning"}
        papers_s2orc = {
            "machine learning for nlp",
            "deep learning models",
            "nlp research",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "intermediate.jsonl"

            results = search_papers._search_papers_fuzzy(
                papers_search,
                papers_s2orc,
                min_fuzzy=50,
                output_intermediate_file=output_file,
                parallel=False,
            )

            assert len(results) == 2
            assert all(isinstance(r, search_papers.Paper) for r in results)
            assert any(r.query == "machine learning" for r in results)
            assert any(r.query == "deep learning" for r in results)


class TestACL:
    """Test the acl module."""

    def test_main_empty_conferences(self) -> None:
        """Test that main raises SystemExit with empty conferences."""
        with pytest.raises(SystemExit):
            acl.main(query="test", conferences=[], year="2023", limit=10)


class TestFilesizes:
    """Test the filesizes module."""

    def test_bytes_to_gib(self) -> None:
        """Test bytes to GiB conversion."""
        assert filesizes._bytes_to_gib(1024 * 1024 * 1024) == 1.0
        assert filesizes._bytes_to_gib(512 * 1024 * 1024) == 0.5
        assert filesizes._bytes_to_gib(0) == 0.0

    def test_get_file_size_sync(self) -> None:
        """Test getting file size calculation logic synchronously."""
        # Testing the conversion function instead
        assert filesizes._bytes_to_gib(1024 * 1024 * 1024) == 1.0
        assert filesizes._bytes_to_gib(2048 * 1024 * 1024) == 2.0


@pytest.mark.asyncio
async def test_async_file_context_manager() -> None:
    """Test AsyncFile context manager."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.bin"

        async with datasets.AsyncFile(test_file) as f:
            bytes_written = await f.write(b"test data")
            assert bytes_written == 9

        # Verify file was written
        assert test_file.exists()
        assert test_file.read_bytes() == b"test data"
