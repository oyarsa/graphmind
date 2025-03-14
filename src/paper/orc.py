"""Fetch conference paper data from the OpenReview API and download LaTeX from arXiv.

The process for retrieving the whole data is running the subcommands in this order:
- reviews
- latex
- parse
- preprocess

To download all the relevant conference information, use:
- download-all
- latex-all
- parse-all
- preprocess

Use the same output/data directory for all of them.
"""

# pyright: basic
import contextlib
import dataclasses as dc
import datetime as dt
import io
import itertools
import json
import logging
import multiprocessing as mp
import re
import subprocess
import tarfile
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from functools import partial
from pathlib import Path
from typing import Annotated, Any, overload

import arxiv  # type: ignore
import backoff
import nltk  # type: ignore
import openreview as openreview_v1
import requests
import typer
from openreview import api as openreview_v2
from tqdm import tqdm

from paper import peerread as pr
from paper.util import Timer, groupby, setup_logging
from paper.util.serde import save_data

logger = logging.getLogger("paper.openreview")

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
    help=__doc__,
)


@app.command(no_args_is_help=True)
def latex(
    reviews_file: Annotated[
        Path,
        typer.Option(
            "--arxiv-ids",
            "-i",
            help="Path to data from arXiv used to download the LaTeX files.",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output", "-o", help="Output directory for arXiv LaTeX source files."
        ),
    ],
    max_papers: Annotated[
        int | None,
        typer.Option(
            "--num-papers",
            "-n",
            help="How many papers to process. If None, processes all.",
        ),
    ] = None,
    clean_run: Annotated[
        bool,
        typer.Option("--clean", help="If True, ignore previously downloaded files."),
    ] = False,
) -> None:
    """Download LaTeX source files from arXiv data.

    The arXiv data is fetched with the `arxiv` subcommand.

    By default, skips re-downloading files that already exist in the output directory.
    You can override this with `--clean`.
    """
    papers: list[dict[str, str]] = json.loads(reviews_file.read_text())[:max_papers]
    arxiv_results = [
        ArxivResult(
            openreview_title=p["openreview_title"],
            arxiv_title=p["arxiv_title"],
            id=p["arxiv_id"],
        )
        for p in papers
    ]

    if clean_run:
        downloaded_prev = set()
    else:
        downloaded_prev = {
            path.stem for path in output_dir.glob("*.tar.gz") if path.is_file()
        }

    downloaded_n = 0
    skipped_n = 0
    failed_n = 0
    output_dir.mkdir(exist_ok=True, parents=True)

    for result in tqdm(arxiv_results, desc="Downloading LaTeX sources"):
        if result.arxiv_title in downloaded_prev:
            skipped_n += 1
            continue

        try:
            if data := _download_latex_source(result.id):
                (output_dir / f"{result.arxiv_title}.tar.gz").write_bytes(data)
                downloaded_n += 1
            else:
                logger.warning(f"Invalid tar.gz file for {result.arxiv_title}")
                failed_n += 1
        except Exception as e:
            logger.warning(
                f"Error downloading LaTeX source for {result.arxiv_title}"
                f" - {type(e).__name__}: {e}"
            )
            failed_n += 1

    logger.info(f"Downloaded : {downloaded_n}")
    logger.info(f"Skipped    : {skipped_n}")
    logger.info(f"Failed     : {failed_n}")


@dc.dataclass(frozen=True, kw_only=True)
class ArxivResult:
    """Result of querying the arXiv API with a paper title from OpenReview."""

    openreview_title: str
    arxiv_title: str
    id: str


def _get_arxiv(openreview_titles: list[str], batch_size: int) -> dict[str, ArxivResult]:
    """Get mapping of OpenReview paper title (casefold) to arXiv ID that exist on arXiv."""
    arxiv_client = arxiv.Client()

    arxiv_results: list[ArxivResult] = []
    for openreview_title_batch in tqdm(
        list(itertools.batched(openreview_titles, batch_size)),
        desc="Querying arXiv",
    ):
        arxiv_results.extend(_batch_search_arxiv(arxiv_client, openreview_title_batch))

    return {_normalise_title(r.openreview_title): r for r in arxiv_results}


def _batch_search_arxiv(
    client: arxiv.Client, openreview_titles: Sequence[str]
) -> list[ArxivResult]:
    """Search multiple OpenReview titles at once on arXiv and return matching results."""
    or_queries = " OR ".join(
        f'ti:"{openreview_title}"' for openreview_title in openreview_titles
    )
    query = f"({or_queries})"
    results_map: dict[str, ArxivResult] = {}
    openreview_titles = [_normalise_title(t) for t in openreview_titles]

    try:
        for result in client.results(
            arxiv.Search(query=query, max_results=len(openreview_titles))
        ):
            arxiv_title = result.title
            for openreview_title in openreview_titles:
                if _similar_titles(openreview_title, arxiv_title):
                    results_map[openreview_title] = ArxivResult(
                        id=result.entry_id.split("/")[-1],
                        openreview_title=openreview_title,
                        arxiv_title=arxiv_title,
                    )
                    break
    except Exception as e:
        logger.warning(f"Error during batch search on arXiv: {e}")

    return [
        result
        for openreview_title in openreview_titles
        if (result := results_map.get(openreview_title))
    ]


def _similar_titles(title1: str, title2: str) -> bool:
    """Check if two titles are similar (case-insensitive, stripped)."""
    t1 = title1.casefold().strip()
    t2 = title2.casefold().strip()
    return t1 == t2 or t1 in t2 or t2 in t1


@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def _download_latex_source(arxiv_id: str) -> bytes | None:
    """Download LaTeX source (tar.gz) from arXiv for the given arXiv ID."""
    url = f"https://arxiv.org/src/{arxiv_id}"
    response = requests.get(url)
    response.raise_for_status()
    content = response.content

    if not _is_valid_targz(content):
        return None
    return content


def _is_valid_targz(content: bytes) -> bool:
    """Check if the given content is a valid tar.gz file.

    Args:
        content: Binary content (bytes) to check.

    Returns:
        True if the content is a valid tar.gz file, False otherwise.
    """
    try:
        file_like_object = io.BytesIO(content)
        with tarfile.open(fileobj=file_like_object, mode="r:gz") as tar:
            # If we can list the contents, it's a valid tar.gz file
            tar.getnames()
    except (tarfile.ReadError, tarfile.CompressionError, EOFError):
        return False
    else:
        return True


@app.command(no_args_is_help=True)
def latex_all(
    data_dir: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            help="Path to directory with data downloaded with `download-all`.",
        ),
    ],
    max_papers: Annotated[
        int | None,
        typer.Option(
            "--num-papers",
            "-n",
            help="How many papers to process. If None, processes all.",
        ),
    ] = None,
    clean_run: Annotated[
        bool,
        typer.Option("--clean", help="If True, ignore previously downloaded files."),
    ] = False,
) -> None:
    """Download LaTeX source files from arXiv for data downloaded from `download-all`.

    The `--input` parameter should be the same directory as the `output_dir` from
    `download-all`.

    By default, skips re-downloading files that already exist in the output directory.
    You can override this with `--clean`.
    """
    venue_dirs = list(data_dir.iterdir())
    for i, venue_dir in enumerate(venue_dirs, 1):
        logger.info("\n>>> [%d/%d] %s", i, len(venue_dirs), venue_dir.name)

        arxiv_file = venue_dir / "openreview_arxiv.json"
        if not arxiv_file.exists():
            logger.warning("No arXiv data file for: %s", venue_dir)
            continue

        latex(arxiv_file, venue_dir / "files", max_papers, clean_run)


def get_conference_submissions(
    venue_id: str,
) -> list[openreview_v1.Note | openreview_v2.Note]:
    """Get all submissions with reviews for `venue_id`.

    Tries both API versions and submissions/blind submissions.
    """
    sections = ["Submission", "Blind_Submission"]

    client_v2 = openreview_v2.OpenReviewClient(baseurl="https://api2.openreview.net")
    for section in sections:
        if submissions := client_v2.get_all_notes(
            invitation=f"{venue_id}/-/{section}", details="replies"
        ):
            return submissions

    client_v1 = openreview_v1.Client(baseurl="https://api.openreview.net")
    for section in sections:
        if submissions := client_v1.get_all_notes(
            invitation=f"{venue_id}/-/{section}", details="directReplies"
        ):
            return submissions

    return []


RATING_KEYS = [
    "contribution",
    "technical_novelty_and_significance",
    "empirical_novelty_and_significance",
]
"""Valid keys for target ratings.

ICLR 2024 and 2025, and NeurIPS 2022-2024 use 'contribution'.
ICLR 2022 and 2023 use technical/empirical novelty.

Our target rating is the max of the available ratings.
"""


@app.command(no_args_is_help=True)
def reviews(
    output_dir: Annotated[
        Path, typer.Argument(help="Output directory for OpenReview reviews file.")
    ],
    venue_id: Annotated[str, typer.Option("--venue", help="Venue ID to fetch data.")],
    batch_size: Annotated[
        int, typer.Option(help="Batch size to query the arXiv API.")
    ] = 50,
    max_papers: Annotated[
        int | None,
        typer.Option(
            "--max-papers",
            "-n",
            help="Maximum number of papers to query. If None, use all.",
        ),
    ] = None,
    query_arxiv: Annotated[
        bool, typer.Option("--arxiv/--no-arxiv", help="Query arXiv for papers")
    ] = True,
) -> None:
    """Download all reviews and metadata for papers from a conference in OpenReview.

    Also queries arXiv to find which papers are available there, adding their arXiv IDs.

    Requires the following environment variables to be set:
    - OPENREVIEW_USERNAME
    - OPENREVIEW_PASSWORD

    These are the standard credentials you use to log into the OpenReview website.

    Supports conference using both v1 and v2 APIs.
    Example venue IDs:

    API v1:
    - ICLR.cc/2022/Conference
    - ICLR.cc/2023/Conference
    - NeurIPS.cc/2022/Conference
    - NeurIPS.cc/2023/Conference

    API v2:
    - ICLR.cc/2024/Conference
    - ICLR.cc/2025/Conference
    - NeurIPS.cc/2024/Conference
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    submissions_raw = get_conference_submissions(venue_id)
    if not submissions_raw:
        logger.warning("No submissions available for %s", venue_id)
        return

    submissions_all = [_note_to_dict(s) for s in submissions_raw]
    logger.info("Submissions - all: %d", len(submissions_all))
    (output_dir / "openreview_all.json").write_text(json.dumps(submissions_all))

    submissions_valid = [s for s in submissions_all if _is_valid(s, RATING_KEYS)]
    logger.info("Submissions - valid: %d", len(submissions_valid))
    (output_dir / "openreview_valid.json").write_text(json.dumps(submissions_valid))

    if not submissions_valid:
        logger.warning("No valid submissions for %s", venue_id)
        return

    openreview_titles = [
        openreview_title
        for paper in submissions_valid[:max_papers]
        if (openreview_title := _get_value(paper["content"], "title"))
    ]

    if not query_arxiv:
        logger.info("Skipping arXiv query.")
        return

    logger.info("Querying arXiv for %d paper titles", len(openreview_titles))
    openreview_to_arxiv = _get_arxiv(openreview_titles, batch_size)
    logger.info("Found %d papers on arXiv", len(openreview_to_arxiv))
    (output_dir / "openreview_arxiv_raw.json").write_text(
        json.dumps([dc.asdict(v) for v in openreview_to_arxiv.values()])
    )

    submissions_with_arxiv: list[dict[str, Any]] = []
    for paper in submissions_valid:
        openreview_title = _get_value(paper["content"], "title") or ""
        if arxiv_result := openreview_to_arxiv.get(_normalise_title(openreview_title)):
            submissions_with_arxiv.append({
                **paper,
                "arxiv_id": arxiv_result.id,
                "openreview_title": arxiv_result.openreview_title,
                "arxiv_title": arxiv_result.arxiv_title,
            })

    logger.info("Submissions with arXiv IDs: %d", len(submissions_with_arxiv))

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "openreview_arxiv.json").write_text(
        json.dumps(submissions_with_arxiv)
    )


@app.command(no_args_is_help=True)
def download_all(
    output_dir: Annotated[
        Path, typer.Argument(help="Output directory for OpenReview reviews file.")
    ],
    query_arxiv: Annotated[
        bool, typer.Option("--arxiv/--no-arxiv", help="Query arXiv for papers")
    ] = True,
) -> None:
    """Download reviews and arXiv IDs for the following conferences.

    - ICLR 2022, 2023, 2024, 2025
    - NeurIPS 2022, 2023, 2024

    Each one gets its own subdirectory under `output_dir`.
    """
    # <venue ID, directory name>
    conferences = [
        ("ICLR.cc/2022", "iclr2022"),
        ("ICLR.cc/2023", "iclr2023"),
        ("ICLR.cc/2024", "iclr2024"),
        ("ICLR.cc/2025", "iclr2025"),
        ("NeurIPS.cc/2022", "neurips2022"),
        ("NeurIPS.cc/2023", "neurips2023"),
        ("NeurIPS.cc/2024", "neurips2024"),
    ]

    with tqdm(total=len(conferences)) as pbar:
        for venue_id, dir_name in conferences:
            pbar.set_description(f"{venue_id}")

            dir_path = output_dir / dir_name
            if dir_path.exists():
                pbar.update(1)
                continue

            reviews(dir_path, f"{venue_id}/Conference", query_arxiv=query_arxiv)
            pbar.update(1)


def _normalise_title(title: str) -> str:
    return title.casefold()


def _note_to_dict(note: openreview_v1.Note | openreview_v2.Note) -> dict[str, Any]:
    """Convert OpenReview API `Note` to dict with additional `details` object."""
    return note.to_json() | {"details": note.details}


def _is_valid(paper: dict[str, Any], rating_keys: Iterable[str]) -> bool:
    """Check if paper has at least one review with `rating`, PDF, title and abstract."""
    return all((
        any(_has_rating(paper, key) for key in rating_keys),
        _has_rating(paper, "decision"),
        _has_field(paper, "pdf"),
        _has_field(paper, "title"),
        _has_field(paper, "abstract"),
    ))


def _get_reviews(paper: dict[str, Any]) -> list[dict[str, Any]]:
    """Get reviews from paper object: either `replies` or `directReplies`."""
    details = paper["details"]
    if "replies" in details:
        return paper["details"]["replies"]
    if "directReplies" in details:
        return paper["details"]["directReplies"]

    raise ValueError("Paper does not have review replies")


def _has_rating(paper: dict[str, Any], name: str) -> bool:
    """Check if any review in `paper` has the rating with given `name`."""
    return any(r["content"].get(name) for r in _get_reviews(paper))


def _get_value(item: dict[str, Any], key: str) -> Any | None:
    value = item.get(key, {})
    if isinstance(value, dict):
        return value.get("value")
    return value


def _has_field(paper: dict[str, Any], name: str) -> bool:
    """Check if the `paper` has a field with `name` and non-empty value."""
    value = _get_value(paper["content"], name)
    if isinstance(value, str):
        value = value.strip()
    return bool(value)


@dc.dataclass(frozen=True, kw_only=True)
class Section:
    """A section in the paper."""

    heading: str
    content: str


@dc.dataclass(frozen=True, kw_only=True)
class Reference:
    """A bibliographic reference."""

    title: str
    year: str | None
    authors: list[str]
    citation_contexts: list[str] = dc.field(default_factory=list)


@dc.dataclass(frozen=True, kw_only=True)
class LatexPaper:
    """Parsed paper content in Markdown with reference citations."""

    title: str
    sections: list[Section]
    references: list[Reference]


class SentenceSplitter:
    """A class to handle sentence splitting with NLTK.

    Handles initialization of NLTK resources and provides a simple interface
    for sentence tokenization.
    """

    def __init__(self) -> None:
        """Initialise the sentence splitter.

        Use NLTK's `punkt` tokeniser for this. Automatically downloads it if unavailable.
        """
        try:
            # Try to use the tokenizer
            nltk.tokenize.sent_tokenize("Test")  # type: ignore
        except LookupError:
            # If the tokenizer is not available, download it
            nltk.download("punkt")  # type: ignore
            nltk.download("punkt_tab")  # type: ignore

    def split(self, text: str) -> list[str]:
        """Split text into sentences.

        Args:
            text: The text to split into sentences.

        Returns:
            A list of sentences.
        """
        return nltk.tokenize.sent_tokenize(text)  # type: ignore


# Citation patterns for various LaTeX citation commands
CITATION_PATTERNS = [
    r"\\cite\{[^}]+\}",
    r"\\citep\{[^}]+\}",
    r"\\citet\{[^}]+\}",
    r"\\citealp\{[^}]+\}",
    r"\\citealt\{[^}]+\}",
    r"\\citeauthor\{[^}]+\}",
    r"\\citeyear\{[^}]+\}",
    r"\\parencite\{[^}]+\}",
    r"\\textcite\{[^}]+\}",
]

# Compile regex patterns for reuse
CITATION_REGEX = re.compile("|".join(CITATION_PATTERNS))
LATEX_CMD_PATTERN = re.compile(
    r"\\(?!cite|citep|citet|citealp|citealt|citeauthor|citeyear|parencite|textcite)[a-zA-Z]+(\[[^\]]*\])?(\{[^}]*\})?"
)
MATH_ENV_PATTERN = re.compile(r"\$\$.*?\$\$|\$.*?\$", re.DOTALL)


def _clean_latex(text: str) -> str:
    """Remove LaTeX commands except citation commands.

    Args:
        text: The LaTeX text to clean.

    Returns:
        The cleaned text with LaTeX commands removed.
    """
    # Replace math environments with placeholders
    text = MATH_ENV_PATTERN.sub(" MATH_PLACEHOLDER ", text)

    # Remove general LaTeX commands (except citation commands)
    text = LATEX_CMD_PATTERN.sub(" ", text)

    # Remove LaTeX environments
    text = re.sub(r"\\begin\{[^}]+\}.*?\\end\{[^}]+\}", " ", text, flags=re.DOTALL)

    # Normalize whitespace
    return re.sub(r"\s+", " ", text).strip()


def _get_citation_keys(sentence: str) -> set[str]:
    """Extract citation keys from a sentence.

    Args:
        sentence: The sentence to extract citation keys from.

    Returns:
        A set of citation keys found in the sentence.
    """
    keys: set[str] = set()
    for citation in CITATION_REGEX.finditer(sentence):
        citation_text = citation.group(0)
        # Extract the keys between curly braces
        if match := re.search(r"\{([^}]+)\}", citation_text):
            citation_keys_str = match.group(1)
            # Handle multiple keys separated by commas
            for key in citation_keys_str.split(","):
                if key := key.strip():
                    keys.add(key)
    return keys


def _extract_citation_sentences(
    splitter: SentenceSplitter, paragraph: str
) -> dict[str, list[str]]:
    """Extract sentences containing citations from a paragraph.

    Args:
        paragraph: The LaTeX paragraph to extract from.
        splitter: The sentence splitter to use.

    Returns:
        A dictionary mapping citation keys to the sentences that cite them.
    """
    cleaned_text = _clean_latex(paragraph)
    sentences = splitter.split(cleaned_text)
    citation_sentences: dict[str, list[str]] = defaultdict(list)
    for sentence in sentences:
        if CITATION_REGEX.search(sentence):
            for key in _get_citation_keys(sentence):
                citation_sentences[key].append(sentence)

    return citation_sentences


def _find_main_tex(directory: Path) -> Path:
    """Find entrypoint TeX file containing include directives for sections."""
    tex_files = list(directory.glob("**/*.tex"))

    if not tex_files:
        raise FileNotFoundError("No .tex files found")
    if len(tex_files) == 1:
        return tex_files[0]

    main_candidates = [
        tex_file
        for tex_file in tex_files
        if "\\begin{document}" in tex_file.read_text(errors="ignore")
    ]

    if not main_candidates:
        raise FileNotFoundError("No main .tex file found in the directory.")
    if len(main_candidates) == 1:
        return main_candidates[0]

    # Prefer a file explicitly named "main.tex"
    for candidate in main_candidates:
        if candidate.name.lower() == "main.tex":
            return candidate

    # Otherwise, choose the candidate with the largest file size
    main_candidates.sort(key=lambda path: path.stat().st_size, reverse=True)
    return main_candidates[0]


def _process_latex_file(
    file_path: Path, root_dir: Path, processed_files: set[Path] | None = None
) -> str:
    """Recursively process TeX inclusion directives."""
    if processed_files is None:
        processed_files = set()

    abs_path = file_path.resolve()
    if abs_path in processed_files:
        return ""  # Avoid circular references
    processed_files.add(abs_path)

    try:
        content = abs_path.read_text(errors="replace")
        content = _remove_latex_comments(content)
    except Exception as e:
        logger.debug(f"Error reading {abs_path}: {e}")
        return ""

    include_pattern = re.compile(r"\\(?:input|include)\{([^}]+)\}")

    def include_replacer(match: re.Match[str]) -> str:
        included_path = Path(match.group(1).strip())
        if not included_path.name.endswith(".tex"):
            included_path = included_path.with_name(f"{included_path.name}.tex")

        return _process_latex_file(root_dir / included_path, root_dir, processed_files)

    return include_pattern.sub(include_replacer, content)


def _remove_arxiv_styling(latex_content: str) -> str:
    """Remove custom arxiv styling directives and problematic LaTeX commands."""
    # Remove lines with \usepackage or \RequirePackage that reference 'arxiv'
    no_package = re.sub(
        r"^(\\(?:usepackage|RequirePackage)\s*(\[[^\]]*\])?\s*\{[^}]*arxiv[^}]*\}.*\n?)",
        "",
        latex_content,
        flags=re.MULTILINE,
    )

    # Remove lines that explicitly include arxiv.sty
    no_arxiv = re.sub(
        r"^(\\(?:input|include)\s*\{[^}]*arxiv\.sty[^}]*\}.*\n?)",
        "",
        no_package,
        flags=re.MULTILINE,
    )

    # Remove tcolorbox package
    no_tcolorbox_pkg = re.sub(
        r"^(\\(?:usepackage|RequirePackage)\s*(\[[^\]]*\])?\s*\{[^}]*tcolorbox[^}]*\}.*\n?)",
        "",
        no_arxiv,
        flags=re.MULTILINE,
    )

    # Remove \newtcolorbox declarations
    no_newtcolorbox = re.sub(
        r"\\newtcolorbox\{[^}]+\}(\[[^\]]*\])?\{[^}]+\}",
        "",
        no_tcolorbox_pkg,
        flags=re.MULTILINE,
    )

    # Remove tcolorbox environments
    no_tcolorbox_env = re.sub(
        r"\\begin\{tcolorbox\}(\[[^\]]*\])?.*?\\end\{tcolorbox\}",
        "",
        no_newtcolorbox,
        flags=re.DOTALL,
    )

    return no_tcolorbox_env  # noqa: RET504


def _remove_latex_comments(tex_string: str) -> str:
    """Remove lines that are entirely commented out from a TeX document string.

    A line is considered fully commented if it only contains whitespace before the
    comment character '%'.
    """
    return "\n".join(
        line for line in tex_string.splitlines() if not line.lstrip().startswith("%")
    )


def _convert_latex_to_markdown(latex_content: str, title: str) -> str | None:
    """Convert LaTeX file to Markdown with pandoc."""
    with tempfile.TemporaryDirectory() as tmp_dir_:
        tmp_dir = Path(tmp_dir_)
        latex_file = tmp_dir / "input.tex"
        markdown_file = tmp_dir / "output.md"

        latex_file.write_text(latex_content)

        # Run pandoc to convert LaTeX to Markdown
        pandoc_cmd = [
            "pandoc",
            "--quiet",
            "--wrap=none",
            "-f",
            "latex",
            "-t",
            "markdown",
            "-o",
            str(markdown_file),
            str(latex_file),
        ]

        try:
            subprocess.run(pandoc_cmd, check=True, timeout=PANDOC_CMD_TIMEOUT)
            return markdown_file.read_text(errors="ignore")
        except subprocess.TimeoutExpired:
            logger.warning("Command timeout during pandoc conversion. Paper: %s", title)
            return None
        except subprocess.CalledProcessError as e:
            logger.warning(
                "Error during pandoc conversion. Paper: %s. Error: %s", title, e
            )
            return None


def _find_bib_files(base_dir: Path, latex_content: str) -> list[Path]:
    """Find all bibliography files referenced in the LaTeX document."""
    bib_paths: list[Path] = []
    bib_pattern = re.compile(r"\\bibliography\{([^}]+)\}")
    bib_matches = bib_pattern.findall(latex_content)

    for match in bib_matches:
        # Multiple bibliography files can be comma-separated
        for bib_name_ in match.split(","):
            bib_name = bib_name_.strip()
            # If no extension, add .bib
            if not Path(bib_name).suffix:
                bib_name = f"{bib_name}.bib"

            # Search for the .bib file in the project directory
            for bib_path in base_dir.glob(f"**/{bib_name}"):
                if bib_path.exists():
                    bib_paths.append(bib_path)
                    break

    return bib_paths


def _extract_bibliography_from_bibfiles(
    bib_paths: list[Path], tmpdir: Path
) -> dict[str, Reference]:
    """Extract bibliography entries from .bib files."""
    references: dict[str, Reference] = {}

    for bib_path in bib_paths:
        if not bib_path.exists():
            continue

        logger.debug("Processing bib file: %s", bib_path)
        try:
            bib_content = bib_path.read_text(errors="ignore")

            # Fix problematic citations with accents
            bib_content = re.sub(
                r"@\w+\{[^{,]*?\\['`^\"]\w+",
                lambda m: m.group(0).replace("\\", ""),
                bib_content,
            )
            # Remove lines with question marks at the beginning of fields
            bib_content = re.sub(r"\s+\?\s*(\w+)\s*=\s*\{[^}]*\},", "", bib_content)

            tmp_file = tmpdir / "clean.bib"
            tmp_file.write_text(bib_content)

            pandoc_cmd = [
                "pandoc",
                "--quiet",
                "-f",
                "bibtex",
                "-t",
                "csljson",
                str(tmp_file),
            ]
            result = subprocess.run(
                pandoc_cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=PANDOC_CMD_TIMEOUT,
            )
            bib_data = json.loads(result.stdout)
        except subprocess.TimeoutExpired:
            logger.warning("Command timeout during bibliography file processing.")
            continue
        except subprocess.CalledProcessError as e:
            logger.debug(f"Error processing bibliography file {bib_path}: {e.stderr}")
            continue
        except json.JSONDecodeError as e:
            logger.debug(f"Error parsing bibliography data from {bib_path}: {e}")
            continue

        for entry in bib_data:
            # Extract the citation key (bib id)
            citation_key = entry.get("id", "")
            if not citation_key:
                continue

            # Extract reference information
            title: str | None = entry.get("title")
            if title is None:
                continue

            # Extract year from issued date-parts if available
            year: str | None = None
            if "issued" in entry and "date-parts" in entry["issued"]:
                date_parts = entry["issued"]["date-parts"]
                if date_parts and date_parts[0]:
                    year = date_parts[0][0]

            # Extract author names
            authors: list[str] = []
            if "author" in entry:
                for author in entry["author"]:
                    family = author.get("family", "")
                    given = author.get("given", "")
                    if family and given:
                        authors.append(f"{given} {family}")
                    elif family:
                        authors.append(family)
                    elif given:
                        authors.append(given)

            references[citation_key] = Reference(
                title=title,
                year=year,
                authors=authors,
            )

    return references


def _extract_bibliography_from_bibitems(latex_content: str) -> dict[str, Reference]:
    r"""Extract bibliography entries from \\bibitem commands in the LaTeX content."""
    references: dict[str, Reference] = {}

    # Find the bibliography environment
    bibenv_pattern = re.compile(
        r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", re.DOTALL
    )
    bibenv_match = bibenv_pattern.search(latex_content)

    if not bibenv_match:
        return references

    bibenv_content = bibenv_match.group(0)

    # Extract bibitem entries
    bibitem_pattern = re.compile(
        r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}(.*?)(?=\\bibitem|\n\n|\\end\{thebibliography\}|$)",
        re.DOTALL,
    )

    for match in bibitem_pattern.finditer(bibenv_content):
        citation_key = match.group(1).strip()
        entry_text = match.group(2).strip()

        if not citation_key or not entry_text:
            continue

        # Extract year (look for 4 consecutive digits, possibly in parentheses)
        year = None
        year_match = re.search(r"(?:\()?\b(\d{4})\b(?:\))?", entry_text)
        if year_match:
            year = year_match.group(1)

        # Extract title (common patterns in bibliography entries)
        title = "Unknown Title"
        title_patterns = [
            r'"([^"]+)"',  # "Title"
            r"``([^']+)''",  # ``Title''
            r"``([^']+)\"",  # ``Title"
            r"\"([^']+)''",  # "Title''
            r"\{\\em ([^}]+)\}",  # {\em Title}
            r"\{\\it ([^}]+)\}",  # {\it Title}
            r"\\textit\{([^}]+)\}",  # \textit{Title}
            r"\\emph\{([^}]+)\}",  # \emph{Title}
        ]

        for pattern in title_patterns:
            title_match = re.search(pattern, entry_text)
            if title_match:
                title = title_match.group(1).strip()
                break

        # Extract authors (heuristic)
        authors: list[str] = []

        # Get text before the title or year marker
        author_text = entry_text
        if title in entry_text and title != "Unknown Title":
            author_text = entry_text.split(title)[0]
        elif year and year in entry_text:
            author_text = entry_text.split(year)[0]

        # Look for patterns like "Author1, Author2, and Author3"
        author_text = re.sub(
            r"\\[a-z]+(\[[^\]]*\])?(\{[^}]*\})?", "", author_text
        )  # Remove LaTeX commands
        author_text = author_text.split(".")[0]  # Authors often end with a period

        # Split by common separators
        if " and " in author_text.lower():
            parts = re.split(r" and ", author_text, flags=re.IGNORECASE)
            authors = [p.strip() for p in parts if p.strip()]
        else:
            # If no "and", try splitting by commas
            parts = [p.strip() for p in author_text.split(",")]
            authors = [p for p in parts if p and not p.isdigit() and len(p) > 1]

        references[citation_key] = Reference(
            title=title,
            year=year,
            authors=authors,
        )

    return references


def _extract_citations_and_contexts(
    splitter: SentenceSplitter, latex_content: str
) -> dict[str, list[str]]:
    """Extract citation keys and their context sentences from the LaTeX content."""
    citation_contexts: dict[str, list[str]] = defaultdict(list)
    paragraphs = re.split(r"\n\s*\n", latex_content)

    for paragraph_ in paragraphs:
        paragraph = paragraph_.strip()
        if not paragraph or not CITATION_REGEX.search(paragraph):
            continue

        # Extract citations at sentence level only if citations exist
        paragraph_citations = _extract_citation_sentences(splitter, paragraph)

        # Merge with overall citation contexts
        for key, sentences in paragraph_citations.items():
            for sentence in sentences:
                if sentence not in citation_contexts[key]:
                    citation_contexts[key].append(sentence)

    return citation_contexts


def _split_markdown_sections(markdown_content: str) -> list[Section]:
    """Split markdown content by top-level sections only."""
    sections: list[Section] = []

    # Split the markdown by top-level headings (# heading)
    # This regex looks for lines that start with exactly one # followed by a space
    section_pattern = re.compile(r"^# (.+?)(\s*\{[^}]+\})?$", re.MULTILINE)

    # Find all section headings
    section_matches = list(section_pattern.finditer(markdown_content))

    # If no sections found, return the entire document as one section
    if not section_matches:
        return [Section(heading="Document", content=markdown_content.strip())]

    # Process each section
    for i, match in enumerate(section_matches):
        # Get heading text, ignoring any LaTeX label
        heading_with_label = match.group(1).strip()
        heading = re.sub(r"\s*\{[^}]+\}$", "", heading_with_label)
        start_pos = match.start()

        # If this is the last section, content goes to the end of the document
        if i == len(section_matches) - 1:
            content = markdown_content[start_pos:].strip()
        else:
            # Otherwise, content goes until the start of the next section
            next_start = section_matches[i + 1].start()
            content = markdown_content[start_pos:next_start].strip()

        sections.append(Section(heading=heading, content=content))

    return sections


def _process_latex(
    splitter: SentenceSplitter, title: str, input_file: Path
) -> LatexPaper | None:
    """Read LaTeX repository from `input_file` and parse into `Paper`.

    LaTeX is converted to Markdown, which is split into sections. The references and
    citations are extracted from bibliography files and sections.
    """
    logger.debug("Processing file: %s", input_file)

    with tempfile.TemporaryDirectory() as tmpdir_:
        tmpdir = Path(tmpdir_)

        with tarfile.open(input_file, "r:gz") as tar:
            tar.extractall(path=tmpdir, filter="data")

        try:
            main_tex = _find_main_tex(tmpdir)
            logger.debug("Using main tex file: %s", main_tex)
        except FileNotFoundError:
            logger.debug("Could not find main file")
            return None

        consolidated_content = _process_latex_file(main_tex, tmpdir)
        if not consolidated_content:
            logger.debug("No content processed. Aborting.")
            return None

        consolidated_content = _remove_arxiv_styling(consolidated_content)

        bib_files = _find_bib_files(tmpdir, consolidated_content)
        citationkey_to_reference = _extract_bibliography_from_bibfiles(
            bib_files, tmpdir
        )
        if not citationkey_to_reference:
            logger.debug("No references from bib files. Trying bib items.")
            citationkey_to_reference = _extract_bibliography_from_bibitems(
                consolidated_content
            )

    citation_contexts = _extract_citations_and_contexts(splitter, consolidated_content)

    # Match citations to references and populate contexts
    for citation_key, contexts in citation_contexts.items():
        if citation_key in citationkey_to_reference:
            # Create a new Reference with updated citation_contexts
            ref = citationkey_to_reference[citation_key]
            citationkey_to_reference[citation_key] = Reference(
                title=ref.title,
                year=ref.year,
                authors=ref.authors,
                citation_contexts=contexts,
            )
        else:
            # If we found a citation but no corresponding reference,
            # add it to the references with placeholder data
            citationkey_to_reference[citation_key] = Reference(
                title="Unknown Reference",
                year=None,
                authors=[],
                citation_contexts=contexts,
            )

    # Remove references that don't have any matching citations
    references = [
        ref for ref in citationkey_to_reference.values() if ref.citation_contexts
    ]

    markdown_content = _convert_latex_to_markdown(consolidated_content, title)
    if markdown_content is None:
        logger.debug("Error converting LaTeX to Markdown. Aborting.")
        return None

    sections = _split_markdown_sections(markdown_content)

    return LatexPaper(title=title, sections=sections, references=references)


def _process_tex_files(
    input_files: list[Path], num_workers: int | None, output_dir: Path
) -> int:
    """Process all TeX `input_files` and save the results to `output_dir`."""
    splitter = SentenceSplitter()

    if num_workers is None or num_workers == 0:
        num_workers = mp.cpu_count()

    logger.debug("Using %d workers", num_workers)

    if num_workers == 1:
        return sum(
            _process_tex_file(splitter, output_dir, input_file)
            for input_file in tqdm(input_files, desc="Converting LaTeX files")
        )

    # We need to re-initialise logging for each subprocess, or nothing is logged
    with mp.Pool(processes=num_workers, initializer=setup_logging) as pool:
        process_func = partial(_process_tex_file, splitter, output_dir)
        return sum(
            tqdm(
                pool.imap_unordered(process_func, input_files),
                total=len(input_files),
                desc="Converting LaTeX files",
            )
        )


# Maximum time allowed for a Pandoc command, in seconds
PANDOC_CMD_TIMEOUT = 30


@app.command(no_args_is_help=True)
def parse_all(
    data_dir: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            help="Path to directory with data downloaded with `download-all`.",
        ),
    ],
    max_items: Annotated[
        int | None,
        typer.Option(
            "--max-items",
            "-n",
            help="Number of items to process. If None, process all.",
        ),
    ] = None,
    num_workers: Annotated[
        int,
        typer.Option(
            "--workers",
            "-j",
            help="Number of workers for parallel processing. Set 0 for all CPUs.",
        ),
    ] = 1,
    clean: Annotated[
        bool, typer.Option(help="Ignore existing files, reprocessing everything.")
    ] = False,
) -> None:
    """Parse LaTeX code from data directory from `latex-all`.

    By default, we avoid reprocessing files that already have processed versions in
    `output_dir`. Override that with `--clean` to reprocess everything.

    Note: for each paper, we combine all TeX code files into a single one and use pandoc
    to convert that to Markdown. However, not all LaTeX files can be parsed. We try to
    remove some offending commands, but sometimes pandoc simply cannot process the LaTeX
    file. In those cases, we just print a warning and give up on that paper. This also
    applies to bib files. This seems to affect about 10% of the input files from arXiv.
    """
    venue_dirs = list(data_dir.iterdir())
    for i, venue_dir in enumerate(venue_dirs, 1):
        logger.info("\n>>> [%d/%d] %s", i, len(venue_dirs), venue_dir.name)

        latex_dir = venue_dir / "files"
        if not latex_dir.exists():
            logger.warning("No downloaded LaTeX data files for: %s", venue_dir)
            continue

        parse(latex_dir, venue_dir / "parsed", max_items, num_workers, clean)


@app.command(no_args_is_help=True)
def parse(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            help="Path to the tar.gz file(s) with LaTeX code. If the path is a"
            " directory, checks for .tar.gz files inside it.",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Directory to save the JSON output file with parsed data.",
        ),
    ],
    max_items: Annotated[
        int | None,
        typer.Option(
            "--max-items",
            "-n",
            help="Number of items to process. If None, process all.",
        ),
    ] = None,
    num_workers: Annotated[
        int,
        typer.Option(
            "--workers",
            "-j",
            help="Number of workers for parallel processing. Set 0 for all CPUs.",
        ),
    ] = 1,
    clean: Annotated[
        bool, typer.Option(help="Ignore existing files, reprocessing everything.")
    ] = False,
) -> None:
    """Parse LaTeX code from directory into JSON with sections and references.

    By default, we avoid reprocessing files that already have processed versions in
    `output_dir`. Override that with `--clean` to reprocess everything.

    Note: for each paper, we combine all TeX code files into a single one and use pandoc
    to convert that to Markdown. However, not all LaTeX files can be parsed. We try to
    remove some offending commands, but sometimes pandoc simply cannot process the LaTeX
    file. In those cases, we just print a warning and give up on that paper. This also
    applies to bib files. This seems to affect about 10% of the input files from arXiv.
    """
    if input_path.is_file():
        input_files = [input_path]
    else:
        input_files = list(input_path.glob("*.tar.gz"))
    input_files = input_files[:max_items]

    output_dir.mkdir(parents=True, exist_ok=True)

    if clean:
        done_titles: set[str] = set()
    else:
        done_titles = {
            _title_from_filename(file, ".json") for file in output_dir.glob("*.json")
        }

    skip_files = {
        file
        for file in input_files
        if _title_from_filename(file, ".tar.gz") in done_titles
    }
    input_files = [file for file in input_files if file not in skip_files]

    with Timer() as timer:
        successful_n = _process_tex_files(input_files, num_workers, output_dir)
    logger.info(timer)

    logger.info("Processed  : %d", len(input_files))
    logger.info("Skipped    : %d", len(skip_files))
    logger.info("Successful : %d", successful_n)


def _title_from_filename(input_file: Path, ext: str) -> str:
    """Get paper title from a file name and an extension (e.g. `.tar.gz` or `.json`).

    This is useful because `Path.stem` doesn't work for `.tar.gz`.
    """
    return input_file.name.removesuffix(ext)


def _process_tex_file(
    splitter: SentenceSplitter, output_dir: Path, input_file: Path
) -> bool:
    """Parse LaTeX files into a paper. Returns True if the conversion was successful."""
    title = _title_from_filename(input_file, ".tar.gz")

    if paper := _process_latex(splitter, title, input_file):
        (output_dir / f"{title}.json").write_text(
            json.dumps(dc.asdict(paper), indent=2)
        )
        return True

    return False


@app.command(no_args_is_help=True)
def preprocess(
    input_dir: Annotated[
        Path,
        typer.Option(
            "--input", "-i", help="Directory containing the data from all conferences."
        ),
    ],
    output_file: Annotated[
        Path, typer.Option("--output", "-o", help="Path to output JSON file.")
    ],
    num_papers: Annotated[
        int | None,
        typer.Option(
            "--num-papers", "-n", help="Number of papers to keep in the output."
        ),
    ] = None,
) -> None:
    """Merge data from all conferences, including reviews and parsed paper content.

    Expects that `input_dir` contains a directory structure like this:

    input_dir
    ├── iclr2024
    │  ├── openreview_arxiv.json
    │  └── parsed
    │     ├── paper1.json
    │     └── paper2.json
    └── iclr2025
       ├── openreview_arxiv.json
       └── parsed
          └── paper3.json

    Where the papers inside `parsed` directories are named after the arXiv title.

    The output is a JSON with an array of pr.Paper.
    """
    if num_papers == 0:
        num_papers = None

    papers_raw = _process_conferences(input_dir)
    papers_processed = [
        _process_paper(paper) for paper in tqdm(papers_raw, "Processing raw papers")
    ]
    papers_valid = [p for p in papers_processed if p]
    papers_dedup = _deduplicate_papers(papers_valid)
    papers_saved = papers_dedup[:num_papers]

    logger.info("Raw papers: %d", len(papers_raw))
    logger.info("Processed papers: %d", len(papers_processed))
    logger.info("Valid papers: %d", len(papers_valid))
    logger.info("Deduplicated papers: %d", len(papers_dedup))
    logger.info("Saving papers: %d.", len(papers_saved))

    save_data(output_file, papers_saved)


def _process_conferences(base_dir: Path) -> list[dict[str, Any]]:
    """Process reviews files and paper contents from conferences in `base_dir`."""
    all_papers: list[dict[str, Any]] = []

    conference_dirs = [d for d in base_dir.iterdir() if d.is_dir()]

    for conf_path in conference_dirs:
        conference = conf_path.name
        arxiv_file = conf_path / "openreview_arxiv.json"
        parsed_dir = conf_path / "parsed"

        logger.info(f"Processing {conference}...")

        # Skip if required files/directories don't exist
        if not arxiv_file.exists() or not parsed_dir.exists():
            logger.info(f"Skipping {conference} - missing required files")
            continue

        papers: list[dict[str, Any]] = json.loads(arxiv_file.read_bytes())
        # Mapping of paper titles (arXiv) to parsed JSON files
        title_to_path = {f.stem: f for f in parsed_dir.glob("*.json")}

        matched = 0

        for paper in tqdm(papers, desc=f"Processing papers in {conference}"):
            arxiv_title = paper.get("arxiv_title")
            if not arxiv_title:
                continue

            if matched_file := title_to_path.get(arxiv_title):
                with contextlib.suppress(Exception):
                    content = json.loads(matched_file.read_bytes())
                    matched += 1

                all_papers.append({
                    **paper,
                    "paper_content": content,
                    "conference": conference,
                })

        logger.info(f"Matched: {matched}. Unmatched: {len(papers) - matched}.")

    return all_papers


def _process_paper(paper_raw: dict[str, Any]) -> pr.Paper | None:
    """Transform a raw paper into a `pr.Paper`.

    Returns None if there are no valid reviews with `RATING_KEYS` or there are
    less than 5 valid references (excludes "Unknown Reference").
    """
    reviews = _get_reviews(paper_raw)
    parsed: dict[str, Any] = paper_raw["paper_content"]

    reviews_processed = _process_reviews(reviews)
    references = _process_references(parsed["references"])
    if not reviews_processed or len(references) < 5:
        return None

    sections = [
        pr.PaperSection(heading=s["heading"], text=s["content"])
        for s in parsed["sections"]
    ]
    approval = _find_approval(reviews)

    content: dict[str, Any] = paper_raw["content"]
    abstract = _value(str, content, "abstract", "")
    authors = _value(list, content, "authors", [])
    # Year from creation timestamp (in ms)
    year = dt.datetime.fromtimestamp(paper_raw["cdate"] / 1000, tz=dt.UTC).year

    return pr.Paper(
        title=parsed["title"],
        reviews=reviews_processed,
        abstract=abstract,
        authors=authors,
        sections=sections,
        approval=approval,
        conference=paper_raw["conference"],
        references=references,
        year=year,
    )


def _deduplicate_papers(papers: Iterable[pr.Paper]) -> list[pr.Paper]:
    """Remove paper duplicates by title taking the earliest paper by year."""
    return [
        min(paper_group, key=lambda p: p.year if p.year is not None else float("inf"))
        for paper_group in groupby(papers, key=lambda x: x.title).values()
    ]


def _find_approval(reviews: list[dict[str, Any]]) -> bool | None:
    """Find the review with a decision, if it exists."""
    for reply in reviews:
        content = reply["content"]
        if decision := _get_value(content, "decision"):
            return decision.lower() != "reject"

    return None


def _process_references(references: list[dict[str, Any]]) -> list[pr.PaperReference]:
    """Transform a raw reference into a `pr.PaperReference`.

    Ignores references with title "Unknown Reference". It's possible that the output
    list is empty.
    """
    output: list[pr.PaperReference] = []

    for ref in references:
        if ref["title"] == "Unknown Reference":
            continue

        output.append(
            pr.PaperReference(
                title=ref["title"],
                year=ref["year"] or 0,
                authors=ref["authors"],
                contexts=[
                    pr.CitationContext(sentence=sentence, polarity=None)
                    for sentence in ref["citation_contexts"]
                ],
            )
        )

    return output


def _process_reviews(reviews: list[dict[str, Any]]) -> list[pr.PaperReview]:
    """Transform raw reviews into a list of `pr.PaperReview`.

    If a review doesn't contain a valid `contribution`, it's skipped.
    The rationale is a combination of the review's sections: summary, strengths,
    weaknesses, questions and limitations. If none of these are given, the review will
    be empty.
    Other integer ratings will be stored in a separate dictionary.
    """
    output: list[pr.PaperReview] = []

    for review in reviews:
        content = review["content"]

        rating = _max_value(_value(_rating, content, key) for key in RATING_KEYS)
        if rating is None:
            continue

        confidence = _value(_rating, content, "confidence")
        rationale = "\n\n".join(
            f"{key.capitalize()}: {value}"
            for key in [
                "summary",
                "strengths",
                "weaknesses",
                "questions",
                "limitations",
            ]
            if (value := _value(str, content, key))
        )
        other_ratings: dict[str, int] = {
            key: value
            for key, item in content.items()
            if (value := _rating(_nested_value(item)))
        }

        output.append(
            pr.PaperReview(
                rating=rating,
                confidence=confidence,
                rationale=rationale,
                other_ratings=other_ratings,
            )
        )

    return output


def _max_value(values: Iterable[int | None]) -> int | None:
    """Find the maximum value in an iterable, ignoring None values.

    Returns None if the iterable is empty or contains only None values.
    """
    filtered_values = [v for v in values if v is not None]
    return max(filtered_values) if filtered_values else None


@overload
def _value[T](
    type_: Callable[[Any], T], item: dict[str, Any], key: str, default: T
) -> T: ...


@overload
def _value[T](
    type_: Callable[[Any], T], item: dict[str, Any], key: str, default: None = None
) -> T | None: ...


def _value[T](
    type_: Callable[[Any], T], item: dict[str, Any], key: str, default: T | None = None
) -> T | None:
    """Take the value under `key.value`, as is common with the OpenReview API.

    As the value type is Any, you can use `type_` to make sure the output is of a given
    type, including a custom conversion function.
    """
    value = _nested_value(item.get(key, {}), default)
    if value is None:
        return None
    return type_(value)


def _nested_value(
    value: Any | dict[str, Any], default: Any | None = None
) -> Any | None:
    """If `value` is a dict, gets its nested `value` field. Otherwise, returns as-is."""
    if isinstance(value, dict):
        return value.get("value", default)
    return value


def _rating(x: Any) -> int | None:
    """Parse a rating from a value.

    If the rating is an int, return it directly. Otherwise, try to extract the rating
    from a string, such as "4 - good" or "1: poor".
    """
    if isinstance(x, int):
        return x

    try:
        fst, _ = re.split(r"[\s\W]+", str(x), maxsplit=1)
        return int(fst)
    except ValueError as e:
        logger.debug("Could not convert rating to int: %s", e)
        return None


@app.callback(help=__doc__)
def main() -> None:
    """Empty callback for documentation."""
    setup_logging()


@app.command(no_args_is_help=True, name="all")
def all_(
    data_dir: Annotated[
        Path, typer.Argument(help="Output directory for OpenReview reviews file.")
    ],
    output_file: Annotated[
        Path, typer.Argument(help="Path to save the output JSON file.")
    ],
    num_papers: Annotated[
        int | None,
        typer.Option(
            "--num-papers",
            "-n",
            help="How many papers to process. If None, processes all.",
        ),
    ] = None,
    clean_run: Annotated[
        bool,
        typer.Option("--clean", help="If True, ignore previously downloaded files."),
    ] = False,
) -> None:
    """Run full ORC pipeline.

    - Download reviews from the OpenReview API.
    - Download LaTeX code from arXiv.
    - Transform the LaTeX code into Markdown and JSON with citations.
    - Merge and transform into a single file.

    These are the conferences used:

    - ICLR 2022, 2023, 2024, 2025
    - NeurIPS 2022, 2023, 2024

    ICLR 2022 and 2023 have "technical" and "empirical" novelty as ratings. We use the
    higher one for the target. For the rest, we use the "contribution" rating.

    These are all numerical ratings from 1 to 4. We also convert them to binary, with
    1-2 being not novel and 3-4 being novel.
    """
    download_all(data_dir, query_arxiv=False)
    latex_all(data_dir, max_papers=num_papers, clean_run=clean_run)
    parse_all(data_dir, max_items=num_papers, clean=clean_run)
    preprocess(data_dir, output_file, num_papers=num_papers)


if __name__ == "__main__":
    app()
