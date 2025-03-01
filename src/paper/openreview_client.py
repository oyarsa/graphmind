"""Fetch conference paper data from the OpenReview API and download LaTeX from arXiv.

Requires the following environment variables to be set:
- OPENREVIEW_USERNAME
- OPENREVIEW_PASSWORD

These are the standard credentials you use to log into the OpenReview website.
Example venue IDs:
- ICLR.cc/2024/Conference
- ICLR.cc/2025/Conference
- NeurIPS.cc/2024/Conference

The process for retrieving the whole data is running the subcommands in this order:
- `reviews`: get the available paper information for a given conference
- `arxiv`: find the arXiv IDs for the papers that are available there
- `latex`: use the arXiv IDs to download the LaTeX code for the papers
- `parse`: convert LaTeX code to parsed paper with sections and references
"""

# pyright: basic
import dataclasses as dc
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
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Annotated, Any

import arxiv  # type: ignore
import backoff
import nltk  # type: ignore
import requests
import typer
from openreview import api
from tqdm import tqdm

from paper.util import Timer, setup_logging
from paper.util.cli import die

logger = logging.getLogger("paper.openreview")

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
    help=__doc__,
)


@app.command(name="arxiv", no_args_is_help=True)
def query_arxiv(
    reviews_file: Annotated[
        Path,
        typer.Option(
            "--papers",
            "-i",
            help="Path to paper data from OpenReview. Can be a text file with one title"
            " per line, or the actual JSON from the `reviews` subcommand.",
        ),
    ],
    output_file: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output path for JSON file with the arXiv information.",
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
    batch_size: Annotated[
        int, typer.Option(help="Batch size to query the arXiv API.")
    ] = 50,
) -> None:
    """Query the arXiv API to find which papers are available.

    Saves an `arxiv.json` file with an array of objects with `id` and `title` fields.
    Use this with the `latex` subcommand to download the LaTeX files.
    """
    if reviews_file.suffix == ".json":
        papers = json.loads(reviews_file.read_text())[:max_papers]
        titles: list[str] = [
            title
            for paper in papers
            if (title := paper.get("content", {}).get("title", {}).get("value"))
        ]
    else:
        titles = [
            title
            for line in reviews_file.read_text().splitlines()[:max_papers]
            if (title := line.strip())
        ]

    if not titles:
        die("No valid titles found.")
    logger.info(f"Found {len(titles)} papers in input file")

    arxiv_results = _get_arxiv(titles, batch_size)
    logger.info(f"Found {len(arxiv_results)} papers on arXiv")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps([dc.asdict(r) for r in arxiv_results]))


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
    skip_file: Annotated[
        Path | None,
        typer.Option("--skip", help="File with paper titles to skip, one per line."),
    ] = None,
) -> None:
    """Download LaTeX source files from arXiv data.

    The arXiv data is fetched with the `arxiv` subcommand.

    By default, skips re-downloading files that already exist in the output directory.
    You can override this with `--clean` and `--skip`.
    """
    papers: list[dict[str, str]] = json.loads(reviews_file.read_text())[:max_papers]
    arxiv_results = [ArxivResult(title=p["title"], id=p["id"]) for p in papers]

    if clean_run:
        downloaded_prev = set()
    elif skip_file is not None:
        downloaded_prev = {
            name
            for line in skip_file.read_text().splitlines()
            if (name := line.strip())
        }
    else:
        downloaded_prev = {
            path.stem for path in output_dir.glob("*.tar.gz") if path.is_file()
        }

    downloaded_n = 0
    skipped_n = 0
    failed_n = 0
    output_dir.mkdir(exist_ok=True, parents=True)

    for result in tqdm(arxiv_results, desc="Downloading LaTeX sources"):
        if result.title in downloaded_prev:
            skipped_n += 1
            continue

        try:
            if data := _download_latex_source(result.id):
                (output_dir / f"{result.title}.tar.gz").write_bytes(data)
                downloaded_n += 1
            else:
                logger.warning(f"Invalid tar.gz file for {result.title}")
                failed_n += 1
        except Exception as e:
            logger.warning(
                f"Error downloading LaTeX source for {result.title}"
                f" - {type(e).__name__}: {e}"
            )
            failed_n += 1

    logger.info(f"Downloaded : {downloaded_n}")
    logger.info(f"Skipped    : {skipped_n}")
    logger.info(f"Failed     : {failed_n}")


@dc.dataclass(frozen=True, kw_only=True)
class ArxivResult:
    """Result of querying the arXiv API with a paper title from OpenReview."""

    title: str
    id: str


def _get_arxiv(paper_titles: list[str], batch_size: int) -> list[ArxivResult]:
    """Get arXiv information for the papers that are present there."""
    arxiv_client = arxiv.Client()

    arxiv_results: list[ArxivResult] = []
    for title_batch in tqdm(
        list(itertools.batched(paper_titles, batch_size)), desc="Querying arXiv"
    ):
        arxiv_results.extend(_batch_search_arxiv(arxiv_client, title_batch))

    return arxiv_results


def _batch_search_arxiv(
    client: arxiv.Client, titles: Sequence[str]
) -> list[ArxivResult]:
    """Search multiple titles at once on arXiv and return matching results."""
    or_queries = " OR ".join(f'ti:"{title}"' for title in titles)
    query = f"({or_queries})"
    results_map: dict[str, ArxivResult] = {}

    try:
        for result in client.results(
            arxiv.Search(query=query, max_results=len(titles))
        ):
            result_title = result.title.lower()
            for original_title in titles:
                if _similar_titles(original_title, result_title):
                    results_map[original_title.lower()] = ArxivResult(
                        id=result.entry_id.split("/")[-1],
                        title=result.title,
                    )
                    break
    except Exception as e:
        logger.warning(f"Error during batch search on arXiv: {e}")

    return [result for title in titles if (result := results_map.get(title.lower()))]


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
def reviews(
    output_dir: Annotated[
        Path, typer.Argument(help="Output directory for OpenReview reviews file.")
    ],
    venue_id: Annotated[str, typer.Option("--venue", help="Venue ID to fetch data.")],
) -> None:
    """Download all reviews and metadata for papers from a conference in OpenReview."""
    client = api.OpenReviewClient(baseurl="https://api2.openreview.net")
    submissions_raw = client.get_all_notes(
        invitation=f"{venue_id}/-/Submission", details="replies"
    )
    if not submissions_raw:
        die("Empty submissions list")

    submissions_all = [_note_to_dict(s) for s in submissions_raw]
    submissions_valid = [s for s in submissions_all if _is_valid(s, "contribution")]

    logger.info("Submissions - all: %d", len(submissions_all))
    logger.info("Submissions - valid: %d", len(submissions_valid))

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "openreview_all.json").write_text(json.dumps(submissions_all))
    (output_dir / "openreview_valid.json").write_text(json.dumps(submissions_valid))


def _note_to_dict(note: api.Note) -> dict[str, Any]:
    """Convert OpenReview API `Note` to dict with additional `details` object."""
    return note.to_json() | {"details": note.details}


def _is_valid(paper: dict[str, Any], rating: str) -> bool:
    """Check if paper has at least one review with `rating`, PDF, title and abstract."""
    return all((
        _has_rating(paper, rating),
        _has_field(paper, "pdf"),
        _has_field(paper, "title"),
        _has_field(paper, "abstract"),
    ))


def _review_has_rating(review: dict[str, Any], name: str) -> bool:
    """Check if the review has the rating with given `name`.

    Checks whether the `content.{name}` field is non-empty.
    """
    return bool(review["content"].get(name))


def _has_rating(paper: dict[str, Any], name: str) -> bool:
    """Check if any review in `paper` has the rating with given `name`."""
    return any(_review_has_rating(r, name) for r in paper["details"]["replies"])


def _has_field(paper: dict[str, Any], name: str) -> bool:
    """Check if the `paper` has a field with `name` and non-empty value."""
    value = paper["content"].get(name, {}).get("value")
    if isinstance(value, str):
        value = value.strip()
    return bool(value)


"""Read a LaTeX directory, convert it to Markdown and extract sections and references.

The script takes input and output paths. The input is a directory containing tex and
other related files. We look for a "main" tex file that includes the others and start
from it. We recursively include all tex files so we're left with a single tex file to
process. We also look for `.bib` files referenced in the tex code.

The output is a JSON file containing the extracted sections and references. The sections
are the top-level sections with their numbers (or letters in the appendix). The
references are all the papers from the `.bib` files, that are mentioned in the main text.
Each reference also includes the paragraphs where it's cited in the text.
"""


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
class Paper:
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


def clean_latex(text: str) -> str:
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


def get_citation_keys(sentence: str) -> set[str]:
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


def extract_citation_sentences(
    splitter: SentenceSplitter, paragraph: str
) -> dict[str, list[str]]:
    """Extract sentences containing citations from a paragraph.

    Args:
        paragraph: The LaTeX paragraph to extract from.
        splitter: The sentence splitter to use.

    Returns:
        A dictionary mapping citation keys to the sentences that cite them.
    """
    cleaned_text = clean_latex(paragraph)
    sentences = splitter.split(cleaned_text)
    citation_sentences: dict[str, list[str]] = defaultdict(list)
    for sentence in sentences:
        if CITATION_REGEX.search(sentence):
            for key in get_citation_keys(sentence):
                citation_sentences[key].append(sentence)

    return citation_sentences


def find_main_tex(directory: Path) -> Path:
    """Find entrypoint TeX file containing include directives for sections."""
    tex_files = list(directory.glob("**/*.tex"))

    if not tex_files:
        raise FileNotFoundError("No .tex files found")
    if len(tex_files) == 1:
        return tex_files[0]

    main_candidates = [
        tex_file
        for tex_file in tex_files
        if "\\begin{document}" in tex_file.read_text()
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


def process_latex_file(
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
        content = abs_path.read_text()
        content = remove_latex_comments(content)
    except Exception as e:
        logger.debug(f"Error reading {abs_path}: {e}")
        return ""

    include_pattern = re.compile(r"\\(?:input|include)\{([^}]+)\}")

    def include_replacer(match: re.Match[str]) -> str:
        included_path = Path(match.group(1).strip())
        if not included_path.name.endswith(".tex"):
            included_path = included_path.with_name(f"{included_path.name}.tex")

        return process_latex_file(root_dir / included_path, root_dir, processed_files)

    return include_pattern.sub(include_replacer, content)


def remove_arxiv_styling(latex_content: str) -> str:
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


def remove_latex_comments(tex_string: str) -> str:
    """Remove lines that are entirely commented out from a TeX document string.

    A line is considered fully commented if it only contains whitespace before the
    comment character '%'.
    """
    return "\n".join(
        line for line in tex_string.splitlines() if not line.lstrip().startswith("%")
    )


def convert_to_markdown(latex_content: str) -> str | None:
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
            subprocess.run(pandoc_cmd, check=True)
            return markdown_file.read_text()
        except subprocess.CalledProcessError as e:
            logger.warning("Error during pandoc conversion: %s", e)
            return None


def find_bib_files(base_dir: Path, latex_content: str) -> list[Path]:
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


def extract_bibliography_from_bibfiles(
    bib_paths: list[Path], tmpdir: Path
) -> dict[str, Reference]:
    """Extract bibliography entries from .bib files."""
    references: dict[str, Reference] = {}

    for bib_path in bib_paths:
        if not bib_path.exists():
            continue

        logger.debug("Processing bib file: %s", bib_path)
        try:
            bib_content = bib_path.read_text(errors="replace")

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
                pandoc_cmd, check=True, capture_output=True, text=True
            )
            bib_data = json.loads(result.stdout)
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


def extract_bibliography_from_bibitems(latex_content: str) -> dict[str, Reference]:
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


def extract_citations_and_contexts(
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
        paragraph_citations = extract_citation_sentences(splitter, paragraph)

        # Merge with overall citation contexts
        for key, sentences in paragraph_citations.items():
            for sentence in sentences:
                if sentence not in citation_contexts[key]:
                    citation_contexts[key].append(sentence)

    return citation_contexts


def split_by_sections(markdown_content: str) -> list[Section]:
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


def process_latex(
    splitter: SentenceSplitter, title: str, input_file: Path
) -> Paper | None:
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
            main_tex = find_main_tex(tmpdir)
            logger.debug("Using main tex file: %s", main_tex)
        except FileNotFoundError:
            logger.debug("Could not find main file")
            return None

        consolidated_content = process_latex_file(main_tex, tmpdir)
        if not consolidated_content:
            logger.debug("No content processed. Aborting.")
            return None

        consolidated_content = remove_arxiv_styling(consolidated_content)

        bib_files = find_bib_files(tmpdir, consolidated_content)
        citationkey_to_reference = extract_bibliography_from_bibfiles(bib_files, tmpdir)
        if not citationkey_to_reference:
            logger.debug("No references from bib files. Trying bib items.")
            citationkey_to_reference = extract_bibliography_from_bibitems(
                consolidated_content
            )

    citation_contexts = extract_citations_and_contexts(splitter, consolidated_content)

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

    markdown_content = convert_to_markdown(consolidated_content)
    if markdown_content is None:
        logger.debug("Error converting LaTeX to Markdown. Aborting.")
        return None

    sections = split_by_sections(markdown_content)

    return Paper(title=title, sections=sections, references=references)


def process_tex_files(
    input_files: list[Path], num_workers: int | None, output_dir: Path
) -> int:
    """Process all TeX `inpt_files` and save the results to `output_dir`."""
    splitter = SentenceSplitter()

    if num_workers is None:
        num_workers = mp.cpu_count()

    logger.debug("Using %d workers", num_workers)

    if num_workers == 1:
        return sum(
            process_tex_file(splitter, output_dir, input_file)
            for input_file in tqdm(input_files, desc="Converting LaTeX files")
        )

    # We need to re-initialise logging for each subprocess, or nothing is logged
    with mp.Pool(processes=num_workers, initializer=setup_logging) as pool:
        process_func = partial(process_tex_file, splitter, output_dir)
        return sum(
            tqdm(
                pool.imap_unordered(process_func, input_files),
                total=len(input_files),
                desc="Converting LaTeX files",
            )
        )


@app.command(help=__doc__, no_args_is_help=True)
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
        int | None,
        typer.Option(
            "--workers", "-j", help="Number of workers for parallel processsing."
        ),
    ] = None,
) -> None:
    """Parse LaTeX code from directory into JSON with sections and references."""
    if input_path.is_file():
        input_files = [input_path]
    else:
        input_files = list(input_path.glob("*.tar.gz"))
    input_files = input_files[:max_items]

    output_dir.mkdir(parents=True, exist_ok=True)

    with Timer() as timer:
        successful_n = process_tex_files(input_files, num_workers, output_dir)
    logger.info(timer)

    logger.info("Processed  : %d", len(input_files))
    logger.info("Successful : %d", successful_n)


def process_tex_file(
    splitter: SentenceSplitter, output_dir: Path, input_file: Path
) -> bool:
    """Parse LaTeX files into a paper. Returns True if the conversion was successful."""
    title = input_file.name.removesuffix(".tar.gz")

    if paper := process_latex(splitter, title, input_file):
        (output_dir / f"{title}.json").write_text(
            json.dumps(dc.asdict(paper), indent=2)
        )
        return True

    return False


@app.callback(help=__doc__)
def main() -> None:
    """Empty callback for documentation."""
    setup_logging()


if __name__ == "__main__":
    app()
