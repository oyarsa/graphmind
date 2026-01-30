"""Parse arXiv LaTeX files into structured data."""

import dataclasses as dc
import logging
import multiprocessing as mp
import re
import subprocess
import tarfile
import tempfile
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Annotated

import nltk  # type: ignore
import orjson
import typer
from tqdm import tqdm

from paper.peerread.model import CitationContext, PaperReference, PaperSection
from paper.util import Timer, setup_logging
from paper.util.cli import die
from paper.util.serde import save_data

logger = logging.getLogger(__name__)

# Maximum time allowed for a Pandoc command, in seconds
PANDOC_CMD_TIMEOUT = 30


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
    _check_pandoc()

    if input_path.is_file():
        input_files = [input_path]
    else:
        input_files = list(input_path.glob("*.tar.gz"))
    input_files = input_files[:max_items]

    output_dir.mkdir(parents=True, exist_ok=True)

    if clean:
        done_titles: set[str] = set()
    else:
        # Check for both .json and .json.zst files
        json_files = list(output_dir.glob("*.json")) + list(
            output_dir.glob("*.json.zst")
        )
        done_titles = {
            _title_from_filename(
                file, ".json.zst" if file.suffix == ".zst" else ".json"
            )
            for file in json_files
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


def parse_all(
    data_dir: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            help="Path to directory with data downloaded with `reviews-all`.",
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


def _check_pandoc() -> None:
    """Check if pandoc is installed and available in PATH."""
    try:
        subprocess.run(["pandoc", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        die(
            "pandoc is not installed but required for LaTeX parsing. "
            "Please install it from https://pandoc.org/installing.html"
        )


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
    citation_contexts: list[str] = dc.field(default_factory=list[str])


@dc.dataclass(frozen=True, kw_only=True)
class LatexPaper:
    """Parsed paper content in Markdown with reference citations."""

    title: str
    sections: list[Section]
    references: list[Reference]


def latex_paper_to_peerread(
    latex_paper: LatexPaper,
) -> tuple[list[PaperSection], list[PaperReference]]:
    """Convert LatexPaper to PeerRead format sections and references.

    Returns:
        Tuple of (sections, references) compatible with PaperSection and PaperReference.
    """

    sections = [
        PaperSection(heading=section.heading, text=section.content)
        for section in latex_paper.sections
    ]

    references: list[PaperReference] = []
    for ref in latex_paper.references:
        contexts = [
            CitationContext.new(sentence=context, polarity=None)
            for context in ref.citation_contexts
        ]

        # Parse year from string to int if possible
        year = 0  # Default year
        if ref.year:
            try:
                year = int(ref.year)
            except (ValueError, TypeError):
                year = 0

        references.append(
            PaperReference(
                title=ref.title,
                year=year,
                authors=ref.authors,
                contexts=contexts,
            )
        )

    return sections, references


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


def _process_tex_file(
    splitter: SentenceSplitter, output_dir: Path, input_file: Path
) -> bool:
    """Parse LaTeX files into a paper. Returns True if the conversion was successful."""
    title = _title_from_filename(input_file, ".tar.gz")

    if paper := process_latex(splitter, title, input_file):
        # Convert dataclass to dict for serialization
        paper_dict = orjson.loads(orjson.dumps(paper, default=vars))
        save_data(output_dir / f"{title}.json", paper_dict)
        return True

    return False


def _title_from_filename(input_file: Path, ext: str) -> str:
    """Get paper title from a file name and an extension (e.g. `.tar.gz` or `.json`).

    This is useful because `Path.stem` doesn't work for `.tar.gz`.
    """
    return input_file.name.removesuffix(ext)


def process_latex(
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
    logger.debug("References before matching: %d", len(citationkey_to_reference))
    references = [
        ref for ref in citationkey_to_reference.values() if ref.citation_contexts
    ]
    logger.debug("References with contexts: %d", len(references))

    markdown_content = _convert_latex_to_markdown(consolidated_content, title)
    if markdown_content is None:
        logger.debug("Error converting LaTeX to Markdown. Aborting.")
        return None

    sections = _split_markdown_sections(markdown_content)

    return LatexPaper(title=title, sections=sections, references=references)


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
    return re.sub(
        r"\\begin\{tcolorbox\}(\[[^\]]*\])?.*?\\end\{tcolorbox\}",
        "",
        no_newtcolorbox,
        flags=re.DOTALL,
    )


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

        logger.debug("Converting LaTeX to Markdown with pandoc: %s", title)
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
            subprocess.run(
                pandoc_cmd,
                check=True,
                timeout=PANDOC_CMD_TIMEOUT,
                capture_output=True,  # Capture stdout and stderr
                text=True,  # Return output as string instead of bytes
            )
            return markdown_file.read_text(errors="ignore")
        except subprocess.TimeoutExpired:
            logger.warning("Command timeout during pandoc conversion. Paper: %s", title)
            return None
        except subprocess.CalledProcessError as e:
            logger.warning(
                "Error during pandoc conversion. Paper: %s. Error: %s\nStderr: %s\nStdout: %s",
                title,
                e,
                e.stderr,
                e.stdout,
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

            logger.debug("Parsing bibtex data with pandoc: %s", bib_path)
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
            bib_data = orjson.loads(result.stdout)
        except subprocess.TimeoutExpired:
            logger.warning("Command timeout during bibliography file processing.")
            continue
        except subprocess.CalledProcessError as e:
            logger.debug(f"Error processing bibliography file {bib_path}: {e.stderr}")
            continue
        except orjson.JSONDecodeError as e:
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


# Compiled regex patterns for bibitem parsing
_RE_BIBENV = re.compile(
    r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", re.DOTALL
)
_RE_BIBITEM = re.compile(
    r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}(.*?)(?=\\bibitem|\n\n|\\end\{thebibliography\}|$)",
    re.DOTALL,
)
_RE_YEAR = re.compile(r"(?:\()?\b(\d{4})\b(?:\))?")
_RE_NEWBLOCK_TITLE = re.compile(r"\\newblock\s+([^{\\]+?)\.(?:\s|\\newblock|$)")
_RE_JOURNAL_LIKE = re.compile(r"^(arXiv|CoRR|Proc|In\s|pages?\s|\d{4})", re.IGNORECASE)
_RE_TRAILING_YEAR = re.compile(r",?\s*\d{4}$")
_RE_LATEX_COMMANDS = re.compile(r"\\[a-z]+(\[[^\]]*\])?(\{[^}]*\})?")
_RE_AND_SEPARATOR = re.compile(r" and ", re.IGNORECASE)


def extract_title_from_bibitem_entry(entry_text: str) -> str:
    r"""Extract paper title from a bibitem entry text.

    Extracts title from \newblock pattern common in BibTeX-generated bibitems.
    The title is expected after the first \newblock, ending with a period.

    Args:
        entry_text: The raw text content of a bibitem entry (after the citation key).

    Returns:
        Extracted title, or "Unknown Title" if no title could be found.
    """
    # Extract title after \newblock (common in bibtex-generated bibitems)
    # e.g., "\newblock Layer normalization."
    newblock_match = _RE_NEWBLOCK_TITLE.search(entry_text)
    if newblock_match:
        candidate = newblock_match.group(1).strip()
        # Filter out journal-like patterns (year, arXiv, CoRR, Proc, etc.)
        if candidate and not _RE_JOURNAL_LIKE.match(candidate):
            # Clean up: collapse whitespace, remove trailing year
            cleaned = " ".join(candidate.split())
            return _RE_TRAILING_YEAR.sub("", cleaned)

    return "Unknown Title"


def _extract_bibliography_from_bibitems(latex_content: str) -> dict[str, Reference]:
    r"""Extract bibliography entries from \\bibitem commands in the LaTeX content."""
    references: dict[str, Reference] = {}

    # Find the bibliography environment
    bibenv_match = _RE_BIBENV.search(latex_content)

    if not bibenv_match:
        return references

    bibenv_content = bibenv_match.group(0)

    for match in _RE_BIBITEM.finditer(bibenv_content):
        citation_key = match.group(1).strip()
        entry_text = match.group(2).strip()

        if not citation_key or not entry_text:
            continue

        # Extract year (look for 4 consecutive digits, possibly in parentheses)
        year = None
        year_match = _RE_YEAR.search(entry_text)
        if year_match:
            year = year_match.group(1)

        # Extract title using the pure function
        title = extract_title_from_bibitem_entry(entry_text)

        # Extract authors (heuristic)
        authors: list[str] = []

        # Get text before the title or year marker
        author_text = entry_text
        if title in entry_text and title != "Unknown Title":
            author_text = entry_text.split(title)[0]
        elif year and year in entry_text:
            author_text = entry_text.split(year)[0]

        # Look for patterns like "Author1, Author2, and Author3"
        author_text = _RE_LATEX_COMMANDS.sub("", author_text)  # Remove LaTeX commands
        author_text = author_text.split(".")[0]  # Authors often end with a period

        # Split by common separators
        if " and " in author_text.lower():
            parts = _RE_AND_SEPARATOR.split(author_text)
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
