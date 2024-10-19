"""Extract the information we care about from the merged ASAP JSON file.

We extract the title, abstract, full paper sections, ratings, approval decision and
references (including their context in the paper).

Since the context sentences sometimes don't give a lot of information so we can later
determine whether they're positive or negative, we also try to expand the context a
little bit. We take --context-sentences before and after the context given, stopping if
we find a sentence that contains another citation.
"""

import argparse
import json
import multiprocessing
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from difflib import get_close_matches
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

from spacy.language import Language as SpacyModel

from paper_hypergraph.asap import process_sections
from paper_hypergraph.util import load_spacy_model

_SPACY_MODEL = "en_core_web_sm"
_CONTEXT_MIN_FUZZY = 0.8


def _parse_rating(rating: str) -> int | None:
    """Parse rating text into a number (e.g., "8: Accept" -> 8).

    Returns:
        The rating number or None if the rating text cannot be parsed.
    """
    try:
        return int(rating.split(":")[0].strip())
    except ValueError:
        return None


def _parse_approval(approval: str) -> bool:
    """Parse approval text into a bool ("Reject" -> False, everything else -> True)."""
    return approval.strip().lower() != "reject"


_REGEX_CITATION = re.compile(r"\([A-Z][a-z]+(\s+et al\.)?\.?,\s+\d{4}\)")
"""Used to search for citations."""


def _split_paragraph(model: SpacyModel, paragraph: str) -> list[str]:
    """Split a paragraph into sentences using a spaCy model."""
    doc = model(paragraph)
    return [sent.text.strip() for sent in doc.sents]


def _expand_citation_contexts(
    spacy_model: SpacyModel,
    sections: Sequence[process_sections.Section],
    contexts: Iterable[str],
    min_fuzzy: float,
    n: int,
) -> list[str]:
    """Expand the given contexts in the paragraph by `n` sentences.

    See `_expand_citation_context` for more information.
    """
    expanded: list[str] = []

    for context in contexts:
        if paragraph := _find_context_paragraph(sections, context):
            expanded.append(
                _expand_citation_context(
                    spacy_model,
                    paragraph,
                    context,
                    min_fuzzy,
                    n=n,
                )
            )
        else:
            expanded.append(context)

    return expanded


def _expand_citation_context(
    spacy_model: SpacyModel,
    paragraph: str,
    citation_sentence: str,
    min_fuzzy: float,
    *,
    n: int,
) -> str:
    """Expand the given context in the paragraph by `n` sentences.

    We try to expand the context within the given paragraph. We get `n` sentences before
    and after the original context sentence, stopping early if we find another citation.

    Args:
        paragraph: Piece of text where we'll locate the citation sentence and try to
            expand it.
        citation_sentence: Original context given in the ASAP dataset.
        n: Number of sentences to expand the context, before and after.
        min_fuzzy: Minimum fuzzy ratio (in [0, 1]) to accept a citation sentence
            candidate.

    Returns:
        The expanded context. This contains at least the original citation sentence we
        match from the paragraph.

    Raises:
        ValueError if the `citation_sentence` cannot be found in the paragraph. Even
            tries to find a fuzzy match (ratio greater or equal to `min_fuzzy`) before
            it bails.
    """
    sentences = _split_paragraph(spacy_model, paragraph)

    # Find the index of the citation sentence using fuzzy matching
    close_matches = get_close_matches(
        citation_sentence, sentences, n=1, cutoff=min_fuzzy
    )
    if not close_matches:
        return citation_sentence

    citation_sentence_match = close_matches[0]
    citation_index = sentences.index(citation_sentence_match)
    context = [citation_sentence_match]

    # Add sentences before the citation
    for i in range(1, n + 1):
        if citation_index - i < 0:
            break

        sentence = sentences[citation_index - i]
        if _REGEX_CITATION.search(sentence):
            break

        context.insert(0, sentence)

    # Add sentences after the citation
    for i in range(1, n + 1):
        if citation_index + i >= len(sentences):
            break

        sentence = sentences[citation_index + i]
        if _REGEX_CITATION.search(sentence):
            break

        context.append(sentence)

    # If we didn't find anything else, we return the original sentence as-is.
    if len(context) == 1:
        return citation_sentence

    return "\n".join(context)


def _find_context_paragraph(
    sections: Iterable[process_sections.Section], context: str
) -> str | None:
    """Find the first paragraph that contains the citation context sentence.

    Args:
        sections: grouped sections from the main paper (see
            paper_hypergraph.asap.process_sections).
        context: context sentence from referenceMentions.

    Returns:
        The paragraph if found. None, otherwise.
    """
    for section in sections:
        for paragraph in section.text.splitlines():
            if context in paragraph:
                return paragraph

    return None


def _process_references(
    spacy_model: SpacyModel,
    paper: dict[str, Any],
    context_sentences: int,
    sections: Sequence[process_sections.Section],
    min_fuzzy: float,
) -> list[dict[str, Any]]:
    class ReferenceKey(NamedTuple):
        title: str
        authors: Sequence[str]
        year: int

    references = paper["references"]
    references_output: defaultdict[ReferenceKey, set[str]] = defaultdict(set)

    for ref_mention in paper["referenceMentions"]:
        ref_id = ref_mention["referenceID"]

        if not (0 <= ref_id < len(references)):
            continue

        ref_original = references[ref_id]
        ref_author = sorted(ref_original["author"])

        ref_key = ReferenceKey(
            ref_original["title"], tuple(ref_author), ref_original["year"]
        )
        references_output[ref_key].add(ref_mention["context"].strip())

    return [
        {
            "title": ref.title,
            "authors": ref.authors,
            "year": ref.year,
            "contexts": list(contexts),
            "contexts_expanded": _expand_citation_contexts(
                spacy_model, sections, contexts, min_fuzzy, context_sentences
            ),
        }
        for ref, contexts in references_output.items()
    ]


def _process_paper(
    item: dict[str, Any],
    spacy_model: SpacyModel,
    context_sentences: int,
    min_fuzzy: float,
) -> dict[str, Any] | None:
    """Process a single paper item."""
    paper = item["paper"]

    sections = process_sections.group_sections(paper["sections"])
    if not sections:
        return None

    ratings = [r for review in item["review"] if (r := _parse_rating(review["rating"]))]

    return {
        "title": paper["title"],
        "abstract": paper["abstractText"],
        "ratings": ratings,
        "sections": [asdict(section) for section in sections],
        "approval": _parse_approval(item["approval"]),
        "references": _process_references(
            spacy_model, paper, context_sentences, sections, min_fuzzy
        ),
    }


def extract_interesting(
    input_file: Path, output_file: Path, context_sentences: int, min_fuzzy: float
) -> None:
    """Extract information from the input JSON file and write to the output JSON file.

    The input file is the output of `paper_hypergraph.asap.merge`.
    """
    spacy_model = load_spacy_model(_SPACY_MODEL)
    data = json.loads(input_file.read_text())

    with multiprocessing.Pool() as pool:
        results = pool.map(
            partial(
                _process_paper,
                spacy_model=spacy_model,
                context_sentences=context_sentences,
                min_fuzzy=min_fuzzy,
            ),
            data,
        )

    results_valid = [res for res in results if res]

    print("no.  input papers:", len(data))
    print(
        "no. output papers:",
        len(results_valid),
        f"({len(results_valid) / len(data):.2%})",
    )

    output_file.write_text(json.dumps(results_valid, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", type=Path, help="Path to input (filtered) JSON file")
    parser.add_argument("output", type=Path, help="Path to output extracted JSON file")
    parser.add_argument(
        "--context-sentences",
        type=int,
        default=1,
        help="Maximum number of sentences to expand the context (before and after)",
    )
    parser.add_argument(
        "--min-fuzzy",
        type=float,
        default=_CONTEXT_MIN_FUZZY,
        help="Minimum fuzzy ratio to accept candidate citation sentences matches. "
        "Value in [0, 1].",
    )
    args = parser.parse_args()
    extract_interesting(args.input, args.output, args.context_sentences, args.min_fuzzy)


if __name__ == "__main__":
    main()
