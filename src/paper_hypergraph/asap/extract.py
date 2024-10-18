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
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from difflib import get_close_matches
from pathlib import Path
from typing import Any, NamedTuple

from paper_hypergraph.asap import process_sections


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


_REGEX_SENTENCE = re.compile(r"(?<=[.!?])\s+")
"""Used to split a paragraph into sentences."""
_REGEX_CITATION = re.compile(r"\([A-Z][a-z]+(\s+et al\.)?\.?,\s+\d{4}\)")
"""Used to search for citations."""


def _expand_citation_context(paragraph: str, citation_sentence: str, *, n: int) -> str:
    """Expand the given context in the paragraph by `n` sentences.

    We try to expand the context within the given paragraph. We get `n` sentences before
    and after the original context sentence, stopping early if we find another citation.

    Args:
        paragraph: Piece of text where we'll locate the citation sentence and try to
            expand it.
        citation_sentence: Original context given in the ASAP dataset.
        n: Number of sentences to expand the context, before and after.

    Returns:
        The expanded context. This contains at least the original citation sentence we
        match from the paragraph.

    Raises:
        ValueError if the `citation_sentence` cannot be found in the paragraph. Even
            tries to find a fuzzy match (ratio over 0.8) before it bails.
    """
    sentences = _REGEX_SENTENCE.split(paragraph)

    # Find the index of the citation sentence using fuzzy matching
    close_matches = get_close_matches(citation_sentence, sentences, n=1, cutoff=0.8)
    if not close_matches:
        raise ValueError("Citation sentence not found in the paragraph")

    citation_sentence_match = close_matches[0]
    citation_index = sentences.index(citation_sentence_match)
    context = [citation_sentence_match]

    # Add sentences before the citation
    for i in range(1, n + 1):
        if citation_index - i < 0:
            break

        sentence = sentences[citation_index - i]
        if not _REGEX_CITATION.search(sentence):
            context.insert(0, sentence)
        else:
            break

    # Add sentences after the citation
    for i in range(1, n + 1):
        if citation_index + i >= len(sentences):
            break

        sentence = sentences[citation_index + i]
        if not _REGEX_CITATION.search(sentence):
            context.append(sentence)
        else:
            break

    # If we didn't find anything else, we return the original sentence as-is.
    if len(context) == 1:
        return citation_sentence

    return "\n".join(context)


def _process_references(
    paper: dict[str, Any], context_sentences: int
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
            "contexts_expanded": [
                _expand_citation_context(context, context, n=context_sentences)
                for context in contexts
            ],
        }
        for ref, contexts in references_output.items()
    ]


def extract_interesting(
    input_file: Path, output_file: Path, context_sentences: int
) -> None:
    """Extract information from the input JSON file and write to the output JSON file.

    The input file is the output of `paper_hypergraph.asap.merge`.
    """
    data = json.loads(input_file.read_text())

    output: list[dict[str, Any]] = []

    for item in data:
        paper = item["paper"]

        sections = process_sections.group_sections(paper["sections"])
        if not sections:
            continue

        ratings = [
            r for review in item["review"] if (r := _parse_rating(review["rating"]))
        ]

        output.append(
            {
                "title": paper["title"],
                "abstract": paper["abstractText"],
                "ratings": ratings,
                "sections": [asdict(section) for section in sections],
                "approval": _parse_approval(item["approval"]),
                "references": _process_references(paper, context_sentences),
            }
        )

    print("no.  input papers:", len(data))
    print("no. output papers:", len(output), f"({len(output) / len(data):.2%})")

    output_file.write_text(json.dumps(output, indent=2))


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
    args = parser.parse_args()
    extract_interesting(args.input, args.output, args.context_sentences)


if __name__ == "__main__":
    main()
