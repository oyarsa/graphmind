"""Find the index where the citation contexts appear in the original paper.

The goal is to understand what `startOffset` and `endOffset` in `referenceMentions` mean.

The --verbose option enables printing the occurrences where the `context` somehow does
not appear in any of the sections. Otherwise, we just print the counts in the end.
"""

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
)


@app.command(help=__doc__)
def main(
    input_file: Annotated[
        Path, typer.Argument(help="Path to input JSON file (asap_merged.json).")
    ],
    output_file: Annotated[Path, typer.Argument(help="Path to output JSON file.")],
    verbose: Annotated[bool, typer.Option(help="Enable verbose output")] = False,
) -> None:
    input_data = TypeAdapter(list[Entry]).validate_json(input_file.read_bytes())

    output_data: list[ContextFound] = []
    count_section_not_found = 0
    count_char_not_found = 0
    count_word_not_found = 0
    count_total = 0

    for entry in input_data:
        paper = entry.paper
        for reference in paper.reference_mentions:
            count_total += 1

            context = reference.context
            section = _find_context_section(paper.sections, context)

            if section is None:
                count_section_not_found += 1
                if verbose:
                    print(
                        f"Not found {count_section_not_found}: title={paper.title} ---"
                        f" {context=}\n"
                    )
                continue

            if char_range := _find_char_range(section.text, context):
                context_char_start_idx, context_char_end_idx = char_range
            else:
                context_char_start_idx, context_char_end_idx = None, None
                count_char_not_found += 1

            if word_range := _find_word_range(section.text, context):
                context_word_start_idx, context_word_end_idx = word_range
            else:
                context_word_start_idx, context_word_end_idx = None, None
                count_word_not_found += 1

            output_data.append(
                ContextFound(
                    context=context,
                    start_offset=reference.start_offset,
                    end_offset=reference.end_offset,
                    section_heading=section.heading,
                    section_text=section.text,
                    context_start_char_idx=context_char_start_idx,
                    context_end_char_idx=context_char_end_idx,
                    context_start_word_idx=context_word_start_idx,
                    context_end_word_idx=context_word_end_idx,
                )
            )

    output_file.write_bytes(
        TypeAdapter(list[ContextFound]).dump_json(output_data, indent=2)
    )

    print(f"Total: {count_total}")
    print(
        f"Section not found: {count_section_not_found} ({count_section_not_found/count_total:.2%})"
    )
    print(
        f"Char not found: {count_char_not_found} ({count_char_not_found/count_total:.2%})"
    )
    print(
        f"Word not found: {count_word_not_found} ({count_word_not_found/count_total:.2%})"
    )


# Input data types following the format from asap_merged.json
class Section(BaseModel):
    model_config = ConfigDict(frozen=True)

    heading: str | None
    text: str


class ReferenceMention(BaseModel):
    model_config = ConfigDict(frozen=True)

    reference_id: int = Field(alias="referenceID")
    start_offset: int = Field(alias="startOffset")
    end_offset: int = Field(alias="endOffset")
    context: str


class Paper(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str
    sections: Sequence[Section]
    reference_mentions: Sequence[ReferenceMention] = Field(alias="referenceMentions")


class Entry(BaseModel):
    model_config = ConfigDict(frozen=True)

    paper: Paper


class ContextFound(BaseModel):
    """Output data containg the input context/section/etc. and possible offsets."""

    model_config = ConfigDict(frozen=True)

    context: str
    start_offset: int
    end_offset: int
    section_heading: str | None
    section_text: str
    context_start_char_idx: int | None
    context_end_char_idx: int | None
    context_start_word_idx: int | None
    context_end_word_idx: int | None


def _find_context_section(sections: Sequence[Section], context: str) -> Section | None:
    """Find the first section that contains the citaton context sentence."""
    for section in sections:
        if context in section.text:
            return section

    return None


def _find_char_range(text: str, sentence: str) -> tuple[int, int] | None:
    """Find the (start, end) range by character index where `sentence` appears in `text`.

    Returns:
        (start, start+len(sentence)) if `sentence` is in `text`. Otherwise, None.
    """
    try:
        start = text.index(sentence)
    except ValueError:
        return None
    else:
        end = start + len(sentence)
        return start, end


def _find_word_range(text: str, sentence: str) -> tuple[int, int] | None:
    r"""Find the (start, end) range by word index where `sentence` appears in `text`.

    Splits both text and sentence by the "\w+" regex.

    Returns:
        (start, start+len(sentence_words)) if `sentence` is in `text`. Otherwise, None.
    """
    paragraph_words = re.findall(r"\w+", text)
    sentence_words = re.findall(r"\w+", sentence)

    for i in range(len(paragraph_words) - len(sentence_words) + 1):
        subsentence = paragraph_words[i : i + len(sentence_words)]
        if subsentence == sentence_words:
            return i, i + len(sentence_words)

    return None


if __name__ == "__main__":
    app()
