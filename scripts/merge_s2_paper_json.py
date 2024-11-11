"""Merge two JSON files containing paper data.

Every object in both files should either have a `paper_id` or `paperId` key.
"""

import copy
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from paper.util import HelpOnErrorArgumentParser, JSONObject, JSONValue


def merge_values(val1: JSONValue, val2: JSONValue) -> Any:
    """Recursively merge two values based on their types.

    Lists are merged, objects are recursively merged, primitives keep first value.
    """
    # If both are lists, merge them
    if isinstance(val1, list) and isinstance(val2, list):
        return val1 + val2

    # If both are dictionaries, merge them recursively
    if isinstance(val1, Mapping) and isinstance(val2, Mapping):
        return merge_objects(val1, val2)

    # For primitive types or mismatched types, keep the first value
    return val1


def merge_objects(obj1: JSONObject, obj2: JSONObject) -> JSONObject:
    """Recursively merge two objects.

    Keeping first value for primitives, merging lists, and recursively merging nested
    objects.
    """
    result = copy.deepcopy(obj1)

    for key, val2 in obj2.items():
        if key not in result:
            result[key] = val2
        else:
            result[key] = merge_values(result[key], val2)

    return result


def merge_paper_lists(
    papers1: list[JSONObject], papers2: list[JSONObject]
) -> list[JSONObject]:
    """Merge two lists of paper objects using `paperId`."""
    # Create a dictionary of papers by paper_id from the first list
    merged_dict = {paper["paperId"]: paper for paper in papers1}

    # Merge in papers from the second list
    for paper2 in papers2:
        paper_id = paper2["paperId"]
        if paper_id in merged_dict:
            merged_dict[paper_id] = merge_objects(merged_dict[paper_id], paper2)
        else:
            merged_dict[paper_id] = paper2

    # Convert back to list
    return list(merged_dict.values())


def main(file1_path: Path, file2_path: Path, output_path: Path) -> None:
    """Merge two JSON files containing arrays of paper objects.

    Quits with an error message if either file:
    - Doesn't have the right format (array of objects).
    - Is an empty array.
    - Has an object that missing the `paper_id`/`paperId` key.

    Args:
        file1_path: Path to first JSON file
        file2_path: Path to second JSON file
        output_path: Path to write merged results
    """
    # Read first file
    with open(file1_path) as f:
        papers1 = json.load(f)

    # Read second file
    with open(file2_path) as f:
        papers2 = json.load(f)

    # Validate input format
    if not isinstance(papers1, list) or not isinstance(papers2, list):
        sys.exit("Both JSON files must contain arrays at the root level")

    papers1 = cast(list[JSONObject], papers1)
    papers2 = cast(list[JSONObject], papers2)
    _check_paper_id(file1_path, papers1)
    _check_paper_id(file2_path, papers2)

    if not papers1 or not papers2:
        sys.exit("Input files must be non-empty.")

    merged_papers = merge_paper_lists(papers1, papers2)
    print(f"{len(papers1)=}")
    print(f"{len(papers2)=}")
    print(
        f"{len(merged_papers)=} ({len(merged_papers)/(len(papers1) + len(papers2)):.2%})"
    )

    with open(output_path, "w") as f:
        json.dump(merged_papers, f, indent=2)


def _check_paper_id(file: Path, papers: list[JSONObject]) -> None:
    """Check whether all objects in the file have the `paperId` key."""
    if not all("paperId" in paper for paper in papers):
        sys.exit(f"All objects in file '{file}' must contain a paperId key")


if __name__ == "__main__":
    parser = HelpOnErrorArgumentParser(__doc__)
    parser.add_argument("file1", type=Path, help="Path to first JSON file")
    parser.add_argument("file2", type=Path, help="Path to second JSON file")
    parser.add_argument("output", type=Path, help="Output file path (optional)")
    args = parser.parse_args()

    main(args.file1, args.file2, args.output)
