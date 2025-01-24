"""Find items that are different between the first and second JSON files.

Prints the data on both sides of the mismatch. If `--ref` is provided, also prints
that reference values.
"""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Any, TypeGuard

import typer

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command(help=__doc__, no_args_is_help=True)
def main(
    first_file: Annotated[
        Path, typer.Option("-1", "--first", help="Path to the first model result file.")
    ],
    second_file: Annotated[
        Path,
        typer.Option("-2", "--second", help="Path to the second model result file."),
    ],
    output_file: Annotated[
        Path | None,
        typer.Option(
            "-o", "--output", help="Path to output file with incompatible results."
        ),
    ] = None,
    first_path: Annotated[
        str, typer.Option("-f", "--first-path", help="Path to paper item in result.")
    ] = "item",
    second_path: Annotated[
        str, typer.Option("-s", "--second-path", help="Path to paper item in result.")
    ] = "item",
    ref_key: Annotated[
        str | None,
        typer.Option(
            "-r", "--ref", help="Reference value to print alongside differences."
        ),
    ] = None,
    diff_key: Annotated[
        str,
        typer.Option("-d", "--diff", help="Value to check for difference."),
    ] = "y_true",
    sort_key: Annotated[
        str,
        typer.Option("-k", "--sort", help="Key to sort the objects before comparison."),
    ] = "title",
    verbose: Annotated[
        bool,
        typer.Option(
            "-v", "--verbose", help="Print full information on different fields."
        ),
    ] = False,
) -> None:
    """Find items that are different between two JSON files."""
    first = _process_input(first_file, first_path, sort_key)
    second = _process_input(second_file, second_path, sort_key)

    assert len(first) == len(second)
    print(f"Items: {len(first)}")

    diffs: list[dict[str, Any]] = []
    for f, s in zip(first, second):
        assert f[sort_key] == s[sort_key]
        if ref_key:
            assert f.get(ref_key) == s.get(ref_key)

        if f[diff_key] != s[diff_key]:
            diffs.append({"first": f, "second": s})

    print(f"Different items: {len(diffs)}")
    print()

    if output_file:
        output_file.write_text(json.dumps(diffs, indent=2))

    for diff in diffs:
        keys = compare_dicts(diff["first"], diff["second"])
        if verbose:
            if real := diff["first"].get(ref_key):
                print(">>> real:", real)

            for key in keys:
                if "/" in key:
                    print("nested:", key)
                    print()
                    continue

                f = diff["first"][key]
                s = diff["second"][key]

                print(">>> first:", f)
                print(">>> second:", s)
                print()

            print()
            print("-" * 80)
            print()
        else:
            print(diff["first"]["title"])
            print(f"  {", ".join(keys)}")
            print()


def _process_input(file: Path, key_path: str, sort_key: str) -> list[dict[str, Any]]:
    data = json.loads(file.read_text())
    items = (_get_path(x, key_path) for x in data)
    return sorted(items, key=lambda x: x[sort_key])


def _get_path(d: dict[str, Any], path: str) -> Any:
    if not path:
        return d

    curr = d
    for key in path.strip(".").split("."):
        curr = curr[key]
    return curr


def _is_dict(val: object) -> TypeGuard[dict[str, Any]]:
    return isinstance(val, dict)


def _is_list(val: object) -> TypeGuard[list[Any]]:
    return isinstance(val, list)


def compare_dicts(
    dict1: dict[str, Any], dict2: dict[str, Any], path: str = ""
) -> list[str]:
    """Compare difference between dictonaries. Returns list of paths that differ."""
    differences: list[str] = []

    for key in sorted(dict1.keys() | dict2.keys()):
        sub_path = _join_key(path, key)

        # If the key is missing on one side, report it
        if key not in dict1:
            differences.append(sub_path)
            continue
        if key not in dict2:
            differences.append(sub_path)
            continue

        val1 = dict1[key]
        val2 = dict2[key]

        # Build the sub-path for this key
        sub_path = _join_key(path, key)

        # If both are dictionaries, recurse
        if _is_dict(val1) and _is_dict(val2):
            differences.extend(compare_dicts(val1, val2, sub_path))

        # If both are lists, compare lists
        elif _is_list(val1) and _is_list(val2):
            differences.extend(_compare_lists(val1, val2, sub_path))

        # Otherwise, compare by value (with special note that type mismatch is also a difference)
        elif val1 != val2:
            differences.append(sub_path)

    return differences


def _compare_lists(list1: list[Any], list2: list[Any], path: str) -> list[str]:
    """Compare two lists according to the rules.

    - If lengths differ, record a difference at 'path'.
    - If they're both purely scalar, treat them as unordered (sort or multiset)
      so that [1, 2, 3] vs [3, 2, 1] is considered equal.
    - Otherwise, compare element by element, recursing into dictionaries/lists.
    """
    if len(list1) != len(list2):
        # If lengths differ, short-circuit
        return [path]

    # Check if both lists are purely scalar
    if _all_scalars(list1) and _all_scalars(list2):
        # Compare as sorted (or via Counter) to ignore ordering
        if sorted(list1) != sorted(list2):
            return [path]
        return []

    # Otherwise, compare element-by-element in order
    differences: list[str] = []
    for i, (v1, v2) in enumerate(zip(list1, list2)):
        sub_path = f"{path}[{i}]"
        # If both are dicts
        if _is_dict(v1) and _is_dict(v2):
            differences.extend(compare_dicts(v1, v2, sub_path))
        # If both are lists
        elif _is_list(v1) and _is_list(v2):
            differences.extend(_compare_lists(v1, v2, sub_path))
        # Compare by direct equality
        elif v1 != v2:
            differences.append(sub_path)
    return differences


def _all_scalars(lst: Iterable[Any]) -> bool:
    """Return True if every element in lst is a 'scalar' (not a list, not a dict).

    We could refine 'scalar' further if needed, but this suffices for the tests.
    """
    return all(not isinstance(x, list | dict) for x in lst)


def _join_key(base: str, key: str) -> str:
    """Join a dictionary key to the existing path using a slash."""
    if not base:
        return key
    return f"{base}/{key}"


def test_simple_value_difference() -> None:
    """Test that a simple value difference is detected."""
    assert compare_dicts({"a": 1}, {"a": 2}) == ["a"]


def test_missing_keys_any_side() -> None:
    """Test that if a key is present on one side but not the other, it's always reported."""
    # 'a' is only in the left; 'b' is only in the right.
    # Expect: both are reported.
    assert compare_dicts({"a": 1}, {"b": 2}) == ["a", "b"]


def test_nested_dict_difference() -> None:
    """Test that differences in nested dictionaries are detected."""
    assert compare_dicts({"a": {"b": 1}}, {"a": {"b": 2}}) == ["a/b"]


def test_unordered_list_equality() -> None:
    """Test that lists with same elements in different order are equal."""
    assert compare_dicts({"a": [1, 2, 3]}, {"a": [3, 2, 1]}) == []


def test_list_different_elements() -> None:
    """Test that lists with different elements are detected."""
    assert compare_dicts({"a": [1, 2]}, {"a": [1, 3]}) == ["a"]


def test_type_mismatch() -> None:
    """Test that type mismatches are detected as differences."""
    assert compare_dicts({"a": 1}, {"a": "1"}) == ["a"]


def test_multiple_differences() -> None:
    """Test detection of multiple differences in nested structure."""
    assert compare_dicts(
        {"a": 1, "b": {"c": 2, "d": 3}}, {"a": 2, "b": {"c": 2, "d": 4}}
    ) == ["a", "b/d"]


def test_empty_dicts() -> None:
    """Test comparison of empty dictionaries."""
    assert compare_dicts({}, {}) == []


def test_deep_nesting() -> None:
    """Test differences in deeply nested structures."""
    assert compare_dicts(
        {"a": {"b": {"c": {"d": 1}}}}, {"a": {"b": {"c": {"d": 2}}}}
    ) == ["a/b/c/d"]


def test_nested_list_item() -> None:
    """Test differences in nested list items."""
    assert compare_dicts(
        {"a": [1, {"b": 2}]},
        {"a": [1, {"b": 3}]},
    ) == ["a[1]/b"]


def test_list_of_dicts() -> None:
    """Test differences in lists of dictionaries."""
    d1 = {"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}
    d2 = {"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bobby"}]}
    assert compare_dicts(d1, d2) == ["users[1]/name"]


def test_list_length_difference() -> None:
    """Test that lists of different lengths are detected as different."""
    d1 = {
        "users": [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
        ]
    }
    d2 = {
        "users": [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
            {"id": 3, "name": "Charlie"},
        ]
    }
    assert compare_dicts(d1, d2) == ["users"]


if __name__ == "__main__":
    app()
