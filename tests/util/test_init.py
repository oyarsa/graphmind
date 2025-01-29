import logging
from typing import Any, Callable

import pytest

from paper.util import (
    format_bullet_list,
    format_numbered_list,
    get_icase,
    remove_parenthetical,
    groupby,
    fix_spaces_before_punctuation,
    on_exception,
)


@pytest.mark.parametrize(
    "items,prefix,indent,expected",
    [
        pytest.param(
            ["apple", "banana", "cherry"],
            "-",
            0,
            "- apple\n- banana\n- cherry",
            id="basic_formatting",
        ),
        pytest.param(
            ["first", "second"],
            "•",
            0,
            "• first\n• second",
            id="custom_prefix",
        ),
        pytest.param(
            ["item1", "item2"],
            "-",
            4,
            "    - item1\n    - item2",
            id="with_indent",
        ),
        pytest.param(
            [],
            "-",
            0,
            "",
            id="empty_list",
        ),
        pytest.param(
            ["item\nwith\nnewlines", "item\twith\ttabs", "item with spaces"],
            "-",
            0,
            "- item\nwith\nnewlines\n- item\twith\ttabs\n- item with spaces",
            id="special_chars",
        ),
        pytest.param(
            ["item1"],
            "-",
            0,
            "- item1",
            id="zero_indent",
        ),
    ],
)
def test_format_bullet_list(items: list[str], prefix: str, indent: int, expected: str):
    """Test format_bullet_list with various inputs and expected outputs."""
    assert format_bullet_list(items, prefix=prefix, indent=indent) == expected


@pytest.mark.parametrize(
    "data, key, default, expected",
    [
        pytest.param(
            {"Hello": 1},
            "HELLO",
            -1,
            1,
            id="existing_insensitive_with_default",
        ),
        pytest.param(
            {"Hello": 1},
            "missing",
            -1,
            -1,
            id="non_existing_insensitive_with_default",
        ),
        pytest.param(
            {"key": 1, "KEY": 2},
            "Key",
            -1,
            1,
            id="case_conflict_returns_first",
        ),
        pytest.param(
            {"Name": "Alice"}, "name", None, "Alice", id="insensitive_without_default"
        ),
        pytest.param(
            {"Name": "Alice"},
            "missing",
            None,
            None,
            id="missing_key_without_default",
        ),
        pytest.param(
            {"café": "coffee"},
            "CAFÉ",
            None,
            "coffee",
            id="unicode_case",
        ),
    ],
)
def test_get_icase(data: dict[str, Any], key: str, default: Any, expected: Any) -> None:
    assert get_icase(data, key, default) == expected


@pytest.mark.parametrize(
    "input_text,expected",
    [
        pytest.param("Example", "Example", id="no_change"),
        pytest.param(
            "Another example (with items).", "Another example.", id="easy_case"
        ),
        pytest.param("With (stuff) in the middle", "With in the middle", id="middle"),
        pytest.param("", "", id="empty_string"),
        pytest.param("()", "", id="empty_parens"),
        pytest.param("(complete)", "", id="all_in_parens"),
        pytest.param("Multiple (sets) of (parentheses)", "Multiple of", id="multiple"),
        pytest.param("Nested (outer (inner) outer)", "Nested", id="nested"),
        pytest.param("Incomplete (", "Incomplete (", id="incomplete"),
        pytest.param(
            "Incomplete (some text",
            "Incomplete (some text",
            id="incomplete_with_leftover",
        ),
        pytest.param(
            "Incomplete(should be removed) (some text",
            "Incomplete (some text",
            id="valid_before_incomplete",
        ),
        pytest.param(
            "Nested(remains (removed)", "Nested(remains", id="nested_incomplete"
        ),
        pytest.param("Nested(outer (inner) (removed))", "Nested", id="nested_2"),
        pytest.param("Incomplete )", "Incomplete)", id="incomplete_closing"),
        pytest.param(
            "Special chars ($@#) here", "Special chars here", id="special_chars"
        ),
        pytest.param(
            "   Whitespace   (test)   here   ", "Whitespace here", id="whitespace"
        ),
    ],
)
def test_remove_parenthetical(input_text: str, expected: str):
    assert remove_parenthetical(input_text) == expected


@pytest.mark.parametrize(
    "items, key_func, expected",
    [
        pytest.param(
            ["apple", "banana", "avocado", "blueberry"],
            lambda x: x[0],  # type: ignore
            {"a": ["apple", "avocado"], "b": ["banana", "blueberry"]},
            id="by_first_letter",
        ),
        pytest.param(
            [-2, -1, 0, 1, 2, 3],
            lambda x: None if x < 0 else "even" if x % 2 == 0 else "odd",  # type: ignore
            {"even": [0, 2], "odd": [1, 3]},
            id="even_odd",
        ),
        pytest.param([], lambda x: x, {}, id="empty"),  # type: ignore
    ],
)
def test_groupby(
    items: list[Any], key_func: Callable[[Any], Any | None], expected: dict[str, Any]
):
    assert groupby(items, key_func) == expected


@pytest.mark.parametrize(
    "input_text, expected",
    [
        pytest.param("Hello , world !", "Hello, world!", id="comma"),
        pytest.param(
            "Multiple   spaces    .", "Multiple   spaces.", id="multiple spaces"
        ),
        pytest.param("(text  )", "(text)", id="closing paren"),
        pytest.param('He said   "quote"   .', 'He said   "quote".', id="final period"),
        pytest.param(
            "Mixed case: ! ? . ,", "Mixed case:!?.,", id="multiple punctuation"
        ),
        pytest.param("No changes needed!", "No changes needed!", id="no changes"),
        pytest.param("", "", id="empty"),
    ],
)
def test_fix_punctuation_spaces(input_text: str, expected: str):
    assert fix_spaces_before_punctuation(input_text) == expected


@pytest.mark.parametrize(
    "items,prefix,suffix,indent,start,sep,expected",
    [
        pytest.param(
            ["apple", "banana", "cherry"],
            "",
            ".",
            0,
            1,
            "\n",
            "1. apple\n2. banana\n3. cherry",
            id="basic_formatting",
        ),
        pytest.param(
            ["first", "second"],
            "",
            ")",
            0,
            1,
            "\n",
            "1) first\n2) second",
            id="custom_prefix",
        ),
        pytest.param(
            ["item1", "item2"],
            "",
            ".",
            4,
            1,
            "\n",
            "    1. item1\n    2. item2",
            id="with_indent",
        ),
        pytest.param(
            [],
            "",
            ".",
            0,
            1,
            "\n",
            "",
            id="empty_list",
        ),
        pytest.param(
            ["item1"],
            "",
            ".",
            0,
            5,
            "\n",
            "5. item1",
            id="custom_start",
        ),
        pytest.param(
            ["a", "b"],
            "",
            ".",
            0,
            1,
            " ",
            "1. a 2. b",
            id="custom_separator",
        ),
        pytest.param(
            ["item\nwith\nnewlines", "item\twith\ttabs"],
            "",
            ".",
            0,
            1,
            "\n",
            "1. item\nwith\nnewlines\n2. item\twith\ttabs",
            id="special_chars",
        ),
        pytest.param(
            ["a", "b"],
            "1.",
            ".",
            2,
            1,
            "\n",
            "  1.1. a\n  1.2. b",
            id="nested_number",
        ),
    ],
)
def test_format_numbered_list(
    items: list[str],
    prefix: str,
    suffix: str,
    indent: int,
    start: int,
    sep: str,
    expected: str,
):
    """Test format_numbered_list with various inputs and expected outputs."""
    assert (
        format_numbered_list(
            items, prefix=prefix, suffix=suffix, indent=indent, start=start, sep=sep
        )
        == expected
    )


# Testing `on_exception` decorator


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (10, 2, 5),
        (10, 0, 0),  # Tests exception case
        (-8, 2, -4),
    ],
)
def test_on_exception_divide(a: float, b: float, expected: float):
    @on_exception(default=0.0)
    def divide(a: float, b: float) -> float:
        return a / b

    assert divide(a, b) == expected


def test_on_exception_custom_default():
    @on_exception(default="error")
    def failing_function() -> str:
        raise ValueError()

    assert failing_function() == "error"


def test_on_exception_preserves_successful_result():
    @on_exception(default="error")
    def success_function() -> str:
        return "success"

    assert success_function() == "success"


def test_on_exception_logger_captures_exception(caplog: pytest.LogCaptureFixture):
    test_logger = logging.getLogger("test")

    @on_exception(default="error", logger=test_logger)
    def failing_function() -> str:
        raise ValueError("test error")

    with caplog.at_level(logging.ERROR):
        result = failing_function()

    assert result == "error"
    assert "Error suppressed with `on_exception`" in caplog.text
    assert "ValueError: test error" in caplog.text


def test_on_exception_no_logger_swallows_exception():
    @on_exception(default="error")
    def failing_function() -> str:
        raise ValueError()

    result = failing_function()
    assert result == "error"
