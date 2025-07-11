"""Test util module."""

import asyncio
import logging
from collections.abc import Callable
from typing import Any

import pytest

from paper.util import (
    extract_task_name,  # type: ignore
    fix_spaces_before_punctuation,
    format_bullet_list,
    format_numbered_list,
    get_icase,
    get_in,
    groupby,
    on_exception,
    remove_parenthetical,
    removeprefix_icase,
)


@pytest.mark.parametrize(
    ("items", "prefix", "indent", "expected"),
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
    ("data", "key", "default", "expected"),
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
    ("input_text", "expected"),
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
    ("items", "key_func", "expected"),
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
    ("input_text", "expected"),
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
    ("items", "prefix", "suffix", "indent", "start", "sep", "expected"),
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
    ("a", "b", "expected"),
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
        raise ValueError("Fail.")

    assert failing_function() == "error"


def test_on_exception_preserves_successful_result():
    @on_exception(default="error")
    def success_function() -> str:
        return "success"

    assert success_function() == "success"


@pytest.mark.parametrize(
    "level",
    ["warning", "info"],
)
def test_on_exception_logger_captures_exception(
    caplog: pytest.LogCaptureFixture, level: str
):
    test_logger = logging.getLogger("test")

    @on_exception(default="error", logger=test_logger, level=level)
    def failing_function() -> str:
        raise ValueError("test error")

    with caplog.at_level(level.upper()):
        result = failing_function()

    assert result == "error"
    assert "Error suppressed with `on_exception`" in caplog.text


@pytest.mark.parametrize(
    ("data", "path", "expected"),
    [
        ({"a": 1}, "a", 1),
        ({"a": {"b": 2}}, "a.b", 2),
        ({"a": {"b": {"c": 3}}}, "a.b.c", 3),
        ({"a": 1}, "b", None),
        ({"a": 1}, "", {"a": 1}),
        ({"a": {"b": 2}}, "a.c.d", None),
        ({"a": 1}, "a.b", None),
        (
            {"user": {"profile": {"name": "John", "age": 30}}},
            "user.profile.name",
            "John",
        ),
        ({"a": {"b": None}}, "a.b", None),
        ({"a": {"b": False}}, "a.b", False),
        ({}, "a", None),
        ({"a": {}}, "a", {}),
    ],
)
def test_get_by_path_parametrized(
    data: dict[str, Any], path: str, expected: Any | None
):
    assert get_in(data, path) == expected


@pytest.mark.parametrize(
    ("string", "prefix", "expected"),
    [
        ("ABCdef", "abc", "def"),
        ("aBcdef", "abc", "def"),
        ("abcdef", "abc", "def"),
        ("xyz", "abc", "xyz"),
        ("", "abc", ""),
    ],
)
def test_removeprefix_icase(string: str, prefix: str, expected: str) -> None:
    assert removeprefix_icase(string, prefix) == expected


class TestExtractTaskName:
    """Test extract_task_name with various awaitable types."""

    @staticmethod
    def sync_function() -> str:
        """Test sync function."""
        return "result"

    @staticmethod
    def another_sync_function(x: int, y: int) -> int:
        """Another test sync function."""
        return x + y

    @staticmethod
    async def another_async_function(x: int) -> int:
        """Another test async function."""
        return x * 2

    @staticmethod
    async def async_function() -> str:
        """Test async function."""
        return "async_result"

    @pytest.mark.asyncio
    async def test_coroutine_direct(self) -> None:
        """Test extraction from direct coroutine."""
        coro = self.async_function()
        assert extract_task_name(coro) == "async_function"
        await coro

    @pytest.mark.asyncio
    async def test_coroutine_with_args(self) -> None:
        """Test extraction from coroutine with arguments."""
        coro = self.another_async_function(5)
        assert extract_task_name(coro) == "another_async_function"
        await coro

    @pytest.mark.asyncio
    async def test_asyncio_task(self) -> None:
        """Test extraction from asyncio.Task."""
        task = asyncio.create_task(self.async_function())
        assert extract_task_name(task) == "async_function"
        await task

    @pytest.mark.asyncio
    async def test_asyncio_to_thread(self) -> None:
        """Test extraction from asyncio.to_thread."""
        to_thread_coro = asyncio.to_thread(self.sync_function)
        name = extract_task_name(to_thread_coro)
        await to_thread_coro
        assert name == "sync_function"

    @pytest.mark.asyncio
    async def test_asyncio_to_thread_with_args(self) -> None:
        """Test extraction from asyncio.to_thread with arguments."""
        to_thread_coro = asyncio.to_thread(self.another_sync_function, 3, 4)
        name = extract_task_name(to_thread_coro)
        await to_thread_coro

        # Should correctly extract the function name from wrapped to_thread
        assert name == "another_sync_function"

    @pytest.mark.asyncio
    async def test_asyncio_gather_single(self) -> None:
        """Test extraction from asyncio.gather with single coroutine."""
        gather_awaitable = asyncio.gather(self.async_function())
        name = extract_task_name(gather_awaitable)
        await gather_awaitable
        assert name == "async_function"

    @pytest.mark.asyncio
    async def test_asyncio_gather_multiple(self) -> None:
        """Test extraction from asyncio.gather with multiple coroutines."""
        gather_awaitable = asyncio.gather(
            self.async_function(),
            self.another_async_function(10),
            self.async_function(),
        )
        name = extract_task_name(gather_awaitable)
        await gather_awaitable
        assert name == "async_function"

    @pytest.mark.asyncio
    async def test_asyncio_task_wrapping_to_thread(self) -> None:
        """Test extraction from asyncio.Task wrapping asyncio.to_thread."""
        to_thread_coro = asyncio.to_thread(self.sync_function)
        task = asyncio.create_task(to_thread_coro)
        name = extract_task_name(task)
        await task

        # Should correctly extract the function name from wrapped to_thread
        assert name == "sync_function"

    @pytest.mark.asyncio
    async def test_asyncio_gather_with_to_thread(self) -> None:
        """Test extraction from asyncio.gather containing asyncio.to_thread."""
        gather_awaitable = asyncio.gather(
            asyncio.to_thread(self.sync_function), self.async_function()
        )
        name = extract_task_name(gather_awaitable)
        await gather_awaitable

        # Should extract from first awaitable (the to_thread call)
        assert name == "sync_function"
