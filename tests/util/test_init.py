from typing import Any
import pytest
from paper.util import format_bullet_list, get_icase


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
