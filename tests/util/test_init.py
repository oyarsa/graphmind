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
