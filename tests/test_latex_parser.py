"""Tests for LaTeX parser functions."""

import pytest

from paper.orc.latex_parser import extract_title_from_bibitem_entry


class TestExtractTitleFromBibitemEntry:
    """Tests for extract_title_from_bibitem_entry function."""

    @pytest.mark.parametrize(
        ("entry_text", "expected_title"),
        [
            # BibTeX-style with \newblock - common case
            (
                r"""Jimmy~Lei Ba, Jamie~Ryan Kiros, and Geoffrey~E Hinton.
\newblock Layer normalization.
\newblock {\em arXiv preprint arXiv:1607.06450}, 2016.""",
                "Layer normalization",
            ),
            # Multiline title in \newblock
            (
                r"""Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio.
\newblock Neural machine translation by jointly learning to align and
  translate.
\newblock {\em CoRR}, abs/1409.0473, 2014.""",
                "Neural machine translation by jointly learning to align and translate",
            ),
            # Title with colon
            (
                r"""Author Name.
\newblock Xception: Deep learning with depthwise separable convolutions.
\newblock {\em CoRR}, abs/1610.02357, 2016.""",
                "Xception: Deep learning with depthwise separable convolutions",
            ),
            # Title ending with year should have year stripped
            (
                r"""Some Author.
\newblock Gradient flow in recurrent nets: the difficulty of learning long-term
  dependencies, 2001.
\newblock In {\em Proc}.""",
                "Gradient flow in recurrent nets: the difficulty of learning long-term dependencies",
            ),
            # Journal-like pattern in \newblock should be skipped (arXiv)
            (
                r"""Author Name.
\newblock arXiv preprint arXiv:1234.5678.
\newblock Some other text.""",
                "Unknown Title",
            ),
            # Journal-like pattern in \newblock should be skipped (Proc)
            (
                r"""Author Name.
\newblock Proc. of NAACL.
\newblock Some other text.""",
                "Unknown Title",
            ),
            # Journal-like pattern in \newblock should be skipped (In)
            (
                r"""Author Name.
\newblock In Proceedings of NeurIPS.
\newblock Some other text.""",
                "Unknown Title",
            ),
            # No \newblock - should return Unknown Title
            (
                r"""Author Name. Some random text without proper formatting, 2020.""",
                "Unknown Title",
            ),
            # Empty entry
            (
                "",
                "Unknown Title",
            ),
            # Only whitespace
            (
                "   \n\t  ",
                "Unknown Title",
            ),
        ],
        ids=[
            "basic_newblock",
            "multiline_title",
            "title_with_colon",
            "title_with_trailing_year",
            "skip_arxiv_pattern",
            "skip_proc_pattern",
            "skip_in_pattern",
            "no_newblock",
            "empty_entry",
            "whitespace_only",
        ],
    )
    def test_extract_title(self, entry_text: str, expected_title: str) -> None:
        """Test title extraction from various bibitem formats."""
        result = extract_title_from_bibitem_entry(entry_text)
        assert result == expected_title
