"""Tests for LaTeX parser functions."""

import re
from unittest.mock import patch

import pytest

from paper.orc.latex_parser import (
    _convert_latex_to_markdown,
    _match_braced_group,
    _rebalance_braces,
    _remove_command_with_braced_args,
    _sanitise_for_pandoc,
    _strip_problematic_packages,
    extract_title_from_bibitem_entry,
)


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


class TestMatchBracedGroup:
    """Tests for _match_braced_group."""

    def test_simple(self) -> None:
        assert _match_braced_group("{hello}", 0) == 7

    def test_nested(self) -> None:
        assert _match_braced_group("{a{b}c}", 0) == 7

    def test_deeply_nested(self) -> None:
        text = r"{a{\textbf{bold}}c}"
        assert _match_braced_group(text, 0) == len(text)

    def test_escaped_brace(self) -> None:
        assert _match_braced_group(r"{a\}b}", 0) == 6

    def test_unmatched(self) -> None:
        assert _match_braced_group("{hello", 0) == -1

    def test_not_a_brace(self) -> None:
        assert _match_braced_group("hello", 0) == -1

    def test_offset(self) -> None:
        assert _match_braced_group("xx{hi}yy", 2) == 6


class TestRebalanceBraces:
    """Tests for _rebalance_braces."""

    def test_already_balanced(self) -> None:
        assert _rebalance_braces("{hello}") == "{hello}"

    def test_appends_missing_closing(self) -> None:
        result = _rebalance_braces("{hello")
        assert result == "{hello}"
        assert result.count("{") == result.count("}")

    def test_drops_orphan_closing(self) -> None:
        result = _rebalance_braces("text } more")
        assert "}" not in result
        assert "text" in result
        assert "more" in result

    def test_nested_unclosed(self) -> None:
        result = _rebalance_braces("{a{b}")
        assert result.count("{") == result.count("}")

    def test_escaped_braces_ignored(self) -> None:
        result = _rebalance_braces(r"\{a, b\}")
        assert result == r"\{a, b\}"
        assert result.count("{") == result.count("}")

    def test_mixed_escaped_and_real(self) -> None:
        result = _rebalance_braces(r"{\textbf{set \{a\}}}")
        assert result.count("{") == result.count("}")
        assert r"\{a\}" in result

    def test_empty(self) -> None:
        assert _rebalance_braces("") == ""

    def test_no_braces(self) -> None:
        assert _rebalance_braces("plain text") == "plain text"


class TestRemoveCommandWithBracedArgs:
    """Tests for _remove_command_with_braced_args."""

    def test_simple_newcommand(self) -> None:
        pat = re.compile(r"\\newcommand\*?\{[^}]+\}(\[\d+\])?")
        text = r"\newcommand{\foo}[1]{body with {nested}}"
        result = _remove_command_with_braced_args(text, pat)
        assert result == ""

    def test_preserves_surrounding_text(self) -> None:
        pat = re.compile(r"\\newcommand\*?\{[^}]+\}(\[\d+\])?")
        text = r"before \newcommand{\foo}{body} after"
        result = _remove_command_with_braced_args(text, pat)
        assert result == "before  after"

    def test_multiline_brace_body(self) -> None:
        pat = re.compile(r"\\newtcolorbox\{[^}]+\}(\[\d+\])?(\[[^\]]*\])?")
        text = "before\n\\newtcolorbox{mybox}[2][]\n{ colback=red }\nafter"
        result = _remove_command_with_braced_args(text, pat)
        assert "newtcolorbox" not in result
        assert "colback" not in result
        assert "before" in result
        assert "after" in result

    def test_multiline_with_comment(self) -> None:
        pat = re.compile(r"\\newcommand\*?\{[^}]+\}(\[\d+\])?")
        text = "\\newcommand{\\foo}[1]\n% definition follows\n{body}\nrest"
        result = _remove_command_with_braced_args(text, pat)
        assert "\\foo" not in result
        assert "body" not in result
        assert "rest" in result


class TestStripProblematicPackages:
    """Tests for _strip_problematic_packages."""

    def test_newtcolorbox_simple(self) -> None:
        tex = r"\newtcolorbox{mybox}{colback=red}" + "\nOK"
        result = _strip_problematic_packages(tex)
        assert "newtcolorbox" not in result
        assert "OK" in result

    def test_newtcolorbox_nested_braces(self) -> None:
        tex = (
            r"\newtcolorbox{mybox}{colback=red,fonttitle={\bfseries}}"
            "\nOK"
        )
        result = _strip_problematic_packages(tex)
        assert "newtcolorbox" not in result
        assert "OK" in result
        # Should not leave orphan braces
        assert result.count("{") == result.count("}")

    def test_newtcolorbox_with_optional_args(self) -> None:
        tex = (
            r"\newtcolorbox{mybox}[2][]{colback=#2,title={\textbf{#1}}}"
            "\nOK"
        )
        result = _strip_problematic_packages(tex)
        assert "newtcolorbox" not in result
        assert "OK" in result

    def test_newtcolorbox_multiline_body(self) -> None:
        tex = "\\newtcolorbox{mybox}[2][]\n{ colback=#2 }\nOK"
        result = _strip_problematic_packages(tex)
        assert "newtcolorbox" not in result
        assert "colback" not in result
        assert "OK" in result

    def test_tcolorbox_env_removal(self) -> None:
        tex = "before\n\\begin{tcolorbox}\ncontent\n\\end{tcolorbox}\nafter"
        result = _strip_problematic_packages(tex)
        assert "tcolorbox" not in result
        assert "before" in result
        assert "after" in result

    def test_usepackage_arxiv_removal(self) -> None:
        tex = "\\usepackage{arxiv}\n\\begin{document}\nHello."
        result = _strip_problematic_packages(tex)
        assert "arxiv" not in result
        assert "Hello" in result


class TestSanitiseForPandoc:
    """Tests for _sanitise_for_pandoc."""

    def test_extracts_document_body(self) -> None:
        tex = "\\documentclass{article}\n\\begin{document}\nHello.\n\\end{document}"
        result = _sanitise_for_pandoc(tex)
        assert "documentclass" not in result
        assert "Hello" in result

    def test_balances_missing_closing_braces(self) -> None:
        tex = "\\begin{document}\n\\textbf{unclosed\n\\end{document}"
        result = _sanitise_for_pandoc(tex)
        assert result.count("{") == result.count("}")

    def test_removes_orphan_closing_braces(self) -> None:
        tex = "\\begin{document}\ntext\n}\nmore\n\\end{document}"
        result = _sanitise_for_pandoc(tex)
        assert result.count("{") == result.count("}")

    def test_escaped_braces_not_counted(self) -> None:
        tex = "\\begin{document}\nThe set \\{a, b\\} is finite.\n\\end{document}"
        result = _sanitise_for_pandoc(tex)
        assert "\\{a, b\\}" in result
        assert result.count("{") == result.count("}")

    def test_removes_def_with_body(self) -> None:
        tex = (
            "\\begin{document}\nbefore\n\\def\\add#1#2{#1 + #2}\nafter\n\\end{document}"
        )
        result = _sanitise_for_pandoc(tex)
        assert "\\def" not in result
        assert "#1 + #2" not in result
        assert "before" in result
        assert "after" in result
        assert result.count("{") == result.count("}")

    def test_removes_def_without_params(self) -> None:
        tex = "\\begin{document}\n\\def\\myname{Alice}\nHello.\n\\end{document}"
        result = _sanitise_for_pandoc(tex)
        assert "\\def" not in result
        assert "Alice" not in result
        assert "Hello" in result

    def test_removes_spaced_def_with_params(self) -> None:
        tex = (
            "\\begin{document}\nbefore\n"
            "\\def \\add #1 #2 {#1 + #2}\nafter\n\\end{document}"
        )
        result = _sanitise_for_pandoc(tex)
        assert "\\def" not in result
        assert "#1 + #2" not in result
        assert "before" in result
        assert "after" in result
        assert result.count("{") == result.count("}")

    def test_removes_spaced_def_without_params(self) -> None:
        tex = "\\begin{document}\n\\def \\myname{Alice}\nHello.\n\\end{document}"
        result = _sanitise_for_pandoc(tex)
        assert "\\def" not in result
        assert "Alice" not in result
        assert "Hello" in result

    def test_removes_gdef(self) -> None:
        tex = "\\begin{document}\n\\gdef\\foo{bar}\nHello.\n\\end{document}"
        result = _sanitise_for_pandoc(tex)
        assert "\\gdef" not in result
        assert "bar" not in result
        assert "Hello" in result

    def test_removes_global_def(self) -> None:
        tex = "\\begin{document}\n\\global\\def \\foo #1 {#1}\nHello.\n\\end{document}"
        result = _sanitise_for_pandoc(tex)
        assert "\\global" not in result
        assert "\\def" not in result
        assert "Hello" in result
        assert result.count("{") == result.count("}")

    def test_removes_long_gdef(self) -> None:
        tex = "\\begin{document}\n\\long\\gdef \\foo #1 {#1}\nHello.\n\\end{document}"
        result = _sanitise_for_pandoc(tex)
        assert "\\long" not in result
        assert "\\gdef" not in result
        assert "Hello" in result
        assert result.count("{") == result.count("}")

    def test_removes_algorithm_env(self) -> None:
        tex = (
            "\\begin{document}\nbefore\n"
            "\\begin{algorithm}\ncode\n\\end{algorithm}\n"
            "after\n\\end{document}"
        )
        result = _sanitise_for_pandoc(tex)
        assert "algorithm" not in result
        assert "before" in result
        assert "after" in result


class TestConvertLatexToMarkdown:
    """Tests for _convert_latex_to_markdown fallback chain."""

    @patch("paper.orc.latex_parser._run_pandoc")
    def test_raw_succeeds_no_fallback(self, mock_pandoc: object) -> None:
        """When raw content works, no preprocessing is applied."""
        mock_pandoc.return_value = ("# Hello", "")  # type: ignore[union-attr]
        result = _convert_latex_to_markdown(
            "\\section{Hello}", "\\section{Hello}", "test"
        )
        assert result == "# Hello"
        mock_pandoc.assert_called_once()  # type: ignore[union-attr]

    @patch("paper.orc.latex_parser._run_pandoc")
    def test_falls_through_to_stripped(self, mock_pandoc: object) -> None:
        """When raw fails, stripped is tried next."""
        mock_pandoc.side_effect = [  # type: ignore[union-attr]
            (None, "exit 64"),
            ("# Cleaned", ""),
            ("# Sanitised", ""),
        ]
        result = _convert_latex_to_markdown(
            "\\section{Hello}", "\\section{Hello}", "test"
        )
        assert result == "# Cleaned"
        assert mock_pandoc.call_count == 2  # type: ignore[union-attr]

    @patch("paper.orc.latex_parser._run_pandoc")
    def test_falls_through_to_sanitised(self, mock_pandoc: object) -> None:
        """When raw and stripped both fail, sanitised is tried."""
        mock_pandoc.side_effect = [  # type: ignore[union-attr]
            (None, "exit 64"),
            (None, "exit 64"),
            ("# Last resort", ""),
        ]
        result = _convert_latex_to_markdown(
            "\\section{Hello}", "\\section{Hello}", "test"
        )
        assert result == "# Last resort"
        assert mock_pandoc.call_count == 3  # type: ignore[union-attr]

    @patch("paper.orc.latex_parser._run_pandoc")
    def test_falls_through_to_full_clean(self, mock_pandoc: object) -> None:
        """When all lighter strategies fail, full-clean is tried."""
        mock_pandoc.side_effect = [  # type: ignore[union-attr]
            (None, "exit 64"),
            (None, "exit 64"),
            (None, "exit 64"),
            ("# Full clean", ""),
        ]
        result = _convert_latex_to_markdown(
            "\\section{Hello}", "\\section{Hello}", "test"
        )
        assert result == "# Full clean"
        assert mock_pandoc.call_count == 4  # type: ignore[union-attr]

    @patch("paper.orc.latex_parser._run_pandoc")
    def test_all_strategies_fail(self, mock_pandoc: object) -> None:
        """Returns None when every strategy fails."""
        mock_pandoc.return_value = (None, "exit 64")  # type: ignore[union-attr]
        result = _convert_latex_to_markdown(
            "\\section{Hello}", "\\section{Hello}", "test"
        )
        assert result is None
        assert mock_pandoc.call_count == 4  # type: ignore[union-attr]

    @pytest.mark.parametrize(
        ("name", "tex"),
        [
            (
                "gdef-spaced",
                "\\begin{document}\n\\gdef \\foo #1 {#1}\nText\n\\end{document}",
            ),
            (
                "gdef-spaced-two-params",
                "\\begin{document}\n\\gdef \\foo #1 #2 {#1 #2}\nText\n\\end{document}",
            ),
            (
                "gdef-newline-body",
                "\\begin{document}\n\\gdef \\foo #1\n{#1}\nText\n\\end{document}",
            ),
            (
                "gdef-newline-params",
                "\\begin{document}\n\\gdef \\foo\n#1 {#1}\nText\n\\end{document}",
            ),
            (
                "global-def-spaced",
                "\\begin{document}\n\\global\\def \\foo #1 {#1}\nText\n\\end{document}",
            ),
            (
                "global-def-multispace",
                "\\begin{document}\n\\global   \\def   \\foo   #1   {#1}\nText\n\\end{document}",
            ),
            (
                "global-def-newline-body",
                "\\begin{document}\n\\global\\def \\foo #1\n{#1}\nText\n\\end{document}",
            ),
            (
                "long-gdef-spaced",
                "\\begin{document}\n\\long\\gdef \\foo #1 {#1}\nText\n\\end{document}",
            ),
        ],
        ids=[
            "gdef_spaced",
            "gdef_spaced_two_params",
            "gdef_newline_body",
            "gdef_newline_params",
            "global_def_spaced",
            "global_def_multispace",
            "global_def_newline_body",
            "long_gdef_spaced",
        ],
    )
    def test_breaking_cases_def_family_variants(self, name: str, tex: str) -> None:
        """Known def-family variants should still yield usable markdown."""
        result = _convert_latex_to_markdown(tex, tex, name)
        assert result is not None
        assert "Text" in result
