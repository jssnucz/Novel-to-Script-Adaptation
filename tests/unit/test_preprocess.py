"""Tests for the Chinese text preprocessing module (Task 1.2)."""

import pytest

from engine.preprocess import (
    normalize_paragraphs,
    preprocess,
    strip_bom,
    unify_quotes,
)
from engine.models import PreprocessArtifact


# ---------------------------------------------------------------------------
# unify_quotes
# ---------------------------------------------------------------------------


class TestUnifyQuotes:
    def test_converts_corner_brackets_to_double(self):
        """「」→ standard ""."""
        result = unify_quotes("他说「你好吗」？")
        assert result == '他说"你好吗"？'

    def test_converts_white_corner_to_single(self):
        """『』→ standard ''."""
        result = unify_quotes("她问『几点了』？")
        assert result == "她问'几点了'？"

    def test_preserves_standard_quotes(self):
        """Already-standard "" and '' are left unchanged."""
        text = '他说"你好"，她回答\'是的\'。'
        result = unify_quotes(text)
        assert result == text

    def test_mixed_all_four_styles(self):
        """All four quote styles in a single string.

        Each corner-bracket character maps independently to its standard
        counterpart, so ``「「`` becomes ``""`` (two opening double-quotes).
        """
        text = '「「outer」」『inner』"std"\'sstd\''
        result = unify_quotes(text)
        assert result == '""outer""\'inner\'"std"\'sstd\''

    def test_empty_string(self):
        """Empty string returns empty string."""
        assert unify_quotes("") == ""

    def test_no_quotes(self):
        """String without any quotes is unchanged."""
        text = "这是一个测试。"
        assert unify_quotes(text) == text


# ---------------------------------------------------------------------------
# normalize_paragraphs
# ---------------------------------------------------------------------------


class TestNormalizeParagraphs:
    def test_collapses_multiple_blank_lines(self):
        """Three or more consecutive newlines become two."""
        text = "第一段\n\n\n\n\n第二段"
        result = normalize_paragraphs(text)
        assert result == "第一段\n\n第二段"

    def test_strips_trailing_whitespace_per_line(self):
        """Each line has trailing spaces/tabs removed."""
        text = "第一段  \n第二段\t\n第三段  \t "
        result = normalize_paragraphs(text)
        assert result == "第一段\n第二段\n第三段"

    def test_normalizes_crlf_to_lf(self):
        """\r\n → \n."""
        text = "第一段\r\n第二段\r\n第三段"
        result = normalize_paragraphs(text)
        assert result == "第一段\n第二段\n第三段"

    def test_normalizes_cr_to_lf(self):
        """Standalone \r → \n."""
        text = "第一段\r第二段\r第三段"
        result = normalize_paragraphs(text)
        assert result == "第一段\n第二段\n第三段"

    def test_mixed_newline_styles(self):
        """Handles \r\n, \r, \n mixed together."""
        text = "A\r\nB\rC\nD\r\n\r\r\r\n\nE"
        result = normalize_paragraphs(text)
        # After CRLF->LF and CR->LF: "A\nB\nC\nD\n\n\n\n\nE"
        # Then collapse 3+ -> 2: "A\nB\nC\nD\n\nE"
        assert result == "A\nB\nC\nD\n\nE"

    def test_empty_string(self):
        """Empty string returns empty string."""
        assert normalize_paragraphs("") == ""

    def test_no_changes_needed(self):
        """Already-normalized text is unchanged."""
        text = "第一段\n第二段"
        assert normalize_paragraphs(text) == text


# ---------------------------------------------------------------------------
# strip_bom
# ---------------------------------------------------------------------------


class TestStripBom:
    def test_removes_bom(self):
        """UTF-8 BOM is stripped from the beginning."""
        result = strip_bom("﻿测试文本")
        assert result == "测试文本"

    def test_no_bom(self):
        """String without BOM is unchanged."""
        text = "测试文本"
        assert strip_bom(text) == text

    def test_empty_string(self):
        """Empty string returns empty string."""
        assert strip_bom("") == ""


# ---------------------------------------------------------------------------
# preprocess (full pipeline)
# ---------------------------------------------------------------------------


class TestPreprocess:
    def test_returns_preprocess_artifact(self):
        """Returns a PreprocessArtifact instance."""
        result = preprocess("测试", "test.txt")
        assert isinstance(result, PreprocessArtifact)

    def test_has_correct_fields(self):
        """Artifact has schema_version, original_path, cleaned_text, total_chars."""
        result = preprocess("测试", "test.txt")
        assert result.schema_version == "1.0"
        assert result.original_path == "test.txt"

    def test_preserves_content_text(self):
        """Cleaned text preserves meaningful content (after quote unification)."""
        text = "他说「你好」"
        result = preprocess(text, "test.txt")
        assert '"你好"' in result.cleaned_text

    def test_strips_bom(self):
        """BOM character is removed from cleaned_text."""
        result = preprocess("﻿测试文本", "test.txt")
        assert result.cleaned_text == "测试文本"

    def test_total_chars_equals_len_of_cleaned_text(self):
        """total_chars matches the final cleaned text length."""
        result = preprocess("测试文本", "test.txt")
        assert result.total_chars == len(result.cleaned_text)

    def test_full_pipeline(self):
        """End-to-end: BOM stripped, quotes unified, paragraphs normalized."""
        text = "﻿他说「你好」\r\n\r\n\r\n\r\n她说『再见』  \t"
        result = preprocess(text, "novel.txt")
        assert result.schema_version == "1.0"
        assert result.original_path == "novel.txt"
        assert "﻿" not in result.cleaned_text
        assert '"你好"' in result.cleaned_text
        assert "'再见'" in result.cleaned_text  # 『』→ single quotes
        assert "\r\n" not in result.cleaned_text
        assert result.cleaned_text.count("\n\n") <= 1  # no triple-blank lines
        assert result.total_chars == len(result.cleaned_text)

    def test_empty_text(self):
        """Empty text produces artifact with empty cleaned_text and 0 total_chars."""
        result = preprocess("", "empty.txt")
        assert result.cleaned_text == ""
        assert result.total_chars == 0
