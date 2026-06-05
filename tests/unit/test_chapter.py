"""Tests for the chapter detection and splitting module (Task 1.3)."""

import pytest

from engine.chapter import detect_chapter_boundaries, split_chapters
from engine.models import PreprocessArtifact, ChapterArtifact


# ---------------------------------------------------------------------------
# detect_chapter_boundaries
# ---------------------------------------------------------------------------


class TestDetectChapterBoundaries:
    def test_standard_chinese_chapter(self):
        r"""Pattern 1 (``第X章``) is detected with confidence 1.0."""
        text = "开头\n第一章 斗之气三段\n正文内容\n第二章 新的旅程\n更多"
        result = detect_chapter_boundaries(text)
        assert len(result) == 3  # sentinel + 2 chapters
        assert result[0] == (0, "(开头)", 1.0)
        assert result[1] == (1, "第一章 斗之气三段", 1.0)
        assert result[2] == (3, "第二章 新的旅程", 1.0)

    def test_english_chapter(self):
        r"""Pattern 2 (``Chapter X``) is detected with confidence 1.0
        (case-sensitive, capital C)."""
        text = "start\nChapter 1: The Beginning\nmore\nChapter 2\nend"
        result = detect_chapter_boundaries(text)
        assert len(result) == 3
        assert result[1] == (1, "Chapter 1: The Beginning", 1.0)
        assert result[2] == (3, "Chapter 2", 1.0)

    def test_english_chapter_lowercase_not_matched(self):
        """Lowercase ``chapter`` does NOT match Pattern 2."""
        text = "start\nchapter 1\ncontent"
        result = detect_chapter_boundaries(text)
        assert len(result) == 1  # only the sentinel

    def test_chinese_hui_chapter(self):
        r"""Pattern 3 (``第X回``) is detected with confidence 0.9."""
        text = "开头\n第一回 风雪山神庙\n正文"
        result = detect_chapter_boundaries(text)
        assert len(result) == 2
        assert result[1] == (1, "第一回 风雪山神庙", 0.9)

    def test_zhang_prefix_chapter(self):
        r"""Pattern 4 (``章X``) is detected with confidence 0.8."""
        text = "开头\n章 一 新的旅程\n正文"
        result = detect_chapter_boundaries(text)
        assert len(result) == 2
        assert result[1][1] == "章 一 新的旅程"
        assert result[1][2] == 0.8

    def test_numbered_heading(self):
        r"""Pattern 5 (``X、`` numbered heading) is detected with
        confidence 0.6."""
        text = "开头\n一、章节标题\n正文\n二．继续\n更多"
        result = detect_chapter_boundaries(text)
        assert len(result) == 3
        assert result[1][2] == 0.6
        assert result[2][2] == 0.6

    def test_special_chapter(self):
        r"""Pattern 6 (special markers like 序章/尾声) is detected with
        confidence 0.5."""
        text = "序章\n正文内容\n尾声\n结尾"
        result = detect_chapter_boundaries(text)
        assert len(result) == 3
        # Line 0 matches "序章" → sentinel plus match at same line
        assert result[0] == (0, "(开头)", 1.0)
        assert result[1] == (0, "序章", 0.5)
        assert result[2] == (2, "尾声", 0.5)

    def test_no_chapter_fallback(self):
        """Text with no recognizable chapter markers returns only the
        sentinel boundary."""
        text = "这是一篇普通的连续文本。\n没有章节标题。\n只有段落。"
        result = detect_chapter_boundaries(text)
        assert result == [(0, "(开头)", 1.0)]

    def test_empty_string(self):
        """Empty text returns only the sentinel boundary."""
        result = detect_chapter_boundaries("")
        assert result == [(0, "(开头)", 1.0)]

    def test_chinese_digits_in_title(self):
        """Pattern matching works with Chinese digit characters
        (零一二三四五六七八九十百千)."""
        text = "百章 零章\n第十二章 终章\n第十章 结局"
        result = detect_chapter_boundaries(text)
        # "百章 零章" does NOT match pattern 1 (doesn't start with 第)
        # "第十二章 终章" matches (starts with 第)
        # "第十章 结局" matches (starts with 第)
        # Also check pattern 4 for "百章"
        assert len(result) >= 2  # sentinel + at least one match


# ---------------------------------------------------------------------------
# split_chapters
# ---------------------------------------------------------------------------


class TestSplitChapters:
    def test_splits_basic_novel_into_chapters(self, basic_novel):
        """The basic 3-chapter novel yields at least 2 chapters."""
        artifact = PreprocessArtifact(
            schema_version="1.0",
            original_path="test.txt",
            cleaned_text=basic_novel,
            total_chars=len(basic_novel),
        )
        result = split_chapters(artifact)
        assert isinstance(result, ChapterArtifact)
        assert len(result.chapters) >= 2

    def test_chapter_ids_are_ch01_ch02(self, basic_novel):
        """First two chapters are numbered CH01, CH02."""
        artifact = PreprocessArtifact(
            schema_version="1.0",
            original_path="test.txt",
            cleaned_text=basic_novel,
            total_chars=len(basic_novel),
        )
        result = split_chapters(artifact)
        assert result.chapters[0].chapter_id == "CH01"
        assert result.chapters[1].chapter_id == "CH02"

    def test_standard_chapter_confidence_is_1(self, basic_novel):
        """Chapters matched by the ``第X章`` pattern have confidence 1.0."""
        artifact = PreprocessArtifact(
            schema_version="1.0",
            original_path="test.txt",
            cleaned_text=basic_novel,
            total_chars=len(basic_novel),
        )
        result = split_chapters(artifact)
        for ch in result.chapters:
            if "第" in ch.title and "章" in ch.title:
                assert ch.confidence == 1.0

    def test_fallback_marker_confidence_less_than_1(self):
        """A chapter detected via a non-standard pattern (e.g. 序章 with
        confidence 0.5) preserves that confidence on the chapter."""
        text = "序章 序言\n这是一段内容\n第二回 继续\n更多正文"
        artifact = PreprocessArtifact(
            schema_version="1.0",
            original_path="test.txt",
            cleaned_text=text,
            total_chars=len(text),
        )
        result = split_chapters(artifact)
        # Two chapters: CH01 (序章, confidence 0.5), CH02 (第二回, confidence 0.9)
        assert len(result.chapters) == 2
        assert result.chapters[0].confidence < 1.0
        assert result.chapters[1].confidence < 1.0

    def test_no_chapter_text_single_chapter_confidence_0_3(self):
        """Text with no detected chapter markers yields a single chapter
        with confidence <= 0.3."""
        text = "这是一篇没有章节标记的纯文本内容。\n全文连续。"
        artifact = PreprocessArtifact(
            schema_version="1.0",
            original_path="test.txt",
            cleaned_text=text,
            total_chars=len(text),
        )
        result = split_chapters(artifact)
        assert len(result.chapters) == 1
        assert result.chapters[0].chapter_id == "CH01"
        assert result.chapters[0].confidence <= 0.3

    def test_empty_chapters_skipped(self):
        """Chapters whose content is empty between boundaries are skipped
        and do not produce output chapters."""
        text = "第一章\n\n\n\n第二章\n内容"
        # Lines: ["第一章", "", "", "", "第二章", "内容"]
        # Boundaries: (0, 开头, 1.0), (0, 第一章, 1.0), (4, 第二章, 1.0)
        # Chapter 1 content = lines[1:4] = ["", "", ""] → empty → skip
        # Chapter 2 content = lines[5:] = ["内容"] → CH01
        artifact = PreprocessArtifact(
            schema_version="1.0",
            original_path="test.txt",
            cleaned_text=text,
            total_chars=len(text),
        )
        result = split_chapters(artifact)
        assert len(result.chapters) == 1
        assert result.chapters[0].chapter_id == "CH01"
        assert result.chapters[0].content.strip() == "内容"

    def test_returns_chapter_artifact(self, basic_novel):
        """Returns a proper ChapterArtifact with schema_version."""
        artifact = PreprocessArtifact(
            schema_version="1.0",
            original_path="test.txt",
            cleaned_text=basic_novel,
            total_chars=len(basic_novel),
        )
        result = split_chapters(artifact)
        assert isinstance(result, ChapterArtifact)
        assert result.schema_version == "1.0"

    def test_chapter_has_content_title_and_lines(self, basic_novel):
        """Each chapter has non-empty content, a meaningful title,
        and valid line ranges."""
        artifact = PreprocessArtifact(
            schema_version="1.0",
            original_path="test.txt",
            cleaned_text=basic_novel,
            total_chars=len(basic_novel),
        )
        result = split_chapters(artifact)
        for ch in result.chapters:
            assert ch.content, f"Chapter {ch.chapter_id} has empty content"
            assert ch.title, f"Chapter {ch.chapter_id} has empty title"
            assert ch.start_line >= 0
            assert ch.end_line > ch.start_line
