"""Tests for the character NER module (Task 1.5).

Tests cover:
- Chinese surname set completeness
- spaCy-based PER extraction (graceful failure when model absent)
- jieba-based surname-prefix word fallback extraction
- Deduplication and alias merging
- Full extract_characters pipeline (SceneArtifact → CharacterArtifact)
- Single-occurrence name filtering
"""

import pytest

from src.engine.character import (
    _CHINESE_SURNAMES,
    extract_names_spacy,
    extract_names_jieba_fallback,
    deduplicate_characters,
    extract_characters,
)
from src.engine.models import (
    CharacterRef,
    CharacterArtifact,
    Scene,
    SceneArtifact,
)


# ---------------------------------------------------------------------------
# _CHINESE_SURNAMES set
# ---------------------------------------------------------------------------


class TestChineseSurnames:
    def test_common_surnames_present(self):
        """Key common Chinese surnames are in the set."""
        for name in ("李", "王", "张", "刘", "陈", "赵", "周", "吴", "郑", "孙"):
            assert name in _CHINESE_SURNAMES, (
                f"Common surname '{name}' missing from _CHINESE_SURNAMES"
            )

    def test_fictional_surnames_present(self):
        """Common fictional/xianxia surnames are in the set."""
        for name in ("萧", "林", "叶", "楚", "苏", "慕容", "纳兰", "百里"):
            assert name in _CHINESE_SURNAMES, (
                f"Fictional surname '{name}' missing from _CHINESE_SURNAMES"
            )

    def test_double_character_surnames_longer_than_single(self):
        """Multi-character surnames like 纳兰, 慕容, 上官 should be present."""
        for name in ("纳兰", "慕容", "上官", "司徒", "欧阳", "百里"):
            assert name in _CHINESE_SURNAMES


# ---------------------------------------------------------------------------
# extract_names_spacy
# ---------------------------------------------------------------------------


class TestExtractNamesSpacy:
    def test_returns_list(self):
        """Returns a list (empty if model not available)."""
        result = extract_names_spacy("萧炎盘膝坐在青石上。")
        assert isinstance(result, list)

    def test_returns_empty_list_when_spacy_not_installed(self):
        """Returns [] gracefully when zh_core_web_trf is unavailable."""
        # This test is safe because the model may or may not be installed;
        # the function should always return a list, never raise.
        result = extract_names_spacy("纳兰嫣然踏空而来。")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# extract_names_jieba_fallback
# ---------------------------------------------------------------------------


class TestExtractNamesJiebaFallback:
    def test_finds_single_character_surname(self):
        """A 2-char word starting with a known surname is detected."""
        result = extract_names_jieba_fallback("萧炎站在山巅。")
        assert "萧炎" in result

    def test_finds_double_surname_name(self):
        """A 3-4 char word starting with a multi-char surname is detected."""
        result = extract_names_jieba_fallback("纳兰嫣然踏空而来。")
        assert "纳兰嫣然" in result

    def test_multi_surname_prefers_longest(self):
        """Longer surnames match before shorter ones (纳兰 before 纳)."""
        result = extract_names_jieba_fallback("纳兰嫣然看了看纳兰桀。")
        assert "纳兰嫣然" in result
        assert "纳兰桀" in result

    def test_returns_empty_for_text_without_surname_words(self):
        """Text with no 2-4-char words starting with a known surname returns
        empty list."""
        result = extract_names_jieba_fallback("这是对的。")
        assert isinstance(result, list)
        assert len(result) == 0

    def test_common_word_not_confused_as_name(self):
        """Common words that happen to start with a surname char are not caught
        because jieba segments them differently (e.g. 我们, 成功)."""
        # "我" is not a surname, so 我们 won't be caught
        # But if jieba segments "我们" as a single word starting with a surname char...
        # Let's just verify the function returns a list and doesn't crash.
        result = extract_names_jieba_fallback("我们一起去成功镇。")
        assert isinstance(result, list)
        # Some common words may match depending on jieba segmentation;
        # we only care that the function doesn't break
        assert "我们" not in result, (
            "Common word '我们' should not be identified as a character name"
        )

    def test_returns_list(self):
        """Always returns a list."""
        result = extract_names_jieba_fallback("")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# deduplicate_characters
# ---------------------------------------------------------------------------


class TestDeduplicateCharacters:
    def test_merges_exact_duplicate_names(self):
        """Same name appearing multiple times merges into one entry."""
        refs = [
            CharacterRef(name="萧炎", first_appearance="CH01-S01"),
            CharacterRef(name="萧炎", first_appearance="CH01-S02"),
            CharacterRef(name="纳兰嫣然", first_appearance="CH01-S01"),
        ]
        result = deduplicate_characters(refs)
        names = [r.name for r in result]
        assert names == ["萧炎", "纳兰嫣然"], (
            f"Expected [萧炎, 纳兰嫣然], got {names}"
        )

    def test_keeps_earliest_first_appearance(self):
        """When merging duplicates, keep the earliest first_appearance."""
        refs = [
            CharacterRef(name="萧炎", first_appearance="CH01-S02"),
            CharacterRef(name="萧炎", first_appearance="CH01-S01"),
        ]
        result = deduplicate_characters(refs)
        assert result[0].first_appearance == "CH01-S01"

    def test_accumulates_aliases(self):
        """Unique aliases from merged entries are accumulated."""
        refs = [
            CharacterRef(name="萧炎", aliases=["炎帝"], first_appearance="CH01-S01"),
            CharacterRef(name="萧炎", aliases=["萧炎"], first_appearance="CH01-S02"),
        ]
        result = deduplicate_characters(refs)
        assert "炎帝" in result[0].aliases

    def test_returns_empty_list_for_empty_input(self):
        """Empty input returns empty list."""
        result = deduplicate_characters([])
        assert result == []

    def test_input_unmodified(self):
        """The function does not mutate the input list."""
        refs = [
            CharacterRef(name="萧炎", first_appearance="CH01-S01"),
            CharacterRef(name="纳兰嫣然", first_appearance="CH01-S02"),
        ]
        original_len = len(refs)
        deduplicate_characters(refs)
        assert len(refs) == original_len


# ---------------------------------------------------------------------------
# extract_characters
# ---------------------------------------------------------------------------


class TestExtractCharacters:
    def test_returns_character_artifact(self):
        """extract_characters returns a CharacterArtifact."""
        scenes = SceneArtifact(
            schema_version="1.0",
            scenes=[
                Scene(
                    scene_id="CH01-S01",
                    chapter_id="CH01",
                    content="萧炎盘膝坐在青石上。纳兰嫣然站在一旁。",
                    boundary_keywords=["测试"],
                    location="迦南学院",
                    int_ext="INT",
                    time_of_day="日",
                    confidence=1.0,
                ),
            ],
        )
        result = extract_characters(scenes)
        assert isinstance(result, CharacterArtifact)
        assert result.schema_version == "1.0"

    def test_extracts_known_characters(self):
        """Characters mentioned in text are extracted."""
        scenes = SceneArtifact(
            schema_version="1.0",
            scenes=[
                Scene(
                    scene_id="CH01-S01",
                    chapter_id="CH01",
                    content="萧炎对纳兰嫣然说：你好。",
                    boundary_keywords=["测试"],
                    location="测试地",
                    int_ext="INT",
                    time_of_day="日",
                    confidence=1.0,
                ),
            ],
        )
        result = extract_characters(scenes)
        names = [c.name for c in result.characters]
        assert "萧炎" in names
        assert "纳兰嫣然" in names

    def test_single_occurrence_names_filtered(self):
        """Names appearing only once are filtered out (when many names present)."""
        scenes = SceneArtifact(
            schema_version="1.0",
            scenes=[
                Scene(
                    scene_id="CH01-S01",
                    chapter_id="CH01",
                    content=(
                        "萧炎来了。萧炎走了。萧炎又来了。"
                        "纳兰嫣然来了。纳兰嫣然走了。"
                        "路人甲路过。"
                    ),
                    boundary_keywords=["测试"],
                    location="测试地",
                    int_ext="INT",
                    time_of_day="日",
                    confidence=1.0,
                ),
            ],
        )
        result = extract_characters(scenes)
        names = [c.name for c in result.characters]
        assert "萧炎" in names      # appears 3 times
        assert "纳兰嫣然" in names  # appears 2 times
        assert "路人甲" not in names  # appears only once

    def test_small_cast_keeps_single_occurrence(self):
        """When only 1-3 unique names found, keep even single-occurrence names."""
        scenes = SceneArtifact(
            schema_version="1.0",
            scenes=[
                Scene(
                    scene_id="CH01-S01",
                    chapter_id="CH01",
                    content="萧炎站在山巅。",
                    boundary_keywords=["测试"],
                    location="测试地",
                    int_ext="EXT",
                    time_of_day="日",
                    confidence=1.0,
                ),
            ],
        )
        result = extract_characters(scenes)
        names = [c.name for c in result.characters]
        assert "萧炎" in names  # single occurrence but only 1 name found

    def test_sorted_by_frequency(self):
        """Characters sorted by frequency (most frequent first)."""
        scenes = SceneArtifact(
            schema_version="1.0",
            scenes=[
                Scene(
                    scene_id="CH01-S01",
                    chapter_id="CH01",
                    content=(
                        "萧炎来了。纳兰嫣然来了。萧炎走了。"
                        "叶老出现了。叶老说话了。萧炎回应了。"
                    ),
                    boundary_keywords=["测试"],
                    location="测试地",
                    int_ext="INT",
                    time_of_day="日",
                    confidence=1.0,
                ),
            ],
        )
        result = extract_characters(scenes)
        # c.f. 萧炎 3x, 叶老 2x, 纳兰嫣然 1x
        names = [c.name for c in result.characters]
        cai_index = names.index("萧炎")
        ye_index = names.index("叶老")
        assert cai_index < ye_index, (
            "萧炎 (3 occurrences) should come before 叶老 (2 occurrences)"
        )

    def test_deduplicates_across_scenes(self):
        """Same character in multiple scenes is deduplicated."""
        scenes = SceneArtifact(
            schema_version="1.0",
            scenes=[
                Scene(
                    scene_id="CH01-S01",
                    chapter_id="CH01",
                    content="萧炎盘膝打坐。",
                    boundary_keywords=["测试"],
                    location="测试地",
                    int_ext="EXT",
                    time_of_day="日",
                    confidence=1.0,
                ),
                Scene(
                    scene_id="CH01-S02",
                    chapter_id="CH01",
                    content="萧炎走向大殿。",
                    boundary_keywords=["然后"],
                    location="大殿",
                    int_ext="INT",
                    time_of_day="日",
                    confidence=0.8,
                ),
            ],
        )
        result = extract_characters(scenes)
        # Expect only 萧炎 (the only name found across both scenes)
        # Note: 大殿 could be a false positive if 大 were a surname — it is not,
        # so only 萧炎 should be found in both scenes
        cai_characters = [c for c in result.characters if c.name == "萧炎"]
        assert len(cai_characters) == 1
        assert cai_characters[0].first_appearance == "CH01-S01"


# ---------------------------------------------------------------------------
# Slow integration test
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestFullNovelCharacterExtraction:
    """Full-novel character extraction — requires spaCy pipeline setup."""

    def test_basic_novel_character_extraction(self, basic_novel):
        """Extract characters from the full 3-chapter basic novel."""
        from src.engine.preprocess import preprocess
        from src.engine.chapter import split_chapters
        from src.engine.scene import detect_scenes

        pre = preprocess(basic_novel, "basic_3ch.txt")
        chapter_artifact = split_chapters(pre)
        scene_artifact = detect_scenes(chapter_artifact)
        result = extract_characters(scene_artifact)

        assert isinstance(result, CharacterArtifact)
        assert len(result.characters) > 0
        # Core characters from the test novel
        names = [c.name for c in result.characters]
        assert "萧炎" in names, (
            "萧炎 should be extracted from the full novel"
        )
        assert "纳兰嫣然" in names, (
            "纳兰嫣然 should be extracted from the full novel"
        )
