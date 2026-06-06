"""Edge-case unit tests for the pipeline converter.

Tests cover _assemble branches, _build_scene_lines corner cases, and
resume/cache helpers that are not exercised by test_converter.py.
"""

import pytest

from engine.converter import Pipeline
from engine.models import (
    ChapterArtifact,
    CharacterArtifact,
    CharacterRef,
    DialogueArtifact,
    DialogueLine,
    PreprocessArtifact,
    Scene,
    SceneArtifact,
)


class TestAssembleTitleEdge:
    def test_title_skips_chapter_marker(self):
        pre = PreprocessArtifact(
            schema_version="1.0", original_path="test.txt",
            cleaned_text="第一章 陨落的天才\n斗破苍穹\n...",
            total_chars=30,
        )
        ch = ChapterArtifact(schema_version="1.0", chapters=[])
        sc = SceneArtifact(schema_version="1.0", scenes=[])
        ca = CharacterArtifact(schema_version="1.0", characters=[])
        da = DialogueArtifact(schema_version="1.0", dialogues=[])
        result = Pipeline()._assemble(pre, ch, sc, ca, da)
        assert result.title == "斗破苍穹"

    def test_title_all_markers_fallback(self):
        pre = PreprocessArtifact(
            schema_version="1.0", original_path="/p/novel_file.txt",
            cleaned_text="第一章\n第二章\n第三章",
            total_chars=30,
        )
        ch = ChapterArtifact(schema_version="1.0", chapters=[])
        sc = SceneArtifact(schema_version="1.0", scenes=[])
        ca = CharacterArtifact(schema_version="1.0", characters=[])
        da = DialogueArtifact(schema_version="1.0", dialogues=[])
        result = Pipeline()._assemble(pre, ch, sc, ca, da)
        assert result.title == "novel_file"


class TestAssembleEmptyInputs:
    def test_empty_characters(self):
        pre = PreprocessArtifact(
            schema_version="1.0", original_path="t.txt",
            cleaned_text="T\nC", total_chars=4,
        )
        ch = ChapterArtifact(schema_version="1.0", chapters=[])
        sc = SceneArtifact(schema_version="1.0", scenes=[
            Scene(scene_id="CH01-S01", chapter_id="CH01", content="x",
                  boundary_keywords=[], location="L", int_ext="INT",
                  time_of_day="日", confidence=1.0),
        ])
        ca = CharacterArtifact(schema_version="1.0", characters=[])
        da = DialogueArtifact(schema_version="1.0", dialogues=[])
        result = Pipeline()._assemble(pre, ch, sc, ca, da)
        assert result.characters == []

    def test_empty_scenes(self):
        pre = PreprocessArtifact(
            schema_version="1.0", original_path="t.txt",
            cleaned_text="T\nC", total_chars=4,
        )
        ch = ChapterArtifact(schema_version="1.0", chapters=[])
        sc = SceneArtifact(schema_version="1.0", scenes=[])
        ca = CharacterArtifact(schema_version="1.0", characters=[])
        da = DialogueArtifact(schema_version="1.0", dialogues=[])
        result = Pipeline()._assemble(pre, ch, sc, ca, da)
        assert result.scenes == []

    def test_scene_empty_content(self):
        pre = PreprocessArtifact(
            schema_version="1.0", original_path="t.txt",
            cleaned_text="T", total_chars=1,
        )
        ch = ChapterArtifact(schema_version="1.0", chapters=[])
        sc = SceneArtifact(schema_version="1.0", scenes=[
            Scene(scene_id="CH01-S01", chapter_id="CH01", content="",
                  boundary_keywords=[], location="UNKNOWN",
                  int_ext="UNKNOWN", time_of_day="UNKNOWN", confidence=1.0),
        ])
        ca = CharacterArtifact(schema_version="1.0", characters=[])
        da = DialogueArtifact(schema_version="1.0", dialogues=[])
        result = Pipeline()._assemble(pre, ch, sc, ca, da)
        assert len(result.scenes) == 1
        assert result.scenes[0].lines == []


class TestAssembleConfidenceEdge:
    def test_all_below_threshold(self):
        pre = PreprocessArtifact(
            schema_version="1.0", original_path="t.txt",
            cleaned_text="T\nC", total_chars=4,
        )
        ch = ChapterArtifact(schema_version="1.0", chapters=[])
        sc = SceneArtifact(schema_version="1.0", scenes=[
            Scene(scene_id="CH01-S01", chapter_id="CH01",
                  content='\"line1\" \"line2\"',
                  boundary_keywords=[], location="L", int_ext="INT",
                  time_of_day="日", confidence=1.0),
        ])
        ca = CharacterArtifact(schema_version="1.0", characters=[
            CharacterRef(name="萧炎", first_appearance="CH01-S01"),
        ])
        da = DialogueArtifact(schema_version="1.0", dialogues=[
            DialogueLine(dialogue_id="D01", scene_id="CH01-S01", line_index=0,
                         speaker="萧炎", line="line1", quote_style="double",
                         confidence=0.3, attribution_method="prev_speaker"),
            DialogueLine(dialogue_id="D02", scene_id="CH01-S01", line_index=1,
                         speaker="萧炎", line="line2", quote_style="double",
                         confidence=0.2, attribution_method="unattributed"),
        ])
        result = Pipeline()._assemble(
            pre, ch, sc, ca, da, confidence_threshold=0.5,
        )
        dl = [l for l in result.scenes[0].lines if l.type == "dialogue"]
        assert len(dl) == 2
        assert all(d.character is None for d in dl)


class TestBuildSceneLines:
    def test_quote_not_found_emitted(self):
        lines = Pipeline._build_scene_lines(
            "无关文本。",
            [DialogueLine(dialogue_id="D01", scene_id="S01", line_index=0,
                          speaker="萧炎", line="你好", quote_style="double",
                          confidence=0.85, attribution_method="prefix_match")],
            0.0,
        )
        dial = [l for l in lines if l.type == "dialogue"]
        assert len(dial) == 1
        assert dial[0].content == "你好"

    def test_quote_not_found_threshold(self):
        lines = Pipeline._build_scene_lines(
            "无关。",
            [DialogueLine(dialogue_id="D01", scene_id="S01", line_index=0,
                          speaker="路人", line="t", quote_style="double",
                          confidence=0.2, attribution_method="unattributed")],
            0.5,
        )
        dial = [l for l in lines if l.type == "dialogue"]
        assert dial[0].character is None

    def test_duplicate_quote(self):
        lines = Pipeline._build_scene_lines(
            '"你好" 中 "你好"',
            [
                DialogueLine(dialogue_id="D01", scene_id="S01", line_index=0,
                             speaker="A", line="你好", quote_style="double",
                             confidence=0.85, attribution_method="prefix_match"),
                DialogueLine(dialogue_id="D02", scene_id="S01", line_index=1,
                             speaker="B", line="你好", quote_style="double",
                             confidence=0.85, attribution_method="prefix_match"),
            ],
            0.0,
        )
        dial = [l for l in lines if l.type == "dialogue"]
        assert len(dial) == 2

    def test_empty_content_no_dial(self):
        assert Pipeline._build_scene_lines("", [], 0.0) == []

    def test_empty_content_with_dial(self):
        lines = Pipeline._build_scene_lines(
            "",
            [DialogueLine(dialogue_id="D01", scene_id="S01", line_index=0,
                          speaker="A", line="hi", quote_style="double",
                          confidence=0.85, attribution_method="prefix_match")],
            0.0,
        )
        assert len(lines) == 1


class TestComputeResumeSet:
    def test_none(self):
        assert Pipeline._compute_resume_set(None) == set()

    def test_preprocess(self):
        assert Pipeline._compute_resume_set("preprocess") == {
            "preprocess", "chapter", "scene", "character", "dialogue"
        }

    def test_chapter(self):
        assert Pipeline._compute_resume_set("chapter") == {
            "chapter", "scene", "character", "dialogue"
        }

    def test_scene(self):
        assert Pipeline._compute_resume_set("scene") == {
            "scene", "character", "dialogue"
        }

    def test_character(self):
        assert Pipeline._compute_resume_set("character") == {"character"}

    def test_dialogue(self):
        assert Pipeline._compute_resume_set("dialogue") == {"dialogue"}

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            Pipeline._compute_resume_set("bad")


class TestDeserializeCache:
    def test_valid(self):
        r = Pipeline._deserialize_cache("preprocess", {
            "schema_version": "1.0", "original_path": "t.txt",
            "cleaned_text": "hi", "total_chars": 2,
        })
        assert r is not None

    def test_invalid_stage(self):
        assert Pipeline._deserialize_cache("bad", {}) is None

    def test_invalid_data(self):
        assert Pipeline._deserialize_cache("preprocess", {"schema_version": "99.0"}) is None

    def test_chapter(self):
        assert Pipeline._deserialize_cache("chapter", {
            "schema_version": "1.0", "chapters": []
        }) is not None

    def test_dialogue(self):
        assert Pipeline._deserialize_cache("dialogue", {
            "schema_version": "1.0", "dialogues": []
        }) is not None
