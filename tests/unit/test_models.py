"""Tests for Pydantic data models (src/engine/models.py)."""

import pytest
from pydantic import ValidationError
from typing import Literal

from src.engine.models import (
    PreprocessArtifact,
    Chapter,
    ChapterArtifact,
    Scene,
    SceneArtifact,
    CharacterRef,
    CharacterArtifact,
    DialogueLine,
    DialogueArtifact,
    CharacterProfile,
    ScriptLine,
    ScriptScene,
    ScriptOutput,
)


# ---------------------------------------------------------------------------
# PreprocessArtifact
# ---------------------------------------------------------------------------

class TestPreprocessArtifact:
    def test_valid_artifact(self):
        """Construct a valid PreprocessArtifact."""
        art = PreprocessArtifact(
            schema_version="1.0",
            original_path="/path/to/novel.txt",
            cleaned_text="Some cleaned text",
            total_chars=42,
        )
        assert art.schema_version == "1.0"
        assert art.original_path == "/path/to/novel.txt"
        assert art.cleaned_text == "Some cleaned text"
        assert art.total_chars == 42

    def test_invalid_schema_version(self):
        """Passing a schema_version other than '1.0' raises ValidationError."""
        with pytest.raises(ValidationError):
            PreprocessArtifact(
                schema_version="2.0",  # not allowed
                original_path="/path.txt",
                cleaned_text="x",
                total_chars=1,
            )

    def test_missing_fields(self):
        """Omitting required fields raises ValidationError."""
        with pytest.raises(ValidationError):
            PreprocessArtifact()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Chapter
# ---------------------------------------------------------------------------

class TestChapter:
    def test_default_confidence(self):
        """Chapter should default confidence to 1.0."""
        ch = Chapter(
            chapter_id="CH01",
            title="第一章",
            content="Some content",
            start_line=1,
            end_line=100,
        )
        assert ch.confidence == 1.0

    def test_low_confidence(self):
        """Explicitly pass a low confidence value."""
        ch = Chapter(
            chapter_id="CH02",
            title="第二章",
            content="More content",
            start_line=101,
            end_line=200,
            confidence=0.6,
        )
        assert ch.confidence == 0.6


# ---------------------------------------------------------------------------
# ChapterArtifact
# ---------------------------------------------------------------------------

class TestChapterArtifact:
    def test_empty_chapters(self):
        """ChapterArtifact can hold an empty list of chapters."""
        art = ChapterArtifact(schema_version="1.0", chapters=[])
        assert art.chapters == []

    def test_with_chapters(self):
        """ChapterArtifact with a list of Chapter instances."""
        ch = Chapter(
            chapter_id="CH01",
            title="第一章",
            content="Content",
            start_line=1,
            end_line=50,
        )
        art = ChapterArtifact(schema_version="1.0", chapters=[ch])
        assert len(art.chapters) == 1
        assert art.chapters[0].chapter_id == "CH01"


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

class TestScene:
    def test_defaults_are_unknown(self):
        """location, int_ext, time_of_day default to 'UNKNOWN'."""
        sc = Scene(
            scene_id="CH01-S01",
            chapter_id="CH01",
            content="Scene content",
            boundary_keywords=["keyword"],
        )
        assert sc.location == "UNKNOWN"
        assert sc.int_ext == "UNKNOWN"
        assert sc.time_of_day == "UNKNOWN"
        assert sc.confidence == 1.0

    def test_classified_scene(self):
        """A fully classified scene with INT/location/time."""
        sc = Scene(
            scene_id="CH01-S02",
            chapter_id="CH01",
            content="Interior scene",
            boundary_keywords=["房间"],
            location="卧室",
            int_ext="INT",
            time_of_day="夜",
            confidence=0.95,
        )
        assert sc.int_ext == "INT"
        assert sc.location == "卧室"
        assert sc.time_of_day == "夜"

    def test_invalid_int_ext(self):
        """Passing an invalid int_ext value raises ValidationError."""
        with pytest.raises(ValidationError):
            Scene(
                scene_id="CH01-S03",
                chapter_id="CH01",
                content="Bad int_ext",
                boundary_keywords=[],
                int_ext="OUTDOOR",  # not in Literal
            )


# ---------------------------------------------------------------------------
# CharacterRef
# ---------------------------------------------------------------------------

class TestCharacterRef:
    def test_minimal_no_aliases(self):
        """CharacterRef with no aliases defaults to empty list."""
        cr = CharacterRef(name="张三", first_appearance="CH01-S01")
        assert cr.name == "张三"
        assert cr.aliases == []

    def test_with_aliases(self):
        """CharacterRef with explicit aliases."""
        cr = CharacterRef(
            name="张三",
            aliases=["张", "三哥"],
            first_appearance="CH01-S01",
        )
        assert len(cr.aliases) == 2
        assert "三哥" in cr.aliases


# ---------------------------------------------------------------------------
# CharacterArtifact
# ---------------------------------------------------------------------------

class TestCharacterArtifact:
    def test_empty_characters(self):
        """CharacterArtifact can hold an empty list."""
        art = CharacterArtifact(schema_version="1.0", characters=[])
        assert art.characters == []

    def test_with_characters(self):
        """CharacterArtifact with one CharacterRef."""
        cr = CharacterRef(name="李四", first_appearance="CH01-S02")
        art = CharacterArtifact(schema_version="1.0", characters=[cr])
        assert len(art.characters) == 1


# ---------------------------------------------------------------------------
# DialogueLine
# ---------------------------------------------------------------------------

class TestDialogueLine:
    def test_full_attribution(self):
        """DialogueLine with a known speaker and parenthetical."""
        dl = DialogueLine(
            dialogue_id="D001",
            scene_id="CH01-S01",
            line_index=5,
            speaker="张三",
            line="你好吗？",
            quote_style="double",
            parenthetical="低声",
            confidence=0.95,
            attribution_method="prefix_match",
        )
        assert dl.speaker == "张三"
        assert dl.parenthetical == "低声"
        assert dl.confidence == 0.95

    def test_unattributed(self):
        """DialogueLine with speaker=None (unattributed line)."""
        dl = DialogueLine(
            dialogue_id="D002",
            scene_id="CH01-S01",
            line_index=6,
            speaker=None,
            line="...",
            quote_style="double",
            confidence=0.5,
            attribution_method="unattributed",
        )
        assert dl.speaker is None

    def test_parenthetical_extraction(self):
        """Parenthetical is extracted and stored separately from the line."""
        dl = DialogueLine(
            dialogue_id="D003",
            scene_id="CH01-S02",
            line_index=10,
            speaker="李四",
            line="我走了",
            quote_style="double",
            parenthetical="站起身",
            confidence=0.9,
            attribution_method="prefix_match",
        )
        assert dl.line == "我走了"
        assert dl.parenthetical == "站起身"


# ---------------------------------------------------------------------------
# DialogueArtifact
# ---------------------------------------------------------------------------

class TestDialogueArtifact:
    def test_empty_dialogues(self):
        """DialogueArtifact can hold an empty list."""
        art = DialogueArtifact(schema_version="1.0", dialogues=[])
        assert art.dialogues == []

    def test_with_dialogues(self):
        """DialogueArtifact with one DialogueLine."""
        dl = DialogueLine(
            dialogue_id="D001",
            scene_id="CH01-S01",
            line_index=1,
            line="Hello",
            quote_style="double",
            confidence=0.9,
            attribution_method="prefix_match",
        )
        art = DialogueArtifact(schema_version="1.0", dialogues=[dl])
        assert len(art.dialogues) == 1


# ---------------------------------------------------------------------------
# CharacterProfile
# ---------------------------------------------------------------------------

class TestCharacterProfile:
    def test_minimal_profile(self):
        """Minimal CharacterProfile with role=None and description=None."""
        cp = CharacterProfile(
            name="张三",
            first_appearance="CH01-S01",
            appearance_count=5,
            dialogue_count=10,
            scenes=["CH01-S01", "CH01-S02"],
        )
        assert cp.role is None
        assert cp.description is None
        assert cp.name == "张三"

    def test_with_role(self):
        """CharacterProfile with a role set."""
        cp = CharacterProfile(
            name="李四",
            role="男二号",
            description="A side character",
            first_appearance="CH01-S02",
            appearance_count=3,
            dialogue_count=7,
            scenes=["CH01-S02"],
        )
        assert cp.role == "男二号"
        assert cp.description == "A side character"


# ---------------------------------------------------------------------------
# ScriptLine
# ---------------------------------------------------------------------------

class TestScriptLine:
    def test_action_line_no_character(self):
        """Action lines have character=None."""
        sl = ScriptLine(type="action", content="风吹过窗帘。")
        assert sl.character is None
        assert sl.parenthetical is None
        assert sl.confidence == 1.0

    def test_dialogue_line_with_character(self):
        """Dialogue line with character and parenthetical."""
        sl = ScriptLine(
            type="dialogue",
            content="我不会放弃的！",
            character="张三",
            parenthetical="坚定地",
            confidence=0.95,
        )
        assert sl.character == "张三"
        assert sl.parenthetical == "坚定地"


# ---------------------------------------------------------------------------
# ScriptScene
# ---------------------------------------------------------------------------

class TestScriptScene:
    def test_heading_is_not_stored(self):
        """ScriptScene does NOT have a 'heading' field; heading is assembled
        from decomposed fields at export time."""
        sc = ScriptScene(
            scene_id="CH01-S01",
            chapter_id="CH01",
            int_ext="INT",
            location="卧室",
            time_of_day="夜",
            lines=[],
            characters_in_scene=["张三"],
        )
        # No 'heading' attribute should exist
        with pytest.raises(AttributeError):
            _ = sc.heading  # type: ignore[attr-defined]

    def test_heading_format(self):
        """Verify the heading format convention: 'INT. 卧室 - 夜'."""
        sc = ScriptScene(
            scene_id="CH01-S01",
            chapter_id="CH01",
            int_ext="INT",
            location="卧室",
            time_of_day="夜",
            lines=[],
            characters_in_scene=["张三"],
        )
        expected = f"{sc.int_ext}. {sc.location} - {sc.time_of_day}"
        assert expected == "INT. 卧室 - 夜"


# ---------------------------------------------------------------------------
# ScriptOutput
# ---------------------------------------------------------------------------

class TestScriptOutput:
    def test_minimal_valid(self):
        """Minimal valid ScriptOutput with empty characters/scenes lists."""
        out = ScriptOutput(
            schema_version="1.0",
            title="Test Script",
            source_novel="Test Novel",
            characters=[],
            scenes=[],
        )
        assert out.title == "Test Script"
        assert out.characters == []
        assert out.scenes == []

    def test_full_output(self):
        """ScriptOutput with characters and scenes populated."""
        cp = CharacterProfile(
            name="张三",
            first_appearance="CH01-S01",
            appearance_count=5,
            dialogue_count=10,
            scenes=["CH01-S01"],
        )
        sc = ScriptScene(
            scene_id="CH01-S01",
            chapter_id="CH01",
            int_ext="INT",
            location="卧室",
            time_of_day="夜",
            lines=[
                ScriptLine(
                    type="action",
                    content="灯亮了。",
                ),
                ScriptLine(
                    type="dialogue",
                    content="你来了。",
                    character="张三",
                ),
            ],
            characters_in_scene=["张三"],
        )
        out = ScriptOutput(
            schema_version="1.0",
            title="Test Script",
            source_novel="Test Novel",
            characters=[cp],
            scenes=[sc],
        )
        assert len(out.characters) == 1
        assert len(out.scenes) == 1
        assert len(out.scenes[0].lines) == 2
        assert out.scenes[0].lines[0].type == "action"
        assert out.scenes[0].lines[1].character == "张三"
