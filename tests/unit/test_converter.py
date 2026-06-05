"""Tests for the pipeline converter module (Task 1.7).

Tests focus on ``Pipeline._assemble()`` (testable without file I/O) and
the cache read/write logic within ``Pipeline.run()``.
"""

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from engine.converter import Pipeline
from engine.models import (
    ChapterArtifact,
    CharacterArtifact,
    CharacterProfile,
    CharacterRef,
    DialogueArtifact,
    DialogueLine,
    PreprocessArtifact,
    Scene,
    SceneArtifact,
    ScriptOutput,
    ScriptScene,
    ScriptLine,
)

# ===================================================================
# _assemble tests  --  no file I/O required
# ===================================================================


class TestAssembleTitle:
    """Pipeline._assemble extracts the correct title."""

    @staticmethod
    def _empty_artifacts():
        """Minimal empty artifacts for tests that don't exercise scenes/chars."""
        return (
            ChapterArtifact(schema_version="1.0", chapters=[]),
            SceneArtifact(schema_version="1.0", scenes=[]),
            CharacterArtifact(schema_version="1.0", characters=[]),
            DialogueArtifact(schema_version="1.0", dialogues=[]),
        )

    def test_title_from_first_non_empty_line(self):
        """Title is the first non-empty line of cleaned_text."""
        pre = PreprocessArtifact(
            schema_version="1.0",
            original_path="test.txt",
            cleaned_text="斗破苍穹\n\n第一章 陨落\n...",
            total_chars=20,
        )
        ch, sc, ca, da = self._empty_artifacts()
        result = Pipeline()._assemble(pre, ch, sc, ca, da)
        assert result.title == "斗破苍穹"

    def test_title_ignores_leading_blank_lines(self):
        """Leading blank lines in cleaned_text are ignored."""
        pre = PreprocessArtifact(
            schema_version="1.0",
            original_path="test.txt",
            cleaned_text="\n\n\n真正的标题\n内容",
            total_chars=20,
        )
        ch, sc, ca, da = self._empty_artifacts()
        result = Pipeline()._assemble(pre, ch, sc, ca, da)
        assert result.title == "真正的标题"

    def test_title_is_empty_when_cleaned_text_empty(self):
        """Empty cleaned_text yields an empty title."""
        pre = PreprocessArtifact(
            schema_version="1.0",
            original_path="test.txt",
            cleaned_text="",
            total_chars=0,
        )
        ch, sc, ca, da = self._empty_artifacts()
        result = Pipeline()._assemble(pre, ch, sc, ca, da)
        assert result.title == ""


class TestAssembleSourceNovel:
    """source_novel maps from original_path."""

    def test_source_novel_from_original_path(self):
        """source_novel matches preprocessed.original_path."""
        pre = PreprocessArtifact(
            schema_version="1.0",
            original_path="E:/novels/my_story.txt",
            cleaned_text="Title\nContent",
            total_chars=15,
        )
        chapters = ChapterArtifact(schema_version="1.0", chapters=[])
        scenes = SceneArtifact(schema_version="1.0", scenes=[])
        characters = CharacterArtifact(schema_version="1.0", characters=[])
        dialogues = DialogueArtifact(schema_version="1.0", dialogues=[])

        result = Pipeline()._assemble(pre, chapters, scenes, characters, dialogues)
        assert result.source_novel == "E:/novels/my_story.txt"


class TestAssembleCharacterProfile:
    """CharacterProfile aggregation from CharacterArtifact + DialogueArtifact."""

    def _make_artifacts(
        self,
        characters: list | None = None,
        dialogues: list | None = None,
        scene_count: int = 3,
    ):
        """Helper that builds all five artifacts from raw lists."""
        scenes = SceneArtifact(
            schema_version="1.0",
            scenes=[
                Scene(
                    scene_id=f"CH01-S{i + 1:02d}",
                    chapter_id="CH01",
                    content=f"Scene {i + 1} content.",
                    boundary_keywords=[],
                    location="迦南学院",
                    int_ext="INT",
                    time_of_day="日",
                    confidence=1.0,
                )
                for i in range(scene_count)
            ],
        )
        pre = PreprocessArtifact(
            schema_version="1.0",
            original_path="test.txt",
            cleaned_text="斗破苍穹\ncontent",
            total_chars=15,
        )
        chapters = ChapterArtifact(schema_version="1.0", chapters=[])
        char_art = CharacterArtifact(
            schema_version="1.0", characters=characters or []
        )
        dial_art = DialogueArtifact(
            schema_version="1.0", dialogues=dialogues or []
        )
        return pre, chapters, scenes, char_art, dial_art

    def test_single_character_appearance_and_dialogue_counts(self):
        """appearance_count counts scenes from first_appearance onward."""
        pre, ch, sc, ca, da = self._make_artifacts(
            characters=[CharacterRef(name="萧炎", first_appearance="CH01-S01")],
            scene_count=3,
        )
        result = Pipeline()._assemble(pre, ch, sc, ca, da)

        assert len(result.characters) == 1
        char = result.characters[0]
        assert char.name == "萧炎"
        assert char.appearance_count == 3  # S01, S02, S03
        assert char.dialogue_count == 0
        assert char.scenes == ["CH01-S01", "CH01-S02", "CH01-S03"]

    def test_multiple_characters_with_different_first_appearance(self):
        """Each character's appearance is counted from its first_appearance."""
        pre, ch, sc, ca, da = self._make_artifacts(
            characters=[
                CharacterRef(name="萧炎", first_appearance="CH01-S01"),
                CharacterRef(name="纳兰嫣然", first_appearance="CH01-S02"),
            ],
            dialogues=[
                DialogueLine(
                    dialogue_id="D01",
                    scene_id="CH01-S01",
                    line_index=0,
                    speaker="萧炎",
                    line="你好",
                    quote_style="double",
                    confidence=0.85,
                    attribution_method="prefix_match",
                ),
            ],
            scene_count=3,
        )
        result = Pipeline()._assemble(pre, ch, sc, ca, da)

        xiao = next(c for c in result.characters if c.name == "萧炎")
        assert xiao.appearance_count == 3
        assert xiao.dialogue_count == 1
        assert xiao.scenes == ["CH01-S01", "CH01-S02", "CH01-S03"]

        nalan = next(c for c in result.characters if c.name == "纳兰嫣然")
        assert nalan.appearance_count == 2
        assert nalan.dialogue_count == 0
        assert nalan.scenes == ["CH01-S02", "CH01-S03"]

    def test_aliases_preserved_in_profile(self):
        """CharacterProfile includes aliases from CharacterRef."""
        pre = PreprocessArtifact(
            schema_version="1.0",
            original_path="test.txt",
            cleaned_text="Title\nContent",
            total_chars=10,
        )
        chapters = ChapterArtifact(schema_version="1.0", chapters=[])
        scenes = SceneArtifact(
            schema_version="1.0",
            scenes=[
                Scene(
                    scene_id="CH01-S01",
                    chapter_id="CH01",
                    content="Test",
                    boundary_keywords=[],
                    location="Loc",
                    int_ext="INT",
                    time_of_day="日",
                    confidence=1.0,
                ),
            ],
        )
        characters = CharacterArtifact(
            schema_version="1.0",
            characters=[
                CharacterRef(
                    name="萧炎",
                    aliases=["炎帝", "萧炎"],
                    first_appearance="CH01-S01",
                ),
            ],
        )
        dialogues = DialogueArtifact(schema_version="1.0", dialogues=[])

        result = Pipeline()._assemble(pre, chapters, scenes, characters, dialogues)
        assert "炎帝" in result.characters[0].aliases

    def test_character_ordered_by_appearance(self):
        """Characters are ordered by first_appearance (ascending)."""
        pre, ch, sc, ca, da = self._make_artifacts(
            characters=[
                CharacterRef(name="萧炎", first_appearance="CH01-S02"),
                CharacterRef(name="纳兰嫣然", first_appearance="CH01-S01"),
            ],
            scene_count=2,
        )
        result = Pipeline()._assemble(pre, ch, sc, ca, da)
        assert result.characters[0].name == "纳兰嫣然"  # appears first
        assert result.characters[1].name == "萧炎"


class TestAssembleScriptScene:
    """ScriptScene assembly from Scene models."""

    def _minimal_artifacts(self, scenes: list, characters: list | None = None, dialogues: list | None = None):
        pre = PreprocessArtifact(
            schema_version="1.0",
            original_path="test.txt",
            cleaned_text="Title\nContent",
            total_chars=10,
        )
        chapters = ChapterArtifact(schema_version="1.0", chapters=[])
        sc = SceneArtifact(schema_version="1.0", scenes=scenes)
        ca = CharacterArtifact(schema_version="1.0", characters=characters or [])
        da = DialogueArtifact(schema_version="1.0", dialogues=dialogues or [])
        return pre, chapters, sc, ca, da

    def test_scene_fields_map_correctly(self):
        """All Scene fields map to ScriptScene fields."""
        pre, ch, sc, ca, da = self._minimal_artifacts([
            Scene(
                scene_id="CH01-S01",
                chapter_id="CH01",
                content="Test content.",
                boundary_keywords=[],
                location="迦南学院",
                int_ext="INT",
                time_of_day="日",
                confidence=1.0,
            ),
            Scene(
                scene_id="CH01-S02",
                chapter_id="CH01",
                content="More content.",
                boundary_keywords=["三天后"],
                location="山巅",
                int_ext="EXT",
                time_of_day="夜",
                confidence=0.7,
            ),
        ])
        result = Pipeline()._assemble(pre, ch, sc, ca, da)

        assert len(result.scenes) == 2

        s1 = result.scenes[0]
        assert s1.scene_id == "CH01-S01"
        assert s1.chapter_id == "CH01"
        assert s1.int_ext == "INT"
        assert s1.location == "迦南学院"
        assert s1.time_of_day == "日"
        assert s1.location_note is None

        s2 = result.scenes[1]
        assert s2.scene_id == "CH01-S02"
        assert s2.chapter_id == "CH01"
        assert s2.int_ext == "EXT"
        assert s2.location == "山巅"
        assert s2.time_of_day == "夜"

    def test_characters_in_scene_derived_from_first_appearance(self):
        """characters_in_scene includes characters whose first_appearance <= scene_id."""
        pre, ch, sc, ca, da = self._minimal_artifacts(
            scenes=[
                Scene(scene_id="CH01-S01", chapter_id="CH01", content="S1",
                      boundary_keywords=[], location="Loc", int_ext="INT",
                      time_of_day="日", confidence=1.0),
                Scene(scene_id="CH01-S02", chapter_id="CH01", content="S2",
                      boundary_keywords=[], location="Loc", int_ext="INT",
                      time_of_day="日", confidence=1.0),
                Scene(scene_id="CH02-S01", chapter_id="CH02", content="S3",
                      boundary_keywords=[], location="Loc", int_ext="INT",
                      time_of_day="日", confidence=1.0),
            ],
            characters=[
                CharacterRef(name="萧炎", first_appearance="CH01-S01"),
                CharacterRef(name="纳兰嫣然", first_appearance="CH01-S02"),
            ],
        )
        result = Pipeline()._assemble(pre, ch, sc, ca, da)

        assert result.scenes[0].characters_in_scene == ["萧炎"]
        assert result.scenes[1].characters_in_scene == ["萧炎", "纳兰嫣然"]
        assert result.scenes[2].characters_in_scene == ["萧炎", "纳兰嫣然"]

    def test_scene_no_dialogues_has_action_line(self):
        """A scene with no dialogues gets a single action line."""
        pre, ch, sc, ca, da = self._minimal_artifacts([
            Scene(scene_id="CH01-S01", chapter_id="CH01",
                  content="萧炎走在山路上。秋风吹过，落叶纷飞。",
                  boundary_keywords=[], location="山路", int_ext="EXT",
                  time_of_day="日", confidence=1.0),
        ])
        result = Pipeline()._assemble(pre, ch, sc, ca, da)

        lines = result.scenes[0].lines
        assert len(lines) == 1
        assert lines[0].type == "action"
        assert lines[0].content == "萧炎走在山路上。秋风吹过，落叶纷飞。"
        assert lines[0].character is None

    def test_dialogue_and_action_lines_interleaved(self):
        """Dialogues and action lines are properly interleaved."""
        pre, ch, sc, ca, da = self._minimal_artifacts(
            scenes=[
                Scene(
                    scene_id="CH01-S01", chapter_id="CH01",
                    content='萧炎说道："你好。"纳兰嫣然答道："再见。"萧炎转身离开。',
                    boundary_keywords=[], location="大殿", int_ext="INT",
                    time_of_day="日", confidence=1.0,
                ),
            ],
            characters=[
                CharacterRef(name="萧炎", first_appearance="CH01-S01"),
                CharacterRef(name="纳兰嫣然", first_appearance="CH01-S01"),
            ],
            dialogues=[
                DialogueLine(
                    dialogue_id="D01", scene_id="CH01-S01", line_index=0,
                    speaker="萧炎", line="你好。", quote_style="double",
                    confidence=0.85, attribution_method="prefix_match",
                ),
                DialogueLine(
                    dialogue_id="D02", scene_id="CH01-S01", line_index=1,
                    speaker="纳兰嫣然", line="再见。", quote_style="double",
                    confidence=0.75, attribution_method="suffix_match",
                ),
            ],
        )
        result = Pipeline()._assemble(pre, ch, sc, ca, da)

        lines = result.scenes[0].lines
        # Expect: action, dialogue(你好), action, dialogue(再见), action
        assert len(lines) == 5

        # Action before first dialogue
        assert lines[0].type == "action"
        assert "萧炎说道" in lines[0].content

        # First dialogue
        assert lines[1].type == "dialogue"
        assert lines[1].content == "你好。"
        assert lines[1].character == "萧炎"

        # Action between dialogues
        assert lines[2].type == "action"
        assert "纳兰嫣然答道" in lines[2].content

        # Second dialogue
        assert lines[3].type == "dialogue"
        assert lines[3].content == "再见。"
        assert lines[3].character == "纳兰嫣然"

        # Action after last dialogue
        assert lines[4].type == "action"
        assert "萧炎转身离开" in lines[4].content

    def test_dialogue_line_structure_in_script_scene(self):
        """Each ScriptLine from a dialogue has the correct structure."""
        pre, ch, sc, ca, da = self._minimal_artifacts(
            scenes=[
                Scene(
                    scene_id="CH01-S01", chapter_id="CH01",
                    content='萧炎（叹气道）："好吧。"',
                    boundary_keywords=[], location="大殿", int_ext="INT",
                    time_of_day="日", confidence=1.0,
                ),
            ],
            characters=[
                CharacterRef(name="萧炎", first_appearance="CH01-S01"),
            ],
            dialogues=[
                DialogueLine(
                    dialogue_id="D01", scene_id="CH01-S01", line_index=0,
                    speaker="萧炎", line="好吧。", quote_style="double",
                    parenthetical="叹气道",
                    confidence=0.85, attribution_method="prefix_match",
                ),
            ],
        )
        result = Pipeline()._assemble(pre, ch, sc, ca, da)

        dline = [l for l in result.scenes[0].lines if l.type == "dialogue"][0]
        assert dline.content == "好吧。"
        assert dline.character == "萧炎"
        assert dline.parenthetical == "叹气道"
        assert dline.confidence == 0.85


class TestAssembleConfidenceThreshold:
    """Confidence threshold filters low-confidence speakers."""

    def test_low_confidence_speaker_set_to_none(self):
        """Dialogue with confidence below threshold has speaker=None."""
        pre = PreprocessArtifact(
            schema_version="1.0",
            original_path="test.txt",
            cleaned_text="Title\nContent",
            total_chars=10,
        )
        chapters = ChapterArtifact(schema_version="1.0", chapters=[])
        scenes = SceneArtifact(
            schema_version="1.0",
            scenes=[
                Scene(
                    scene_id="CH01-S01", chapter_id="CH01",
                    content='某人说："低置信。"萧炎说："高置信。"',
                    boundary_keywords=[], location="大殿", int_ext="INT",
                    time_of_day="日", confidence=1.0,
                ),
            ],
        )
        characters = CharacterArtifact(
            schema_version="1.0",
            characters=[
                CharacterRef(name="萧炎", first_appearance="CH01-S01"),
            ],
        )
        dialogues = DialogueArtifact(
            schema_version="1.0",
            dialogues=[
                DialogueLine(
                    dialogue_id="D01", scene_id="CH01-S01", line_index=0,
                    speaker="萧炎", line="低置信。", quote_style="double",
                    confidence=0.3, attribution_method="nearest_name",
                ),
                DialogueLine(
                    dialogue_id="D02", scene_id="CH01-S01", line_index=1,
                    speaker="萧炎", line="高置信。", quote_style="double",
                    confidence=0.85, attribution_method="prefix_match",
                ),
            ],
        )

        result = Pipeline()._assemble(pre, chapters, scenes, characters, dialogues,
                                      confidence_threshold=0.5)
        dialines = [l for l in result.scenes[0].lines if l.type == "dialogue"]
        assert dialines[0].character is None
        assert dialines[1].character == "萧炎"

    def test_default_threshold_zero_keeps_all_speakers(self):
        """Default confidence_threshold=0.0 keeps all speakers intact."""
        pre = PreprocessArtifact(
            schema_version="1.0",
            original_path="test.txt",
            cleaned_text="Title\nContent",
            total_chars=10,
        )
        chapters = ChapterArtifact(schema_version="1.0", chapters=[])
        scenes = SceneArtifact(
            schema_version="1.0",
            scenes=[
                Scene(
                    scene_id="CH01-S01", chapter_id="CH01",
                    content='某人说："低置信。"',
                    boundary_keywords=[], location="大殿", int_ext="INT",
                    time_of_day="日", confidence=1.0,
                ),
            ],
        )
        characters = CharacterArtifact(
            schema_version="1.0",
            characters=[
                CharacterRef(name="萧炎", first_appearance="CH01-S01"),
            ],
        )
        dialogues = DialogueArtifact(
            schema_version="1.0",
            dialogues=[
                DialogueLine(
                    dialogue_id="D01", scene_id="CH01-S01", line_index=0,
                    speaker="萧炎", line="低置信。", quote_style="double",
                    confidence=0.0, attribution_method="unattributed",
                ),
            ],
        )

        result = Pipeline()._assemble(pre, chapters, scenes, characters, dialogues,
                                      confidence_threshold=0.0)
        dialines = [l for l in result.scenes[0].lines if l.type == "dialogue"]
        assert dialines[0].character == "萧炎"


class TestAssembleScriptOutputValidation:
    """Assembled ScriptOutput passes Pydantic validation."""

    def test_model_validate_passes(self):
        """Assembled output validates as a proper ScriptOutput."""
        pre = PreprocessArtifact(
            schema_version="1.0",
            original_path="test.txt",
            cleaned_text="Title\nContent",
            total_chars=10,
        )
        chapters = ChapterArtifact(schema_version="1.0", chapters=[])
        scenes = SceneArtifact(
            schema_version="1.0",
            scenes=[
                Scene(
                    scene_id="CH01-S01", chapter_id="CH01",
                    content="Content",
                    boundary_keywords=[], location="Loc", int_ext="INT",
                    time_of_day="日", confidence=1.0,
                ),
            ],
        )
        characters = CharacterArtifact(
            schema_version="1.0",
            characters=[
                CharacterRef(name="萧炎", first_appearance="CH01-S01"),
            ],
        )
        dialogues = DialogueArtifact(schema_version="1.0", dialogues=[])

        result = Pipeline()._assemble(pre, chapters, scenes, characters, dialogues)
        validated = ScriptOutput.model_validate(result.model_dump())
        assert validated.schema_version == "1.0"
        assert validated.title == "Title"
        assert len(validated.characters) == 1
        assert len(validated.scenes) == 1


# ===================================================================
# Cache tests  --  require temporary directories
# ===================================================================


class TestCacheCreation:
    """Cache files are created correctly after a pipeline run."""

    def test_all_cache_files_created(self, tmp_path):
        """All .json and .meta files are created after a successful run."""
        novel = tmp_path / "novel.txt"
        novel.write_text("测试标题\n\n正文内容", encoding="utf-8")
        cache = tmp_path / "cache"

        Pipeline().run(str(novel), cache_dir=str(cache))

        for name in ("preprocess", "chapters", "scenes", "characters", "dialogues"):
            assert (cache / f"{name}.json").exists(), f"{name}.json missing"
            assert (cache / f"{name}.meta").exists(), f"{name}.meta missing"

    def test_meta_contains_correct_sha256(self, tmp_path):
        """Meta file contains the correct SHA256 of the input file."""
        novel = tmp_path / "novel.txt"
        content = "测试标题\n\n正文内容"
        novel.write_text(content, encoding="utf-8")
        cache = tmp_path / "cache"

        # Compute SHA256 from the file on disk (accounts for OS newline conv.)
        expected_sha = hashlib.sha256(novel.read_bytes()).hexdigest()

        Pipeline().run(str(novel), cache_dir=str(cache))

        meta = json.loads((cache / "preprocess.meta").read_text("utf-8"))
        assert meta["input_sha256"] == expected_sha
        assert meta["pipeline_version"] == "1.0"
        assert "created_at" in meta

    def test_no_cache_flag_skips_writes(self, tmp_path):
        """With no_cache=True, no cache files are written."""
        novel = tmp_path / "novel.txt"
        novel.write_text("测试标题\n\n正文内容", encoding="utf-8")
        cache = tmp_path / "cache"

        Pipeline().run(str(novel), cache_dir=str(cache), no_cache=True)

        assert not cache.exists() or not list(cache.iterdir()), (
            "No cache files should exist when no_cache=True"
        )


class TestCacheHit:
    """Valid cache skips module execution."""

    def test_cache_hit_skips_module_calls(self, tmp_path):
        """When valid cache exists, module functions are NOT called."""
        novel = tmp_path / "novel.txt"
        novel.write_text("测试标题\n\n正文内容", encoding="utf-8")
        cache = tmp_path / "cache"

        # Patch *before* first run so that the real functions are wrapped and
        # we can verify they are NOT called on the second (cached) run.
        import engine.converter as conv

        import engine.preprocess as pre_mod
        import engine.chapter as ch_mod
        import engine.scene as sc_mod
        import engine.character as char_mod
        import engine.dialogue as dial_mod

        with (
            patch.object(conv, "preprocess", wraps=pre_mod.preprocess) as m_pre,
            patch.object(conv, "split_chapters", wraps=ch_mod.split_chapters) as m_ch,
            patch.object(conv, "detect_scenes", wraps=sc_mod.detect_scenes) as m_sc,
            patch.object(conv, "extract_characters", wraps=char_mod.extract_characters) as m_char,
            patch.object(conv, "extract_dialogues", wraps=dial_mod.extract_dialogues) as m_dial,
        ):
            # First run — populates cache
            Pipeline().run(str(novel), cache_dir=str(cache))
            m_pre.assert_called_once()
            m_ch.assert_called_once()
            m_sc.assert_called_once()
            m_char.assert_called_once()
            m_dial.assert_called_once()

            # Reset call counts for the second run
            m_pre.reset_mock()
            m_ch.reset_mock()
            m_sc.reset_mock()
            m_char.reset_mock()
            m_dial.reset_mock()

            # Second run — all modules should be skipped (cache hit)
            Pipeline().run(str(novel), cache_dir=str(cache))
            m_pre.assert_not_called()
            m_ch.assert_not_called()
            m_sc.assert_not_called()
            m_char.assert_not_called()
            m_dial.assert_not_called()


class TestCacheInvalidation:
    """Cache is invalidated on SHA256 or version mismatch."""

    def test_sha256_mismatch_re_runs_preprocess(self, tmp_path):
        """Changing input file content invalidates the cache."""
        novel = tmp_path / "novel.txt"
        novel.write_text("原始内容", encoding="utf-8")
        cache = tmp_path / "cache"

        import engine.converter as conv
        import engine.preprocess as pre_mod

        with patch.object(conv, "preprocess", wraps=pre_mod.preprocess) as m_pre:
            # First run to populate cache
            Pipeline().run(str(novel), cache_dir=str(cache))
            m_pre.assert_called_once()
            m_pre.reset_mock()

            # Change input content
            novel.write_text("修改后的内容", encoding="utf-8")

            # Second run — cache invalidated, preprocess called again
            Pipeline().run(str(novel), cache_dir=str(cache))
            m_pre.assert_called_once()

    def test_version_mismatch_re_runs_preprocess(self, tmp_path):
        """Changing pipeline_version in .meta invalidates cache."""
        novel = tmp_path / "novel.txt"
        novel.write_text("测试标题\n\n正文内容", encoding="utf-8")
        cache = tmp_path / "cache"

        import engine.converter as conv
        import engine.preprocess as pre_mod

        with patch.object(conv, "preprocess", wraps=pre_mod.preprocess) as m_pre:
            # First run to populate cache
            Pipeline().run(str(novel), cache_dir=str(cache))
            m_pre.assert_called_once()
            m_pre.reset_mock()

        # Corrupt the meta version *outside* the patch context
        meta_path = cache / "preprocess.meta"
        meta = json.loads(meta_path.read_text("utf-8"))
        meta["pipeline_version"] = "0.5"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

        with patch.object(conv, "preprocess", wraps=pre_mod.preprocess) as m_pre:
            # Second run — cache invalidated, preprocess called
            Pipeline().run(str(novel), cache_dir=str(cache))
            m_pre.assert_called_once()


class TestResumeFrom:
    """resume_from controls which stages run fresh."""

    def test_resume_from_chapter_skips_preprocess(self, tmp_path):
        """resume_from='chapter' loads preprocess from cache, runs rest."""
        novel = tmp_path / "novel.txt"
        novel.write_text("测试标题\n\n正文内容", encoding="utf-8")
        cache = tmp_path / "cache"

        import engine.converter as conv
        import engine.preprocess as pre_mod
        import engine.chapter as ch_mod
        import engine.scene as sc_mod
        import engine.character as char_mod
        import engine.dialogue as dial_mod

        with (
            patch.object(conv, "preprocess", wraps=pre_mod.preprocess) as m_pre,
            patch.object(conv, "split_chapters", wraps=ch_mod.split_chapters) as m_ch,
            patch.object(conv, "detect_scenes", wraps=sc_mod.detect_scenes) as m_sc,
            patch.object(conv, "extract_characters", wraps=char_mod.extract_characters) as m_char,
            patch.object(conv, "extract_dialogues", wraps=dial_mod.extract_dialogues) as m_dial,
        ):
            # First run — populate caches
            Pipeline().run(str(novel), cache_dir=str(cache))
            m_pre.assert_called_once()
            m_ch.assert_called_once()
            m_sc.assert_called_once()
            m_char.assert_called_once()
            m_dial.assert_called_once()
            m_pre.reset_mock()
            m_ch.reset_mock()
            m_sc.reset_mock()
            m_char.reset_mock()
            m_dial.reset_mock()

            # Second run — resume from chapter
            Pipeline().run(str(novel), cache_dir=str(cache), resume_from="chapter")
            m_pre.assert_not_called()
            m_ch.assert_called_once()
            m_sc.assert_called_once()
            m_char.assert_called_once()
            m_dial.assert_called_once()

    def test_resume_from_scene_skips_preprocess_and_chapter(self, tmp_path):
        """resume_from='scene' loads preprocess/chapter caches, runs scene+."""
        novel = tmp_path / "novel.txt"
        novel.write_text("测试标题\n\n正文内容", encoding="utf-8")
        cache = tmp_path / "cache"

        import engine.converter as conv
        import engine.preprocess as pre_mod
        import engine.chapter as ch_mod
        import engine.scene as sc_mod
        import engine.character as char_mod
        import engine.dialogue as dial_mod

        with (
            patch.object(conv, "preprocess", wraps=pre_mod.preprocess) as m_pre,
            patch.object(conv, "split_chapters", wraps=ch_mod.split_chapters) as m_ch,
            patch.object(conv, "detect_scenes", wraps=sc_mod.detect_scenes) as m_sc,
            patch.object(conv, "extract_characters", wraps=char_mod.extract_characters) as m_char,
            patch.object(conv, "extract_dialogues", wraps=dial_mod.extract_dialogues) as m_dial,
        ):
            # First run — populate caches
            Pipeline().run(str(novel), cache_dir=str(cache))
            m_pre.reset_mock()
            m_ch.reset_mock()
            m_sc.reset_mock()
            m_char.reset_mock()
            m_dial.reset_mock()

            # Second run — resume from scene
            Pipeline().run(str(novel), cache_dir=str(cache), resume_from="scene")
            m_pre.assert_not_called()
            m_ch.assert_not_called()
            m_sc.assert_called_once()
            m_char.assert_called_once()
            m_dial.assert_called_once()

    def test_resume_from_character_runs_only_character(self, tmp_path):
        """resume_from='character' runs character; dialogue from cache."""
        novel = tmp_path / "novel.txt"
        novel.write_text("测试标题\n\n正文内容", encoding="utf-8")
        cache = tmp_path / "cache"

        import engine.converter as conv
        import engine.preprocess as pre_mod
        import engine.chapter as ch_mod
        import engine.scene as sc_mod
        import engine.character as char_mod
        import engine.dialogue as dial_mod

        with (
            patch.object(conv, "preprocess", wraps=pre_mod.preprocess) as m_pre,
            patch.object(conv, "split_chapters", wraps=ch_mod.split_chapters) as m_ch,
            patch.object(conv, "detect_scenes", wraps=sc_mod.detect_scenes) as m_sc,
            patch.object(conv, "extract_characters", wraps=char_mod.extract_characters) as m_char,
            patch.object(conv, "extract_dialogues", wraps=dial_mod.extract_dialogues) as m_dial,
        ):
            # First run — populate caches
            Pipeline().run(str(novel), cache_dir=str(cache))
            m_pre.reset_mock()
            m_ch.reset_mock()
            m_sc.reset_mock()
            m_char.reset_mock()
            m_dial.reset_mock()

            # Second run — resume from character
            Pipeline().run(str(novel), cache_dir=str(cache), resume_from="character")
            m_pre.assert_not_called()
            m_ch.assert_not_called()
            m_sc.assert_not_called()
            m_char.assert_called_once()
            m_dial.assert_not_called()

    def test_resume_from_dialogue_runs_only_dialogue(self, tmp_path):
        """resume_from='dialogue' runs dialogue; character from cache."""
        novel = tmp_path / "novel.txt"
        novel.write_text("测试标题\n\n正文内容", encoding="utf-8")
        cache = tmp_path / "cache"

        import engine.converter as conv
        import engine.preprocess as pre_mod
        import engine.chapter as ch_mod
        import engine.scene as sc_mod
        import engine.character as char_mod
        import engine.dialogue as dial_mod

        with (
            patch.object(conv, "preprocess", wraps=pre_mod.preprocess) as m_pre,
            patch.object(conv, "split_chapters", wraps=ch_mod.split_chapters) as m_ch,
            patch.object(conv, "detect_scenes", wraps=sc_mod.detect_scenes) as m_sc,
            patch.object(conv, "extract_characters", wraps=char_mod.extract_characters) as m_char,
            patch.object(conv, "extract_dialogues", wraps=dial_mod.extract_dialogues) as m_dial,
        ):
            # First run — populate caches
            Pipeline().run(str(novel), cache_dir=str(cache))
            m_pre.reset_mock()
            m_ch.reset_mock()
            m_sc.reset_mock()
            m_char.reset_mock()
            m_dial.reset_mock()

            # Second run — resume from dialogue
            Pipeline().run(str(novel), cache_dir=str(cache), resume_from="dialogue")
            m_pre.assert_not_called()
            m_ch.assert_not_called()
            m_sc.assert_not_called()
            m_char.assert_not_called()
            m_dial.assert_called_once()


# ===================================================================
# Integration test  —  full pipeline against a real novel
# ===================================================================


class TestPipelineIntegration:
    """End-to-end run using the basic_3ch test fixture."""

    def test_full_pipeline_returns_script_output(self, basic_novel, tmp_path):
        """Full pipeline produces a valid ScriptOutput."""
        novel = tmp_path / "novel.txt"
        novel.write_text(basic_novel, encoding="utf-8")

        result = Pipeline().run(str(novel), cache_dir=str(tmp_path / "cache"))

        assert isinstance(result, ScriptOutput)
        assert result.schema_version == "1.0"
        assert result.title
        assert result.source_novel == str(novel)
        assert len(result.scenes) >= 1

    def test_output_yaml_serialization(self, basic_novel, tmp_path):
        """ScriptOutput is correctly serialized to YAML when output_path set."""
        novel = tmp_path / "novel.txt"
        novel.write_text(basic_novel, encoding="utf-8")
        output = tmp_path / "output.yaml"

        result = Pipeline().run(
            str(novel), output_path=str(output), cache_dir=str(tmp_path / "cache")
        )

        assert output.exists()
        with open(output, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["title"] == result.title
        assert data["schema_version"] == "1.0"
        assert "scenes" in data
        assert "characters" in data
