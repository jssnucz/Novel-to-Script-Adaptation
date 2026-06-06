"""Tests for AI character profiling (Phase 2 follow-up)."""

import pytest

from engine.ai_enhancer import _heuristic_role, profile_characters, _fallback_profiles


class TestHeuristicRole:
    """Statistical fallback for role classification."""

    def test_protagonist_high_dialogue_share(self):
        role, conf = _heuristic_role(
            dialogue_count=50,
            appearance_count=10,
            total_dialogues=100,
            total_appearances=30,
        )
        assert role == "主角"
        assert conf == 0.7

    def test_supporting_moderate_dialogue_share(self):
        role, conf = _heuristic_role(
            dialogue_count=15,
            appearance_count=5,
            total_dialogues=100,
            total_appearances=30,
        )
        assert role == "配角"
        assert conf == 0.6

    def test_extra_low_dialogue_share(self):
        role, conf = _heuristic_role(
            dialogue_count=2,
            appearance_count=1,
            total_dialogues=100,
            total_appearances=30,
        )
        assert role == "龙套"
        assert conf == 0.8

    def test_boundary_exactly_40_percent(self):
        """40% is not > 0.4, so it falls to 配角."""
        role, conf = _heuristic_role(
            dialogue_count=40,
            appearance_count=10,
            total_dialogues=100,
            total_appearances=30,
        )
        assert role == "配角"
        assert conf == 0.6

    def test_boundary_exactly_10_percent(self):
        """10% is not > 0.1, so it falls to 龙套."""
        role, conf = _heuristic_role(
            dialogue_count=10,
            appearance_count=2,
            total_dialogues=100,
            total_appearances=30,
        )
        assert role == "龙套"
        assert conf == 0.8

    def test_zero_total_dialogues_does_not_divide_by_zero(self):
        """Edge case: no dialogues at all."""
        role, conf = _heuristic_role(
            dialogue_count=0,
            appearance_count=1,
            total_dialogues=0,
            total_appearances=10,
        )
        assert role == "龙套"
        assert conf == 0.8

    def test_single_character_all_dialogues(self):
        """When there's only one character, they're the protagonist."""
        role, conf = _heuristic_role(
            dialogue_count=10,
            appearance_count=3,
            total_dialogues=10,
            total_appearances=3,
        )
        assert role == "主角"
        assert conf == 0.7


class TestFallbackProfiles:
    """Fallback profile generation when AI is unavailable."""

    def test_returns_correct_structure(self):
        profiles = [
            {"name": "萧炎", "dialogue_count": 30, "appearance_count": 3,
             "first_appearance_text": "测试山巅..."},
            {"name": "纳兰嫣然", "dialogue_count": 6, "appearance_count": 3,
             "first_appearance_text": "远处传来..."},
            {"name": "路人甲", "dialogue_count": 1, "appearance_count": 1,
             "first_appearance_text": "人群中..."},
        ]
        result = _fallback_profiles(profiles)
        assert len(result) == 3
        for item in result:
            assert "name" in item
            assert "role" in item
            assert "role_confidence" in item
            # description is always None in fallback mode
            assert item["description"] is None

    def test_roles_assigned_correctly(self):
        profiles = [
            {"name": "主角", "dialogue_count": 50, "appearance_count": 10,
             "first_appearance_text": "..."},
            {"name": "配角", "dialogue_count": 10, "appearance_count": 5,
             "first_appearance_text": "..."},
            {"name": "龙套", "dialogue_count": 2, "appearance_count": 1,
             "first_appearance_text": "..."},
        ]
        result = _fallback_profiles(profiles)
        assert result[0]["role"] == "主角"
        assert result[1]["role"] == "配角"
        assert result[2]["role"] == "龙套"

    def test_empty_profiles(self):
        result = _fallback_profiles([])
        assert result == []


class TestProfileCharacters:
    """AI character profiling function."""

    def test_empty_profiles_returns_empty_list(self):
        result = profile_characters([])
        assert result == []

    def test_no_api_key_uses_fallback(self, monkeypatch):
        """When NOVEL2SCRIPT_API_KEY is not set, fallback is used."""
        monkeypatch.delenv("NOVEL2SCRIPT_API_KEY", raising=False)
        # Force re-initialization
        import engine.ai_enhancer as mod
        mod._client = None
        mod._client_init_attempted = False

        profiles = [
            {"name": "萧炎", "dialogue_count": 30, "appearance_count": 3,
             "first_appearance_text": "测试山巅，云海翻腾。"},
            {"name": "纳兰嫣然", "dialogue_count": 6, "appearance_count": 3,
             "first_appearance_text": "远处传来一声清脆的呼喊。"},
        ]
        result = profile_characters(profiles)
        assert len(result) == 2
        # 萧炎 has 30/36 ≈ 83% dialogue share → 主角
        assert result[0]["name"] == "萧炎"
        assert result[0]["role"] == "主角"
        assert result[0]["description"] is None
        # 纳兰嫣然 has 6/36 ≈ 17% → 配角
        assert result[1]["name"] == "纳兰嫣然"
        assert result[1]["role"] == "配角"
        assert result[1]["description"] is None

    def test_fallback_preserves_name_order(self):
        """Fallback results should preserve input order."""
        import engine.ai_enhancer as mod
        mod._client = None
        mod._client_init_attempted = False

        profiles = [
            {"name": "C", "dialogue_count": 1, "appearance_count": 1,
             "first_appearance_text": ""},
            {"name": "A", "dialogue_count": 10, "appearance_count": 3,
             "first_appearance_text": ""},
            {"name": "B", "dialogue_count": 3, "appearance_count": 2,
             "first_appearance_text": ""},
        ]
        result = profile_characters(profiles)
        assert [r["name"] for r in result] == ["C", "A", "B"]

    def test_result_structure(self):
        """Each result dict has the expected keys."""
        import engine.ai_enhancer as mod
        mod._client = None
        mod._client_init_attempted = False

        profiles = [
            {"name": "测试角色", "dialogue_count": 5, "appearance_count": 2,
             "first_appearance_text": "测试内容"},
        ]
        result = profile_characters(profiles)
        assert len(result) == 1
        keys = set(result[0].keys())
        assert keys == {"name", "role", "role_confidence", "description"}


class TestConverterIntegration:
    """Test that converter._assemble uses profiling data correctly."""

    def test_assemble_without_profiles_leaves_role_null(self):
        """Without AI profiling, role and description stay None."""
        from engine.converter import Pipeline
        from engine.models import (
            PreprocessArtifact, ChapterArtifact, SceneArtifact,
            CharacterArtifact, CharacterRef, DialogueArtifact,
            Scene,
        )

        pipeline = Pipeline()
        preprocessed = PreprocessArtifact(
            schema_version="1.0",
            original_path="/test/novel.txt",
            cleaned_text="测试小说\n萧炎站在山巅。",
            total_chars=10,
        )
        chapters = ChapterArtifact(schema_version="1.0", chapters=[])
        scenes = SceneArtifact(schema_version="1.0", scenes=[
            Scene(scene_id="CH01-S01", chapter_id="CH01",
                  content="萧炎站在山巅。", boundary_keywords=[],
                  location="山巅", int_ext="EXT", time_of_day="日"),
        ])
        characters = CharacterArtifact(schema_version="1.0", characters=[
            CharacterRef(name="萧炎", first_appearance="CH01-S01"),
        ])
        dialogues = DialogueArtifact(schema_version="1.0", dialogues=[])

        result = pipeline._assemble(
            preprocessed, chapters, scenes, characters, dialogues,
        )
        assert len(result.characters) == 1
        assert result.characters[0].role is None
        assert result.characters[0].description is None

    def test_assemble_with_profiles_fills_role_and_description(self):
        """With AI profiling data, role and description are populated."""
        from engine.converter import Pipeline
        from engine.models import (
            PreprocessArtifact, ChapterArtifact, SceneArtifact,
            CharacterArtifact, CharacterRef, DialogueArtifact,
            Scene,
        )

        pipeline = Pipeline()
        preprocessed = PreprocessArtifact(
            schema_version="1.0",
            original_path="/test/novel.txt",
            cleaned_text="斗破苍穹\n萧炎站在山巅。",
            total_chars=15,
        )
        chapters = ChapterArtifact(schema_version="1.0", chapters=[])
        scenes = SceneArtifact(schema_version="1.0", scenes=[
            Scene(scene_id="CH01-S01", chapter_id="CH01",
                  content="萧炎站在山巅。", boundary_keywords=[],
                  location="山巅", int_ext="EXT", time_of_day="日"),
        ])
        characters = CharacterArtifact(schema_version="1.0", characters=[
            CharacterRef(name="萧炎", first_appearance="CH01-S01"),
        ])
        dialogues = DialogueArtifact(schema_version="1.0", dialogues=[])

        profiles = [
            {
                "name": "萧炎",
                "role": "主角",
                "role_confidence": 0.95,
                "description": "萧家修炼天才，性格坚毅沉稳。",
            },
        ]

        result = pipeline._assemble(
            preprocessed, chapters, scenes, characters, dialogues,
            profiles=profiles,
        )
        assert len(result.characters) == 1
        assert result.characters[0].role == "主角"
        assert result.characters[0].description == "萧家修炼天才，性格坚毅沉稳。"

    def test_assemble_partial_profiles_only_fills_known(self):
        """Characters without profiling data keep None values."""
        from engine.converter import Pipeline
        from engine.models import (
            PreprocessArtifact, ChapterArtifact, SceneArtifact,
            CharacterArtifact, CharacterRef, DialogueArtifact,
            Scene,
        )

        pipeline = Pipeline()
        preprocessed = PreprocessArtifact(
            schema_version="1.0",
            original_path="/test/novel.txt",
            cleaned_text="测试\n萧炎站在山巅。纳兰嫣然走来。",
            total_chars=20,
        )
        chapters = ChapterArtifact(schema_version="1.0", chapters=[])
        scenes = SceneArtifact(schema_version="1.0", scenes=[
            Scene(scene_id="CH01-S01", chapter_id="CH01",
                  content="萧炎站在山巅。纳兰嫣然走来。",
                  boundary_keywords=[], location="山巅",
                  int_ext="EXT", time_of_day="日"),
        ])
        characters = CharacterArtifact(schema_version="1.0", characters=[
            CharacterRef(name="萧炎", first_appearance="CH01-S01"),
            CharacterRef(name="纳兰嫣然", first_appearance="CH01-S01"),
        ])
        dialogues = DialogueArtifact(schema_version="1.0", dialogues=[])

        # Only 萧炎 has profiling data
        profiles = [
            {
                "name": "萧炎",
                "role": "主角",
                "role_confidence": 0.9,
                "description": "主角描述",
            },
        ]

        result = pipeline._assemble(
            preprocessed, chapters, scenes, characters, dialogues,
            profiles=profiles,
        )
        assert len(result.characters) == 2

        xiao_yan = next(c for c in result.characters if c.name == "萧炎")
        assert xiao_yan.role == "主角"
        assert xiao_yan.description == "主角描述"

        na_lan = next(c for c in result.characters if c.name == "纳兰嫣然")
        assert na_lan.role is None
        assert na_lan.description is None
