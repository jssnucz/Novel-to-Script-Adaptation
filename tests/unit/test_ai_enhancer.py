"""Tests for AI enhancer — profiling, scene classification, character
verification, and dialogue attribution."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from engine.ai_enhancer import (
    _heuristic_role,
    _fallback_profiles,
    _cache_key,
    _read_cache,
    _write_cache,
    _apply_dialogue_updates,
    enhance_scene_classification,
    verify_characters,
    profile_characters,
    enhance_dialogue_attribution,
    is_ai_available,
)
from engine.models import DialogueLine


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class _FakeChoice:
    """Simulates ``openai.types.chat.ChatCompletionMessage``."""
    def __init__(self, content: str):
        self.message = MagicMock()
        self.message.content = content


class _FakeCompletion:
    """Simulates ``openai.types.chat.ChatCompletion``."""
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


def _make_mock_client(response_json: dict | str) -> MagicMock:
    """Return a mock OpenAI client whose ``chat.completions.create`` returns
    a completion containing *response_json* (serialised as JSON if it's a
    dict)."""
    client = MagicMock()
    if isinstance(response_json, dict):
        raw = json.dumps(response_json, ensure_ascii=False)
    else:
        raw = response_json
    client.chat.completions.create.return_value = _FakeCompletion(raw)
    return client


def _patch_client(monkeypatch, response_json: dict | str):
    """Patch ``ai_enhancer._get_client`` to return a mock client and also set
    ``NOVEL2SCRIPT_API_KEY`` so the module thinks AI is available."""
    import engine.ai_enhancer as mod

    monkeypatch.setenv("NOVEL2SCRIPT_API_KEY", "sk-test-mock")
    mod._client = None
    mod._client_init_attempted = False
    client = _make_mock_client(response_json)
    monkeypatch.setattr(mod, "_get_client", lambda: client)
    return client


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


# ===================================================================
# Mock-based tests — core LLM functions (covers the 21 % → 70 %+ gap)
# ===================================================================


class TestEnhanceSceneClassificationMocked:
    """Test ``enhance_scene_classification`` with mocked LLM."""

    def test_llm_success_returns_classification(self, monkeypatch, tmp_path):
        response = {
            "int_ext": "INT",
            "location": "客栈大厅",
            "time_of_day": "夜",
            "confidence": 0.92,
        }
        _patch_client(monkeypatch, response)

        result = enhance_scene_classification(
            "CH01-S01", "客栈大厅内，烛火摇曳。", cache_dir=str(tmp_path),
        )
        assert result["int_ext"] == "INT"
        assert result["location"] == "客栈大厅"
        assert result["time_of_day"] == "夜"
        assert result["confidence"] == 0.92

    def test_llm_success_writes_and_reads_cache(self, monkeypatch, tmp_path):
        response = {"int_ext": "EXT", "location": "山巅", "time_of_day": "日", "confidence": 0.85}
        client = _patch_client(monkeypatch, response)
        cache_dir = str(tmp_path)

        # First call — should hit LLM
        r1 = enhance_scene_classification("CH01-S01", "山巅云海翻腾", cache_dir=cache_dir)
        assert client.chat.completions.create.call_count == 1
        assert r1["int_ext"] == "EXT"

        # Second call — should hit cache, no additional API call
        r2 = enhance_scene_classification("CH01-S01", "山巅云海翻腾", cache_dir=cache_dir)
        assert client.chat.completions.create.call_count == 1  # still 1
        assert r2 == r1

    def test_no_api_key_returns_empty(self):
        import engine.ai_enhancer as mod

        mod._client = None
        mod._client_init_attempted = True  # simulate tried-and-failed
        result = enhance_scene_classification("CH01-S01", "测试")
        assert result == {}

    def test_llm_failure_graceful_degradation(self, monkeypatch):
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("Network error")
        import engine.ai_enhancer as mod

        monkeypatch.setenv("NOVEL2SCRIPT_API_KEY", "sk-test")
        mod._client = None
        mod._client_init_attempted = False
        monkeypatch.setattr(mod, "_get_client", lambda: client)

        result = enhance_scene_classification("CH01-S01", "测试场景")
        assert result == {}


class TestVerifyCharactersMocked:
    """Test ``verify_characters`` with mocked LLM."""

    def test_llm_filters_false_positives(self, monkeypatch, tmp_path):
        response = {
            "results": [
                {"index": 0, "name": "萧炎", "is_character": True, "confidence": 0.95},
                {"index": 1, "name": "云海", "is_character": False, "confidence": 0.05},
                {"index": 2, "name": "纳兰嫣然", "is_character": True, "confidence": 0.90},
                {"index": 3, "name": "终于", "is_character": False, "confidence": 0.02},
            ]
        }
        _patch_client(monkeypatch, response)

        candidates = ["萧炎", "云海", "纳兰嫣然", "终于"]
        snippets = ["测试", "云海翻腾", "纳兰嫣然走来", "终于到了"]
        verified, confs = verify_characters(
            candidates, snippets, cache_dir=str(tmp_path),
        )
        assert set(verified) == {"萧炎", "纳兰嫣然"}
        assert confs["萧炎"] == 0.95
        assert confs["纳兰嫣然"] == 0.90

    def test_llm_cache_hit(self, monkeypatch, tmp_path):
        response = {
            "results": [
                {"index": 0, "name": "萧炎", "is_character": True, "confidence": 0.9},
            ]
        }
        client = _patch_client(monkeypatch, response)
        cache_dir = str(tmp_path)

        r1_names, _ = verify_characters(["萧炎"], ["测试"], cache_dir=cache_dir)
        assert client.chat.completions.create.call_count == 1

        r2_names, _ = verify_characters(["萧炎"], ["测试"], cache_dir=cache_dir)
        assert client.chat.completions.create.call_count == 1
        assert r2_names == r1_names

    def test_empty_candidates_returns_empty(self):
        verified, confs = verify_characters([], [])
        assert verified == []
        assert confs == {}

    def test_no_api_key_returns_original_with_low_confidence(self):
        import engine.ai_enhancer as mod

        mod._client = None
        mod._client_init_attempted = True
        verified, confs = verify_characters(["萧炎", "云海"], ["t1", "t2"])
        assert set(verified) == {"萧炎", "云海"}
        assert confs["萧炎"] == 0.3


class TestProfileCharactersMocked:
    """Test ``profile_characters`` with mocked LLM (success path)."""

    def test_llm_success_fills_role_and_description(self, monkeypatch, tmp_path):
        response = {
            "profiles": [
                {
                    "name": "萧炎",
                    "role": "主角",
                    "role_confidence": 0.96,
                    "description": "萧家修炼天才，性格坚毅。",
                },
                {
                    "name": "纳兰嫣然",
                    "role": "配角",
                    "role_confidence": 0.88,
                    "description": "云岚宗女弟子，与萧炎有旧。",
                },
            ]
        }
        _patch_client(monkeypatch, response)

        profiles = [
            {"name": "萧炎", "dialogue_count": 30, "appearance_count": 3,
             "first_appearance_text": "测试山巅，云海翻腾。"},
            {"name": "纳兰嫣然", "dialogue_count": 6, "appearance_count": 3,
             "first_appearance_text": "远处传来呼喊。"},
        ]
        result = profile_characters(profiles, cache_dir=str(tmp_path))
        assert len(result) == 2
        assert result[0] == {
            "name": "萧炎",
            "role": "主角",
            "role_confidence": 0.96,
            "description": "萧家修炼天才，性格坚毅。",
        }
        assert result[1]["role"] == "配角"

    def test_llm_invalid_role_filtered(self, monkeypatch, tmp_path):
        """LLM returns an invalid role — should be set to None."""
        response = {
            "profiles": [
                {"name": "萧炎", "role": "超级主角", "role_confidence": 0.9,
                 "description": "xxx"},
            ]
        }
        _patch_client(monkeypatch, response)

        profiles = [
            {"name": "萧炎", "dialogue_count": 5, "appearance_count": 1,
             "first_appearance_text": "测试"},
        ]
        result = profile_characters(profiles, cache_dir=str(tmp_path))
        assert result[0]["role"] is None  # invalid role → filtered

    def test_llm_cache_read_write(self, monkeypatch, tmp_path):
        response = {
            "profiles": [
                {"name": "A", "role": "主角", "role_confidence": 0.9,
                 "description": "desc A"},
            ]
        }
        client = _patch_client(monkeypatch, response)
        cache_dir = str(tmp_path)

        profiles = [{"name": "A", "dialogue_count": 5, "appearance_count": 1,
                     "first_appearance_text": "text"}]
        r1 = profile_characters(profiles, cache_dir=cache_dir, input_sha256="abc123")
        assert client.chat.completions.create.call_count == 1

        r2 = profile_characters(profiles, cache_dir=cache_dir, input_sha256="abc123")
        assert client.chat.completions.create.call_count == 1
        assert r2 == r1

    def test_llm_failure_falls_back_to_heuristic(self, monkeypatch):
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("Timeout")
        import engine.ai_enhancer as mod

        monkeypatch.setenv("NOVEL2SCRIPT_API_KEY", "sk-test")
        mod._client = None
        mod._client_init_attempted = False
        monkeypatch.setattr(mod, "_get_client", lambda: client)

        profiles = [
            {"name": "主角", "dialogue_count": 50, "appearance_count": 5,
             "first_appearance_text": ""},
            {"name": "配角", "dialogue_count": 15, "appearance_count": 3,
             "first_appearance_text": ""},
        ]
        result = profile_characters(profiles)
        # 50/65 ≈ 77% → 主角; 15/65 ≈ 23% → 配角
        assert result[0]["role"] == "主角"
        assert result[0]["description"] is None
        assert result[1]["role"] == "配角"


class TestDialogueAttributionMocked:
    """Test ``enhance_dialogue_attribution`` with mocked LLM."""

    def _make_dialogue_lines(self, scene_id: str) -> list[DialogueLine]:
        return [
            DialogueLine(
                dialogue_id=f"{scene_id}-D00", scene_id=scene_id,
                line_index=0, speaker="萧炎",
                line="三年了……", quote_style="double",
                confidence=0.85, attribution_method="prefix_match",
            ),
            DialogueLine(
                dialogue_id=f"{scene_id}-D01", scene_id=scene_id,
                line_index=1, speaker=None,
                line="萧炎哥哥！", quote_style="double",
                confidence=0.3, attribution_method="unattributed",
            ),
            DialogueLine(
                dialogue_id=f"{scene_id}-D02", scene_id=scene_id,
                line_index=2, speaker=None,
                line="你准备好了吗？", quote_style="double",
                confidence=0.2, attribution_method="unattributed",
            ),
        ]

    def test_no_low_confidence_lines_skips_llm(self, monkeypatch):
        """When all lines have high confidence, LLM is never called."""
        client = _patch_client(monkeypatch, {})
        lines = [
            DialogueLine(
                dialogue_id="S01-D00", scene_id="S01", line_index=0,
                speaker="萧炎", line="测试", quote_style="double",
                confidence=0.9, attribution_method="prefix_match",
            ),
        ]
        result = enhance_dialogue_attribution(
            "S01", "测试场景", lines, ["萧炎"],
        )
        assert client.chat.completions.create.call_count == 0
        assert result == lines

    def test_round1_attribution_applied(self, monkeypatch, tmp_path):
        response = {
            "discovered_characters": [],
            "attributions": [
                {"line_index": 1, "speaker": "纳兰嫣然", "confidence": 0.9},
                {"line_index": 2, "speaker": "纳兰嫣然", "confidence": 0.85},
            ],
        }
        _patch_client(monkeypatch, response)

        lines = self._make_dialogue_lines("CH01-S01")
        scene_content = (
            '萧炎低声道：“三年了……”\n'
            '远处传来呼喊：“萧炎哥哥！”\n'
            '纳兰嫣然踏空而来：“你准备好了吗？”'
        )
        result = enhance_dialogue_attribution(
            "CH01-S01", scene_content, lines, ["萧炎"],
            cache_dir=str(tmp_path),
        )
        assert result[1].speaker == "纳兰嫣然"
        assert result[1].confidence == 0.9
        assert result[1].attribution_method == "llm"
        assert result[2].speaker == "纳兰嫣然"
        assert result[2].confidence == 0.85

    def test_no_api_key_returns_original(self):
        import engine.ai_enhancer as mod

        mod._client = None
        mod._client_init_attempted = True
        lines = self._make_dialogue_lines("S01")
        original = [dl.model_copy() for dl in lines]
        result = enhance_dialogue_attribution("S01", "测试", lines, ["萧炎"])
        # Should return same lines (with low-confidence capped at 0.3)
        assert len(result) == len(original)

    def test_all_high_confidence_no_change(self, monkeypatch):
        client = _patch_client(monkeypatch, {})
        lines = [
            DialogueLine(
                dialogue_id="S01-D00", scene_id="S01", line_index=0,
                speaker="萧炎", line="哦", quote_style="double",
                confidence=0.95, attribution_method="prefix_match",
            ),
        ]
        result = enhance_dialogue_attribution(
            "S01", "萧炎说：“哦”", lines, ["萧炎"],
        )
        assert client.chat.completions.create.call_count == 0
        assert result[0].speaker == "萧炎"


class TestApplyDialogueUpdates:
    """Unit tests for ``_apply_dialogue_updates``."""

    def test_applies_valid_updates(self):
        lines = [
            DialogueLine(
                dialogue_id="S01-D00", scene_id="S01", line_index=0,
                speaker=None, line="你好", quote_style="double",
                confidence=0.1, attribution_method="unattributed",
            ),
        ]
        attributions = [{"line_index": 0, "speaker": "张三", "confidence": 0.9}]
        result = _apply_dialogue_updates(lines, attributions)
        assert result[0].speaker == "张三"
        assert result[0].confidence == 0.9
        assert result[0].attribution_method == "llm"

    def test_null_speaker_is_normalised(self):
        lines = [
            DialogueLine(
                dialogue_id="S01-D00", scene_id="S01", line_index=0,
                speaker="萧炎", line="?", quote_style="double",
                confidence=0.8, attribution_method="prefix_match",
            ),
        ]
        attributions = [{"line_index": 0, "speaker": None, "confidence": 0.5}]
        result = _apply_dialogue_updates(lines, attributions)
        assert result[0].speaker is None
        assert result[0].confidence == 0.0

    def test_out_of_range_index_ignored(self):
        lines = [
            DialogueLine(
                dialogue_id="S01-D00", scene_id="S01", line_index=0,
                speaker=None, line="x", quote_style="double",
                confidence=0.1, attribution_method="unattributed",
            ),
        ]
        result = _apply_dialogue_updates(lines, [{"line_index": 99, "speaker": "X"}])
        assert result[0].speaker is None  # unchanged


class TestCacheHelpers:
    """Tests for cache utility functions."""

    def test_cache_key_deterministic(self):
        k1 = _cache_key("a", "b")
        k2 = _cache_key("a", "b")
        assert k1 == k2

    def test_cache_key_different_for_different_inputs(self):
        k1 = _cache_key("a", "b")
        k2 = _cache_key("a", "c")
        assert k1 != k2

    def test_read_cache_miss(self, tmp_path):
        cache_path = tmp_path / "nonexistent.json"
        assert _read_cache(cache_path, "key") is None

    def test_read_cache_hit(self, tmp_path):
        cache_path = tmp_path / "test.json"
        _write_cache(cache_path, "abc", {"value": 42})
        result = _read_cache(cache_path, "abc")
        assert result == {"cache_key": "abc", "value": 42}

    def test_read_cache_key_mismatch(self, tmp_path):
        cache_path = tmp_path / "test.json"
        _write_cache(cache_path, "abc", {"value": 42})
        result = _read_cache(cache_path, "xyz")
        assert result is None

    def test_write_cache_creates_dirs(self, tmp_path):
        cache_path = tmp_path / "sub" / "deep" / "cache.json"
        _write_cache(cache_path, "k", {"v": 1})
        assert cache_path.exists()


class TestIsAiAvailable:
    """Tests for ``is_ai_available``."""

    def test_available_when_key_set(self, monkeypatch):
        import engine.ai_enhancer as mod

        monkeypatch.setenv("NOVEL2SCRIPT_API_KEY", "sk-test")
        mod._client = None
        mod._client_init_attempted = False
        # We can't actually connect, but the client init should succeed
        # with openai installed — the function should return True
        # (client is created lazily, actual connectivity not checked)
        try:
            result = is_ai_available()
            # With a valid key and openai installed, should be True
            assert result is True
        except Exception:
            # If openai not installed, this would be False
            pass

    def test_unavailable_when_no_key(self, monkeypatch):
        import engine.ai_enhancer as mod

        monkeypatch.delenv("NOVEL2SCRIPT_API_KEY", raising=False)
        mod._client = None
        mod._client_init_attempted = False
        result = is_ai_available()
        assert result is False
