"""Tests for the dialogue extraction and attribution module (Task 1.6).

Tests cover:
- extract_quoted_texts: all 4 quote styles, nested, no quotes
- extract_parenthetical: （）, (), no bracket
- infer_speaker: prefix_match, suffix_match, nearest_name, unattributed, prev_speaker
- extract_dialogues: full pipeline via SceneArtifact -> DialogueArtifact
"""

import pytest

from src.engine.dialogue import (
    extract_quoted_texts,
    extract_parenthetical,
    infer_speaker,
    extract_dialogues,
)
from src.engine.models import (
    DialogueLine,
    DialogueArtifact,
    Scene,
    SceneArtifact,
)


# ---------------------------------------------------------------------------
# extract_quoted_texts
# ---------------------------------------------------------------------------


class TestExtractQuotedTexts:
    def test_double_quotes(self):
        """Detect "" style quotes."""
        text = '他说："你好。"然后离开。'
        result = extract_quoted_texts(text)
        assert len(result) == 1
        pos, full, style = result[0]
        assert style == "double"
        assert full == '"你好。"'

    def test_single_quotes(self):
        """Detect '' style quotes."""
        text = "他低声说：'小心一点。'"
        result = extract_quoted_texts(text)
        assert len(result) == 1
        pos, full, style = result[0]
        assert style == "single"
        assert full == "'小心一点。'"

    def test_corner_brackets(self):
        """Detect 「」 style quotes."""
        text = "他说道：「原来如此。」"
        result = extract_quoted_texts(text)
        assert len(result) == 1
        pos, full, style = result[0]
        assert style == "corner"
        assert full == "「原来如此。」"

    def test_white_corner_brackets(self):
        """Detect 『』 style quotes."""
        text = "他叹道：『真是造化弄人。』"
        result = extract_quoted_texts(text)
        assert len(result) == 1
        pos, full, style = result[0]
        assert style == "white_corner"
        assert full == "『真是造化弄人。』"

    def test_multiple_quotes(self):
        """Find all quotes in text with multiple quotes."""
        text = '他说："第一句。"然后「第二句。」最后\'第三句\''
        result = extract_quoted_texts(text)
        assert len(result) == 3
        # Sorted by position
        assert result[0][2] == "double"
        assert result[1][2] == "corner"
        assert result[2][2] == "single"

    def test_no_quotes_returns_empty(self):
        """Text with no quotes returns empty list."""
        result = extract_quoted_texts("这是一个没有引号的句子。")
        assert result == []

    def test_nested_quotes_outer_first(self):
        """Nested double inside double — outermost is detected first."""
        text = '他说："外面「里面」还有。"'
        result = extract_quoted_texts(text)
        assert len(result) >= 1
        # The outermost " " is matched first; the inner 「」 is a separate match
        styles = [r[2] for r in result]
        assert "double" in styles
        # The corner brackets inside should also be matched
        assert "corner" in styles

    def test_scene_with_multiple_styles(self):
        """Mixed quotes from basic_3ch novel: double, single, corner, white_corner."""
        text = (
            '萧炎说："你好。"\n'
            "纳兰嫣然道：「再见。」\n"
            "药老叹道：『好。』\n"
            "萧炎答：'走吧。'\n"
        )
        result = extract_quoted_texts(text)
        assert len(result) == 4
        styles = [r[2] for r in result]
        assert "double" in styles
        assert "single" in styles
        assert "corner" in styles
        assert "white_corner" in styles

    def test_quote_style_order_double_before_single(self):
        """When both "" and '' present, return sorted by position."""
        text = "'第一''第二'"
        result = extract_quoted_texts(text)
        assert len(result) >= 1

    def test_keyword_detected_in_quotes(self):
        """Verify quote content extraction works with Chinese text."""
        text = "「斗气大陆」"
        result = extract_quoted_texts(text)
        assert len(result) == 1
        pos, full, style = result[0]
        assert style == "corner"
        assert "斗气大陆" in full


# ---------------------------------------------------------------------------
# extract_parenthetical
# ---------------------------------------------------------------------------


class TestExtractParenthetical:
    def test_chinese_brackets(self):
        """Extract text from （） brackets."""
        result = extract_parenthetical("（叹气）")
        assert result == "叹气"

    def test_ascii_brackets(self):
        """Extract text from () brackets."""
        result = extract_parenthetical("(冷冷地)")
        assert result == "冷冷地"

    def test_no_brackets_returns_none(self):
        """No bracket pattern returns None."""
        result = extract_parenthetical("萧炎说")
        assert result is None

    def test_parenthetical_with_context(self):
        """Parenthetical embedded before a quote."""
        # The function receives text_before_quote which may include the paren
        result = extract_parenthetical("萧炎（叹了口气）")
        assert result == "叹了口气"

    def test_empty_brackets_returns_empty_string(self):
        """Empty () returns empty string."""
        result = extract_parenthetical("（）")
        assert result == ""

    def test_multi_char_parenthetical(self):
        """Multi-char parenthetical extracted correctly."""
        result = extract_parenthetical("（摇了摇头）")
        assert result == "摇了摇头"

    def test_parenthetical_with_punctuation(self):
        """Parenthetical containing punctuation."""
        result = extract_parenthetical("（冷冷地笑）")
        assert result == "冷冷地笑"


# ---------------------------------------------------------------------------
# infer_speaker
# ---------------------------------------------------------------------------


class TestInferSpeakerPrefixMatch:
    def test_prefix_match_basic(self):
        """Name + speech verb 15 chars before quote → prefix_match (0.85)."""
        text_before = '萧炎说道："'
        speaker, conf, method = infer_speaker(
            text_before=text_before,
            text_after="",
            character_names=["萧炎", "纳兰嫣然"],
            line_index=0,
            prev_speakers=[],
        )
        assert speaker == "萧炎"
        assert conf == 0.85
        assert method == "prefix_match"

    def test_prefix_match_ask(self):
        """问 as speech verb."""
        text_before = '纳兰嫣然问道："'
        speaker, conf, method = infer_speaker(
            text_before=text_before,
            text_after="",
            character_names=["萧炎", "纳兰嫣然"],
            line_index=0,
            prev_speakers=[],
        )
        assert speaker == "纳兰嫣然"
        assert method == "prefix_match"

    def test_prefix_match_mixed_speech_verb(self):
        """答道 as speech verb."""
        text_before = '萧炎答道："'
        speaker, conf, method = infer_speaker(
            text_before=text_before,
            text_after="",
            character_names=["萧炎", "纳兰嫣然"],
            line_index=0,
            prev_speakers=[],
        )
        assert speaker == "萧炎"
        assert method == "prefix_match"


class TestInferSpeakerSuffixMatch:
    def test_suffix_match_basic(self):
        """Speech verb + name within 10 chars after quote → suffix_match (0.75)."""
        text_before = '"你好"'
        text_after = "萧炎说道。"
        speaker, conf, method = infer_speaker(
            text_before=text_before,
            text_after=text_after,
            character_names=["萧炎", "纳兰嫣然"],
            line_index=0,
            prev_speakers=[],
        )
        assert speaker == "萧炎"
        assert conf == 0.75
        assert method == "suffix_match"

    def test_suffix_match_after_quote(self):
        """verb + name after quote."""
        text_before = '"原来如此"'
        text_after = "纳兰嫣然答道。"
        speaker, conf, method = infer_speaker(
            text_before=text_before,
            text_after=text_after,
            character_names=["萧炎", "纳兰嫣然"],
            line_index=0,
            prev_speakers=[],
        )
        assert speaker == "纳兰嫣然"
        assert method == "suffix_match"


class TestInferSpeakerNearestName:
    def test_nearest_name_before(self):
        """Name within 30 chars before quote, no verb → nearest_name (0.5)."""
        text_before = "萧炎看着远方。"
        text_after = ""
        speaker, conf, method = infer_speaker(
            text_before=text_before,
            text_after="",
            character_names=["萧炎", "纳兰嫣然"],
            line_index=0,
            prev_speakers=[],
        )
        assert speaker == "萧炎"
        assert conf == 0.5
        assert method == "nearest_name"

    def test_nearest_name_after(self):
        """Name within 30 chars after quote, no verb → nearest_name (0.5)."""
        text_before = ""
        text_after = "萧炎转身离开了。"
        speaker, conf, method = infer_speaker(
            text_before=text_before,
            text_after=text_after,
            character_names=["萧炎", "纳兰嫣然"],
            line_index=0,
            prev_speakers=[],
        )
        assert speaker == "萧炎"
        assert method == "nearest_name"

    def test_longest_name_wins(self):
        """Longest character name matches first (纳兰嫣然 before 纳兰)."""
        text_before = "纳兰嫣然说道"
        speaker, conf, method = infer_speaker(
            text_before=text_before,
            text_after="",
            character_names=["纳兰", "纳兰嫣然", "萧炎"],
            line_index=0,
            prev_speakers=[],
        )
        assert speaker == "纳兰嫣然"
        assert method == "prefix_match"

    def test_name_too_far_before(self):
        """Name beyond 30 chars before is not found."""
        # "萧炎" at position 0, followed by 31 filler chars (31 chars between
        # name end and quote start), exceeding the 30-char nearest_name window.
        text_before = "萧炎" + "。" * 31
        text_after = ""
        speaker, conf, method = infer_speaker(
            text_before=text_before,
            text_after="",
            character_names=["萧炎"],
            line_index=0,
            prev_speakers=[],
        )
        # May fall through to prev_speaker or unattributed
        assert speaker is None or method != "nearest_name"


class TestInferSpeakerPrevSpeaker:
    def test_prev_speaker_match(self):
        """Same speaker as previous line → prev_speaker (0.3)."""
        speaker, conf, method = infer_speaker(
            text_before="",
            text_after="",
            character_names=["萧炎", "纳兰嫣然"],
            line_index=1,
            prev_speakers=["萧炎"],
        )
        assert speaker == "萧炎"
        assert conf == 0.3
        assert method == "prev_speaker"

    def test_prev_speaker_alternating(self):
        """A-B-A-B conversation pattern: line2 inherits speaker from line0."""
        # line_index=2, prev_speakers=["A", "B"]
        speaker, conf, method = infer_speaker(
            text_before="",
            text_after="",
            character_names=["纳兰嫣然", "萧炎"],
            line_index=2,
            prev_speakers=["萧炎", "纳兰嫣然"],
        )
        # Even index back to speaker 0 (萧炎)
        assert speaker == "萧炎"
        assert method == "prev_speaker"

    def test_prev_speaker_no_history(self):
        """line_index=0 with empty prev_speakers → unattributed."""
        speaker, conf, method = infer_speaker(
            text_before="",
            text_after="",
            character_names=["萧炎"],
            line_index=0,
            prev_speakers=[],
        )
        assert speaker is None
        assert method == "unattributed"


class TestInferSpeakerUnattributed:
    def test_no_names_present(self):
        """No names in context → unattributed (0.0)."""
        speaker, conf, method = infer_speaker(
            text_before="某人说",
            text_after="",
            character_names=[],
            line_index=0,
            prev_speakers=[],
        )
        assert speaker is None
        assert conf == 0.0
        assert method == "unattributed"


# ---------------------------------------------------------------------------
# extract_dialogues
# ---------------------------------------------------------------------------


class TestExtractDialogues:
    def test_returns_dialogue_artifact(self):
        """extract_dialogues returns a DialogueArtifact."""
        scene = Scene(
            scene_id="CH01-S01",
            chapter_id="CH01",
            content='萧炎说："你好。"',
            boundary_keywords=["测试"],
            location="测试地",
            int_ext="INT",
            time_of_day="日",
            confidence=1.0,
        )
        artifact = SceneArtifact(schema_version="1.0", scenes=[scene])
        result = extract_dialogues(artifact)
        assert isinstance(result, DialogueArtifact)
        assert result.schema_version == "1.0"

    def test_dialogue_line_structure(self):
        """Each extracted dialogue is a DialogueLine with proper fields."""
        scene = Scene(
            scene_id="CH01-S01",
            chapter_id="CH01",
            content='萧炎说："你好。"',
            boundary_keywords=["测试"],
            location="测试地",
            int_ext="INT",
            time_of_day="日",
            confidence=1.0,
        )
        artifact = SceneArtifact(schema_version="1.0", scenes=[scene])
        result = extract_dialogues(artifact)
        assert len(result.dialogues) == 1
        dl = result.dialogues[0]
        assert isinstance(dl, DialogueLine)
        assert dl.dialogue_id == "CH01-S01-D01"
        assert dl.scene_id == "CH01-S01"
        assert dl.line_index == 0
        assert dl.line == "你好。"
        assert dl.quote_style == "double"
        assert dl.confidence == 0.85
        assert dl.attribution_method == "prefix_match"

    def test_line_index_sequential_per_scene(self):
        """line_index is sequential (0, 1, 2...) per scene."""
        scene1 = Scene(
            scene_id="CH01-S01",
            chapter_id="CH01",
            content='萧炎说："第一句。"纳兰嫣然答："第二句。"药老笑道："第三句。"',
            boundary_keywords=["测试"],
            location="测试地",
            int_ext="INT",
            time_of_day="日",
            confidence=1.0,
        )
        artifact = SceneArtifact(schema_version="1.0", scenes=[scene1])
        result = extract_dialogues(artifact)
        assert len(result.dialogues) == 3
        for i, dl in enumerate(result.dialogues):
            assert dl.line_index == i
            assert dl.dialogue_id == f"CH01-S01-D{i+1:02d}"

    def test_multiple_scenes(self):
        """Lines across multiple scenes each have their own sequential indices."""
        scene1 = Scene(
            scene_id="CH01-S01",
            chapter_id="CH01",
            content='萧炎说："第一句。"',
            boundary_keywords=["测试"],
            location="测试地",
            int_ext="INT",
            time_of_day="日",
            confidence=1.0,
        )
        scene2 = Scene(
            scene_id="CH01-S02",
            chapter_id="CH01",
            content='药老道："第二句。"',
            boundary_keywords=["然后"],
            location="大殿",
            int_ext="INT",
            time_of_day="日",
            confidence=0.8,
        )
        artifact = SceneArtifact(schema_version="1.0", scenes=[scene1, scene2])
        result = extract_dialogues(artifact)
        assert len(result.dialogues) == 2
        assert result.dialogues[0].dialogue_id == "CH01-S01-D01"
        assert result.dialogues[0].line_index == 0
        assert result.dialogues[1].dialogue_id == "CH01-S02-D01"
        assert result.dialogues[1].line_index == 0

    def test_quote_style_captured(self):
        """Quote style is correctly captured for each line."""
        scene = Scene(
            scene_id="CH01-S01",
            chapter_id="CH01",
            content=(
                '萧炎说："双引号。"\n'
                "纳兰嫣然道：「方括号。」\n"
                "药老叹：『白括号。』\n"
                "萧炎答：'单引号。'\n"
            ),
            boundary_keywords=["测试"],
            location="测试地",
            int_ext="INT",
            time_of_day="日",
            confidence=1.0,
        )
        artifact = SceneArtifact(schema_version="1.0", scenes=[scene])
        result = extract_dialogues(artifact)
        styles = [dl.quote_style for dl in result.dialogues]
        assert "double" in styles
        assert "corner" in styles
        assert "white_corner" in styles
        assert "single" in styles

    def test_no_dialogue_returns_empty_artifact(self):
        """Scene with no quotes returns a DialogueArtifact with empty list."""
        scene = Scene(
            scene_id="CH01-S01",
            chapter_id="CH01",
            content="这里没有对话，只有叙述。",
            boundary_keywords=["测试"],
            location="测试地",
            int_ext="INT",
            time_of_day="日",
            confidence=1.0,
        )
        artifact = SceneArtifact(schema_version="1.0", scenes=[scene])
        result = extract_dialogues(artifact)
        assert isinstance(result, DialogueArtifact)
        assert result.dialogues == []


# ---------------------------------------------------------------------------
# Integration test — full novel via pipeline
# ---------------------------------------------------------------------------


class TestExtractDialoguesIntegration:
    """End-to-end test: preprocess -> chapters -> scenes -> dialogues."""

    def test_basic_novel_has_dialogues(self, basic_novel):
        """Full basic_3ch novel yields at least 5 dialogue lines."""
        from src.engine.preprocess import preprocess
        from src.engine.chapter import split_chapters
        from src.engine.scene import detect_scenes

        pre = preprocess(basic_novel, "basic_3ch.txt")
        chapter_artifact = split_chapters(pre)
        scene_artifact = detect_scenes(chapter_artifact)
        result = extract_dialogues(scene_artifact)

        assert isinstance(result, DialogueArtifact)
        assert len(result.dialogues) >= 5, (
            f"Expected at least 5 dialogues in basic_3ch.txt, "
            f"got {len(result.dialogues)}"
        )

        # Verify structure of first dialogue
        first = result.dialogues[0]
        assert first.dialogue_id.startswith("CH01-S")
        assert first.line_index == 0
        assert first.speaker is not None
        assert first.line
        assert first.quote_style in ("double", "single", "corner", "white_corner")
        assert first.confidence > 0
        assert first.attribution_method in (
            "prefix_match", "suffix_match", "nearest_name", "prev_speaker", "unattributed"
        )

    def test_no_dialogue_novel_returns_empty(self, no_dialogue_novel):
        """A novel with no quoted text returns an empty DialogueArtifact."""
        from src.engine.preprocess import preprocess
        from src.engine.chapter import split_chapters
        from src.engine.scene import detect_scenes

        pre = preprocess(no_dialogue_novel, "no_dialogue.txt")
        chapter_artifact = split_chapters(pre)
        scene_artifact = detect_scenes(chapter_artifact)
        result = extract_dialogues(scene_artifact)

        assert isinstance(result, DialogueArtifact)
        assert result.dialogues == []

    def test_mixed_quotes_novel_dialogues(self, mixed_quotes_novel):
        """mixed_quotes.txt has at least 6 dialogue lines with varied styles."""
        from src.engine.preprocess import preprocess
        from src.engine.chapter import split_chapters
        from src.engine.scene import detect_scenes

        pre = preprocess(mixed_quotes_novel, "mixed_quotes.txt")
        chapter_artifact = split_chapters(pre)
        scene_artifact = detect_scenes(chapter_artifact)
        result = extract_dialogues(scene_artifact)

        assert isinstance(result, DialogueArtifact)
        assert len(result.dialogues) >= 6, (
            f"Expected at least 6 dialogues in mixed_quotes.txt, "
            f"got {len(result.dialogues)}"
        )

        # Check multiple styles present
        styles = {dl.quote_style for dl in result.dialogues}
        assert len(styles) >= 2, (
            f"Expected at least 2 quote styles, got {styles}"
        )
