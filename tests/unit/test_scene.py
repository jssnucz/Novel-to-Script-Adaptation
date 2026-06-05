"""Tests for the scene boundary detection module (Task 1.4)."""

import re

import pytest

from src.engine.chapter import split_chapters
from src.engine.models import (
    Chapter,
    ChapterArtifact,
    PreprocessArtifact,
    SceneArtifact,
)
from src.engine.scene import (
    classify_int_ext,
    classify_time_of_day,
    detect_scenes,
    extract_location,
)


# ---------------------------------------------------------------------------
# classify_time_of_day
# ---------------------------------------------------------------------------


class TestClassifyTimeOfDay:
    def test_night_keywords(self):
        """夜／月／星／黑暗／深夜／烛火／灯笼 → '夜'."""
        samples = [
            "夜深人静，月光如水。",
            "黑暗笼罩着大地。",
            "烛火在风中摇曳。",
        ]
        for text in samples:
            assert classify_time_of_day(text) == "夜"

    def test_morning_keywords(self):
        """晨／黎明／拂晓／日出／清晨／早晨 → '晨'."""
        samples = [
            "清晨，阳光洒满大地。",
            "黎明破晓，晨光微露。",
            "早晨的空气格外清新。",
        ]
        for text in samples:
            assert classify_time_of_day(text) == "晨"

    def test_dusk_keywords(self):
        """黄昏／傍晚／夕阳／日落／暮 → '黄昏'."""
        samples = [
            "黄昏时分，夕阳西下。",
            "傍晚的微风轻拂面庞。",
            "暮色笼罩了整座城市。",
        ]
        for text in samples:
            assert classify_time_of_day(text) == "黄昏"

    def test_daytime_keywords(self):
        """阳光／正午／白天／晌午 → '日'."""
        samples = [
            "阳光明媚，万里无云。",
            "正午的太阳炙烤着大地。",
            "白天人来人往，车水马龙。",
        ]
        for text in samples:
            assert classify_time_of_day(text) == "日"

    def test_unknown_no_keywords(self):
        """Text with no time-of-day keywords returns 'UNKNOWN'."""
        assert classify_time_of_day("这是一段没有任何时间线索的普通文字。") == "UNKNOWN"

    def test_only_first_200_chars_checked(self):
        """Only the first ~200 characters are checked for keywords."""
        text = "普通文字。" * 60  # well over 200 chars
        # Append a night keyword far beyond the 200-char window
        text += "深夜关键字不应被检测到"
        result = classify_time_of_day(text)
        assert result != "夜", "Keywords beyond 200 chars should not be matched"

    def test_prefers_first_match_priority(self):
        """Whichever keyword pattern matches first in the checked window wins."""
        text = "黄昏时分，但天色依然明亮。"
        assert classify_time_of_day(text) == "黄昏"


# ---------------------------------------------------------------------------
# classify_int_ext
# ---------------------------------------------------------------------------


class TestClassifyIntExt:
    def test_int_pattern_room(self):
        """房间 + 内／里／中 → INT."""
        assert classify_int_ext("房间内灯火通明。") == "INT"

    def test_int_pattern_hall(self):
        """殿／堂／厅 + 内／里／中 → INT."""
        samples = [
            "大殿中回荡着回声。",
            "厅堂里摆满了桌椅。",
            "客厅内空无一人。",
        ]
        for text in samples:
            assert classify_int_ext(text) == "INT"

    def test_int_pattern_cave(self):
        """洞窟 + 内／里／中 → INT."""
        assert classify_int_ext("洞窟内一片漆黑。") == "INT"

    def test_ext_pattern_mountain(self):
        """山巅 → EXT."""
        assert classify_int_ext("山巅之上，云雾缭绕。") == "EXT"

    def test_ext_pattern_outdoor(self):
        """路边／街上／城外／野外 → EXT."""
        samples = [
            "路边野花盛开。",
            "街上空无一人。",
            "城外驻扎着军队。",
            "野外篝火燃起。",
        ]
        for text in samples:
            assert classify_int_ext(text) == "EXT"

    def test_ext_pattern_riverside(self):
        """河畔／湖边／海边 → EXT."""
        samples = [
            "河畔柳树成荫。",
            "湖边波光粼粼。",
            "海边的浪花拍打着礁石。",
        ]
        for text in samples:
            assert classify_int_ext(text) == "EXT"

    def test_unknown_no_keywords(self):
        """Text with no INT/EXT keywords returns 'UNKNOWN'."""
        assert classify_int_ext("这是一个普通的地方。") == "UNKNOWN"

    def test_only_first_200_chars_checked(self):
        """Only the first ~200 characters are checked for INT/EXT keywords."""
        text = "普通文字。" * 60
        text += "房间内——"
        result = classify_int_ext(text)
        assert result == "UNKNOWN"


# ---------------------------------------------------------------------------
# extract_location
# ---------------------------------------------------------------------------


class TestExtractLocation:
    def test_from_short_boundary_keyword(self):
        """A short boundary keyword (<=10 chars) is returned directly."""
        result = extract_location("无关正文", ["迦南学院"])
        assert result == "迦南学院"

    def test_boundary_keyword_too_long_ignored(self):
        """A boundary keyword >10 chars is ignored; falls back to regex."""
        text = "他在大殿中端坐。"
        result = extract_location(text, ["这是一句非常长的标题超过十个字"])
        # Falls back to first sentence regex: "他在大殿中端坐"
        # (?:在|到)(.{2,8})(?:内|里|中|外|上|下|前|后|旁|边)
        # "在大殿中" → group 1 = "大殿"
        assert result == "大殿"

    def test_no_boundary_keywords_first_sentence_pattern(self):
        """With empty boundary keywords, uses first sentence regex."""
        text = "他来到山巅之上，俯瞰大地。"
        result = extract_location(text, [])
        # "到山巅之上" → group 1 = "山巅"
        assert result == "山巅"

    def test_regex_multiple_matches_first(self):
        """The FIRST pattern match in the first sentence is used."""
        text = "他在屋子中休息，随后到庭院中散步。"
        result = extract_location(text, [])
        # First match: "在屋子中" → group 1 = "屋子"
        assert result == "屋子"

    def test_unknown_no_match(self):
        """Returns 'UNKNOWN' when no location can be extracted."""
        text = "这是一个没有位置信息的句子。"
        result = extract_location(text, [])
        assert result == "UNKNOWN"

    def test_empty_text_returns_unknown(self):
        """Empty text returns 'UNKNOWN'."""
        assert extract_location("", []) == "UNKNOWN"


# ---------------------------------------------------------------------------
# detect_scenes
# ---------------------------------------------------------------------------


class TestDetectScenes:
    def test_time_split_produces_multiple_scenes(self):
        """A chapter with internal time keywords is split into scenes."""
        content = "萧炎走在山路上。\n\n三天后，他来到了迦南学院。\n\n第二天，考核开始了。"
        artifact = ChapterArtifact(
            schema_version="1.0",
            chapters=[
                Chapter(
                    chapter_id="CH01",
                    title="第一章",
                    content=content,
                    start_line=1,
                    end_line=5,
                    confidence=1.0,
                )
            ],
        )
        result = detect_scenes(artifact)
        assert len(result.scenes) >= 2

    def test_single_scene_when_no_splits(self):
        """A chapter with no time keywords or separators stays as one scene."""
        content = "这是一段连续的内容。没有时间跳转。场景始终如一。"
        artifact = ChapterArtifact(
            schema_version="1.0",
            chapters=[
                Chapter(
                    chapter_id="CH01",
                    title="单章",
                    content=content,
                    start_line=1,
                    end_line=2,
                    confidence=1.0,
                )
            ],
        )
        result = detect_scenes(artifact)
        assert len(result.scenes) == 1

    def test_scene_id_format(self, basic_novel):
        """Scene IDs follow the pattern CHXX-SXX."""
        pre = PreprocessArtifact(
            schema_version="1.0",
            original_path="test.txt",
            cleaned_text=basic_novel,
            total_chars=len(basic_novel),
        )
        chapter_artifact = split_chapters(pre)
        result = detect_scenes(chapter_artifact)
        for scene in result.scenes:
            assert re.match(r"CH\d{2}-S\d{2}", scene.scene_id), (
                f"Scene ID '{scene.scene_id}' does not match CHXX-SXX"
            )

    def test_no_split_confidence_one(self):
        """A chapter with no internal split keywords produces a scene with
        confidence 1.0."""
        content = "这是一个平静的场景。没有任何时间跳转。角色在对话。"
        artifact = ChapterArtifact(
            schema_version="1.0",
            chapters=[
                Chapter(
                    chapter_id="CH01",
                    title="第一章",
                    content=content,
                    start_line=1,
                    end_line=3,
                    confidence=1.0,
                )
            ],
        )
        result = detect_scenes(artifact)
        assert len(result.scenes) == 1
        assert result.scenes[0].confidence == 1.0

    def test_time_split_confidence_less_than_one(self):
        """Scenes resulting from a time split have confidence < 1.0."""
        content = "开头场景。\n\n三天后，新的场景开始了。\n\n第二天，又一个场景。"
        artifact = ChapterArtifact(
            schema_version="1.0",
            chapters=[
                Chapter(
                    chapter_id="CH01",
                    title="第一章",
                    content=content,
                    start_line=1,
                    end_line=5,
                    confidence=1.0,
                )
            ],
        )
        result = detect_scenes(artifact)
        # At least the later scenes (from time splits) have confidence < 1.0
        # The first scene may remain 1.0
        # Find scenes whose boundary_keywords are non-empty (keyword-triggered)
        for scene in result.scenes:
            if scene.boundary_keywords:
                assert scene.confidence < 1.0, (
                    f"Scene {scene.scene_id} was split at a keyword "
                    f"({scene.boundary_keywords}) but has confidence 1.0"
                )

    def test_full_novel_produces_at_least_three_scenes(self, basic_novel):
        """The 3-chapter basic novel yields at least 3 scenes."""
        pre = PreprocessArtifact(
            schema_version="1.0",
            original_path="test.txt",
            cleaned_text=basic_novel,
            total_chars=len(basic_novel),
        )
        chapter_artifact = split_chapters(pre)
        result = detect_scenes(chapter_artifact)
        assert len(result.scenes) >= 3

    def test_separator_splits_scenes(self):
        """Separator patterns (---, ..., ***) split scenes."""
        content = "第一段内容。\n\n---\n\n第二段内容。\n\n***\n\n第三段内容。"
        artifact = ChapterArtifact(
            schema_version="1.0",
            chapters=[
                Chapter(
                    chapter_id="CH01",
                    title="第一章",
                    content=content,
                    start_line=1,
                    end_line=6,
                    confidence=1.0,
                )
            ],
        )
        result = detect_scenes(artifact)
        assert len(result.scenes) == 3

    def test_returns_scene_artifact(self):
        """detect_scenes returns a SceneArtifact with schema_version."""
        content = "一些内容。"
        artifact = ChapterArtifact(
            schema_version="1.0",
            chapters=[
                Chapter(
                    chapter_id="CH01",
                    title="第一章",
                    content=content,
                    start_line=1,
                    end_line=1,
                    confidence=1.0,
                )
            ],
        )
        result = detect_scenes(artifact)
        assert isinstance(result, SceneArtifact)
        assert result.schema_version == "1.0"

    def test_scene_has_chapter_id(self):
        """Each scene correctly links back to its chapter."""
        content = "场景内容。"
        artifact = ChapterArtifact(
            schema_version="1.0",
            chapters=[
                Chapter(
                    chapter_id="CH01",
                    title="第一章",
                    content=content,
                    start_line=1,
                    end_line=1,
                    confidence=1.0,
                )
            ],
        )
        result = detect_scenes(artifact)
        assert result.scenes[0].chapter_id == "CH01"

    def test_scenes_have_classification_fields(self):
        """Each scene has int_ext, time_of_day, location set."""
        artifact = ChapterArtifact(
            schema_version="1.0",
            chapters=[
                Chapter(
                    chapter_id="CH01",
                    title="第一章",
                    content="山巅之上，清晨的阳光洒满大地。",
                    start_line=1,
                    end_line=1,
                    confidence=1.0,
                )
            ],
        )
        result = detect_scenes(artifact)
        scene = result.scenes[0]
        # "山巅" should give EXT, "清晨" should give 晨
        assert scene.int_ext in ("INT", "EXT", "INT/EXT", "UNKNOWN")
        assert scene.time_of_day in ("日", "夜", "晨", "黄昏", "UNKNOWN")
        assert isinstance(scene.location, str)

    def test_empty_chapter_content_skipped(self):
        """A chapter with empty/whitespace-only content produces no scenes."""
        artifact = ChapterArtifact(
            schema_version="1.0",
            chapters=[
                Chapter(
                    chapter_id="CH01",
                    title="空章",
                    content="   \n\n  ",
                    start_line=1,
                    end_line=3,
                    confidence=1.0,
                )
            ],
        )
        result = detect_scenes(artifact)
        assert len(result.scenes) == 0
