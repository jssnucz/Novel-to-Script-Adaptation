"""Scene boundary detection and time-of-day / INT-EXT classification for
the Novel-to-Script Adaptation pipeline.

Every function in this module is a pure function with no side effects and no
file I/O.

- ``classify_time_of_day`` — classify a text snippet as 日/夜/晨/黄昏/UNKNOWN
- ``classify_int_ext`` — classify a text snippet as INT/EXT/UNKNOWN
- ``extract_location`` — extract location from boundary keywords or text regex
- ``detect_scenes`` — split chapter content into Scene objects
"""

import re

from engine.models import (
    Chapter,
    ChapterArtifact,
    Scene,
    SceneArtifact,
)

# ---------------------------------------------------------------------------
# Time-of-day classification
# ---------------------------------------------------------------------------

# Ordered list of (pattern, label). First match wins.
_TIME_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Night (returns "夜")
    (re.compile(r"(?:夜|月|星|黑暗|深夜|烛火|灯笼)"), "夜"),
    # Morning (returns "晨")
    (re.compile(r"(?:晨|黎明|拂晓|日出|清晨|早晨|朝)"), "晨"),
    # Dusk (returns "黄昏")
    (re.compile(r"(?:黄昏|傍晚|夕阳|日落|暮|夕照)"), "黄昏"),
    # Daytime (returns "日")
    (re.compile(r"(?:阳光|正午|白天|白日|日中|日光|晌午)"), "日"),
]


def classify_time_of_day(text: str) -> str:
    """Classify the time of day from the first ~200 characters of *text*.

    Parameters
    ----------
    text : str
        Scene text to classify.

    Returns
    -------
    str
        One of ``"日"``, ``"夜"``, ``"晨"``, ``"黄昏"``, or ``"UNKNOWN"``.
    """
    sample = text[:200]
    for pattern, label in _TIME_PATTERNS:
        if pattern.search(sample):
            return label
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# INT / EXT classification
# ---------------------------------------------------------------------------

_INT_PATTERNS: list[re.Pattern] = [
    # Location word + 内/里/中 — e.g. 房间内, 大殿中, 洞窟里
    re.compile(
        r"(?:房间|屋|殿|堂|厅|室|楼|阁|客栈|酒楼|店铺|洞窟|帐篷)"
        r"(?:内|里|中)"
    ),
]

_EXT_PATTERNS: list[re.Pattern] = [
    re.compile(r"山巅"),
    re.compile(r"路边"),
    re.compile(r"街上"),
    re.compile(r"城外"),
    re.compile(r"野外"),
    re.compile(r"林中|林间"),
    re.compile(r"河畔|湖边|海边"),
]


def classify_int_ext(text: str) -> str:
    """Classify the scene environment as interior (INT) or exterior (EXT).

    Parameters
    ----------
    text : str
        Scene text to classify.

    Returns
    -------
    str
        One of ``"INT"``, ``"EXT"``, or ``"UNKNOWN"``.
    """
    sample = text[:200]

    # Check INT first (more specific)
    for pattern in _INT_PATTERNS:
        if pattern.search(sample):
            return "INT"

    # Check EXT
    for pattern in _EXT_PATTERNS:
        if pattern.search(sample):
            return "EXT"

    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Location extraction
# ---------------------------------------------------------------------------

_LOCATION_RE = re.compile(r"(?:在|到)(.{2,8}?)(?:内|里|中|外|上|下|前|后|旁|边)")

# Boundary keywords that are time/special markers, NOT locations.
# These are the keywords from _TIME_SPLITS and _SEPARATOR_RE patterns.
_NON_LOCATION_KEYWORDS: frozenset[str] = frozenset({
    "三天后", "数日后", "几日后",
    "第二天", "次日", "翌日",
    "与此同时", "另一方面", "镜头一转",
    "夜已深沉", "夜深人静", "深夜",
    "清晨", "早晨", "黎明", "拂晓",
    "黄昏", "傍晚", "夕阳西下", "暮色",
})


def extract_location(text: str, boundary_keywords: list[str]) -> str:
    """Extract a location name from boundary keywords or the first sentence.

    Parameters
    ----------
    text : str
        Scene text to analyse.
    boundary_keywords : list[str]
        Keywords that triggered the scene boundary (e.g. ``["三天后"]``).
        Time keywords and separator markers are filtered out.

    Returns
    -------
    str
        Location name or ``"UNKNOWN"`` if nothing could be determined.
    """
    # Try boundary keywords first, skipping time markers and separators
    for kw in boundary_keywords:
        if len(kw) <= 10 and kw not in _NON_LOCATION_KEYWORDS:
            return kw

    # Fall back to first‑sentence regex
    # Take the first "sentence" up to 。！？ or line break
    first_sentence = text.split("。")[0].split("！")[0].split("？")[0].split("\n")[0]
    m = _LOCATION_RE.search(first_sentence)
    if m:
        location = m.group(1)
        # Strip trailing modifier particles
        location = re.sub(r"[之地的]$", "", location)
        return location

    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Scene split — time keywords and separators
# ---------------------------------------------------------------------------

# Time‑split keyword groups with their associated confidence
# Ordered by regex alternation; each entry maps a pattern to its confidence.
_TIME_SPLITS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"三天后|数日后|几日后"), 0.7),
    (re.compile(r"第二天|次日|翌日"), 0.8),
    (re.compile(r"与此同时|另一方面|镜头一转"), 0.6),
    (re.compile(r"夜已深沉|夜深人静|深夜"), 0.7),
    (re.compile(r"清晨|早晨|黎明|拂晓"), 0.7),
    (re.compile(r"黄昏|傍晚|夕阳西下|暮色"), 0.7),
]

# Separator patterns that indicate a scene break (standalone lines).
_SEPARATOR_RE = re.compile(r"^[.\-*]{3,}\s*$", re.MULTILINE)

_SEPARATOR_CONFIDENCE = 0.5


def _find_split_points(content: str) -> list[tuple[int, str, float]]:
    """Find all scene split points in *content*.

    Returns a list of ``(position, keyword, confidence)`` sorted by position.
    Positions at index 0 are excluded (cannot split at the very beginning).
    """
    points: list[tuple[int, str, float]] = []

    # Time‑keyword matches
    for pattern, confidence in _TIME_SPLITS:
        for m in pattern.finditer(content):
            pos = m.start()
            if pos > 0:  # skip splits at position 0
                points.append((pos, m.group(), confidence))

    # Separator matches (standalone lines)
    for m in _SEPARATOR_RE.finditer(content):
        pos = m.start()
        if pos > 0:
            points.append((pos, m.group().strip(), _SEPARATOR_CONFIDENCE))

    # Deduplicate by position (earliest match wins for same position,
    # but positions should differ — sort by position, keep first per pos)
    points.sort(key=lambda x: x[0])
    deduped: list[tuple[int, str, float]] = []
    seen_positions: set[int] = set()
    for pos, kw, conf in points:
        if pos not in seen_positions:
            deduped.append((pos, kw, conf))
            seen_positions.add(pos)

    return deduped


def _classify_and_tag_chunk(
    chunk: str,
    chapter_id: str,
    scene_number: int,
    boundary_keywords: list[str],
    confidence: float,
) -> Scene:
    """Build a Scene model from a text chunk with classification."""
    return Scene(
        scene_id=f"{chapter_id}-S{scene_number:02d}",
        chapter_id=chapter_id,
        content=chunk,
        boundary_keywords=boundary_keywords,
        location=extract_location(chunk, boundary_keywords),
        int_ext=classify_int_ext(chunk),
        time_of_day=classify_time_of_day(chunk),
        confidence=confidence,
    )


def detect_scenes(artifact: ChapterArtifact) -> SceneArtifact:
    """Detect scenes within each chapter and return classified Scene objects.

    For each chapter in the artifact:

    1. Find split points using time keywords and separator patterns.
    2. Split the chapter content at those points.
    3. Classify each resulting scene's ``int_ext``, ``time_of_day``, and
       ``location``.

    Parameters
    ----------
    artifact : ChapterArtifact
        Output of the chapter splitting phase.

    Returns
    -------
    SceneArtifact
        Pydantic model with a flat list of ``Scene`` objects across all
        chapters.
    """
    all_scenes: list[Scene] = []

    for chapter in artifact.chapters:
        content = chapter.content.strip()
        if not content:
            continue

        split_points = _find_split_points(content)
        ch_scene_num = 0  # per-chapter counter, resets for each chapter

        # No splits found — single scene with confidence 1.0
        if not split_points:
            ch_scene_num += 1
            all_scenes.append(
                _classify_and_tag_chunk(
                    chunk=content,
                    chapter_id=chapter.chapter_id,
                    scene_number=ch_scene_num,
                    boundary_keywords=[],
                    confidence=1.0,
                )
            )
            continue

        # Split at the found points
        prev = 0
        for pos, keyword, kw_confidence in split_points:
            # Chunk from prev up to (but not including) the split point
            chunk = content[prev:pos].strip()
            if chunk:
                # The first chunk (prev == 0) has no preceding keyword
                if prev == 0:
                    ch_scene_num += 1
                    all_scenes.append(
                        _classify_and_tag_chunk(
                            chunk=chunk,
                            chapter_id=chapter.chapter_id,
                            scene_number=ch_scene_num,
                            boundary_keywords=[],
                            confidence=1.0,
                        )
                    )
                else:
                    ch_scene_num += 1
                    all_scenes.append(
                        _classify_and_tag_chunk(
                            chunk=chunk,
                            chapter_id=chapter.chapter_id,
                            scene_number=len(all_scenes) + 1,
                            boundary_keywords=[keyword],
                            confidence=kw_confidence,
                        )
                    )
            prev = pos

        # Remaining content after the last split point
        tail = content[prev:].strip()
        if tail:
            ch_scene_num += 1
            all_scenes.append(
                _classify_and_tag_chunk(
                    chunk=tail,
                    chapter_id=chapter.chapter_id,
                    scene_number=ch_scene_num,
                    boundary_keywords=[split_points[-1][1]],
                    confidence=split_points[-1][2],
                )
            )

    return SceneArtifact(schema_version="1.0", scenes=all_scenes)
