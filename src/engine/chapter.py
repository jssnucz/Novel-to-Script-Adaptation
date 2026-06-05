"""Chapter boundary detection and text splitting for the Novel-to-Script
Adaptation pipeline.

Every function in this module is a pure function with no side effects and no
file I/O.

- ``detect_chapter_boundaries`` — scan text lines for chapter-title patterns
- ``split_chapters`` — divide preprocessed text into ``Chapter`` models
"""

import re

from src.engine.models import Chapter, ChapterArtifact, PreprocessArtifact

# Ordered from highest confidence to lowest. Patterns are tried in order;
# first match on a line wins.
_PATTERNS: list[tuple[re.Pattern, float]] = [
    # 1. 第X章 — standard Chinese chapter (confidence 1.0)
    (re.compile(r"^第[零一二三四五六七八九十百千\d]+\s*章\s*.*$"), 1.0),
    # 2. Chapter X — English chapter (confidence 1.0, case-sensitive)
    (re.compile(r"^Chapter\s+\d+.*$"), 1.0),
    # 3. 第X回 — Chinese "hui" (confidence 0.9)
    (re.compile(r"^第[零一二三四五六七八九十百千\d]+\s*回\s*.*$"), 0.9),
    # 4. 章X — "Zhang"-prefix style (confidence 0.8)
    (re.compile(r"^章\s*[零一二三四五六七八九十\d]+\s*.*$"), 0.8),
    # 5. X、／X．／X. — numbered headings (confidence 0.6)
    (re.compile(r"^[一二三四五六七八九十]+\s*[、．.]\s*\S"), 0.6),
    # 6. Special markers (confidence 0.5)
    (re.compile(r"^(序章|终章|尾声|楔子|番外|后记|前言)\s*.*$"), 0.5),
]

_SENTINEL_TITLE = "(开头)"


def detect_chapter_boundaries(text: str) -> list[tuple[int, str, float]]:
    """Find chapter start positions using ordered regex patterns.

    Patterns are tried in descending confidence order; the first match on
    each line wins.

    Parameters
    ----------
    text : str
        Preprocessed novel text (lines separated by ``\\n``).

    Returns
    -------
    list[tuple[int, str, float]]
        List of ``(line_index, title_text, confidence)`` sorted by position.
        Always includes ``(0, "(开头)", 1.0)`` as the first element.
    """
    lines = text.split("\n")
    boundaries: list[tuple[int, str, float]] = [(0, _SENTINEL_TITLE, 1.0)]

    for i, line in enumerate(lines):
        for pattern, confidence in _PATTERNS:
            if pattern.match(line):
                boundaries.append((i, line, confidence))
                break

    return boundaries


def split_chapters(artifact: PreprocessArtifact) -> ChapterArtifact:
    """Split preprocessed text into chapters based on detected boundaries.

    The text is split at every detected chapter-title line.  The title line
    itself is excluded from the chapter content.

    Parameters
    ----------
    artifact : PreprocessArtifact
        Output of the preprocessing phase containing ``cleaned_text``.

    Returns
    -------
    ChapterArtifact
        Pydantic model with a list of ``Chapter`` objects, each carrying
        ``chapter_id``, ``title``, ``content``, line ranges, and confidence.
    """
    text = artifact.cleaned_text
    lines = text.split("\n")
    boundaries = detect_chapter_boundaries(text)

    chapters: list[Chapter] = []

    # No real chapter boundaries found — treat entire text as one chapter.
    if len(boundaries) == 1:
        chapters.append(
            Chapter(
                chapter_id="CH01",
                title=_SENTINEL_TITLE,
                content=text,
                start_line=1,
                end_line=len(lines),
                confidence=0.3,
            )
        )
        return ChapterArtifact(schema_version="1.0", chapters=chapters)

    for i in range(1, len(boundaries)):
        title_line = boundaries[i][0]
        chapter_title = boundaries[i][1]
        chapter_confidence = boundaries[i][2]

        # Determine content line range: from the line after the title up to
        # the next boundary's title line (exclusive) or end of text.
        if i + 1 < len(boundaries):
            next_title_line = boundaries[i + 1][0]
            content_lines = lines[title_line + 1 : next_title_line]
        else:
            content_lines = lines[title_line + 1 :]

        content = "\n".join(content_lines)

        # Skip chapters with empty/whitespace-only content
        if not content.strip():
            continue

        chapters.append(
            Chapter(
                chapter_id=f"CH{len(chapters) + 1:02d}",
                title=chapter_title,
                content=content,
                start_line=title_line + 1,
                end_line=title_line + 1 + len(content_lines),
                confidence=chapter_confidence,
            )
        )

    # Fallback: all detected chapters were empty
    if not chapters:
        chapters.append(
            Chapter(
                chapter_id="CH01",
                title=_SENTINEL_TITLE,
                content=text,
                start_line=1,
                end_line=len(lines),
                confidence=0.3,
            )
        )

    return ChapterArtifact(schema_version="1.0", chapters=chapters)
