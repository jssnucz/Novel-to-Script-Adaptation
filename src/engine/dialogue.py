"""Dialogue extraction and attribution for the Novel-to-Script Adaptation
pipeline.

Provides quoted-text extraction, parenthetical detection, speaker inference
with a 5-tier attribution priority, and a full ``extract_dialogues`` pipeline
that converts a ``SceneArtifact`` into a ``DialogueArtifact``.

- ``extract_quoted_texts`` — find all quoted spans in text
- ``extract_parenthetical`` — extract parenthetical from text before a quote
- ``infer_speaker`` — 5-tier speaker attribution
- ``extract_dialogues`` — full pipeline (SceneArtifact -> DialogueArtifact)
"""

from __future__ import annotations

import re

from engine.character import extract_names_jieba_fallback, extract_names_spacy
from engine.models import (
    DialogueArtifact,
    DialogueLine,
    SceneArtifact,
)

# ---------------------------------------------------------------------------
# Quote detection — one pattern per style, independent matching
# ---------------------------------------------------------------------------

_DOUBLE_QUOTE_RE = re.compile(r'"[^"]*"')
_SINGLE_QUOTE_RE = re.compile(r"'[^']*'")
_CORNER_QUOTE_RE = re.compile(r"「[^」]*」")  # 「」
_WHITE_CORNER_QUOTE_RE = re.compile(r"『[^』]*』")  # 『』

_QUOTE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (_DOUBLE_QUOTE_RE, "double"),
    (_SINGLE_QUOTE_RE, "single"),
    (_CORNER_QUOTE_RE, "corner"),
    (_WHITE_CORNER_QUOTE_RE, "white_corner"),
]

# ---------------------------------------------------------------------------
# Parenthetical patterns
# ---------------------------------------------------------------------------

_PARENTHETICAL_RE = re.compile(r"（([^）]*)）|\(([^)]*)\)")

# ---------------------------------------------------------------------------
# Speech verb list (longer/more specific first for prefix matching)
# ---------------------------------------------------------------------------

_SPEECH_VERBS: list[str] = [
    "说道",
    "问道",
    "答道",
    "回答道",
    "回答说",
    "开口",
    "喃喃",
    "低语",
    "自语",
    "冷笑",
    "说",
    "道",
    "问",
    "答",
    "喊",
    "叫",
    "嚷",
    "吼",
    "叹",
    "骂",
    "哭",
    "喝",
]

_SORTED_SPEECH_VERBS: list[str] = sorted(_SPEECH_VERBS, key=len, reverse=True)

# ---------------------------------------------------------------------------
# Context window constants (in characters)
# ---------------------------------------------------------------------------

_PREFIX_CONTEXT_CHARS: int = 15
_SUFFIX_CONTEXT_CHARS: int = 10
_NEAREST_CONTEXT_CHARS: int = 30
_DIALOGUE_CONTEXT_CHARS: int = 80


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_quoted_texts(text: str) -> list[tuple[int, str, str]]:
    """Find all quoted spans in *text*.

    Searches independently for each of the four quote styles (double ``"``,
    single ``'``, corner ``「」``, white-corner ``『』``) and
    returns every match sorted by start position.

    Parameters
    ----------
    text : str
        Text to scan for quoted spans.

    Returns
    -------
    list[tuple[int, str, str]]
        List of ``(start_position, full_quote_with_delimiters, quote_style)``
        sorted by start position.
    """
    results: list[tuple[int, str, str]] = []

    for pattern, style in _QUOTE_PATTERNS:
        for m in pattern.finditer(text):
            results.append((m.start(), m.group(), style))

    results.sort(key=lambda x: x[0])
    return results


def extract_parenthetical(text_before_quote: str) -> str | None:
    """Extract parenthetical content from text preceding a quote.

    Handles both full-width ``（...）`` and ASCII ``(...)`` brackets.

    Parameters
    ----------
    text_before_quote : str
        Text preceding the quote that may contain a parenthetical.

    Returns
    -------
    str | None
        Inner text without brackets, or ``None`` if no parenthetical found.
    """
    m = _PARENTHETICAL_RE.search(text_before_quote)
    if m:
        return m.group(1) if m.group(1) is not None else m.group(2)
    return None


def _longest_first(names: list[str]) -> list[str]:
    """Return names sorted by length descending (longest match first)."""
    return sorted(names, key=len, reverse=True)


def _find_prefix_match(
    text_before: str,
    sorted_names: list[str],
) -> tuple[str | None, str | None]:
    """Check for ``name + speech_verb`` pattern within 15 chars before quote.

    Returns ``(speaker, verb)`` or ``(None, None)``.
    """
    if not text_before:
        return None, None

    context = (
        text_before[-_PREFIX_CONTEXT_CHARS:]
        if len(text_before) > _PREFIX_CONTEXT_CHARS
        else text_before
    )

    for verb in _SORTED_SPEECH_VERBS:
        for name in sorted_names:
            if re.search(re.escape(name) + re.escape(verb), context):
                return name, verb

    return None, None


def _find_suffix_match(
    text_after: str,
    sorted_names: list[str],
) -> tuple[str | None, str | None]:
    """Check for speech verb and/or name within 10 chars after quote.

    Tries both ``name + verb`` and ``verb + name`` patterns.

    Returns ``(speaker, verb)`` or ``(None, None)``.
    """
    if not text_after:
        return None, None

    context = text_after[:_SUFFIX_CONTEXT_CHARS]

    for verb in _SORTED_SPEECH_VERBS:
        for name in sorted_names:
            # name + verb (e.g. "萧炎说道")
            if re.search(re.escape(name) + re.escape(verb), context):
                return name, verb
            # verb + name (e.g. "说道萧炎")
            if re.search(re.escape(verb) + re.escape(name), context):
                return name, verb

    return None, None


def _find_nearest_name(
    text_before: str,
    text_after: str,
    sorted_names: list[str],
) -> str | None:
    """Find any character name within 30 chars before or after the quote.

    Returns the name or ``None``.
    """
    for name in sorted_names:
        # Check before: name's end must be within 30 chars of the quote
        idx = text_before.rfind(name)
        if idx != -1:
            chars_to_quote = len(text_before) - (idx + len(name))
            if chars_to_quote <= _NEAREST_CONTEXT_CHARS:
                return name

        # Check after: name's start must be within 30 chars of the quote
        idx = text_after.find(name)
        if idx != -1:
            if idx + len(name) <= _NEAREST_CONTEXT_CHARS:
                return name

    return None


def infer_speaker(
    text_before: str,
    text_after: str,
    character_names: list[str],
    line_index: int,
    prev_speakers: list[str | None],
) -> tuple[str | None, float, str]:
    """Infer speaker for a quoted dialogue line using 5-tier attribution.

    Priority (highest to lowest):

    1. **prefix_match** (0.85) — ``name + speech_verb`` within 15 chars before
       the quote.
    2. **suffix_match** (0.75) — speech verb + name within 10 chars after the
       quote.
    3. **nearest_name** (0.5) — any character name within 30 chars before or
       after.
    4. **prev_speaker** (0.3) — A-B-A-B conversation pattern based on
       previously attributed speakers.
    5. **unattributed** (0.0) — no speaker found.

    Parameters
    ----------
    text_before : str
        Text immediately preceding the quote.
    text_after : str
        Text immediately following the quote.
    character_names : list[str]
        Known character names (longest-first matching is applied internally).
    line_index : int
        Zero-based index of this line within its scene.
    prev_speakers : list[str | None]
        Speakers attributed to previous lines in the same scene, in order.

    Returns
    -------
    tuple[str | None, float, str]
        ``(speaker, confidence, attribution_method)``.
    """
    sorted_names = _longest_first(character_names)

    # 1. Prefix match (0.85)
    name, _verb = _find_prefix_match(text_before, sorted_names)
    if name:
        return name, 0.85, "prefix_match"

    # 2. Suffix match (0.75)
    name, _verb = _find_suffix_match(text_after, sorted_names)
    if name:
        return name, 0.75, "suffix_match"

    # 3. Nearest name (0.5)
    name = _find_nearest_name(text_before, text_after, sorted_names)
    if name:
        return name, 0.5, "nearest_name"

    # 4. Prev speaker (0.3) — A-B-A-B pattern
    if prev_speakers:
        if line_index >= 2 and len(prev_speakers) >= 2:
            pattern_idx = line_index % 2
            if (
                pattern_idx < len(prev_speakers)
                and prev_speakers[pattern_idx] is not None
            ):
                return prev_speakers[pattern_idx], 0.3, "prev_speaker"

        last_speaker = prev_speakers[-1]
        if last_speaker is not None:
            return last_speaker, 0.3, "prev_speaker"

    # 5. Unattributed (0.0)
    return None, 0.0, "unattributed"


def _strip_quote_delimiters(full_quote: str, style: str) -> str:
    """Remove the outermost quote delimiters from a matched quote."""
    if style in ("double", "single", "corner", "white_corner"):
        return full_quote[1:-1]
    return full_quote


def extract_dialogues(artifact: SceneArtifact) -> DialogueArtifact:
    """Extract all dialogue lines from scenes with speaker attribution.

    For each scene:

    1. Extract character names via ``extract_names_spacy`` falling back to
       ``extract_names_jieba_fallback``.
    2. Find all quoted spans via ``extract_quoted_texts``.
    3. For each quote: grab 80-char context before/after, extract
       parenthetical, infer speaker.
    4. Build ``DialogueLine`` with sequential ``line_index`` per scene.

    ``dialogue_id`` format: ``"{scene_id}-D{line_index+1:02d}"``.

    Parameters
    ----------
    artifact : SceneArtifact
        Scene-segmented novel content.

    Returns
    -------
    DialogueArtifact
        All extracted dialogue lines with attribution metadata.
    """
    all_dialogues: list[DialogueLine] = []

    for scene in artifact.scenes:
        # 1. Get character names for this scene
        names = extract_names_spacy(scene.content)
        if not names:
            names = extract_names_jieba_fallback(scene.content)

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_names: list[str] = []
        for n in names:
            if n not in seen:
                seen.add(n)
                unique_names.append(n)

        # 2. Extract all quoted spans
        quotes = extract_quoted_texts(scene.content)

        # 4. Track previous speakers for this scene
        prev_speakers: list[str | None] = []

        for i, (pos, full_quote, style) in enumerate(quotes):
            end = pos + len(full_quote)

            # Text before quote (up to context window)
            context_start = max(0, pos - _DIALOGUE_CONTEXT_CHARS)
            text_before = scene.content[context_start:pos]

            # Text after quote (up to context window)
            text_after = scene.content[end : end + _DIALOGUE_CONTEXT_CHARS]

            # Extract parenthetical from text before
            parenthetical = extract_parenthetical(text_before)

            # Infer speaker
            speaker, confidence, method = infer_speaker(
                text_before=text_before,
                text_after=text_after,
                character_names=unique_names,
                line_index=i,
                prev_speakers=prev_speakers,
            )

            # Strip outer delimiters to get the spoken line
            line = _strip_quote_delimiters(full_quote, style)

            dialogue_id = f"{scene.scene_id}-D{i + 1:02d}"

            dl = DialogueLine(
                dialogue_id=dialogue_id,
                scene_id=scene.scene_id,
                line_index=i,
                speaker=speaker,
                line=line,
                quote_style=style,
                parenthetical=parenthetical,
                confidence=confidence,
                attribution_method=method,
            )
            all_dialogues.append(dl)
            prev_speakers.append(speaker)

    return DialogueArtifact(schema_version="1.0", dialogues=all_dialogues)
