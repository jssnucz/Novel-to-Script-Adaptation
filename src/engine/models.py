"""
Pydantic v2 data models for the Novel-to-Script Adaptation pipeline.

Every artifact model carries a ``schema_version`` literal for pipeline
traceability.  Heuristic outputs include a ``confidence`` field.
"""

from pydantic import BaseModel
from typing import Literal


# ---------------------------------------------------------------------------
# Phase 1 — Preprocessing
# ---------------------------------------------------------------------------

class PreprocessArtifact(BaseModel):
    """Output of the preprocessing phase: cleaned plain text ready for
    chapter / scene segmentation."""

    schema_version: Literal["1.0"]
    original_path: str
    cleaned_text: str
    total_chars: int


# ---------------------------------------------------------------------------
# Phase 1 — Chapter Segmentation
# ---------------------------------------------------------------------------

class Chapter(BaseModel):
    """A single chapter extracted from the novel."""

    chapter_id: str          # e.g. "CH01"
    title: str
    content: str
    start_line: int
    end_line: int
    confidence: float = 1.0


class ChapterArtifact(BaseModel):
    """Artifact wrapping a list of detected chapters."""

    schema_version: Literal["1.0"]
    chapters: list[Chapter]


# ---------------------------------------------------------------------------
# Phase 1 — Scene Segmentation
# ---------------------------------------------------------------------------

class Scene(BaseModel):
    """A scene boundary within a chapter."""

    scene_id: str                   # e.g. "CH01-S01"
    chapter_id: str
    content: str
    boundary_keywords: list[str]
    location: str = "UNKNOWN"
    int_ext: Literal["INT", "EXT", "INT/EXT", "UNKNOWN"] = "UNKNOWN"
    time_of_day: Literal["日", "夜", "晨", "黄昏", "UNKNOWN"] = "UNKNOWN"
    confidence: float = 1.0


class SceneArtifact(BaseModel):
    """Artifact wrapping a list of detected scenes."""

    schema_version: Literal["1.0"]
    scenes: list[Scene]


# ---------------------------------------------------------------------------
# Phase 1 — Character Extraction
# ---------------------------------------------------------------------------

class CharacterRef(BaseModel):
    """A reference to a character as discovered during extraction."""

    name: str
    aliases: list[str] = []
    first_appearance: str   # scene_id


class CharacterArtifact(BaseModel):
    """Artifact wrapping a list of discovered character references."""

    schema_version: Literal["1.0"]
    characters: list[CharacterRef]


# ---------------------------------------------------------------------------
# Phase 1 — Dialogue Extraction
# ---------------------------------------------------------------------------

class DialogueLine(BaseModel):
    """A single line of dialogue with optional attribution."""

    dialogue_id: str
    scene_id: str
    line_index: int          # enables Phase 2 adjacency inference
    speaker: str | None = None
    line: str
    quote_style: Literal["double", "single", "corner", "white_corner"]
    parenthetical: str | None = None
    confidence: float
    attribution_method: Literal["prefix_match", "suffix_match", "nearest_name", "prev_speaker", "unattributed", "llm"]


class DialogueArtifact(BaseModel):
    """Artifact wrapping a list of extracted dialogue lines."""

    schema_version: Literal["1.0"]
    dialogues: list[DialogueLine]


# ---------------------------------------------------------------------------
# Phase 2 — Character Profiling
# ---------------------------------------------------------------------------

class CharacterProfile(BaseModel):
    """Consolidated profile for a character (enriched in Phase 3)."""

    name: str
    aliases: list[str] = []
    role: str | None = None          # filled in Phase 3 (AI role classification)
    description: str | None = None   # filled in Phase 3 (AI character description)
    first_appearance: str
    appearance_count: int
    dialogue_count: int
    scenes: list[str]


# ---------------------------------------------------------------------------
# Phase 2 — Script Assembly
# ---------------------------------------------------------------------------

class ScriptLine(BaseModel):
    """A single line in the final script output."""

    type: Literal["action", "dialogue", "transition", "note"]
    content: str
    character: str | None = None
    parenthetical: str | None = None
    confidence: float = 1.0


class ScriptScene(BaseModel):
    """A scene in the final script.

    The heading is NOT stored as a field.  It is assembled at export time
    from the decomposed fields::

        f"{int_ext}. {location} - {time_of_day}"
    """

    scene_id: str
    chapter_id: str
    int_ext: Literal["INT", "EXT", "INT/EXT", "UNKNOWN"]
    location: str
    time_of_day: Literal["日", "夜", "晨", "黄昏", "UNKNOWN"] = "UNKNOWN"
    location_note: str | None = None
    lines: list[ScriptLine]
    characters_in_scene: list[str]


class ScriptOutput(BaseModel):
    """Top-level output of the adaptation pipeline."""

    schema_version: Literal["1.0"]
    title: str
    source_novel: str
    characters: list[CharacterProfile]
    scenes: list[ScriptScene]
