"""Pipeline orchestrator for the Novel-to-Script Adaptation pipeline.

The ``Pipeline`` class wires together the five module stages (preprocess,
chapter, scene, character, dialogue), provides SHA256-based disk caching
with version validation, and assembles the final ``ScriptOutput``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from engine.chapter import split_chapters
from engine.character import extract_characters
from engine.dialogue import extract_dialogues
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
    ScriptLine,
    ScriptOutput,
    ScriptScene,
)
from engine.preprocess import preprocess
from engine.scene import detect_scenes

logger = logging.getLogger(__name__)

_PIPELINE_VERSION = "1.0"
_MODULE_VERSION = "1.0"

# All valid stage names in dependency order.
_STAGE_ORDER = ["preprocess", "chapter", "scene", "character", "dialogue"]

# Cache file-name suffix per stage.
_STAGE_CACHE_FILE: dict[str, str] = {
    "preprocess": "preprocess",
    "chapter": "chapters",
    "scene": "scenes",
    "character": "characters",
    "dialogue": "dialogues",
}

# Map stage name → its dependency: the previous stage's output variable name.
_STAGE_DEP: dict[str, str] = {
    "preprocess": None,  # reads raw text + path
    "chapter": "preprocessed",
    "scene": "chapters",
    "character": "scenes",
    "dialogue": "scenes",
}


# ---------------------------------------------------------------------------
# Quote-style delimiter maps
# ---------------------------------------------------------------------------

_QUOTE_OPEN: dict[str, str] = {
    "double": '"',
    "single": "'",
    "corner": "「",
    "white_corner": "『",
}

_QUOTE_CLOSE: dict[str, str] = {
    "double": '"',
    "single": "'",
    "corner": "」",
    "white_corner": "』",
}


# ===================================================================
# Pipeline
# ===================================================================


class Pipeline:
    """Orchestrates the novel-to-script conversion pipeline.

    Usage::

        pipeline = Pipeline()
        result = pipeline.run("novel.txt", output_path="script.yaml")
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        input_path: str,
        *,
        output_path: str | None = None,
        cache_dir: str = "./cache",
        no_cache: bool = False,
        resume_from: str | None = None,
        confidence_threshold: float = 0.0,
        verbose: bool = False,
    ) -> ScriptOutput:
        """Execute the full pipeline.

        Parameters
        ----------
        input_path:
            Path to the UTF-8 encoded novel file.
        output_path:
            Optional path for YAML-serialized ``ScriptOutput``.
        cache_dir:
            Directory for stage caches (created if missing).
        no_cache:
            Skip all cache reads and writes.
        resume_from:
            Stage name to resume from (load earlier stages from cache).
            Valid values: ``"preprocess"``, ``"chapter"``, ``"scene"``,
            ``"character"``, ``"dialogue"``.
        confidence_threshold:
            Dialogue lines with confidence below this value have their
            ``speaker`` set to ``None``.
        verbose:
            If ``True``, log each stage execution.

        Returns
        -------
        ScriptOutput
            The assembled script output.
        """
        if verbose:
            logging.basicConfig(level=logging.INFO)

        # 1 -- Compute SHA256 of the input file
        input_sha256 = self._compute_sha256(input_path)

        # 2 -- Read the input file
        with open(input_path, "r", encoding="utf-8") as fh:
            raw_text = fh.read()

        # 3 -- Determine which stages run fresh
        resume_set = self._compute_resume_set(resume_from)

        # 4 -- Execute stages in dependency order
        ctx: dict[str, Any] = {}

        # Preprocess
        if "preprocess" in resume_set or no_cache:
            logger.info("Running preprocess...")
            preprocessed = preprocess(raw_text, input_path)
        else:
            preprocessed = self._load_or_run(
                stage="preprocess",
                input_sha256=input_sha256,
                no_cache=no_cache,
                cache_dir=cache_dir,
                run_fn=lambda: preprocess(raw_text, input_path),
            )
        # Ensure original_path reflects the actual input (cache may be stale)
        preprocessed.original_path = input_path
        ctx["preprocessed"] = preprocessed

        # Chapter
        if "chapter" in resume_set or no_cache:
            logger.info("Running chapter split...")
            chapters = split_chapters(preprocessed)
        else:
            chapters = self._load_or_run(
                stage="chapter",
                input_sha256=input_sha256,
                no_cache=no_cache,
                cache_dir=cache_dir,
                run_fn=lambda: split_chapters(ctx["preprocessed"]),
            )
        ctx["chapters"] = chapters

        # Scene
        if "scene" in resume_set or no_cache:
            logger.info("Running scene detection...")
            scenes = detect_scenes(chapters)
        else:
            scenes = self._load_or_run(
                stage="scene",
                input_sha256=input_sha256,
                no_cache=no_cache,
                cache_dir=cache_dir,
                run_fn=lambda: detect_scenes(ctx["chapters"]),
            )
        ctx["scenes"] = scenes

        # Character (parallel with dialogue)
        if "character" in resume_set or no_cache:
            logger.info("Running character extraction...")
            characters = extract_characters(scenes)
        else:
            characters = self._load_or_run(
                stage="character",
                input_sha256=input_sha256,
                no_cache=no_cache,
                cache_dir=cache_dir,
                run_fn=lambda: extract_characters(ctx["scenes"]),
            )
        ctx["characters"] = characters

        # Dialogue (parallel with character)
        if "dialogue" in resume_set or no_cache:
            logger.info("Running dialogue extraction...")
            dialogues = extract_dialogues(scenes)
        else:
            dialogues = self._load_or_run(
                stage="dialogue",
                input_sha256=input_sha256,
                no_cache=no_cache,
                cache_dir=cache_dir,
                run_fn=lambda: extract_dialogues(ctx["scenes"]),
            )
        ctx["dialogues"] = dialogues

        # 5 -- Assemble
        result = self._assemble(
            preprocessed=ctx["preprocessed"],
            chapters=ctx["chapters"],
            scenes=ctx["scenes"],
            characters=ctx["characters"],
            dialogues=ctx["dialogues"],
            confidence_threshold=confidence_threshold,
        )

        # 6 -- Serialize to YAML
        if output_path is not None:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as fh:
                yaml.dump(
                    result.model_dump(mode="python"),
                    fh,
                    allow_unicode=True,
                    sort_keys=False,
                )

        return result

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def _assemble(
        self,
        preprocessed: PreprocessArtifact,
        chapters: ChapterArtifact,  # noqa: ARG002 (available for future use)
        scenes: SceneArtifact,
        characters: CharacterArtifact,
        dialogues: DialogueArtifact,
        confidence_threshold: float = 0.0,
    ) -> ScriptOutput:
        """Assemble all stage artifacts into a single ``ScriptOutput``.

        Parameters
        ----------
        preprocessed:
            Preprocessing output containing cleaned text and metadata.
        chapters:
            Chapter segmentation output (reserved for future use).
        scenes:
            Scene segmentation output.
        characters:
            Character extraction output.
        dialogues:
            Dialogue extraction output.
        confidence_threshold:
            Dialogue lines with confidence below this threshold get
            ``speaker=None``.

        Returns
        -------
        ScriptOutput
        """
        # ---- Title: first non-empty line of cleaned_text ----
        title = ""
        for line in preprocessed.cleaned_text.split("\n"):
            stripped = line.strip()
            if stripped:
                title = stripped
                break

        # ---- Build lookup maps ----
        scene_list: list[Scene] = scenes.scenes
        dialogue_list: list[DialogueLine] = dialogues.dialogues

        # Group dialogues by scene
        dialogues_by_scene: dict[str, list[DialogueLine]] = {}
        for dl in dialogue_list:
            dialogues_by_scene.setdefault(dl.scene_id, []).append(dl)

        # ---- Build CharacterProfiles ----
        profiles: list[CharacterProfile] = []
        for ref in characters.characters:
            # Scenes where this character is "present"
            # Phase 1 approximation: assumes character present in ALL scenes
            # from first_appearance onward. Phase 2 should use actual
            # dialogue participation and scene-level presence detection.
            char_scenes = sorted(
                sc.scene_id
                for sc in scene_list
                if sc.scene_id >= ref.first_appearance
            )
            # Dialogue lines spoken by this character
            char_dialogue_count = sum(
                1 for dl in dialogue_list if dl.speaker == ref.name
            )

            profiles.append(
                CharacterProfile(
                    name=ref.name,
                    aliases=list(ref.aliases),
                    first_appearance=ref.first_appearance,
                    appearance_count=len(char_scenes),
                    dialogue_count=char_dialogue_count,
                    scenes=char_scenes,
                )
            )

        # Sort profiles by first_appearance (ascending)
        profiles.sort(key=lambda p: p.first_appearance)

        # ---- Build characters_in_scene map (scene_id → list of names) ----
        # Phase 1 approximation: marks character present from first_appearance
        # onward. Phase 2 should use actual scene-level presence (dialogue or
        # action participation) instead of this inclusive range check.
        chars_in_scene: dict[str, list[str]] = {}
        for sc in scene_list:
            names = [
                ref.name
                for ref in characters.characters
                if sc.scene_id >= ref.first_appearance
            ]
            chars_in_scene[sc.scene_id] = names

        # ---- Build ScriptScenes ----
        script_scenes: list[ScriptScene] = []
        for sc in scene_list:
            scene_dialogues = sorted(
                dialogues_by_scene.get(sc.scene_id, []),
                key=lambda d: d.line_index,
            )

            lines = self._build_scene_lines(
                scene_content=sc.content,
                scene_dialogues=scene_dialogues,
                confidence_threshold=confidence_threshold,
            )

            script_scenes.append(
                ScriptScene(
                    scene_id=sc.scene_id,
                    chapter_id=sc.chapter_id,
                    int_ext=sc.int_ext,
                    location=sc.location,
                    time_of_day=sc.time_of_day,
                    location_note=None,
                    lines=lines,
                    characters_in_scene=chars_in_scene.get(sc.scene_id, []),
                )
            )

        return ScriptOutput(
            schema_version="1.0",
            title=title,
            source_novel=preprocessed.original_path,
            characters=profiles,
            scenes=script_scenes,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_sha256(path: str) -> str:
        """Return the SHA-256 hex digest of *path*."""
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _compute_resume_set(resume_from: str | None) -> set[str]:
        """Return the set of stage names that should run fresh.

        Stages before *resume_from* try cache first; the resume target
        and all later stages run fresh.  Character and dialogue are
        treated as independent — resuming to one does **not** cause the
        other to run fresh.
        """
        if resume_from is None:
            return set()

        if resume_from in ("character", "dialogue"):
            return {resume_from}

        try:
            idx = _STAGE_ORDER.index(resume_from)
        except ValueError:
            msg = (
                f"Invalid resume_from value {resume_from!r}. "
                f"Valid values: {_STAGE_ORDER}"
            )
            raise ValueError(msg) from None

        return set(_STAGE_ORDER[idx:])

    @staticmethod
    def _reconstruct_quote(line: str, style: str) -> str:
        """Wrap *line* in the quote delimiters for *style*."""
        open_delim = _QUOTE_OPEN.get(style, '"')
        close_delim = _QUOTE_CLOSE.get(style, '"')
        return f"{open_delim}{line}{close_delim}"

    # ------------------------------------------------------------------
    # Cache I/O
    # ------------------------------------------------------------------

    def _load_or_run(
        self,
        stage: str,
        input_sha256: str,
        no_cache: bool,
        cache_dir: str,
        run_fn,
    ):
        """Try to load a cached artifact; if miss, run *run_fn* and cache.

        When *no_cache* is ``True``, always runs *run_fn* and does
        **not** write cache files.
        """
        if no_cache:
            return run_fn()

        cache_base = Path(cache_dir)
        json_path = cache_base / f"{_STAGE_CACHE_FILE[stage]}.json"
        meta_path = cache_base / f"{_STAGE_CACHE_FILE[stage]}.meta"

        # Attempt cache load
        if json_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text("utf-8"))
            if (
                meta.get("input_sha256") == input_sha256
                and meta.get("pipeline_version") == _PIPELINE_VERSION
            ):
                data = json.loads(json_path.read_text("utf-8"))
                result = self._deserialize_cache(stage, data)
                if result is not None:
                    return result

        # Cache miss — run the module
        result = run_fn()

        # Write cache
        cache_base.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        meta: dict[str, str] = {
            "input_sha256": input_sha256,
            "pipeline_version": _PIPELINE_VERSION,
            "module_version": _MODULE_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return result

    @staticmethod
    def _deserialize_cache(stage: str, data: dict):
        """Deserialize cached JSON data into the correct artifact model.

        Returns ``None`` if validation fails.
        """
        model_map = {
            "preprocess": PreprocessArtifact,
            "chapter": ChapterArtifact,
            "scene": SceneArtifact,
            "character": CharacterArtifact,
            "dialogue": DialogueArtifact,
        }
        model_cls = model_map.get(stage)
        if model_cls is None:
            return None
        try:
            return model_cls.model_validate(data)
        except Exception:
            logger.debug("Cache validation failed for %s", stage, exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Scene line building
    # ------------------------------------------------------------------

    @staticmethod
    def _build_scene_lines(
        scene_content: str,
        scene_dialogues: list[DialogueLine],
        confidence_threshold: float,
    ) -> list[ScriptLine]:
        """Build a list of ``ScriptLine`` from scene content and dialogues.

        Action lines are extracted from text between dialogue quotes.
        """
        if not scene_dialogues:
            content = scene_content.strip()
            if content:
                return [ScriptLine(type="action", content=content)]
            return []

        lines: list[ScriptLine] = []
        cursor = 0

        for dl in scene_dialogues:
            quote = Pipeline._reconstruct_quote(dl.line, dl.quote_style)
            pos = scene_content.find(quote, cursor)

            if pos == -1:
                # Quote not found — try from the beginning
                pos = scene_content.find(quote)
                if pos != -1 and pos < cursor:
                    # Found before cursor — duplicate text, treat as not found
                    pos = -1
            if pos == -1:
                # Still not found — emit dialogue without action context
                speaker = dl.speaker if dl.confidence >= confidence_threshold else None
                lines.append(
                    ScriptLine(
                        type="dialogue",
                        content=dl.line,
                        character=speaker,
                        parenthetical=dl.parenthetical,
                        confidence=dl.confidence,
                    )
                )
                continue

            # Action text before this quote
            if pos > cursor:
                action_text = scene_content[cursor:pos].strip()
                if action_text:
                    lines.append(ScriptLine(type="action", content=action_text))

            # Dialogue line
            speaker = dl.speaker if dl.confidence >= confidence_threshold else None
            lines.append(
                ScriptLine(
                    type="dialogue",
                    content=dl.line,
                    character=speaker,
                    parenthetical=dl.parenthetical,
                    confidence=dl.confidence,
                )
            )

            cursor = pos + len(quote)

        # Trailing action text
        if cursor < len(scene_content):
            action_text = scene_content[cursor:].strip()
            if action_text:
                lines.append(ScriptLine(type="action", content=action_text))

        return lines
