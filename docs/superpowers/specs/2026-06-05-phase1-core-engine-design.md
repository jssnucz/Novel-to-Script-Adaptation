# Phase 1 Core Engine — Design Document

**Date**: 2026-06-05
**Status**: Approved
**Project**: Novel-to-Script Adaptation Agent
**Phase**: Phase 1 — 核心引擎开发（MVP）

---

## Overview

Phase 1 delivers an end-to-end CLI pipeline that converts Chinese novels into structured YAML scripts using pure rule-engine techniques (no AI). It is the MVP per the project plan v2.0.

**Core flow**: `novel.txt → [preprocess → chapter → scene → character → dialogue] → converter → output.yaml`

## Technical Decisions (Confirmed)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Language | Python 3.12+ (using 3.13 at `C:\Users\asus\AppData\Local\Programs\Python\Python313\python.exe`) | NLP ecosystem, FastAPI in later phases |
| NLP | spaCy (`zh_core_web_trf`) + jieba | spaCy for NER/dependency parsing; jieba as fallback tokenizer |
| Schema | Pydantic v2 | Type-safe schema, same stack as FastAPI, auto-generates docs |
| CLI | Typer | Modern Click wrapper, auto-generated `--help`, type validation |
| Testing | pytest + pytest-cov | Standard toolchain |
| Export | PyYAML | Phase 1 only needs YAML output |

## Project Structure (Phase 1 Only)

```
novel-to-script/
├── src/
│   ├── engine/                    # Phase 1 core engine
│   │   ├── __init__.py
│   │   ├── models.py              # All Pydantic models (source of truth)
│   │   ├── preprocess.py          # Chinese text preprocessing (1.2)
│   │   ├── chapter.py             # Chapter recognition & split (1.3)
│   │   ├── scene.py               # Scene boundary detection (1.4)
│   │   ├── character.py           # Character NER (1.5)
│   │   ├── dialogue.py            # Dialogue extraction & attribution (1.6)
│   │   └── converter.py           # Pipeline orchestrator + _assemble (1.7)
│   └── cli/
│       ├── __init__.py
│       └── main.py                # Typer CLI (1.8)
├── tests/
│   ├── fixtures/
│   │   ├── novels/                # Chinese novel fragments for testing
│   │   │   ├── basic_3ch.txt
│   │   │   ├── mixed_quotes.txt
│   │   │   └── no_dialogue.txt
│   │   └── expected/              # Expected YAML outputs
│   │       ├── basic_3ch.yaml
│   │       └── ...
│   ├── unit/
│   │   ├── test_preprocess.py
│   │   ├── test_chapter.py
│   │   ├── test_scene.py
│   │   ├── test_character.py
│   │   ├── test_dialogue.py
│   │   └── test_converter.py
│   ├── integration/
│   │   ├── test_pipeline_e2e.py
│   │   └── test_cli.py
│   └── conftest.py
├── cache/                         # Runtime intermediate JSON (gitignored)
├── pyproject.toml
└── README.md
```

## Pipeline Architecture

### Pipeline Contract

| Layer | Mechanism |
|-------|-----------|
| Module interface | Input Pydantic object → Output Pydantic object. Modules do no file I/O. |
| Output validation | `module_output = Model.model_validate(module_output)` — fail on first error |
| Write to disk | Converter serializes → writes `.json` → writes `.meta` (SHA256 + version) |
| Read from disk | Converter reads `.meta`, compares hash + version → deserializes on match |
| Post-deserialize | `Model.model_validate_json(data)` — second validation gate |
| User control | `--resume-from <stage>`, `--no-cache` |

### Cache Key Strategy

Each intermediate `.json` has a companion `.meta` file:

```json
{
  "input_sha256": "abc123...",
  "pipeline_version": "1.0",
  "module_version": "1.0",
  "created_at": "2026-06-05T12:00:00"
}
```

Converter checks: (1) `.meta` exists, (2) `input_sha256` matches current file, (3) `pipeline_version` matches. All three pass → cache hit. Any fail → cache invalid, re-run.

### Pipeline Flow

```
converter.py  ──  Pipeline.run(input_path)
  │
  ├── 1.2  preprocess.py    novel.txt           → PreprocessArtifact (v1.0)
  │                              ↓ validated by Pydantic
  ├── 1.3  chapter.py       PreprocessArtifact  → ChapterArtifact (v1.0)
  │                              ↓
  ├── 1.4  scene.py         ChapterArtifact     → SceneArtifact (v1.0)
  │                              ↓
  ├── 1.5  character.py     SceneArtifact       → CharacterArtifact (v1.0)
  ├── 1.6  dialogue.py      SceneArtifact       → DialogueArtifact (v1.0)
  │        ↑ 1.5 and 1.6 are independent: both consume SceneArtifact
  │                              ↓
  └── 1.7  _assemble()      PreprocessArtifact + ChapterArtifact + SceneArtifact
                               + CharacterArtifact + DialogueArtifact
                                   ↓
                             ScriptOutput (v1.0)
                                   ↓
                             output.yaml
```

## Pydantic Data Models

All models defined in `src/engine/models.py`. Every artifact model carries `schema_version: Literal["1.0"]`.

### Intermediate Artifacts

```python
class PreprocessArtifact(BaseModel):
    schema_version: Literal["1.0"]
    original_path: str
    cleaned_text: str           # Unified quotes, normalized paragraphs
    total_chars: int

class Chapter(BaseModel):
    chapter_id: str             # "CH01"
    title: str                  # "第一章 斗之气三段"
    content: str                # Chapter plain text
    start_line: int
    end_line: int
    confidence: float = 1.0     # "第X章" = 1.0, "一、" = 0.6

class ChapterArtifact(BaseModel):
    schema_version: Literal["1.0"]
    chapters: list[Chapter]

class Scene(BaseModel):
    scene_id: str               # "CH01-S01"
    chapter_id: str
    content: str
    boundary_keywords: list[str]  # Keywords that triggered the split
    location: str = "UNKNOWN"     # Extracted from boundary_keywords or first sentence
    int_ext: Literal["INT", "EXT", "INT/EXT", "UNKNOWN"] = "UNKNOWN"
    time_of_day: str = "UNKNOWN"  # "日" | "夜" | "晨" | "黄昏" | "UNKNOWN"
    confidence: float = 1.0       # "三天后" = 0.7, "---" = 0.5

class SceneArtifact(BaseModel):
    schema_version: Literal["1.0"]
    scenes: list[Scene]

class CharacterRef(BaseModel):
    name: str
    aliases: list[str] = []
    first_appearance: str       # scene_id

class CharacterArtifact(BaseModel):
    schema_version: Literal["1.0"]
    characters: list[CharacterRef]

class DialogueLine(BaseModel):
    dialogue_id: str
    scene_id: str
    line_index: int             # Position in scene — enables Phase 2 adjacency inference
    speaker: str | None = None  # None = unattributed, Phase 2 fills
    line: str                   # Dialogue text without quotes
    quote_style: str            # "" '' 「」 『』
    parenthetical: str | None = None  # "(冷冷地)" from quote context
    confidence: float           # 0.0–1.0, attribution confidence
    attribution_method: str     # "prefix_match" | "suffix_match" | "nearest_name" | "unattributed"

class DialogueArtifact(BaseModel):
    schema_version: Literal["1.0"]
    dialogues: list[DialogueLine]
```

### Output Models

```python
class CharacterProfile(BaseModel):
    """Aggregated character view for final output"""
    name: str
    aliases: list[str] = []
    role: str | None = None                     # Phase 2 fills: "主角" | "配角" | "龙套"
    description: str | None = None              # Phase 2 fills
    first_appearance: str                       # scene_id
    appearance_count: int                       # scenes appeared in
    dialogue_count: int                         # lines spoken
    scenes: list[str]                           # all scene_ids

class ScriptLine(BaseModel):
    type: Literal["action", "dialogue", "transition", "note"]
    content: str
    character: str | None = None            # populated for dialogue
    parenthetical: str | None = None
    confidence: float = 1.0

class ScriptScene(BaseModel):
    scene_id: str
    chapter_id: str
    int_ext: Literal["INT", "EXT", "INT/EXT", "UNKNOWN"]
    location: str
    time_of_day: str                        # "日" | "夜" | "晨" | "黄昏" | "UNKNOWN"
    location_note: str | None = None        # "雨中" etc.
    lines: list[ScriptLine]
    characters_in_scene: list[str]

class ScriptOutput(BaseModel):
    schema_version: Literal["1.0"]
    title: str
    source_novel: str
    characters: list[CharacterProfile]
    scenes: list[ScriptScene]
```

### Key Design Rules

- **Decomposed fields are the source of truth.** `heading` is NOT stored — it is assembled at export time: `f"{int_ext}. {location} - {time_of_day}"`. No dual-source drift.
- **All heuristic modules carry `confidence`.** Chapter, Scene, and Dialogue all report confidence. Phase 2 AI reads these to prioritize corrections.
- **DialogueLine.line_index** enables Phase 2 adjacency inference without re-scanning source text.
- **DialogueLine.parenthetical** is extracted by Phase 1 from quote context; converter copies directly to ScriptLine.
- **ScriptOutput.title** is derived from the first non-empty line of `PreprocessArtifact.cleaned_text` during `_assemble()`. `ScriptOutput.source_novel` comes from `PreprocessArtifact.original_path`.

## Module Signatures

```python
# 1.2 — Independent, no upstream dependency
def preprocess(text: str, source_path: str) -> PreprocessArtifact

# 1.3 — Depends on 1.2
def split_chapters(artifact: PreprocessArtifact) -> ChapterArtifact

# 1.4 — Depends on 1.3
# Detects scene boundaries + classifies scene metadata (int_ext, time_of_day, location).
def detect_scenes(artifact: ChapterArtifact) -> SceneArtifact

# 1.5 — Depends on 1.4 (needs scene_id)
# NOTE: Internally performs global dedup and merge across all scenes —
# alias resolution, main character identification via frequency, and
# filtering of single-occurrence names (likely false positives/extras).
def extract_characters(artifact: SceneArtifact) -> CharacterArtifact

# 1.6 — Depends on 1.4 (needs scene.content + scene_id)
def extract_dialogues(artifact: SceneArtifact) -> DialogueArtifact

# 1.7 — Pipeline orchestrator
class Pipeline:
    def run(self, input_path: str, *,
            output_path: str | None = None,    # If set, serialize ScriptOutput to YAML file
            cache_dir: str = "./cache",
            no_cache: bool = False,
            resume_from: str | None = None,
            confidence_threshold: float = 0.0,
            verbose: bool = False,
            ) -> ScriptOutput
    # When output_path is None, returns ScriptOutput without writing to disk
    # (useful for tests and library use). CLI layer always passes output_path.

    def _assemble(
        self,
        preprocessed: PreprocessArtifact,
        chapters: ChapterArtifact,
        scenes: SceneArtifact,
        characters: CharacterArtifact,
        dialogues: DialogueArtifact,
        confidence_threshold: float = 0.0,     # Dialogue attributions below this → speaker = None
    ) -> ScriptOutput
```

## CLI Design (Typer)

```bash
novel2script input.txt -o output.yaml                    # Basic usage
novel2script input.txt -o output.yaml --no-cache          # Force re-run
novel2script input.txt -o output.yaml --resume-from scene # Resume from stage
novel2script input.txt -o output.yaml --confidence-threshold 0.5  # Filter low-confidence attributions
novel2script input.txt -o output.yaml --verbose           # Per-stage timing + stats
novel2script --version                                    # Version info
novel2script --schema                                     # Current schema version
```

### `--resume-from` Valid Values

| Value | Cached Stages | Runs |
|-------|---------------|------|
| `preprocess` | (none) | preprocess → chapter → scene → character + dialogue → assemble |
| `chapter` | preprocess | chapter → scene → character + dialogue → assemble |
| `scene` | preprocess, chapter | scene → character + dialogue → assemble |
| `character` | preprocess, chapter, scene | character → assemble (dialogue NOT run) |
| `dialogue` | preprocess, chapter, scene | dialogue → assemble (character NOT run) |

**Note**: `character` and `dialogue` have no mutual dependency. Resuming to one does NOT run the other. To re-run both, call `novel2script` twice or use `--no-cache`.

## Error Handling

| Category | Strategy |
|----------|----------|
| Deterministic rules | Process directly, no error (e.g., "第X章" regex) |
| Heuristic rules | Process + `confidence < 1.0` (e.g., dialogue attribution) |
| Unrecoverable errors | Raise exception → Pipeline terminates (e.g., empty input, encoding corruption) |

Modules never swallow errors. All exceptions propagate to Pipeline for decision.

## Testing Strategy

### Directory Structure

```
tests/
├── fixtures/                        # Shared test data
│   ├── novels/                      # Chinese novel fragments
│   │   ├── basic_3ch.txt
│   │   ├── mixed_quotes.txt
│   │   └── no_dialogue.txt
│   └── expected/                    # Per-input expected YAML
│       ├── basic_3ch.yaml
│       └── ...
├── unit/                            # One file per module
│   ├── test_preprocess.py
│   ├── test_chapter.py
│   ├── test_scene.py
│   ├── test_character.py
│   ├── test_dialogue.py
│   └── test_converter.py
├── integration/
│   ├── test_pipeline_e2e.py
│   └── test_cli.py
└── conftest.py                      # Shared fixture loaders
```

### Unit Test Focus

| Module | Test Points |
|--------|-------------|
| preprocess | Four quote styles unified, blank line normalization, BOM removal |
| chapter | "第X章" / "Chapter X" / "章X" / volume-chapter nesting, confidence values |
| scene | "三天后" / "与此同时" / location-change keywords / separator lines, int_ext + time_of_day classification |
| character | spaCy NER recall, dedup, given+surname combination, aliases recorded but not resolved (Phase 2) |
| dialogue | Four quote styles captured, parenthetical extraction from quote context, prefix/suffix attribution inference, line_index sequential continuity, nested quotes, confidence markings for low-confidence items |
| converter | Schema validation pass, cache hit/miss, version mismatch error, hash mismatch re-run, _assemble character aggregation (CharacterRef → CharacterProfile), confidence-threshold filtering for low-confidence attribution |

### Integration Test Focus

- **test_pipeline_e2e**: Full novel.txt → ScriptOutput → YAML file. Verify YAML is parseable and conforms to Schema.
- **test_cli**: `--help` output, `--no-cache` clears cache, `--resume-from` behavior per stage, error messages on bad input, `--confidence-threshold` parameter propagation.

### `--confidence-threshold` CLI Test

Given `basic_3ch.txt` with a dialogue attribution at confidence 0.3:
- `--confidence-threshold 0.5` → output YAML has `speaker: null` for that dialogue
- `--confidence-threshold 0.2` → output YAML retains original speaker value

This tests the Pipeline.run() → _assemble() parameter propagation interface, not just end-to-end behavior.

## Acceptance Criteria (Phase 1)

- [ ] Input 3 chapters of Chinese novel (TXT/Markdown), one CLI command completes conversion
- [ ] Output is valid YAML, conforms to Schema, parseable by standard YAML parsers
- [ ] Chapter recognition accuracy > 95% (standard "第X章" markers)
- [ ] Scene boundary recall > 75% (on time/location keyword test set)
- [ ] Dialogue extraction rate > 85% (no missed quotes)
- [ ] Dialogue attribution rate > 60% (pure rule engine; low-confidence items annotated not guessed)
- [ ] Ground Truth dataset v1.0 deliverable (3 chapters annotated)
- [ ] Core module unit test coverage > 80%
- [ ] All four Chinese quote styles pass tests
- [ ] CLI provides `--help` and basic error messages

## Dependencies (from project plan)

```
1.1 (Schema) ──→ 1.7 (Converter)
                      ↑
1.2 (Preprocess) → 1.3 (Chapter) → 1.4 (Scene) ──→ 1.5 (Character) ──→ 1.7
                                              ──→ 1.6 (Dialogue)  ──→ 1.7
                                              ↑ 1.5 and 1.6 are independent

                 1.8 (CLI) ←── parallel to 1.3–1.7 ──────────────────────┘

1.9 (Ground Truth) ←──贯穿全程, parallel to 1.3–1.6
```

## Implementation Order

1. **Project scaffolding** — `pyproject.toml`, directory structure, `conftest.py`, test fixtures skeleton.
   **Prerequisite**: `python -m spacy download zh_core_web_trf` (required before running `test_character.py`)
2. **Task 1.1** — `models.py` (Pydantic models, all models defined upfront as source of truth)
3. **Task 1.2** — `preprocess.py` + `test_preprocess.py`
4. **Task 1.3** — `chapter.py` + `test_chapter.py`
5. **Task 1.4** — `scene.py` + `test_scene.py`
6. **Task 1.5** — `character.py` + `test_character.py`  ← 与 1.6 无互相依赖，顺序可交换
7. **Task 1.6** — `dialogue.py` + `test_dialogue.py`    ← 与 1.5 无互相依赖，顺序可交换
8. **Task 1.7** — `converter.py` + `test_converter.py`
9. **Task 1.8** — `cli/main.py` + `test_cli.py`
10. **Task 1.9** — Ground Truth dataset construction (parallel to 1.3–1.6)
