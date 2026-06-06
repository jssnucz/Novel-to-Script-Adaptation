# Novel-to-Script Adaptation

AI-assisted Chinese novel to script adaptation tool. Converts Chinese web novel text into structured YAML screenplays.

## Phase 1 — Core Engine (MVP) ✅

Pure rule-engine pipeline with no AI dependency. Produces valid YAML scripts from Chinese novel text.

## Phase 2 — LLM Enhancement ✅

AI-powered dialogue attribution, scene classification, and character verification via DeepSeek API. Two-round attribution with rule-engine fallback on every LLM call.

### Results (basic_3ch.txt, 3 chapters)

| Dimension | Rule Engine | AI Enhanced |
|-----------|:----------:|:-----------:|
| Character accuracy | 40% (2/5) | 100% (2/2) |
| Scene classification | 0% (0/3) | 100% (3/3) |
| Dialogue attribution | 43% (11/26) | 100% (26/26) |

### Prerequisites

- Python 3.12+
- spaCy Chinese transformer model (optional; falls back to jieba):

```bash
python -m spacy download zh_core_web_trf
```

### Install

```bash
# Rule-engine only
pip install -e ".[dev]"

# With AI enhancement support
pip install -e ".[dev,ai]"
```

### Usage

```bash
# Basic conversion (pure rule engine)
novel2script input.txt -o output.yaml

# AI-enhanced conversion
export NOVEL2SCRIPT_API_KEY=sk-your-deepseek-key
novel2script input.txt -o output.yaml --ai --verbose

# Force re-run all stages (skip cache)
novel2script input.txt -o output.yaml --no-cache

# Resume from a specific stage
novel2script input.txt -o output.yaml --resume-from scene

# Filter low-confidence dialogue attributions
novel2script input.txt -o output.yaml --confidence-threshold 0.6

# Show per-stage timing
novel2script input.txt -o output.yaml --verbose

# Version and schema info
novel2script --version
novel2script --schema
```

### Pipeline

```
                    ┌─ rule engine (always) ─┐
novel.txt → preprocess → chapters → scenes → characters → dialogues ─┤
                    │                         │
                    └─ --ai? → AI enhancer ───┘
                              ├─ Two-round dialogue attribution
                              ├─ Scene classification (INT/EXT/location/time)
                              └─ Character verification (filter false positives)
                                               ↓
                                         assemble → output.yaml
```

All intermediate results are cached with SHA256 validation. Use `--no-cache` to force re-run.
AI calls are independently cached per round; network failures fall back to rule-engine results.

### Project Structure

```
src/
├── engine/
│   ├── models.py         # Pydantic v2 schemas (source of truth)
│   ├── preprocess.py     # Quote unification, paragraph normalization
│   ├── chapter.py        # Chapter boundary detection (6 regex patterns)
│   ├── scene.py          # Scene split + INT/EXT/time classification
│   ├── character.py      # spaCy NER + jieba fallback, dedup, frequency filter
│   ├── dialogue.py       # 4 quote styles, 5-tier speaker attribution
│   ├── converter.py      # Pipeline orchestrator, SHA256 cache, YAML assembly
│   └── ai_enhancer.py    # DeepSeek LLM integration (Phase 2)
└── cli/
    └── main.py           # Typer CLI entry point
tests/
├── fixtures/novels/      # Chinese novel test fragments
├── fixtures/expected/    # Expected YAML outputs
├── fixtures/ground_truth/ # Manually annotated evaluation baseline
├── unit/                 # Per-module unit tests
└── integration/          # E2E pipeline + CLI tests
```

### Test

```bash
# Run all tests (skip slow spaCy-dependent tests)
pytest tests/ -v -k "not slow"

# With coverage
pytest tests/ --cov=src/engine --cov-report=term
```

### Acceptance Criteria

#### Phase 1 ✅

- [x] Input 3 chapters of Chinese novel, one CLI command completes conversion
- [x] Output is valid YAML, conforms to Schema
- [x] Chapter recognition via standard "第X章" markers
- [x] Scene boundary detection via time/location keywords
- [x] Dialogue extraction from four Chinese quote styles
- [x] Dialogue attribution (pure rule engine, low-confidence items annotated)
- [x] Ground Truth dataset skeleton (3 chapters)
- [x] Core module unit test coverage
- [x] CLI provides `--help` and basic error messages

#### Phase 2 ✅

- [x] Two-round LLM dialogue attribution (DeepSeek, OpenAI-compatible API)
- [x] Scene classification: INT/EXT, location, time-of-day via LLM
- [x] Character verification: filter jieba false positives via LLM
- [x] `--ai` CLI switch with API key detection and graceful fallback
- [x] SHA256-cached LLM calls; network failures fall back to rule engine
- [x] Speaker-clue annotation for rapid-fire dialogue tracking
- [x] Title extraction: skip chapter markers, filename stem fallback
- [x] Confidence threshold filtering with actual computed values
- [x] All 202 tests pass

### Upcoming

- **Phase 2 follow-up**: AI role classification and character descriptions
- **Phase 3A**: Web application (FastAPI + React + Docker)
- **Phase 3B**: Advanced features (versioning, comments, collaboration)
