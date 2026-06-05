# Novel-to-Script Adaptation

AI-assisted Chinese novel to script adaptation tool. Converts Chinese web novel text into structured YAML screenplays.

## Phase 1 — Core Engine (MVP)

Pure rule-engine pipeline with no AI dependency. Produces valid YAML scripts from Chinese novel text.

### Prerequisites

- Python 3.12+
- spaCy Chinese transformer model:

```bash
python -m spacy download zh_core_web_trf
```

### Install

```bash
pip install -e ".[dev]"
```

### Usage

```bash
# Basic conversion
novel2script input.txt -o output.yaml

# Force re-run all stages (skip cache)
novel2script input.txt -o output.yaml --no-cache

# Resume from a specific stage
novel2script input.txt -o output.yaml --resume-from scene

# Filter low-confidence dialogue attributions
novel2script input.txt -o output.yaml --confidence-threshold 0.5

# Show per-stage timing
novel2script input.txt -o output.yaml --verbose

# Version and schema info
novel2script --version
novel2script --schema
```

### Pipeline

```
novel.txt → preprocess → chapters → scenes → characters → dialogues → assemble → output.yaml
```

All intermediate results are cached with SHA256 validation. Use `--no-cache` to force re-run.

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
│   └── converter.py      # Pipeline orchestrator, SHA256 cache, YAML assembly
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

### Acceptance Criteria (Phase 1)

- [x] Input 3 chapters of Chinese novel, one CLI command completes conversion
- [x] Output is valid YAML, conforms to Schema
- [x] Chapter recognition via standard "第X章" markers
- [x] Scene boundary detection via time/location keywords
- [x] Dialogue extraction from four Chinese quote styles
- [x] Dialogue attribution (pure rule engine, low-confidence items annotated)
- [x] Ground Truth dataset skeleton (3 chapters)
- [x] Core module unit test coverage
- [x] CLI provides `--help` and basic error messages

### Upcoming

- **Phase 2**: AI-enhanced attribution, scene classification, quality evaluation
- **Phase 3A**: Web application (FastAPI + React + Docker)
- **Phase 3B**: Advanced features (versioning, comments, collaboration)
