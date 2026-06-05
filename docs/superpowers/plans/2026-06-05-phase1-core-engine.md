# Phase 1 Core Engine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the end-to-end CLI pipeline that converts Chinese novels into structured YAML scripts using pure rule-engine techniques.

**Architecture:** Serial pipeline with intermediate JSON cache. Each module is a pure function (Pydantic in → Pydantic out). Converter orchestrates I/O, caching (SHA256 + version metadata), validation, and final assembly.

**Tech Stack:** Python 3.13, Pydantic v2, Typer, spaCy (zh_core_web_trf) + jieba, pytest + pytest-cov, PyYAML

**Python Path:** `C:\Users\asus\AppData\Local\Programs\Python\Python313\python.exe`

---

### Task 0: Project Scaffolding & Prerequisites

**Files:**
- Create: `pyproject.toml`
- Create: `src/__init__.py`
- Create: `src/engine/__init__.py`
- Create: `src/cli/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/unit/__init__.py`
- Create: `tests/integration/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/fixtures/novels/basic_3ch.txt`
- Create: `tests/fixtures/novels/mixed_quotes.txt`
- Create: `tests/fixtures/novels/no_dialogue.txt`
- Create: `tests/fixtures/expected/basic_3ch.yaml`
- Create: `.gitignore`

- [ ] **Step 0: Install spaCy Chinese model**

Run: `C:\Users\asus\AppData\Local\Programs\Python\Python313\python.exe -m spacy download zh_core_web_trf`
Expected: "Download and installation successful"

- [ ] **Step 1: Create directory structure**

Run:
```bash
cd "E:/Novel-to-Script Adaptation"
mkdir -p src/engine src/cli
mkdir -p tests/unit tests/integration tests/fixtures/novels tests/fixtures/expected
mkdir -p cache
```

- [ ] **Step 2: Create pyproject.toml**

Write `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=75.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "novel-to-script"
version = "0.1.0"
description = "AI-assisted Chinese novel to script adaptation tool"
requires-python = ">=3.12"
dependencies = [
    "pydantic>=2.0",
    "typer>=0.15",
    "pyyaml>=6.0",
    "jieba>=0.42",
    "spacy>=3.8",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=6.0",
]

[project.scripts]
novel2script = "src.cli.main:app"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = ["-v", "--tb=short"]
```

- [ ] **Step 3: Create __init__.py files**

Write `src/__init__.py`: (empty file)
Write `src/engine/__init__.py`: (empty file)
Write `src/cli/__init__.py`: (empty file)
Write `tests/__init__.py`: (empty file)
Write `tests/unit/__init__.py`: (empty file)
Write `tests/integration/__init__.py`: (empty file)

- [ ] **Step 4: Create .gitignore**

Write `.gitignore`:

```
__pycache__/
*.pyc
.pytest_cache/
cache/
*.egg-info/
dist/
build/
.coverage
htmlcov/
```

- [ ] **Step 5: Create test fixtures — basic_3ch.txt**

Write `tests/fixtures/novels/basic_3ch.txt`:

```
斗破苍穹

第一章 斗之气三段

测试山巅，云海翻腾。

萧炎盘膝坐在青石之上，双目紧闭，周身天地能量缓缓汇聚而来。
"三年了……终于摸到了斗者门槛。"他低声自语，语气中带着说不清的复杂。

"萧炎哥哥！"远处传来一声清脆的呼喊。

纳兰嫣然踏空而来，白衣飘飘，落在萧炎面前。她神色清冷，眼中却有一丝不易察觉的关切："明日就是迦南学院的考核，你准备好了吗？"

萧炎睁开眼睛，苦笑道："准备？我准备了整整三年。"

"那就好。"纳兰嫣然转身望向云海，「这云岚宗的风景，怕是最后一次看了。」

萧炎站起身，拍了拍衣袍上的尘土：'走吧，下山。'

两人并肩而行，谁也没有再说话。山路两旁的枫叶红得像火，山风卷起落叶，在他们身后飘零。

第二章 迦南学院的考验

三天后，迦南学院考核大殿。

数百名年轻修炼者齐聚一堂，气氛压抑得让人喘不过气。萧炎站在人群中央，感受着四面八方投来的目光——有好奇，有轻蔑，更多的是漠不关心。

"下一个，萧炎！"考核官的声音在大殿中回荡。

『终于轮到我了。』萧炎深吸一口气，走向考核台。

考核台上放着一块测魔石碑，碑面光滑如镜。萧炎将手掌按在碑面上，体内斗气如潮水般涌出。石碑瞬间亮起刺目的光芒——三段斗之气！

"三段斗之气……"考核官皱了皱眉，"修为平平，下一个。"

萧炎默默退下，耳边传来细微的窃窃私语。
"听说他是萧家那个废物？"
"三年前还是天才，现在连四段都没突破……"
"啧啧，这落差，换我早就不练了。"

纳兰嫣然站在人群中，望着萧炎的背影，手指微微攥紧。

第三章 深夜来客

夜已深沉，月光如水。

萧炎独自坐在院落中，手中握着一枚古朴的戒指。戒指在月光下泛着幽幽的光泽，这是他母亲留给他的唯一遗物。

忽然，戒指轻颤了一下。

「小家伙，你倒是沉得住气。」一道苍老的声音从戒指中传出。

萧炎瞳孔骤缩，猛地站起身：「谁？！」

「别慌。」戒指中飘出一缕白烟，缓缓凝聚成一个透明老者的身影。老者负手而立，面带微笑：「老夫名号药老，魂居此戒已千年。你这三年斗气倒退，是你这戒指在吸你的斗气温养老夫。」

『什么？！』萧炎先是震惊，随即涌上怒意：「我三年苦修，全给你做了嫁衣？！」

药老不紧不慢地说：「嘿嘿，作为补偿，老夫可以教你炼药术。这天地间，炼药师的身份可比斗者尊贵得多。」

「炼药师？」萧炎愣了愣，「你是说……那些能炼出丹药，让人脱胎换骨的炼药师？」

「正是。」药老抚须一笑，「怎样，拜师吗？」

萧炎沉默了片刻，眼神渐渐坚定：「拜。」
```

- [ ] **Step 6: Create test fixtures — mixed_quotes.txt**

Write `tests/fixtures/novels/mixed_quotes.txt`:

```
第一章 引号测试

李明推开门走进房间。

"你来了。"张华头也不抬地说。

'我等了很久了。'王芳从角落里站起来，语气冰冷：「你知道现在几点了吗？」

「抱歉，路上堵车。」李明解释道，"下次我会注意的。"

张华终于抬起头，看着两人：『都别吵了，说正事吧。』

王芳冷哼一声：「正事？他现在才来，还说什么正事？」她转过身去，"我不想谈了。"

"别这样，"李明轻声说，（叹了口气）『我确实有错，但今天的会议很重要。』

张华站起身："都坐下。"
```

- [ ] **Step 7: Create test fixtures — no_dialogue.txt**

Write `tests/fixtures/novels/no_dialogue.txt`:

```
第一章 无言之夜

夜色如墨，万籁俱寂。

他站在窗前，望着远方的山峦。月光洒在窗台上，像是铺了一层薄薄的霜。三年前的那个夜晚，也是这样的月色，也是这样的寂静。

风从窗缝中钻进来，带着深秋的凉意。他打了个寒噤，却没有关窗的意思。冷一点好，冷一点能让人清醒。

桌上摊着一封信，信封已经泛黄，边角有些磨损。信的内容他早已背得滚瓜烂熟，但他还是每天都会展开读一遍，仿佛那些字迹里藏着什么他还没发现的秘密。

窗外的梧桐树落下了最后一片叶子。
```

- [ ] **Step 8: Create expected YAML — basic_3ch.yaml (skeleton)**

Write `tests/fixtures/expected/basic_3ch.yaml`:

```yaml
# Expected output skeleton for basic_3ch.txt
# This will be refined after models.py is defined and converter works end-to-end.
# For now, placeholder to establish the file exists.
schema_version: "1.0"
title: "斗破苍穹"
source_novel: ""
characters: []
scenes: []
```

- [ ] **Step 9: Create conftest.py**

Write `tests/conftest.py`:

```python
"""Shared test fixtures for the novel-to-script test suite."""

from pathlib import Path
import pytest


@pytest.fixture
def fixture_path():
    """Return the path to the fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def load_novel(fixture_path):
    """Return a function that loads a test novel by filename."""

    def _load(name: str) -> str:
        filepath = fixture_path / "novels" / name
        return filepath.read_text(encoding="utf-8")

    return _load


@pytest.fixture
def load_expected(fixture_path):
    """Return a function that loads an expected YAML file by filename."""

    def _load(name: str) -> dict:
        import yaml

        filepath = fixture_path / "expected" / name
        return yaml.safe_load(filepath.read_text(encoding="utf-8"))

    return _load


@pytest.fixture
def basic_novel(load_novel):
    """Return the full text of basic_3ch.txt."""
    return load_novel("basic_3ch.txt")


@pytest.fixture
def mixed_quotes_novel(load_novel):
    """Return the full text of mixed_quotes.txt."""
    return load_novel("mixed_quotes.txt")


@pytest.fixture
def no_dialogue_novel(load_novel):
    """Return the full text of no_dialogue.txt."""
    return load_novel("no_dialogue.txt")
```

- [ ] **Step 10: Install project in dev mode and run smoke test**

Run:
```bash
cd "E:/Novel-to-Script Adaptation"
C:\Users\asus\AppData\Local\Programs\Python\Python313\python.exe -m pip install -e ".[dev]"
```

Expected: Package installs successfully.

Run:
```bash
cd "E:/Novel-to-Script Adaptation"
C:\Users\asus\AppData\Local\Programs\Python\Python313\python.exe -m pytest tests/ -v
```

Expected: "no tests ran" (no test files with test functions yet, but pytest discovers the directory correctly).

- [ ] **Step 11: Commit**

```bash
cd "E:/Novel-to-Script Adaptation"
git init
git add -A
git commit -m "chore: scaffold project structure with pyproject.toml, test fixtures, and conftest.py

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 1: Pydantic Data Models (Task 1.1)

**Files:**
- Create: `src/engine/models.py`
- Create: `tests/unit/test_models.py`

**Purpose:** Define all Pydantic models as the source of truth. Every artifact carries `schema_version: Literal["1.0"]`.

- [ ] **Step 1: Write the model validation test**

Write `tests/unit/test_models.py`:

```python
"""Tests for Pydantic data models."""

import pytest
from pydantic import ValidationError
from src.engine.models import (
    PreprocessArtifact, Chapter, ChapterArtifact,
    Scene, SceneArtifact, CharacterRef, CharacterArtifact,
    DialogueLine, DialogueArtifact,
    CharacterProfile, ScriptLine, ScriptScene, ScriptOutput,
)


class TestPreprocessArtifact:
    def test_valid_artifact(self):
        artifact = PreprocessArtifact(
            schema_version="1.0",
            original_path="/path/to/novel.txt",
            cleaned_text="这是预处理后的文本",
            total_chars=10,
        )
        assert artifact.schema_version == "1.0"
        assert artifact.total_chars == 10

    def test_invalid_schema_version_raises(self):
        with pytest.raises(ValidationError):
            PreprocessArtifact(
                schema_version="2.0",
                original_path="/path/to/novel.txt",
                cleaned_text="text",
                total_chars=4,
            )


class TestChapter:
    def test_default_confidence(self):
        ch = Chapter(chapter_id="CH01", title="第一章 测试",
                     content="章节内容", start_line=1, end_line=10)
        assert ch.confidence == 1.0

    def test_low_confidence_chapter_marker(self):
        ch = Chapter(chapter_id="CH02", title="一、序章",
                     content="内容", start_line=1, end_line=5, confidence=0.6)
        assert ch.confidence == 0.6


class TestChapterArtifact:
    def test_empty_chapters(self):
        artifact = ChapterArtifact(schema_version="1.0", chapters=[])
        assert artifact.chapters == []

    def test_with_chapters(self):
        ch = Chapter(chapter_id="CH01", title="第一章", content="内容",
                     start_line=1, end_line=5)
        artifact = ChapterArtifact(schema_version="1.0", chapters=[ch])
        assert len(artifact.chapters) == 1


class TestScene:
    def test_defaults_are_unknown(self):
        scene = Scene(scene_id="CH01-S01", chapter_id="CH01",
                      content="场景内容", boundary_keywords=["三天后"])
        assert scene.int_ext == "UNKNOWN"
        assert scene.time_of_day == "UNKNOWN"
        assert scene.location == "UNKNOWN"
        assert scene.confidence == 1.0

    def test_classified_scene(self):
        scene = Scene(scene_id="CH01-S01", chapter_id="CH01",
                      content="客栈大厅内，烛火摇曳。",
                      boundary_keywords=["客栈"], location="客栈大厅",
                      int_ext="INT", time_of_day="夜", confidence=0.8)
        assert scene.int_ext == "INT"
        assert scene.location == "客栈大厅"


class TestCharacterRef:
    def test_minimal(self):
        ref = CharacterRef(name="萧炎", first_appearance="CH01-S01")
        assert ref.aliases == []

    def test_with_aliases(self):
        ref = CharacterRef(name="萧炎", aliases=["炎帝", "小家伙"],
                          first_appearance="CH01-S01")
        assert "炎帝" in ref.aliases


class TestDialogueLine:
    def test_full_attribution(self):
        dl = DialogueLine(dialogue_id="CH01-S01-D01", scene_id="CH01-S01",
                          line_index=0, speaker="萧炎",
                          line="三年了……终于摸到了斗者门槛。",
                          quote_style="“”", parenthetical=None,
                          confidence=0.9, attribution_method="prefix_match")
        assert dl.speaker == "萧炎"
        assert dl.line_index == 0

    def test_unattributed_dialogue(self):
        dl = DialogueLine(dialogue_id="CH01-S01-D02", scene_id="CH01-S01",
                          line_index=1, speaker=None,
                          line="你准备好了吗？", quote_style="“”",
                          parenthetical=None, confidence=0.0,
                          attribution_method="unattributed")
        assert dl.speaker is None

    def test_parenthetical_extraction(self):
        dl = DialogueLine(dialogue_id="CH01-S01-D03", scene_id="CH01-S01",
                          line_index=2, speaker="李明", line="我确实有错",
                          quote_style="“”", parenthetical="叹了口气",
                          confidence=0.8, attribution_method="suffix_match")
        assert dl.parenthetical == "叹了口气"


class TestCharacterProfile:
    def test_minimal_profile(self):
        profile = CharacterProfile(name="萧炎", first_appearance="CH01-S01",
                                   appearance_count=3, dialogue_count=12,
                                   scenes=["CH01-S01", "CH01-S02"])
        assert profile.role is None
        assert profile.description is None

    def test_with_role(self):
        profile = CharacterProfile(name="萧炎", role="主角",
                                   first_appearance="CH01-S01",
                                   appearance_count=5, dialogue_count=20,
                                   scenes=["CH01-S01"])
        assert profile.role == "主角"


class TestScriptLine:
    def test_action_line(self):
        line = ScriptLine(type="action", content="萧炎盘膝坐在青石之上。")
        assert line.type == "action"
        assert line.character is None

    def test_dialogue_line(self):
        line = ScriptLine(type="dialogue",
                          content="三年了……终于摸到了斗者门槛。",
                          character="萧炎", parenthetical="低声", confidence=0.9)
        assert line.character == "萧炎"


class TestScriptScene:
    def test_heading_is_assembled_not_stored(self):
        scene = ScriptScene(scene_id="CH01-S01", chapter_id="CH01",
                            int_ext="INT", location="客栈大厅", time_of_day="夜",
                            lines=[], characters_in_scene=["萧炎"])
        heading = f"{scene.int_ext}. {scene.location} - {scene.time_of_day}"
        assert heading == "INT. 客栈大厅 - 夜"


class TestScriptOutput:
    def test_minimal_valid_output(self):
        output = ScriptOutput(schema_version="1.0", title="测试小说",
                              source_novel="/path/to/novel.txt",
                              characters=[], scenes=[])
        assert output.title == "测试小说"

    def test_full_output(self):
        profile = CharacterProfile(name="萧炎", first_appearance="CH01-S01",
                                   appearance_count=1, dialogue_count=2,
                                   scenes=["CH01-S01"])
        scene = ScriptScene(scene_id="CH01-S01", chapter_id="CH01",
                            int_ext="EXT", location="山巅", time_of_day="日",
                            lines=[
                                ScriptLine(type="action", content="萧炎盘膝坐在青石之上。"),
                                ScriptLine(type="dialogue", content="三年了……", character="萧炎"),
                            ],
                            characters_in_scene=["萧炎"])
        output = ScriptOutput(schema_version="1.0", title="斗破苍穹",
                              source_novel="test.txt",
                              characters=[profile], scenes=[scene])
        assert len(output.scenes) == 1
        assert len(output.scenes[0].lines) == 2
```

- [ ] **Step 2: Run test — expect import failure**

Run: `C:\Users\asus\AppData\Local\Programs\Python\Python313\python.exe -m pytest tests/unit/test_models.py -v`
Expected: FAIL — "ModuleNotFoundError: No module named 'src.engine.models'"

- [ ] **Step 3: Write models.py**

Write `src/engine/models.py`:

```python
"""Pydantic v2 data models — source of truth for all pipeline schemas.

Every artifact model carries schema_version: Literal["1.0"].
Models are the contract between pipeline stages and between Phase 1 and Phase 2.
"""

from pydantic import BaseModel
from typing import Literal


# ============================================================
# Intermediate Artifacts (pipeline stage outputs)
# ============================================================

class PreprocessArtifact(BaseModel):
    """Output of Chinese text preprocessing (1.2)."""
    schema_version: Literal["1.0"]
    original_path: str
    cleaned_text: str           # Unified quotes, normalized paragraphs
    total_chars: int


class Chapter(BaseModel):
    """A single chapter extracted from the novel."""
    chapter_id: str             # "CH01"
    title: str                  # "第一章 斗之气三段"
    content: str                # Chapter plain text (excluding title line)
    start_line: int
    end_line: int
    confidence: float = 1.0     # "第X章" = 1.0, "一、" = 0.6


class ChapterArtifact(BaseModel):
    """Output of chapter recognition & split (1.3)."""
    schema_version: Literal["1.0"]
    chapters: list[Chapter]


class Scene(BaseModel):
    """A single scene detected within a chapter."""
    scene_id: str               # "CH01-S01"
    chapter_id: str
    content: str
    boundary_keywords: list[str]  # Keywords that triggered the split
    location: str = "UNKNOWN"     # Extracted from boundary_keywords or first sentence
    int_ext: Literal["INT", "EXT", "INT/EXT", "UNKNOWN"] = "UNKNOWN"
    time_of_day: str = "UNKNOWN"  # "日" | "夜" | "晨" | "黄昏" | "UNKNOWN"
    confidence: float = 1.0       # "三天后" = 0.7, "---" = 0.5


class SceneArtifact(BaseModel):
    """Output of scene boundary detection (1.4)."""
    schema_version: Literal["1.0"]
    scenes: list[Scene]


class CharacterRef(BaseModel):
    """A character reference extracted from the novel."""
    name: str
    aliases: list[str] = []
    first_appearance: str       # scene_id


class CharacterArtifact(BaseModel):
    """Output of character NER (1.5)."""
    schema_version: Literal["1.0"]
    characters: list[CharacterRef]


class DialogueLine(BaseModel):
    """A single line of dialogue with attribution metadata."""
    dialogue_id: str
    scene_id: str
    line_index: int             # Position in scene — enables Phase 2 adjacency inference
    speaker: str | None = None  # None = unattributed, Phase 2 fills
    line: str                   # Dialogue text without quotes
    quote_style: str            # "“”" | "‘’" | "「」" | "『』"
    parenthetical: str | None = None  # "(冷冷地)" from quote context
    confidence: float           # 0.0–1.0, attribution confidence
    attribution_method: str     # "prefix_match" | "suffix_match" | "nearest_name" | "unattributed"


class DialogueArtifact(BaseModel):
    """Output of dialogue extraction & attribution (1.6)."""
    schema_version: Literal["1.0"]
    dialogues: list[DialogueLine]


# ============================================================
# Output Models (final ScriptOutput)
# ============================================================

class CharacterProfile(BaseModel):
    """Aggregated character view for final output."""
    name: str
    aliases: list[str] = []
    role: str | None = None                     # Phase 2 fills
    description: str | None = None              # Phase 2 fills
    first_appearance: str                       # scene_id
    appearance_count: int                       # scenes appeared in
    dialogue_count: int                         # lines spoken
    scenes: list[str]                           # all scene_ids


class ScriptLine(BaseModel):
    """A single line in the script output."""
    type: Literal["action", "dialogue", "transition", "note"]
    content: str
    character: str | None = None
    parenthetical: str | None = None
    confidence: float = 1.0


class ScriptScene(BaseModel):
    """A scene in the final script output.

    heading is NOT stored — it is assembled at export time:
        f"{int_ext}. {location} - {time_of_day}"
    """
    scene_id: str
    chapter_id: str
    int_ext: Literal["INT", "EXT", "INT/EXT", "UNKNOWN"]
    location: str
    time_of_day: str
    location_note: str | None = None
    lines: list[ScriptLine]
    characters_in_scene: list[str]


class ScriptOutput(BaseModel):
    """Top-level script output — the final product of Phase 1 pipeline."""
    schema_version: Literal["1.0"]
    title: str
    source_novel: str
    characters: list[CharacterProfile]
    scenes: list[ScriptScene]
```

- [ ] **Step 4: Run tests — all pass**

Run: `C:\Users\asus\AppData\Local\Programs\Python\Python313\python.exe -m pytest tests/unit/test_models.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
cd "E:/Novel-to-Script Adaptation"
git add src/engine/models.py tests/unit/test_models.py
git commit -m "feat: add Pydantic v2 data models as schema source of truth

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Chinese Text Preprocessing (Task 1.2)

**Files:**
- Create: `src/engine/preprocess.py`
- Create: `tests/unit/test_preprocess.py`

**Purpose:** Unify four Chinese quote styles, normalize paragraphs, strip BOM. Pure functions, no I/O.

**Key functions:** `unify_quotes(text)` — converts 「」『』 to standard quotes; `normalize_paragraphs(text)` — collapses blank lines, normalizes line endings; `preprocess(text, source_path) -> PreprocessArtifact`

**TDD flow:**
1. Write `tests/unit/test_preprocess.py` with tests for: CJK corner bracket conversion, white corner bracket conversion, mixed all-four styles, blank line collapse, trailing whitespace strip, CRLF normalization, BOM removal, artifact field correctness
2. Run test → FAIL (ModuleNotFoundError)
3. Write `src/engine/preprocess.py` implementing the three functions
4. Run test → ALL PASS
5. Commit

---

### Task 3: Chapter Recognition & Split (Task 1.3)

**Files:**
- Create: `src/engine/chapter.py`
- Create: `tests/unit/test_chapter.py`

**Purpose:** Identify chapter boundaries via regex patterns. Confidence per pattern type.

**Key functions:** `detect_chapter_boundaries(text) -> list[(line_idx, title, confidence)]`; `split_chapters(artifact: PreprocessArtifact) -> ChapterArtifact`

**Chapter patterns (ordered by confidence):**
1. `第X章` (1.0) — highest confidence
2. `Chapter X` (1.0)
3. `第X回` (0.9)
4. `章X` (0.8)
5. `一、` etc (0.6)
6. 序章/终章/尾声/楔子 (0.5)

**TDD flow:**
1. Write `tests/unit/test_chapter.py`
2. Run test → FAIL
3. Write `src/engine/chapter.py`
4. Run test → ALL PASS
5. Commit

---

### Task 4: Scene Boundary Detection (Task 1.4)

**Files:**
- Create: `src/engine/scene.py`
- Create: `tests/unit/test_scene.py`

**Purpose:** Detect scene boundaries via time keywords, separators. Classify INT/EXT, time_of_day, location.

**Key functions:** `detect_scenes(artifact: ChapterArtifact) -> SceneArtifact`; `classify_time_of_day(text) -> str`; `classify_int_ext(text) -> str`

**TDD flow:**
1. Write `tests/unit/test_scene.py`
2. Run test → FAIL
3. Write `src/engine/scene.py`
4. Run test → ALL PASS
5. Commit

---

### Task 5: Character NER (Task 1.5)

**Files:**
- Create: `src/engine/character.py`
- Create: `tests/unit/test_character.py`

**Prerequisite:** `python -m spacy download zh_core_web_trf`

**Purpose:** Extract character names using spaCy NER with jieba fallback. Global dedup, filter single-occurrence names.

**Key functions:** `extract_names_spacy(text) -> list[str]`; `extract_names_jieba_fallback(text) -> list[str]`; `deduplicate_characters(refs) -> list[CharacterRef]`; `extract_characters(artifact: SceneArtifact) -> CharacterArtifact`

**TDD flow:**
1. Write `tests/unit/test_character.py`
2. Run test (skip slow) → FAIL
3. Write `src/engine/character.py`
4. Run test (skip slow) → ALL PASS
5. Commit

---

### Task 6: Dialogue Extraction & Attribution (Task 1.6)

**Files:**
- Create: `src/engine/dialogue.py`
- Create: `tests/unit/test_dialogue.py`

**Purpose:** Extract dialogue from four quote styles. Attribute via 5-tier priority: prefix_match > suffix_match > nearest_name > prev_speaker > unattributed. Extract parentheticals.

**Key functions:** `extract_quoted_texts(text) -> list[(pos, quoted, style)]`; `infer_speaker(before, after, char_names, line_idx, prev_speakers) -> (speaker, confidence, method)`; `extract_parenthetical(text) -> str|None`; `extract_dialogues(artifact: SceneArtifact) -> DialogueArtifact`

**TDD flow:**
1. Write `tests/unit/test_dialogue.py`
2. Run test → FAIL
3. Write `src/engine/dialogue.py`
4. Run test → ALL PASS
5. Commit

---

### Task 7: Pipeline Converter (Task 1.7)

**Files:**
- Create: `src/engine/converter.py`
- Create: `tests/unit/test_converter.py`

**Purpose:** Orchestrate full pipeline, cache management, Pydantic validation at each stage, final assembly.

**Key class:** `Pipeline`

**Key methods:**
- `run(input_path, *, output_path, cache_dir, no_cache, resume_from, confidence_threshold, verbose) -> ScriptOutput`
- `_assemble(preprocessed, chapters, scenes, characters, dialogues, confidence_threshold) -> ScriptOutput`

**_assemble() logic:**
1. Title from first non-empty line of preprocessed.cleaned_text
2. source_novel from preprocessed.original_path
3. Cross CharacterArtifact + DialogueArtifact → CharacterProfile[] (appearance_count, dialogue_count, scenes)
4. Scene[] → ScriptScene[] (no heading stored — assembled at export)
5. DialogueLine[] → ScriptLine[] (confidence_threshold: if confidence < threshold → speaker = None)
6. Build ScriptOutput

**Cache logic:**
- SHA256 of input file as cache key
- Each stage writes `<stage>.json` + `<stage>.meta`
- .meta: input_sha256, pipeline_version, module_version, created_at
- Cache valid only if SHA256 AND pipeline_version match
- `--no-cache` skips all; `--resume-from <stage>` starts from stage

**TDD flow:**
1. Write `tests/unit/test_converter.py`
2. Run test → FAIL
3. Write `src/engine/converter.py`
4. Run test → ALL PASS
5. Commit

---

### Task 8: CLI Interface (Task 1.8)

**Files:**
- Create: `src/cli/main.py`
- Create: `tests/integration/test_cli.py`

**CLI spec:**
```
novel2script input.txt -o output.yaml
novel2script input.txt -o output.yaml --no-cache
novel2script input.txt -o output.yaml --resume-from scene
novel2script input.txt -o output.yaml --confidence-threshold 0.5
novel2script input.txt -o output.yaml --verbose
novel2script --version
novel2script --schema
```

**TDD flow:**
1. Write `tests/integration/test_cli.py`
2. Run test → FAIL
3. Write `src/cli/main.py` (Typer app, thin layer)
4. Run test → ALL PASS
5. Commit

---

### Task 9: Integration — End-to-End Pipeline Test

**Files:**
- Create: `tests/integration/test_pipeline_e2e.py`

**Tests:**
1. Pipeline on basic_3ch.txt → output.yaml is valid YAML
2. Parse output back via ScriptOutput.model_validate → no ValidationError
3. Chapter/scene/dialogue counts are reasonable
4. All four quote styles correctly converted
5. Second run uses cache (faster, same output)

---

### Task 10: Ground Truth Dataset (Task 1.9)

**Files:**
- Create: `tests/fixtures/ground_truth/chapter_01_annotated.yaml`
- Create: `tests/fixtures/ground_truth/chapter_02_annotated.yaml`
- Create: `tests/fixtures/ground_truth/chapter_03_annotated.yaml`

**Purpose:** Manually annotated 3 chapters with correct scene boundaries, dialogue attributions, character lists. Baseline for Phase 1 self-test and Phase 2 quantification.

---

## Final Verification Checklist

- [ ] Full test suite: `pytest tests/ -v`
- [ ] Coverage: `pytest tests/ --cov=src/engine --cov-report=term`
- [ ] Coverage > 80%
- [ ] CLI smoke test on basic_3ch.txt
- [ ] CLI on mixed_quotes.txt — all quote styles handled
- [ ] CLI on no_dialogue.txt — no crashes on pure narration
