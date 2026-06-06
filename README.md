# Novel-to-Script Adaptation

AI 辅助中文网文转剧本工具。将中文网络小说文本转换为结构化 YAML 剧本。

## Phase 1 — 核心引擎（MVP）✅

纯规则引擎管线，无 AI 依赖。从中文小说文本生成合法 YAML 剧本。

## Phase 2 — LLM 增强 ✅

通过 DeepSeek API 实现 AI 驱动的对话归属、场景分类和角色验证。两轮归因，
每次 LLM 调用失败自动回退规则引擎。

### 效果对比（basic_3ch.txt，3 章）

| 维度 | 规则引擎 | AI 增强 |
|------|:------:|:-----:|
| 角色准确率 | 40% (2/5) | 100% (2/2) |
| 场景分类 | 0% (0/3) | 100% (3/3) |
| 对话归属 | 43% (11/26) | 100% (26/26) |

### 环境要求

- Python 3.12+
- spaCy 中文模型（可选，缺失时自动降级为 jieba 分词）：

```bash
python -m spacy download zh_core_web_trf
```

### 安装

```bash
# 仅规则引擎
pip install -e ".[dev]"

# 含 AI 增强
pip install -e ".[dev,ai]"
```

### 使用

```bash
# 基本转换（纯规则引擎）
novel2script input.txt -o output.yaml

# AI 增强转换
set NOVEL2SCRIPT_API_KEY=sk-你的deepseek密钥
novel2script input.txt -o output.yaml --ai --verbose

# 强制跳过缓存重跑
novel2script input.txt -o output.yaml --no-cache

# 从指定阶段恢复
novel2script input.txt -o output.yaml --resume-from scene

# 过滤低置信度对话归属
novel2script input.txt -o output.yaml --confidence-threshold 0.6

# 查看版本
novel2script --version
novel2script --schema
```

### 管线

```
                    ┌─ 规则引擎（始终运行）─┐
novel.txt → preprocess → chapters → scenes → characters → dialogues ─┤
                    │                         │
                    └─ --ai? → AI 增强 ───────┘
                               ├─ 两轮对话归因
                               ├─ 场景分类（内外景/地点/时间）
                               └─ 角色验证（过滤误判）
                                                ↓
                                          assemble → output.yaml
```

所有中间结果通过 SHA256 缓存。`--no-cache` 强制重跑。
AI 调用按轮次独立缓存；网络失败自动回退规则引擎结果。

### 项目结构

```
src/
├── engine/
│   ├── models.py         # Pydantic v2 数据模型
│   ├── preprocess.py     # 引号统一、段落规范化
│   ├── chapter.py        # 章节边界检测（6 种正则模式）
│   ├── scene.py          # 场景切分 + 内外景/时间分类
│   ├── character.py      # spaCy NER + jieba 降级，去重，频率过滤
│   ├── dialogue.py       # 4 种引号风格，5 级说话人归因
│   ├── converter.py      # 管线编排，SHA256 缓存，YAML 组装
│   └── ai_enhancer.py    # DeepSeek LLM 集成（Phase 2）
└── cli/
    └── main.py           # Typer CLI 入口
tests/
├── fixtures/novels/      # 中文小说测试片段
├── fixtures/expected/    # 预期 YAML 输出
├── fixtures/ground_truth/ # 人工标注评估基准
├── unit/                 # 单模块单元测试
└── integration/          # E2E 管线 + CLI 测试
```

### 测试

```bash
# 跑全部测试（排除 spaCy 慢测试）
pytest tests/ -v -k "not slow"

# 带覆盖率
pytest tests/ --cov=src/engine --cov-report=term
```

### 验收标准

#### Phase 1 ✅

- 输入 3 章中文小说，一条命令完成转换
- 输出为合法 YAML，符合 Schema 定义
- 通过标准"第X章"标记识别章节
- 通过时间/地点关键词检测场景边界
- 支持 4 种中文引号风格的对话提取
- 对话归因（纯规则引擎，低置信度标注）
- Ground Truth 数据集骨架（3 章）
- 核心模块单元测试覆盖
- CLI 提供 `--help` 和基本错误信息

#### Phase 2 ✅

- 两轮 LLM 对话归因（DeepSeek，OpenAI 兼容 API）
- 场景分类：内外景、地点、时间（LLM）
- 角色验证：过滤 jieba 误判的假角色（LLM）
- `--ai` 开关，配 API Key 检测和优雅降级
- SHA256 缓存 LLM 调用；网络失败回退规则引擎
- 说话线索标注，辅助 LLM 追踪快速交替对话
- 标题提取：跳过章节标记，回退文件名
- `--confidence-threshold` 基于实际计算值过滤
- 全部 202 个测试通过

### 后续计划

- **Phase 2 后续**：AI 角色定位分类与角色描述生成
- **Phase 3A**：Web 应用（FastAPI + React + Docker）
- **Phase 3B**：高级功能（版本管理、批注、协作）
