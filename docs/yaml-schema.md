# Novel-to-Script YAML Schema 定义（v1.0）

## 概述

本 Schema 定义中文网络小说经 `novel2script` 管线转换后产出的 YAML 剧本格式。设计目标：

- **人类可读、机器可解析**：既可以直接在文本编辑器中阅读修改，也可以用脚本批量处理
- **保留溯源信息**：每个字段都能追溯到管线中的具体处理阶段，方便调试和迭代
- **兼容编剧行业惯例**：场景标头格式（INT./EXT. 地点 - 时间）参照标准剧本格式，降低编剧的学习成本

---

## 顶层结构

```yaml
schema_version: '1.0'     # Schema 版本号，用于未来格式迁移的兼容性标识
title: 斗破苍穹             # 小说标题，从正文首行非章节标记文本提取
source_novel: novel.txt    # 源文件路径（Web 端为 "网页输入"）
characters: [...]          # 角色画像列表，按首次出场顺序排列
scenes: [...]              # 场景列表，按原文出现顺序排列
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|:----:|------|
| `schema_version` | `Literal["1.0"]` | ✓ | 固定值，用于未来的格式迁移。读取方应先检查此字段再解析后续内容 |
| `title` | `string` | ✓ | 小说标题。从预处理文本中提取，跳过章节标记（`第X章` 等），若无法提取则回退为文件名主干 |
| `source_novel` | `string` | ✓ | 源文件路径。供下游工具追溯原始素材 |
| `characters` | `list[CharacterProfile]` | ✓ | 可为空列表。角色按 `first_appearance` 升序排列 |
| `scenes` | `list[ScriptScene]` | ✓ | 可为空列表。场景按原文出现顺序排列 |

**设计理由**：

- **`schema_version` 放在顶层**：当 Schema 升级时，解析器无需深度遍历即可判断格式版本，避免向前兼容问题。这是 REST API 和配置文件格式的常见实践（如 OpenAPI 的 `openapi: "3.0"`）
- **`title` 与 `source_novel` 分离**：前者是作品的语义标识，后者是文件的物理路径。当同一小说被多次转换（如不同版本、不同格式的源文件），`title` 保持一致而 `source_novel` 可以区分
- **`characters` 放在 `scenes` 之前**：剧本阅读的自然顺序是先认识角色再看戏。技术上这也使单遍解析成为可能——读者先建立角色索引，后续场景中的 `character` 引用即可直接对照

---

## CharacterProfile — 角色画像

```yaml
- name: 萧炎                    # 角色名（规范形式）
  aliases: []                   # 别名列表，如 ["炎帝", "药岩"]
  role: 主角                    # 角色定位，可为 null
  description: 乌坦城萧家少主...  # 1-2 句身份性格描述，可为 null
  first_appearance: CH01-S01   # 首次出场的场景 ID
  appearance_count: 12          # 出场场景数
  dialogue_count: 87            # 台词句数（规则引擎或 LLM 归因结果）
  scenes:                       # 实际出场的场景 ID 列表（检测逻辑见下文）
  - CH01-S01
  - CH01-S03
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|:----:|------|
| `name` | `string` | ✓ | 规范角色名。从 NER 提取的首选名称（最长、频率最高者），去重合并后的权威标识 |
| `aliases` | `list[string]` | ✓ | 可为空。同一角色的别称、简称、尊称等。当前由 NER 去重阶段通过**完全匹配 + 子串包含**合并：若名称 A 是名称 B 的子串且 B 的频率 ≥ A，则将 A 合并为 B 的别名（如"萧"合并到"萧炎"）。此策略的已知局限是无法处理语义等同但字面无关的别名（如"药岩"是萧炎的化名但无法自动关联），也无法区分恰好同姓的不同角色。LLM 辅助的共指消解是解决此问题的可能方向，但当前版本未实现 |
| `role` | `Literal["主角", "配角", "龙套"] \| null` | | `null` 表示未分类（纯规则引擎模式）。AI 增强模式下由 `profile_characters()` 在一次 LLM 调用中批量填充，降级时由统计规则兜底（对话占比 >40% → 主角，>10% → 配角） |
| `description` | `string \| null` | | `null` 表示未生成（纯规则引擎模式或 AI 调用失败后的统计降级）。AI 模式下为 1-2 句中文描述，涵盖身份、性格、与主角的关系 |
| `first_appearance` | `string` | ✓ | 场景 ID。角色在小说中首次被检测到的场景，用于排序和出场时间线 |
| `appearance_count` | `int` | ✓ | ≥ 0。实际出场场景数。检测逻辑：角色在场景中说台词（dialogue 归因）OR 名字在场景原文中被提及（纯文本 `in` 匹配）。**已知局限**：纯文本匹配无法区分"角色确实在本场出场"和"角色仅在被回忆/提及"。例如"萧炎曾在三年前来过这里"会触发名字匹配，但萧炎并未在当前场景中实际出现。数字越高不代表检测越精确——它反映的是提及频率而非出场密度。当前版本不做语义区分 |
| `dialogue_count` | `int` | ✓ | ≥ 0。该角色作为说话人的台词总句数。来源于归因阶段（规则引擎为低置信度估计，AI 增强后为精确值） |
| `scenes` | `list[string]` | | 该角色实际出场的所有场景 ID，按场景出现顺序排列。用于下游的"本章出场角色"统计和角色关系图构建。与 `appearance_count` 共享相同的检测逻辑，因此面临相同的提及-vs-出场歧义 |

**设计理由**：

- **`role` 用中文枚举而非英文映射**：`主角/配角/龙套` 是网文读者和作者的原生分类体系，强制套用 `protagonist/deuteragonist/tertiary` 反而丢失了文化语境。中文枚举直接对接中国编剧的工作语言
- **`description` 与 `role` 分为两字段**：定位分类（主角/配角/龙套）是类型标签，适合程序化过滤（如"只导出主角台词"）；描述是自然语言摘要，适合人工浏览。两者用途不同，不应合并
- **`scenes` 存储为显式列表而非推导字段**：虽然可以从 scenes 数组反推，但显式存储消除了下游消费者的计算负担。对于有 50+ 场景的小说，这个预计算是有意义的去冗余化

---

## ScriptScene — 剧本场景

```yaml
- scene_id: CH01-S01            # 场景唯一标识
  chapter_id: CH01              # 所属章节
  int_ext: INT                  # 内外景分类
  location: 萧家大厅             # 地点名称
  time_of_day: 日               # 时间分类
  location_note: null            # 地点补充说明
  lines: [...]                  # 场景内容行
  characters_in_scene:           # 本场出场角色
  - 萧炎
  - 纳兰嫣然
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|:----:|------|
| `scene_id` | `string` | ✓ | 格式 `CH{章节编号}-S{场景编号}`，如 `CH01-S01`。章节编号从 01 开始补零，场景编号每章从 01 重置 |
| `chapter_id` | `string` | | 所属章节 ID，用于按章节分组和导航 |
| `int_ext` | `Literal["INT", "EXT", "INT/EXT", "UNKNOWN"]` | ✓ | 内外景分类。`INT/EXT` 表示跨内外场景（如一镜从室内走到室外），`UNKNOWN` 为无法判断时的默认值 |
| `location` | `string` | ✓ | 地点名称。从边界关键词或首句地点模式提取，值可为 `"UNKNOWN"` |
| `time_of_day` | `Literal["日", "夜", "晨", "黄昏", "UNKNOWN"]` | ✓ | 时间分类。`UNKNOWN` 为默认值。检测范围为场景前 200 字符 |
| `location_note` | `string \| null` | | 地点的补充说明。预留字段，当前管线未填充。可用于人工标注（如"萧家大厅（二楼书房）"） |
| `lines` | `list[ScriptLine]` | ✓ | 场景内容行。action 和 dialogue 交替排列，保持原文顺序 |
| `characters_in_scene` | `list[string]` | | 本场实际出场的角色名列表，按 `first_appearance` 排序。用于快速生成场景角色表。检测逻辑与 `CharacterProfile.scenes` 一致（对话参与 OR 名字提及），存在相同的提及-vs-出场歧义 |

**设计理由**：

- **场景标头不存储为字符串，而是分解为 `int_ext` + `location` + `time_of_day`**：这是本 Schema 最重要的设计决策之一。合成标头（如 `INT. 萧家大厅 - 日`）是展示层的事情。分解存储允许下游工具独立操作每个维度（按内景/外景过滤、按时间段分组、按地点聚类），而非依赖不稳定的正则解析。同时，枚举约束的 `int_ext` 和 `time_of_day` 保证了值的合法性——LLM 返回 "INTERIOR" 而非 "INT" 时会被 whitelist 校验拦截并回退
- **`scene_id` 按章重置编号**：`CH02-S01` 而非全局递增。这反映原文结构——场景编号只在章节内有序，跨章比较编号无意义。对于 1000+ 章的长篇网文，全局编号会迅速膨胀到难以阅读的数值
- **`characters_in_scene` 是预计算列表**：下游消费者（如前端的"本场角色"标签）无需遍历全部 dialogue line 即可获得角色列表。数据冗余但消除了 O(n) 的遍历成本
- **`location_note` 预留为 `null`**：当前管线不填充此字段，但预留它可以让人工标注者在不修改 `location` 的情况下补充细节（如"萧家大厅（偏厅）"），避免覆盖自动提取的结果

---

## ScriptLine — 剧本行

```yaml
- type: dialogue               # 行类型
  content: 三年了……终于摸到了斗者门槛。  # 纯文本（无引号）
  character: 萧炎               # 说话人，action 行和未归因行可为 null
  parenthetical: 低声            # 括号内的表演提示，可为 null
  confidence: 0.85              # 归因置信度 [0.0, 1.0]
```

```yaml
- type: action                  # 动作/描写行
  content: 萧炎推开门，走入大厅。  # 场景描写文本
  character: null
  parenthetical: null
  confidence: 1.0
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|:----:|------|
| `type` | `Literal["action", "dialogue", "transition", "note"]` | ✓ | 行类型。当前管线仅输出 `action` 和 `dialogue`，`transition` 和 `note` 预留给未来扩展 |
| `content` | `string` | ✓ | 纯文本内容。dialogue 行已剥离引号，只保留说话内容。action 行保留原文段落格式（含换行符） |
| `character` | `string \| null` | | 说话人角色名。action 行始终为 `null`。dialogue 行在归因失败或低于置信度阈值时可为 `null` |
| `parenthetical` | `string \| null` | | 括号内的表演提示文本（如原文 `"（低声）"` → `"低声"`）。**当前支持的格式**：全角圆括号 `（）` 和半角圆括号 `()` 内嵌的简短描述。**已知不支持**：中文网文中的方括号 `【】`、破折号引导的情绪描写（`——愤怒地`）、紧跟在引号后未加括号的动作短语（`"……"萧炎冷笑一声`中的`冷笑一声`）。这些情况下 parenthetical 将为 `null`，表演提示信息丢失 |
| `confidence` | `float` | | 归因置信度，范围 `[0.0, 1.0]`。action 行始终为 `1.0`。dialogue 行取决于归因方法（见下表）。下游可通过 `--confidence-threshold` 过滤低置信度归因 |

### 归因置信度参考

| 方法 | 置信度 | 说明 |
|------|:------:|------|
| `prefix_match` | 0.85 | 引号前 15 字符内匹配到 `角色名 + 说话动词`（如"萧炎说道"） |
| `suffix_match` | 0.75 | 引号后 10 字符内匹配到 `说话动词 + 角色名` |
| `nearest_name` | 0.50 | 引号前后 50 字符内找到角色名 |
| `prev_speaker` | 0.30 | 基于两人交替对话模式的猜测 |
| `llm` | LLM 返回 | AI 增强后的归因，由 LLM 自行评估的置信度 |
| `unattributed` | 0.00 | 所有方法均失败，或无角色名可匹配 |

**置信度的跨方法可比性问题**：上述数值虽然都在 `[0.0, 1.0]` 区间内，但它们的来源不可比。规则引擎的 0.85/0.75/0.50/0.30 是管线中的硬编码设计常数——反映的是设计者对各 tier 相对可靠性的排序判断，而非统计概率。LLM 返回的置信度则是模型对 token 概率的 self-report（如 `role_confidence: 0.95`），不同模型的置信度校准曲线不同，且可能与实际准确率存在系统性偏差。下游如果对两者一视同仁（例如用 `--confidence-threshold 0.7` 统一过滤），LLM 的 0.85 和规则引擎的 0.85 含义完全不同：前者是模型自评，后者是设计者预设。当前 Schema 层面不做归一化——将区分责任留给下游消费者。建议做法：按 `attribution_method` 分层设置阈值，而非用一个全局数值截断。

### ScriptLine 是扁平列表而非嵌套结构

场景内容被展开为扁平的 `lines` 数组，action 和 dialogue 交替排列。这是有意为之——它直接映射到剧本的一行一行阅读顺序。虽然丢失了"某段 action 和紧跟的 dialogue 属于同一叙事单元"的结构信息，但换来了简单性：任何下游渲染器只需遍历 `lines` 并按 `type` 分发即可生成输出，无需递归解析。

---

## 完整示例

以下是一个最小但完整的两场景剧本，展示了 Schema 中所有字段在真实 YAML 中的形态：

```yaml
schema_version: '1.0'
title: 斗破苍穹
source_novel: novel.txt
characters:
- name: 萧炎
  aliases: []
  role: 主角
  description: 乌坦城萧家少主，三年隐忍只为通过迦南学院考核。
  first_appearance: CH01-S01
  appearance_count: 2
  dialogue_count: 3
  scenes:
  - CH01-S01
  - CH01-S02
- name: 纳兰嫣然
  aliases: []
  role: 配角
  description: 云岚宗弟子，与萧炎有旧，性格清冷但暗藏关切。
  first_appearance: CH01-S01
  appearance_count: 1
  dialogue_count: 1
  scenes:
  - CH01-S01
scenes:
- scene_id: CH01-S01
  chapter_id: CH01
  int_ext: EXT
  location: 测试山巅
  time_of_day: 晨
  location_note: null
  lines:
  - type: action
    content: 测试山巅，云海翻腾。萧炎盘膝坐在青石之上。
    character: null
    parenthetical: null
    confidence: 1.0
  - type: dialogue
    content: 三年了……终于摸到了斗者门槛。
    character: 萧炎
    parenthetical: 低声
    confidence: 0.85
  - type: action
    content: 纳兰嫣然踏空而来，落在萧炎面前。
    character: null
    parenthetical: null
    confidence: 1.0
  - type: dialogue
    content: 明日就是迦南学院的考核，你准备好了吗？
    character: 纳兰嫣然
    parenthetical: null
    confidence: 0.75
  characters_in_scene:
  - 萧炎
  - 纳兰嫣然
- scene_id: CH01-S02
  chapter_id: CH01
  int_ext: INT
  location: 迦南学院大殿
  time_of_day: 日
  location_note: null
  lines:
  - type: action
    content: 大殿中人头攒动，数十名考生列队等候。
    character: null
    parenthetical: null
    confidence: 1.0
  - type: dialogue
    content: 下一个，萧炎。
    character: null
    parenthetical: null
    confidence: 0.0
  characters_in_scene:
  - 萧炎
```

这个示例中：

- `纳兰嫣然` 的 `appearance_count` 为 1（只在 CH01-S01 出场），CH01-S02 的 `characters_in_scene` 只有萧炎
- CH01-S02 的对话 "下一个，萧炎。" 归因失败（`confidence: 0.0`, `character: null`）——说话人是考官，但角色列表中没有这个角色
- `parenthetical` 只在萧炎的第一句台词有值（原文为"（低声）"），其余为 null

---

## 设计总览

### 为什么是 YAML

1. **人类可读**：YAML 比 JSON 更适合中文内容——无需转义大部分标点，多行字符串支持友好（`|` 和 `>`）
2. **保留排序**：YAML 的列表项天然有序，剧本场景和行顺序是关键信息，JSON 虽然也有序但容易被人误以为无序
3. **编剧友好**：场景标头格式（`INT. 地点 - 日`）本身接近 YAML 的美学，降低非技术人员直接编辑的心理门槛
4. **Python 原生**：PyYAML 的 `model_dump(mode="python")` 可以无损往返 Pydantic 模型。下游工具可以直接 `yaml.safe_load()` 后构造回 `ScriptOutput.model_validate()`

### 为什么不是标准剧本格式（Final Draft / Fountain / .fdx）

这些格式是为**人类编剧**设计的创作工具格式，强调页面排版、页码、修订标记等印刷时代的概念。本项目的输出定位是**结构化数据**——它是一个中间表示（IR），可以被渲染为任何目标格式。YAML Schema 的设计哲学是"捕捉语义，延迟渲染"：存储 INT/EXT 分类而非 `INT. LOCATION - DAY` 字符串，存储角色列表而非场景标题下的角色名单。任何渲染格式（Fountain、HTML、PDF、甚至 .fdx）都可以从这个 IR 派生。

### 扩展性

- **`location_note`** 和 **`transition`/`note` 行类型**是预留的扩展点，当前不做填充但不破坏 Schema 兼容性
- **`schema_version`** 机制允许未来新增字段或调整类型而不破坏旧文件的解析
- **`confidence` 字段**为所有 heuristics/AI 输出提供了统一的不确定性表达渠道，但使用者应注意跨方法可比性问题（见上文置信度章节）
