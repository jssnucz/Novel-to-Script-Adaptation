"""AI-enhanced pipeline stages for the Novel-to-Script Adaptation pipeline.

Provides three LLM-powered enhancement functions that improve upon the
rule-engine outputs:

- ``enhance_dialogue_attribution`` — re-attribute low-confidence dialogue
  lines using LLM context understanding.
- ``enhance_scene_classification`` — improve INT/EXT, location, and time-of-day
  classification.
- ``verify_characters`` — filter false-positive character names from jieba
  extraction.

Every function follows the same contract:

1. Try the LLM call.
2. On success, update the artifact with LLM results (cached by SHA256).
3. On **any** failure (network, auth, timeout, bad response), keep the
   rule-engine result and set ``confidence`` to 0.3 for the affected items.

DeepSeek API is used via the OpenAI-compatible SDK.  Set the environment
variable ``NOVEL2SCRIPT_API_KEY`` to enable AI enhancement.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from engine.models import DialogueLine, Scene

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DeepSeek client (lazy, thread-safe enough for single-thread pipeline use)
# ---------------------------------------------------------------------------

_client: Any = None  # openai.OpenAI | None
_client_init_attempted: bool = False


def _get_client() -> Any | None:  # -> openai.OpenAI | None
    """Return an OpenAI-compatible DeepSeek client, or ``None`` if not configured.

    Reads ``NOVEL2SCRIPT_API_KEY`` from the environment.  Returns ``None``
    (silently) when the key is missing or the ``openai`` SDK is unavailable.
    """
    global _client, _client_init_attempted

    if _client is not None:
        return _client
    if _client_init_attempted:
        return None

    _client_init_attempted = True

    api_key = os.environ.get("NOVEL2SCRIPT_API_KEY")
    if not api_key:
        logger.debug("NOVEL2SCRIPT_API_KEY not set — AI enhancement disabled")
        return None

    try:
        import openai  # type: ignore[import-untyped]

        _client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
        )
        logger.info("DeepSeek client initialized")
        return _client
    except Exception:
        logger.debug("Failed to init DeepSeek client", exc_info=True)
        return None


def is_ai_available() -> bool:
    """Return ``True`` when AI enhancement is configured and ready."""
    return _get_client() is not None


# ---------------------------------------------------------------------------
# Quote reconstruction (mirrors converter._reconstruct_quote)
# ---------------------------------------------------------------------------

_QUOTE_OPEN: dict[str, str] = {
    "double": '"',
    "single": "'",
    "corner": "「",  # 「
    "white_corner": "『",  # 『
}
_QUOTE_CLOSE: dict[str, str] = {
    "double": '"',
    "single": "'",
    "corner": "」",  # 」
    "white_corner": "』",  # 』
}


def _reconstruct_quote(line: str, style: str) -> str:
    """Wrap *line* in the quote delimiters for *style*."""
    open_delim = _QUOTE_OPEN.get(style, '"')
    close_delim = _QUOTE_CLOSE.get(style, '"')
    return f"{open_delim}{line}{close_delim}"


# ---------------------------------------------------------------------------
# SHA256 cache helpers
# ---------------------------------------------------------------------------


def _cache_key(*parts: str) -> str:
    """Return a SHA256 hex digest of the concatenated *parts*."""
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
    return h.hexdigest()


def _read_cache(cache_path: Path, expected_key: str) -> dict | None:
    """Return cached data if the stored key matches *expected_key*."""
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text("utf-8"))
        if data.get("cache_key") == expected_key:
            return data
    except Exception:
        logger.debug("Cache read failed for %s", cache_path, exc_info=True)
    return None


def _write_cache(cache_path: Path, cache_key_value: str, payload: dict) -> None:
    """Write *payload* to *cache_path* with the given key."""
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"cache_key": cache_key_value, **payload}
        cache_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        logger.debug("Cache write failed for %s", cache_path, exc_info=True)


# ===================================================================
# 1. Dialogue Attribution Enhancement (two-round)
# ===================================================================
#
# Round 1: DISCOVERY — LLM identifies new characters not in the known list,
#           then does a best-effort attribution of low-confidence lines.
# Round 2: ATTRIBUTION (only when new characters were discovered) — LLM
#           re-attributes with a complete character list, focusing purely on
#           tracking who speaks each line.
#
# This separation prevents "discover new characters" and "track who speaks"
# from competing for the LLM's attention, which caused the CH03 collapse.


def enhance_dialogue_attribution(
    scene_id: str,
    scene_content: str,
    dialogue_lines: list[DialogueLine],
    character_names: list[str],
    cache_dir: str = "./cache",
) -> list[DialogueLine]:
    """Re-attribute low-confidence dialogue lines with LLM (two-round).

    **Only** lines where ``confidence < 0.6`` are sent to the LLM.
    High-confidence rule-engine results are kept as-is.

    Two-round strategy
    ------------------
    1. **Round 1 (discovery)**: LLM is asked to first list any new characters
       it notices in the scene (``discovered_characters``), then attribute
       the low-confidence lines.  Focus is on discovery.
    2. If new characters were found, merge them into the known list and run
       **Round 2 (attribution)**.  With the complete character list, the LLM
       only needs to track who speaks — no discovery overhead.

    Cache
    -----
    Round 1 → ``ai_dialogue_{scene_id}_r1.json``
    Round 2 → ``ai_dialogue_{scene_id}_r2.json``

    If round 2 fails, round 1 results are used directly.
    """
    client = _get_client()
    if client is None:
        return dialogue_lines

    low_indices: list[int] = [
        i for i, dl in enumerate(dialogue_lines) if dl.confidence < 0.6
    ]
    if not low_indices:
        return dialogue_lines

    # ---- Round 1: Discovery ----
    r1_result = _run_attribution_round(
        client=client,
        scene_id=scene_id,
        scene_content=scene_content,
        dialogue_lines=dialogue_lines,
        character_names=character_names,
        low_indices=low_indices,
        cache_dir=cache_dir,
        round_num=1,
    )
    if r1_result is None:
        return _fallback_dialogue(dialogue_lines, low_indices)

    discovered: list[str] = r1_result.get("discovered_characters", [])
    r1_attributions: list[dict] = r1_result.get("attributions", [])

    # If no new characters discovered, round 1 results are final
    if not discovered:
        return _apply_dialogue_updates(dialogue_lines, r1_attributions)

    logger.info(
        "Round 1 discovered %d new characters for %s: %s",
        len(discovered), scene_id, ", ".join(discovered),
    )

    # Merge discovered characters into known list
    merged_names = list(dict.fromkeys(character_names + discovered))

    # ---- Round 2: Precise attribution with complete list ----
    r2_result = _run_attribution_round(
        client=client,
        scene_id=scene_id,
        scene_content=scene_content,
        dialogue_lines=dialogue_lines,
        character_names=merged_names,
        low_indices=low_indices,
        cache_dir=cache_dir,
        round_num=2,
    )
    if r2_result is None:
        # Round 2 failed — fall back to round 1 results
        logger.debug("Round 2 failed for %s — using round 1 results", scene_id)
        return _apply_dialogue_updates(dialogue_lines, r1_attributions)

    return _apply_dialogue_updates(dialogue_lines, r2_result.get("attributions", []))


# ---------------------------------------------------------------------------
# Speaker-clue annotation (pre-processes scene text for the LLM)
# ---------------------------------------------------------------------------

# Speech verbs whose presence before a quote signals that the preceding
# character/descriptor is the speaker.
_SPEAKER_CLUE_VERBS: list[str] = sorted(
    ["说道", "问道", "答道", "回答道", "回答说", "开口", "喃喃", "低语",
     "自语", "冷笑", "说", "道", "问", "答", "喊", "叫", "嚷", "吼",
     "叹", "骂", "哭", "喝", "开口说", "轻声说", "淡淡说", "笑着说",
     "不紧不慢地说", "冷冷道", "缓缓说"],
    key=len, reverse=True,
)

# Regex: NAME (2-4 chars) + speech verb — strongest clue
_NAME_VERB_CLUE_RE = re.compile(
    r"([一-鿿]{2,4})\s*("
    + "|".join(re.escape(v) for v in _SPEAKER_CLUE_VERBS)
    + ")"
)

# Regex: "XX的声音" pattern
_VOICE_CLUE_RE = re.compile(r"([一-鿿]{1,6})的声音")

# Regex: descriptor (老者/少年/女子/男子/大汉/声音) + speech verb
_DESCRIPTOR_VERB_RE = re.compile(
    r"((?:老者|少年|女子|男子|大汉|中年|妇人|姑娘|公子的)?声音|"
    r"老者|少年|女子|男子|大汉|中年|妇人|姑娘|公子)"
    r"\s*("
    + "|".join(re.escape(v) for v in _SPEAKER_CLUE_VERBS)
    + ")?"
)


def _annotate_scene_with_speaker_clues(
    scene_content: str,
    dialogue_lines: list[DialogueLine],
    character_names: list[str],
) -> str:
    """Insert ``[说话线索: ...]`` markers before each dialogue in the scene.

    Scans the action text between dialogues for speaker clues (name + speech
    verb, voice indicators, descriptor + verb) and inserts explicit hints
    that help the LLM track who is speaking in rapid-fire exchanges.
    """
    if not dialogue_lines:
        return scene_content

    parts: list[str] = []
    cursor = 0

    for i, dl in enumerate(dialogue_lines):
        quoted = _reconstruct_quote(dl.line, dl.quote_style)
        pos = scene_content.find(quoted, cursor)
        if pos == -1:
            pos = scene_content.find(quoted)
        if pos == -1:
            continue

        # Text between previous cursor and this quote
        action_text = scene_content[cursor:pos]

        # Extract speaker clues from action text
        clues: list[str] = _extract_clues(action_text, character_names)

        # Append action text
        parts.append(action_text)

        # Append clue markers before the dialogue
        for clue in clues:
            parts.append(f"\n[说话线索: {clue}]\n")

        # Append the quoted dialogue
        parts.append(quoted)
        cursor = pos + len(quoted)

    # Trailing text
    if cursor < len(scene_content):
        parts.append(scene_content[cursor:])

    return "".join(parts)


def _extract_clues(
    action_text: str,
    character_names: list[str],
) -> list[str]:
    """Extract speaker clues from action text between dialogues.

    Returns a list of clue strings like ``"萧炎 → 接下来说话人"``.
    """
    clues: list[str] = []

    # 1. Name + speech verb ("萧炎说道", "药老不紧不慢地说")
    for m in _NAME_VERB_CLUE_RE.finditer(action_text):
        name = m.group(1)
        verb = m.group(2)
        if name in character_names or len(name) >= 2:
            clues.append(f"{name}{verb} → 说话人: {name}")

    # 2. Voice pattern ("一道苍老的声音", "考核官的声音")
    for m in _VOICE_CLUE_RE.finditer(action_text):
        voice_owner = m.group(1)
        _voice_blacklist = {"他", "她", "自己", "一阵", "一道", "那", "这", "什么"}
        if voice_owner not in _voice_blacklist and not any(
            voice_owner.startswith(p) for p in _voice_blacklist
        ):
            clues.append(f"{voice_owner}的声音 → 说话人: {voice_owner}")

    # 3. Implicit speaker from reaction patterns
    # "XX瞳孔骤缩，猛地站起身/愣了愣/深吸一口气/沉默/抚须一笑" → XX will speak
    # Allow up to 15 chars between name and reaction verb (Chinese prose often has modifiers between name and action)
    _REACTION_WORDS = (
        r"站起身|愣了愣|深吸|沉默|皱眉|苦笑|怒意|坚定|喊道|吼道|站起身"
        r"|涌上|瞳孔|抚须|一笑|冷哼一声|微微一笑|苦笑一声|开口道|淡淡道"
        r"|冷冷道|怒道|叹道|笑道|问道|答道"
    )
    for name in character_names:
        react_pattern = re.compile(
            re.escape(name) + r".{0,15}?(?:" + _REACTION_WORDS + r")"
        )
        if react_pattern.search(action_text):
            if f"说话人: {name}" not in " ".join(clues):
                clues.append(f"{name}的动作 → 可能是{name}在说或即将说话")

    return clues


# ---------------------------------------------------------------------------
# Round runner (shared by round 1 and round 2)
# ---------------------------------------------------------------------------


def _run_attribution_round(
    *,
    client,
    scene_id: str,
    scene_content: str,
    dialogue_lines: list[DialogueLine],
    character_names: list[str],
    low_indices: list[int],
    cache_dir: str,
    round_num: int,
) -> dict | None:
    """Run one round of LLM dialogue attribution.

    Returns the parsed JSON result dict, or ``None`` on failure.
    """
    cache_key_val = _cache_key(
        scene_content,
        json.dumps(low_indices),
        json.dumps(character_names),
        str(round_num),
    )
    cache_path = Path(cache_dir) / f"ai_dialogue_{scene_id}_r{round_num}.json"

    # Try cache
    cached = _read_cache(cache_path, cache_key_val)
    if cached is not None:
        return cached

    char_list = "、".join(character_names) if character_names else "（从上下文中识别）"

    # Build dialogue manifest
    dialogue_manifest: list[str] = []
    for i in low_indices:
        dl = dialogue_lines[i]
        quoted = _reconstruct_quote(dl.line, dl.quote_style)
        dialogue_manifest.append(
            f"  [{i}] {quoted}  ← 规则引擎猜测: {dl.speaker or 'UNKNOWN'} (conf={dl.confidence})"
        )

    # Annotate scene with pre-extracted speaker clues from action lines
    annotated_scene = _annotate_scene_with_speaker_clues(
        scene_content, dialogue_lines, character_names,
    )

    # Build a conversation transcript prompt — asking the LLM to
    # *reconstruct* the dialogue flow forces turn-by-turn tracking,
    # which is more reliable than labelling numbered indices.
    transcript_guide: list[str] = []
    for i, dl in enumerate(dialogue_lines):
        quoted = _reconstruct_quote(dl.line, dl.quote_style)
        if i in low_indices:
            transcript_guide.append(f"  ???: {quoted}  ← 请标注说话人")
        else:
            speaker = dl.speaker or "???"
            transcript_guide.append(f"  {speaker}: {quoted}  ← 已标注")

    # Different prompts for round 1 vs round 2
    if round_num == 1:
        task_desc = """你的任务分两步：
第一步：通读场景全文，列出文中出现的所有说话人（包括已知角色和未知的新角色）。
第二步：把场景改写为对话脚本，标注每句话的说话人。"""
        output_format = """{
  "discovered_characters": ["考核官", "路人", "药老"],
  "attributions": [
    {"line_index": 0, "speaker": "考核官", "confidence": 0.9},
    ...
  ]
}"""
    else:
        task_desc = """你的任务：把场景改写为对话脚本，标注每句话的说话人。
已知角色列表是完整的，所有说话人都在列表中。请追踪对话的交替模式来判断谁说哪句。"""
        output_format = """{
  "discovered_characters": [],
  "attributions": [
    {"line_index": 0, "speaker": "萧炎", "confidence": 0.95},
    ...
  ]
}"""

    prompt = f"""你是中国网文剧本顾问。{task_desc}

【已知角色】{char_list}

【场景原文（含说话线索标记）】
---
{annotated_scene}
---

【对话脚本（??? 的行需要你标注说话人）】
{chr(10).join(transcript_guide)}

【标注规则】
1. 逐句通读场景原文，追踪对话的交替模式。
2. [说话线索] 标记可信度高（≈90%），请优先参考。
3. 交替对话中，如果上一句是角色A，下一句通常是对手角色B。
4. 如果完全无法判断，写 null。
5. confidence: 0.95=非常确定，0.8=比较确定，0.5=推测。

请只返回 JSON：
{output_format}"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=2000,
        )
        raw = response.choices[0].message.content
        result = json.loads(raw)

        # Write cache (store everything the LLM returned)
        _write_cache(cache_path, cache_key_val, result)
        return result

    except Exception:
        logger.debug(
            "AI dialogue round %d failed for %s",
            round_num, scene_id, exc_info=True,
        )
        return None


def _apply_dialogue_updates(
    dialogue_lines: list[DialogueLine],
    attributions: list[dict],
) -> list[DialogueLine]:
    """Apply LLM attributions to dialogue lines (non-destructive copy)."""
    result = list(dialogue_lines)
    for item in attributions:
        idx = item.get("line_index")
        if idx is None or idx >= len(result):
            continue
        speaker = item.get("speaker")
        conf_raw = item.get("confidence", 0.5)
        try:
            conf = min(max(float(conf_raw), 0.0), 1.0)
        except (TypeError, ValueError):
            conf = 0.5
        # Normalize: null/None speaker stays None
        if speaker is None or speaker == "null":
            speaker = None
            conf = 0.0
        result[idx] = result[idx].model_copy(
            update={
                "speaker": speaker,
                "confidence": conf,
                "attribution_method": "llm",
            }
        )
    return result


def _fallback_dialogue(
    dialogue_lines: list[DialogueLine],
    low_indices: list[int],
) -> list[DialogueLine]:
    """Fallback: cap low-confidence lines at 0.3, keep original speaker."""
    result = list(dialogue_lines)
    for i in low_indices:
        result[i] = result[i].model_copy(update={"confidence": 0.3})
    return result


# ===================================================================
# 2. Scene Classification Enhancement
# ===================================================================


def enhance_scene_classification(
    scene_id: str,
    scene_content: str,
    *,
    cache_dir: str = "./cache",
) -> dict:
    """Use LLM to classify a scene's INT/EXT, location, and time of day.

    Returns a dict with keys ``int_ext``, ``location``, ``time_of_day``,
    and ``confidence``.  On failure, returns an empty dict (caller should
    keep rule-engine values).
    """
    client = _get_client()
    if client is None:
        return {}

    # Only look at first 500 chars for classification (same heuristic as rules)
    sample = scene_content[:500]

    cache_key_val = _cache_key(sample)
    cache_path = Path(cache_dir) / f"ai_scene_{scene_id}.json"

    cached = _read_cache(cache_path, cache_key_val)
    if cached is not None:
        return {
            "int_ext": cached.get("int_ext", "UNKNOWN"),
            "location": cached.get("location", "UNKNOWN"),
            "time_of_day": cached.get("time_of_day", "UNKNOWN"),
            "confidence": cached.get("confidence", 0.5),
        }

    prompt = f"""你是剧本分析助手。请分析以下中国网文场景片段的场景信息。

【场景原文（前500字）】
---
{sample}
---

请返回 JSON：
{{
  "int_ext": "INT" | "EXT" | "UNKNOWN",
  "location": "地点名称（2-8个字，如'迦南学院大殿'、'云岚宗山巅'。无法判断写'UNKNOWN'）",
  "time_of_day": "日" | "夜" | "晨" | "黄昏" | "UNKNOWN",
  "confidence": 0.85
}}

注意：
- INT = 室内/洞窟/大殿/房间/院落/庭院/天井/中庭/宅院等有围墙的封闭或半封闭场景
  （院落、庭院虽露天但四面围墙，属于 INT 不是 EXT）
- EXT = 山巅/野外/街道/树林/河边/海边等完全开放的户外场景
- time_of_day 根据"月光/深夜/烛火→夜"、"日出/清晨→晨"、"阳光/正午→日"、"夕阳/黄昏→黄昏"判断"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=500,
        )
        raw = response.choices[0].message.content
        result = json.loads(raw)

        payload = {
            "int_ext": result.get("int_ext", "UNKNOWN"),
            "location": result.get("location", "UNKNOWN"),
            "time_of_day": result.get("time_of_day", "UNKNOWN"),
            "confidence": float(result.get("confidence", 0.5)),
        }
        _write_cache(cache_path, cache_key_val, payload)
        return payload

    except Exception:
        logger.debug(
            "AI scene classification failed for %s — keeping rule-engine values",
            scene_id,
            exc_info=True,
        )
        return {}


# ===================================================================
# 3. Character Verification
# ===================================================================


def verify_characters(
    candidate_names: list[str],
    context_snippets: list[str],
    *,
    cache_dir: str = "./cache",
    input_sha256: str = "",
) -> tuple[list[str], dict[str, float]]:
    """Use LLM to filter false-positive character names.

    Many names extracted by jieba are not actual characters (e.g. "云海",
    "石碑", "终于").  The LLM reads context snippets and decides which
    candidates are real people.

    Parameters
    ----------
    candidate_names:
        Names extracted by the rule engine (jieba/spaCy).
    context_snippets:
        One snippet per candidate (first 300 chars of the scene where the
        name first appeared).  Should have the same length as
        *candidate_names*.
    cache_dir:
        Directory for LLM result caching.
    input_sha256:
        SHA256 of the source novel file, used to scope the cache file
        per novel (avoids cross-novel cache collisions).

    Returns
    -------
    tuple[list[str], dict[str, float]]
        ``(verified_names, confidences)`` — only real characters are kept.
        On failure, returns the original candidates with confidence 0.3.
    """
    client = _get_client()
    if client is None:
        return candidate_names, {n: 0.3 for n in candidate_names}

    if not candidate_names:
        return [], {}

    cache_key_val = _cache_key(
        json.dumps(candidate_names), json.dumps(context_snippets)
    )
    # Scope cache per novel to prevent cross-novel collisions
    sha_prefix = input_sha256[:16] if input_sha256 else "default"
    cache_path = Path(cache_dir) / f"ai_characters_{sha_prefix}.json"

    cached = _read_cache(cache_path, cache_key_val)
    if cached is not None:
        return cached["verified"], {n: cached["confidences"].get(n, 0.5) for n in cached["verified"]}

    # Build candidate list with context
    candidate_entries: list[str] = []
    for i, name in enumerate(candidate_names):
        ctx = context_snippets[i] if i < len(context_snippets) else ""
        # Truncate context to 300 chars
        ctx_short = ctx[:300].replace("\n", " ")
        candidate_entries.append(
            f"  [{i}] {name}  ← 出处: 「{ctx_short}...」"
        )

    prompt = f"""你是网文编辑。请判断以下候选词中哪些是真实的角色人名，哪些是误判的普通词汇。

{chr(10).join(candidate_entries)}

【判断标准】
- is_character: true → 这是一个人的名字（如"萧炎"、"纳兰嫣然"、"药老"）
- is_character: false → 这是风景/物品/副词/形容词被误判为人名
  例："云海"→风景，"石碑"→物品，"终于"→副词，"苍老"→形容词，"考核官"→身份但也算角色

请返回 JSON：
{{"results": [
  {{"index": 0, "name": "萧炎", "is_character": true, "confidence": 0.95}},
  {{"index": 1, "name": "云海", "is_character": false, "confidence": 0.05}},
  ...
]}}"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=2000,
        )
        raw = response.choices[0].message.content
        result = json.loads(raw)

        verified: list[str] = []
        confidences: dict[str, float] = {}
        for item in result.get("results", []):
            name = item.get("name", "")
            is_char = item.get("is_character", True)
            conf = float(item.get("confidence", 0.5))
            if is_char:
                verified.append(name)
                confidences[name] = min(max(conf, 0.0), 1.0)

        payload = {
            "verified": verified,
            "confidences": confidences,
        }
        _write_cache(cache_path, cache_key_val, payload)
        return verified, confidences

    except Exception:
        logger.debug("AI character verification failed — keeping all candidates", exc_info=True)
        return candidate_names, {n: 0.3 for n in candidate_names}
