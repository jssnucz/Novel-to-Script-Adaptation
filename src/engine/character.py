"""Character Named Entity Recognition for the Novel-to-Script Adaptation
pipeline.

Provides two extraction backends (spaCy and jieba) plus deduplication and
frequency-based filtering.

- ``extract_names_spacy`` — spaCy ``zh_core_web_trf`` PER entity extraction
  (returns [] when the model is not installed).
- ``extract_names_jieba_fallback`` — jieba-based surname-prefix matching.
- ``deduplicate_characters`` — merge duplicate ``CharacterRef`` entries.
- ``extract_characters`` — full pipeline: extract, count, filter, sort.
"""

import functools
import logging
import re
from collections import Counter

from engine.models import CharacterRef, CharacterArtifact, SceneArtifact

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chinese surname data
# ---------------------------------------------------------------------------

# ===================================================================
# Full Chinese surnames (reference — used for _is_valid_character_candidate
# and dialogue-context surname matching)
# ===================================================================

_CHINESE_SURNAMES: set[str] = {
    # ── 百家姓 top 100 (single-character) ──────────────────────────────
    "赵", "钱", "孙", "李", "周", "吴", "郑", "王",
    "冯", "陈", "褚", "卫", "蒋", "沈", "韩", "杨",
    "朱", "秦", "尤", "许", "何", "吕", "施", "张",
    "孔", "曹", "严", "华", "金", "魏", "陶", "姜",
    "戚", "谢", "邹", "喻", "柏", "水", "窦", "章",
    "云", "苏", "潘", "葛", "奚", "范", "彭", "郎",
    "鲁", "韦", "昌", "马", "苗", "凤", "花", "方",
    "俞", "任", "袁", "柳", "酆", "鲍", "史", "唐",
    "费", "廉", "岑", "薛", "雷", "贺", "倪", "汤",
    "滕", "殷", "罗", "毕", "郝", "邬", "安", "常",
    "乐", "于", "时", "傅", "皮", "卞", "齐", "康",
    "伍", "余", "元", "卜", "顾", "孟", "平", "黄",
    "和", "穆", "萧", "尹", "姚", "邵", "湛", "汪",
    "祁", "毛", "禹", "狄", "米", "贝", "明", "臧",
    "计", "伏", "成", "戴", "谈", "宋", "茅", "庞",
    "熊", "纪", "舒", "屈", "项", "祝", "董", "梁",
    "杜", "阮", "蓝", "闵", "席", "季", "麻", "强",
    "贾", "路", "娄", "危", "江", "童", "颜", "郭",
    "梅", "盛", "林", "刁", "钟", "徐", "邱", "骆",
    "高", "夏", "蔡", "田", "樊", "胡", "凌", "霍",
    "虞", "万", "支", "柯", "昝", "管", "卢", "莫",
    "经", "房", "裘", "缪", "干", "解", "应", "宗",
    "丁", "宣", "贲", "邓", "郁", "单", "杭", "洪",
    "包", "诸", "左", "石", "崔", "吉", "钮", "龚",
    "程", "嵇", "邢", "滑", "裴", "陆", "荣", "翁",
    "荀", "羊", "於", "惠", "甄", "曲", "家", "封",
    "芮", "羿", "储", "靳", "汲", "邴", "糜", "松",
    "井", "段", "富", "巫", "乌", "焦", "巴", "弓",
    "牧", "隗", "山", "谷", "车", "侯", "宓", "蓬",
    "全", "郗", "班", "仰", "秋", "仲", "伊", "宫",
    "宁", "仇", "栾", "暴", "甘", "钭", "厉", "戎",
    "祖", "武", "符", "刘", "景", "詹", "束", "龙",
    "叶", "幸", "司", "韶", "郜", "黎", "蓟", "薄",
    "印", "宿", "白", "怀", "蒲", "邰", "从", "鄂",
    "索", "咸", "籍", "赖", "卓", "蔺", "屠", "蒙",
    "池", "乔", "阴", "郁", "胥", "能", "苍", "双",
    "闻", "莘", "党", "翟", "谭", "贡", "劳", "逄",
    "姬", "申", "扶", "堵", "冉", "宰", "郦", "雍",
    "郤", "璩", "桑", "桂", "濮", "牛", "寿", "通",
    "边", "扈", "燕", "冀", "郏", "浦", "尚", "农",
    "温", "别", "庄", "晏", "柴", "瞿", "阎", "充",
    "慕", "连", "茹", "习", "宦", "艾", "鱼", "容",
    "向", "古", "易", "慎", "戈", "廖", "庾", "终",
    "暨", "居", "衡", "步", "都", "耿", "满", "弘",
    "匡", "国", "文", "寇", "广", "禄", "阙", "东",
    "欧", "殳", "沃", "利", "蔚", "越", "夔", "隆",
    "师", "巩", "厍", "聂", "晁", "勾", "敖", "融",
    "冷", "訾", "辛", "阚", "那", "简", "饶", "空",
    "曾", "毋", "沙", "乜", "养", "鞠", "须", "丰",
    "巢", "关", "蒯", "相", "查", "后", "荆", "红",
    "游", "竺", "权", "逯", "盖", "益", "桓", "公",
    # ── Multi-character surnames (百家姓) ──────────────────────────
    "万俟", "司马", "上官", "欧阳", "夏侯", "诸葛",
    "闻人", "东方", "赫连", "皇甫", "尉迟", "公羊",
    "澹台", "公冶", "宗政", "濮阳", "淳于", "单于",
    "太叔", "申屠", "公孙", "仲孙", "轩辕", "令狐",
    "钟离", "宇文", "长孙", "慕容", "鲜于", "闾丘",
    "司徒", "司空", "亓官", "司寇", "子车", "颛孙",
    "端木", "巫马", "公西", "漆雕", "乐正", "壤驷",
    "公良", "拓跋", "夹谷", "宰父", "谷梁", "段干",
    "百里", "东郭", "南门", "呼延", "归海", "羊舌",
    "微生", "岳帅", "缑亢", "况后", "有琴", "梁丘",
    "左丘", "东门", "西门", "商牟", "佘佴", "伯赏",
    "南宫", "墨哈", "谯笪", "年爱", "阳佟", "言福",
    "第五",
    # ── Fictional / wuxia / xianxia surnames ───────────────────────
    "战", "楚", "夜", "风", "凰", "雪", "云", "月",
    "冰", "龙", "玉", "凤", "岚", "剑", "珏", "殇",
    "北堂", "东野", "南郭",
    "独孤", "纳兰", "剑心", "即墨", "公输", "墨翟",
}

# Surnames sorted by length descending for longest-first matching
_SORTED_SURNAMES: list[str] = sorted(_CHINESE_SURNAMES, key=len, reverse=True)

# Chinese punctuation characters that should never be part of a name.
_PUNCTUATION: frozenset[str] = frozenset(
    "，。、；：？！""''「」『』（）【】《》〈〉—…·～\t\n\r　 "
)

# Common non-person words that jieba frequently misidentifies as names.
# These are filtered out before deduplication to reduce false positives
# without requiring LLM verification.
_NAME_BLACKLIST: frozenset[str] = frozenset({
    # Nature / landscape words
    "云海", "石碑", "山巅", "枫叶", "落叶", "云岚",
    "月光", "月", "风", "火", "烛火", "山风",
    "山谷", "山谷间", "山涧", "山岚", "云雾", "云端",
    "溪水", "湖水", "松林", "湖泊", "湖水",
    # Common adverbs / conjunctions / abstract words
    "终于", "忽然", "仿佛", "似乎", "渐渐", "其实",
    "终于轮",  # jieba sometimes emits this for "终于轮到我了"
    # Objects / artifacts / materials
    "戒指", "斗气", "丹药", "三段斗", "三段",
    "测魔", "测魔石", "魔石碑",
    "羊皮纸", "羊皮", "竹篓", "碎石", "火堆", "火星",
    "琥珀", "琥珀色", "光晕", "气泡", "气泡",
    # Common nouns (frequently in scene descriptions)
    "考核", "修炼", "修为", "目光", "背影",
    "大殿", "考核台", "人群", "一方", "三年",
    "一个", "什么", "这个", "那个", "怎么",
    "考核官", "炼药师",  # role titles, detected by dialogue context
    # Generic descriptors / colors / qualities
    "苍老", "透明", "古朴", "刺目", "清脆", "金色",
    "瘦小", "倔强", "蓬松", "优雅", "炽热", "沉重",
    "说不清", "沉得住", "不紧不慢",
    # Numbers / quantities / common verbs
    "数百", "一段", "一名", "一句", "三年",
    "没有回答", "越来越", "没有动", "有的是",
    "应为", "枯死", "那个", "天边",
    # Place names / descriptors falsely detected as persons
    "龙眠之地", "山脚下", "山路上", "山顶", "夜空",
    # Common jieba false positives (from dragon_hunt and similar)
    "那条", "成为", "方向", "金光", "金色", "那条",
    "羊皮纸", "羊皮", "白天", "明天", "怀里", "干粮",
    "时候", "那个", "终点", "通体", "雪白", "边缘",
    "蓬松", "别怕", "步子", "全身", "通过", "那么",
    "平静", "越来越", "后退", "水面", "双眼",
    "龙说",  # "龙说" = dragon said, not a name
    "那句话", "时候", "黎明前", "鱼肚白", "通往",
    "云雾", "山巅", "山脚下", "山上",
    "布满", "山涧", "山岚", "巨石", "石像",
    "那道", "万籁俱寂", "松林", "山头",
    "龙眠",  # place name prefix
})

# Role / occupation vocabulary — descriptive character identifiers that
# are valid character names even though they don't contain surnames.
# Used by dialogue-context extraction to recognise non-surname characters
# (e.g. 少女, 老猎人, 黑袍人).
_ROLE_TITLE_VOCABULARY: frozenset[str] = frozenset({
    # Age + gender descriptors (very common in web novels)
    "少女", "少年", "女孩", "男孩", "女子", "男子",
    "妇人", "姑娘", "公子", "大娘", "大爷",
    # Elder descriptors
    "老人", "老者", "老妇", "老妪", "老翁", "老太",
    # Occupation / role titles
    "猎人", "老猎人", "猎户", "渔夫", "樵夫", "农夫",
    "道人", "僧人", "和尚", "道士", "尼姑", "法师",
    "掌柜", "老板", "伙计", "小二", "店家",
    "将军", "士兵", "护卫", "守卫", "门卫", "哨兵",
    "长老", "掌门", "教主", "帮主", "堂主", "宗主",
    "皇帝", "太子", "王爷", "公主", "皇子", "郡主",
    "丫鬟", "仆人", "侍从", "随从", "家丁",
    "先生", "师父", "徒弟", "师兄", "师姐", "师弟", "师妹",
    "乞丐", "商贩", "郎中", "铁匠", "剑客", "刀客",
    # Descriptive attire (web-novel common)
    "黑袍人", "白衣人", "黑衣人", "灰袍人", "青衫客",
    "蒙面人", "斗笠人", "紫衣", "红衣",
    # Non-human characters (mythology / xianxia / fantasy)
    "狐狸", "白狐", "龙", "老龙", "凤", "麒麟", "虎",
    "白蛇", "黑蛇", "狼", "鹰",
    # Generic but common
    "村长", "族长", "城主", "国师", "军师",
    "路人", "村民", "山民", "渔民",
})

# All role vocabulary terms sorted by length (longest first) for
# greedy matching in dialogue-context extraction.
_SORTED_ROLE_TERMS: list[str] = sorted(_ROLE_TITLE_VOCABULARY, key=len, reverse=True)


# ---------------------------------------------------------------------------
# spaCy extraction  (graceful degradation)
# ---------------------------------------------------------------------------

@functools.cache
def _get_spacy_nlp():
    """Load and cache the spaCy Chinese model. Called once across all scenes."""
    import spacy
    return spacy.load("zh_core_web_trf")


def extract_names_spacy(text: str) -> list[str]:
    """Extract person names from *text* using spaCy ``zh_core_web_trf``.

    If the model is not installed or spaCy is unavailable, returns ``[]``
    silently (logs a debug message).

    Parameters
    ----------
    text : str
        Chinese text to analyse.

    Returns
    -------
    list[str]
        Person names found in the text.
    """
    try:
        nlp = _get_spacy_nlp()
    except Exception:
        logger.debug("spaCy zh_core_web_trf not available, returning []")
        return []

    try:
        doc = nlp(text)
        return [ent.text for ent in doc.ents if ent.label_ == "PER"]
    except Exception:
        logger.debug("spaCy PER extraction failed, returning []", exc_info=True)
        return []


# ---------------------------------------------------------------------------
# jieba fallback  (always works)
# ---------------------------------------------------------------------------

def extract_names_jieba_fallback(text: str) -> list[str]:
    """Extract likely character names using jieba segmentation and surname
    prefix matching.

    Words of length 2–4 that start with one of the known Chinese surnames
    are returned.  Multi-character surnames are matched before single-
    character ones (longest-first) so that "纳兰嫣然" is matched by the
    surname "纳兰" rather than "纳".

    Because jieba sometimes splits a full name into multiple tokens (e.g.
    "纳兰嫣然" -> "纳兰" + "嫣然"), this function reconstructs the full
    name when a multi-character surname word is followed by plausible
    name-part words.

    Returns a *list* (not a set) so that downstream frequency counting
    can observe multiple occurrences of the same name.

    Parameters
    ----------
    text : str
        Chinese text to analyse.

    Returns
    -------
    list[str]
        Likely character names found in the text.
    """
    if not text:
        return []

    import jieba
    words = jieba.lcut(text)
    names: list[str] = []
    n = len(words)

    def _match_surname(s: str) -> str | None:
        """Return the longest surname *s* starts with, or ``None``."""
        for surname in _SORTED_SURNAMES:
            if s.startswith(surname):
                return surname
        return None

    def _looks_like_name_part(w: str) -> bool:
        """Return True when *w* looks like it could be part of a name
        (i.e. not punctuation and not whitespace)."""
        return bool(w) and w not in _PUNCTUATION

    for i in range(n):
        word = words[i]
        if not (2 <= len(word) <= 4):
            continue

        surname = _match_surname(word)

        if surname is not None and len(surname) >= 2:
            # Multi-character surname: extend with adjacent words to
            # reconstruct names split by jieba (e.g. 纳兰 + 嫣然).
            combined = word
            j = i + 1
            while j < n and len(combined) + len(words[j]) <= 4:
                if not _looks_like_name_part(words[j]):
                    break
                combined += words[j]
                j += 1
            names.append(combined)

        elif surname is not None and len(surname) == 1:
            # Single-character surname: jieba usually keeps the full name
            # as one word (e.g. 萧炎, 林动).  Add the word directly.
            names.append(word)

        else:
            # No surname at the start — check whether a suffix of this
            # jieba word (after removing 1–2 characters) starts with a
            # multi-character surname.  Handles cases like "看纳兰桀"
            # where jieba emits "看纳兰" + "桀" and the intended name
            # "纳兰桀" straddles the split.
            for split_pos in range(1, min(3, len(word))):
                suffix = word[split_pos:]
                if len(suffix) < 2:
                    continue
                s2 = _match_surname(suffix)
                if s2 is not None and len(s2) >= 2 and s2 == suffix:
                    # The suffix IS the full surname — try to extend
                    # with the next word(s).
                    combined = suffix
                    j = i + 1
                    while j < n and len(combined) + len(words[j]) <= 4:
                        if not _looks_like_name_part(words[j]):
                            break
                        combined += words[j]
                        j += 1
                    names.append(combined)
                    break

    return names


# ---------------------------------------------------------------------------
# Dialogue-context extraction (non-surname characters)
# ---------------------------------------------------------------------------

# Quote-matching regex.  After preprocessor unify_quotes() runs, all
# CJK corner brackets (U+300C/D, U+300E/F) are converted to ASCII
# double/single quotes (U+0022, U+0027).  This pattern uses only
# ASCII-safe escapes to avoid source-file encoding issues.
_QUOTE_PATTERN = re.compile(
    r'"' r'[^"]*' r'"'                 # ASCII double quotes
    r"|'[^']*'"                        # ASCII single quotes
    r"|\u300c[^\u300d]*\u300d"     # CJK corner brackets (fallback)
    r"|\u300e[^\u300f]*\u300f"     # CJK white corner brackets (fallback)
)

def extract_names_dialogue_context(text: str) -> list[str]:
    """Extract candidate character names from dialogue-context subject
    detection.

    For each quote in the text, this function examines the preceding
    narrative to find role/title vocabulary terms that are likely the
    speaker.  This catches non-surname characters like "少女", "老猎人",
    "黑袍人" that surname-matching misses entirely.

    Only vocabulary terms are returned — surname-based names are handled
    by the main jieba extraction pipeline.

    Parameters
    ----------
    text : str
        Chinese text to analyse.

    Returns
    -------
    list[str]
        Candidate character names found in dialogue contexts.
    """
    if not text:
        return []

    names: list[str] = []

    for m in _QUOTE_PATTERN.finditer(text):
        quote_start = m.start()
        before_start = max(0, quote_start - 80)
        before_text = text[before_start:quote_start]

        if not before_text.strip():
            continue

        # Find role/title vocabulary terms in before_text
        # Match longest first so "老猎人" is found before "猎人"
        for term in _SORTED_ROLE_TERMS:
            if term in before_text:
                names.append(term)
                break  # one match per quote is sufficient

    return names

def deduplicate_characters(refs: list[CharacterRef]) -> list[CharacterRef]:
    """Merge duplicate ``CharacterRef`` entries by name.

    For each unique name:
    - Keep the earliest ``first_appearance``.
    - Accumulate all unique aliases.

    Parameters
    ----------
    refs : list[CharacterRef]
        Character references to deduplicate.

    Returns
    -------
    list[CharacterRef]
        Deduplicated character references.
    """
    seen: dict[str, CharacterRef] = {}

    for ref in refs:
        if ref.name in seen:
            existing = seen[ref.name]
            # Keep earliest first_appearance
            if ref.first_appearance < existing.first_appearance:
                existing.first_appearance = ref.first_appearance
            # Merge aliases
            existing_aliases = set(existing.aliases)
            for alias in ref.aliases:
                if alias not in existing_aliases:
                    existing.aliases.append(alias)
                    existing_aliases.add(alias)
        else:
            seen[ref.name] = ref.model_copy(deep=True)

    return list(seen.values())


# ---------------------------------------------------------------------------
# Jieba POS-tag extraction (person-name detection)
# ---------------------------------------------------------------------------


def _extract_names_jieba_pos(text: str) -> list[str]:
    """Extract person names using jieba's POS tagging (``nr`` tag).

    Jieba's ``posseg`` module tags words with part-of-speech labels.
    The ``nr`` tag means "person name" — this is the highest-quality
    signal available without spaCy.  It catches both surname-based
    names (萧炎/nr) and some non-surname names (老猎人 is typically
    tagged as ``n`` not ``nr``, so dialogue-context extraction is
    still needed for those).

    Parameters
    ----------
    text : str
        Chinese text to analyse.

    Returns
    -------
    list[str]
        Words tagged as person names (``nr``).
    """
    if not text:
        return []

    try:
        import jieba.posseg as pseg
        words = pseg.cut(text)
        return [w for w, flag in words if flag == "nr" and len(w) >= 2]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# POS tag map builder (quality filter helper)
# ---------------------------------------------------------------------------


def _build_pos_tag_map(text: str, candidates: set[str]) -> dict[str, str]:
    """Build a mapping from candidate word → most common POS tag.

    Runs jieba POS tagging once over the full text and records which
    tag each candidate word most frequently receives.

    Parameters
    ----------
    text : str
        Full novel text (all scenes concatenated).
    candidates : set[str]
        Candidate character names to look up.

    Returns
    -------
    dict[str, str]
        ``{word: pos_tag}`` — words not found in the POS output
        default to ``"nr"`` (assume person if unknown).
    """
    if not text or not candidates:
        return {}

    try:
        import jieba.posseg as pseg
        words = pseg.cut(text)
    except Exception:
        return {}

    tag_counts: dict[str, Counter] = {}
    for word, flag in words:
        if word in candidates:
            tag_counts.setdefault(word, Counter()).update([flag])

    result: dict[str, str] = {}
    for word in candidates:
        counts = tag_counts.get(word)
        if counts:
            result[word] = counts.most_common(1)[0][0]
        else:
            result[word] = "nr"  # default: assume person name

    return result


# ---------------------------------------------------------------------------
# Full extraction pipeline
# ---------------------------------------------------------------------------

def extract_characters(artifact: SceneArtifact) -> CharacterArtifact:
    """Extract, count, filter and deduplicate characters from a
    ``SceneArtifact``.

    Pipeline
    --------
    1. For each scene, try ``extract_names_spacy``; if it returns empty,
       fall back to ``extract_names_jieba_fallback``.
    2. Count occurrences of each name globally.
    3. Remove names that appear only **once**, unless the total number of
       unique names is 3 or fewer.
    4. Deduplicate and sort by frequency (most frequent first).

    Parameters
    ----------
    artifact : SceneArtifact
        Scene-segmented novel content.

    Returns
    -------
    CharacterArtifact
        Deduplicated and sorted character list.
    """
    # Phase 1a: extract reliable names from full-text dialogue context
    # (run once on the full novel — per-scene extraction misses quotes
    # that span scene boundaries)
    full_text = "\n".join(sc.content for sc in artifact.scenes)
    if not extract_names_spacy(full_text):
        reliable_names: set[str] = set(
            extract_names_dialogue_context(full_text)
        )
    else:
        reliable_names = set()

    # Phase 1b: extract raw names per scene
    raw_refs: list[CharacterRef] = []
    # First-appearance tracking for reliable names (so they get the
    # correct first_appearance scene_id)
    reliable_first_seen: dict[str, str] = {}

    for scene in artifact.scenes:
        # Try spaCy first
        names = extract_names_spacy(scene.content)
        if not names:
            pos_names = _extract_names_jieba_pos(scene.content)
            surname_names = extract_names_jieba_fallback(scene.content)
            # Use POS names as primary source; supplement with
            # surname-fallback names that POS missed (avoids double-
            # counting names found by both methods)
            pos_name_set = set(pos_names)
            surname_only = [n for n in surname_names if n not in pos_name_set]
            names = pos_names + surname_only

        for name in names:
            raw_refs.append(
                CharacterRef(
                    name=name,
                    first_appearance=scene.scene_id,
                )
            )

        # Track where reliable names first appear in scene content
        for name in reliable_names:
            if name not in reliable_first_seen and name in scene.content:
                reliable_first_seen[name] = scene.scene_id

    # Add reliable names as explicit CharacterRefs (with correct
    # first_appearance — the first scene where their name appears)
    for name in reliable_names:
        raw_refs.append(
            CharacterRef(
                name=name,
                first_appearance=reliable_first_seen.get(
                    name, artifact.scenes[0].scene_id
                ),
            )
        )

    if not raw_refs:
        return CharacterArtifact(schema_version="1.0", characters=[])

    # Phase 2: count global occurrences
    counter = Counter(ref.name for ref in raw_refs)

    # Phase 2b: Quality filter — surname-only matches need corroboration
    # from dialogue context, role vocabulary, or high frequency.
    unique_names = len(counter)

    def _quality_filter(ref: CharacterRef) -> bool:
        name = ref.name
        freq = counter.get(name, 0)
        # Blacklist → always discard
        if name in _NAME_BLACKLIST:
            return False
        # Role vocabulary → always keep (e.g. 少女, 老猎人, 狐狸, 龙)
        if name in _ROLE_TITLE_VOCABULARY:
            return True
        # Dialogue context match → keep (vocabulary found near a quote)
        if name in reliable_names:
            return True
        # Multi-occurrence surname-prefixed name → keep
        if freq >= 2:
            return True
        # Single-occurrence surname-prefixed name with small cast → keep
        # (when the novel has <= 3 unique names, keep all candidates to
        #  avoid losing genuine characters in short test texts)
        if unique_names <= 3:
            return True
        # Low-frequency surname match without corroboration → discard
        # (filters out single-occurrence false positives)
        return False

    raw_refs = [ref for ref in raw_refs if _quality_filter(ref)]

    # Phase 3: filter single-occurrence names (unless cast is very small)
    # Names from role vocabulary or reliable sources are exempt from
    # frequency filtering — they are real characters even if only
    # extracted once (e.g. non-surname characters like 老猎人).
    unique_names = len(counter)
    if unique_names > 3:
        valid_names: set[str] = {
            name for name, count in counter.items()
            if count > 1 or name in _ROLE_TITLE_VOCABULARY or name in reliable_names
        }
        raw_refs = [ref for ref in raw_refs if ref.name in valid_names]

    # Phase 3b: filter known non-person words (blacklist)
    raw_refs = [ref for ref in raw_refs if ref.name not in _NAME_BLACKLIST]

    # Phase 4: deduplicate
    deduped = deduplicate_characters(raw_refs)

    # Phase 5: sort by frequency (descending), then alphabetically for ties
    deduped.sort(
        key=lambda r: (-counter.get(r.name, 0), r.name)
    )

    return CharacterArtifact(schema_version="1.0", characters=deduped)
