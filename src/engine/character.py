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

from src.engine.models import CharacterRef, CharacterArtifact, SceneArtifact

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chinese surname data
# ---------------------------------------------------------------------------

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
# Deduplication
# ---------------------------------------------------------------------------

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
    # Phase 1: extract raw names per scene
    raw_refs: list[CharacterRef] = []

    for scene in artifact.scenes:
        # Try spaCy first
        names = extract_names_spacy(scene.content)
        if not names:
            names = extract_names_jieba_fallback(scene.content)

        for name in names:
            raw_refs.append(
                CharacterRef(
                    name=name,
                    first_appearance=scene.scene_id,
                )
            )

    if not raw_refs:
        return CharacterArtifact(schema_version="1.0", characters=[])

    # Phase 2: count global occurrences
    counter = Counter(ref.name for ref in raw_refs)

    # Phase 3: filter single-occurrence names (unless cast is very small)
    unique_names = len(counter)
    if unique_names > 3:
        # Determine which names appear more than once
        valid_names: set[str] = {name for name, count in counter.items() if count > 1}
        raw_refs = [ref for ref in raw_refs if ref.name in valid_names]

    # Phase 4: deduplicate
    deduped = deduplicate_characters(raw_refs)

    # Phase 5: sort by frequency (descending), then alphabetically for ties
    deduped.sort(
        key=lambda r: (-counter.get(r.name, 0), r.name)
    )

    return CharacterArtifact(schema_version="1.0", characters=deduped)
