"""Class notes (PDF) generated from the final transcript + slide thumbnails.

The LLM (chat function injected by the caller) turns the final, corrected
transcript into study notes in a user-selected language, including a glossary
that annotates technical terms; reportlab renders the notes together with the
per-page slide images into a PDF.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Callable, Optional
from xml.sax.saxutils import escape

log = logging.getLogger("srt_gen.notes")

# key -> language label sent to the LLM ("auto" = transcript's language)
NOTE_LANGS = {
    "auto": "",
    "en": "English",
    "zh-Hans": "Simplified Chinese (use only Simplified characters)",
    "zh-Hant": "Traditional Chinese (use only Traditional characters)",
    "ja": "Japanese",
}

_LANG_NAMES = {"zh": "Chinese", "yue": "Chinese", "en": "English",
               "ja": "Japanese", "ko": "Korean", "fr": "French",
               "de": "German", "es": "Spanish", "pt": "Portuguese",
               "it": "Italian", "ru": "Russian", "vi": "Vietnamese",
               "th": "Thai", "id": "Indonesian"}

_SCRIPT_OF = {"en": "latin", "zh-Hans": "zh-Hans", "zh-Hant": "zh-Hant",
              "ja": "ja"}
_LANG_SCRIPT = {"zh": "zh-Hans", "yue": "zh-Hant", "en": "latin", "ja": "ja",
                "ko": "ko"}

# TrueType (.ttf/.ttc) faces that can be embedded, tried in order per script
_TTF_CANDIDATES = {
    "latin": [("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0),
              ("/usr/share/fonts/truetype/arphic/uming.ttc", 0)],
    "zh-Hans": [("/usr/share/fonts/truetype/arphic/uming.ttc", 0)],
    "zh-Hant": [("/usr/share/fonts/truetype/arphic/uming.ttc", 2),
                ("/usr/share/fonts/truetype/arphic/uming.ttc", 0)],
    "ja": [("/usr/share/fonts/truetype/arphic/uming.ttc", 0)],
    "ko": [],
}
_BOLD_CANDIDATES = {
    "latin": [("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0)],
}
# non-embedded fallbacks (Adobe standard CID fonts, viewer supplies glyphs)
_CID_FALLBACK = {"latin": "Helvetica", "zh-Hans": "STSong-Light",
                 "zh-Hant": "MSung-Light", "ja": "HeiseiMin-W3",
                 "ko": "HYSMyeongJo-Medium"}

# downloadable TrueType families (fonts/ dir, OFL licensed). Noto Serif TC
# follows the Taiwan MOE standard, so e.g. 床 does not fall back to the
# Japanese form 牀 the way AR PL UMing does.
FONTS_DIR = Path(__file__).resolve().parent / "fonts"
NOTE_FONTS = {
    "auto": "Auto (by language)",
    "noto-serif-tc": "Noto Serif TC · 思源宋體",
    "lxgw-wenkai": "LXGW WenKai · 霞鶩文楷",
    "ming": "AR PL UMing · 明體",
}
_FAMILY_FILES = {  # key -> (regular file, bold file) under FONTS_DIR
    "noto-serif-tc": ("NotoSerifTC-Regular.ttf", "NotoSerifTC-Bold.ttf"),
    "lxgw-wenkai": ("LXGWWenKai-Regular.ttf", "LXGWWenKai-Medium.ttf"),
}
# the "auto" choice prefers these downloaded faces for Traditional Chinese
_AUTO_PREFERRED = {
    "zh-Hant": [("NotoSerifTC-Regular.ttf", "NotoSerifTC-Bold.ttf")],
}
_font_cache: dict[tuple[str, str], tuple[str, str]] = {}


def _font_candidates(script: str, font: str) -> tuple[list, list]:
    """(regular, bold) candidate lists of (path, subfontIndex) to try."""
    family = _FAMILY_FILES.get(font)
    if family:
        reg, bld = family
        return ([(str(FONTS_DIR / reg), 0)], [(str(FONTS_DIR / bld), 0)])
    regular: list = []
    bold: list = []
    if font == "auto":
        for reg, bld in _AUTO_PREFERRED.get(script, []):
            regular.append((str(FONTS_DIR / reg), 0))
            bold.append((str(FONTS_DIR / bld), 0))
    regular += _TTF_CANDIDATES.get(script, [])
    bold += _BOLD_CANDIDATES.get(script, [])
    return regular, bold


def _register_fonts(script: str, font: str = "auto") -> tuple[str, str]:
    """Return (regular, bold) font names for a script + family, once."""
    key = (script, font)
    if key in _font_cache:
        return _font_cache[key]
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfbase.ttfonts import TTFont
    regular = bold = ""
    regulars, bolds = _font_candidates(script, font)
    for i, (path, idx) in enumerate(regulars):
        try:
            name = f"notes_{font}_{script}_{i}"
            if not Path(path).exists():
                continue
            pdfmetrics.registerFont(TTFont(name, path, subfontIndex=idx))
            regular = name
            break
        except Exception as e:  # noqa: BLE001
            log.warning("Notes font %s unusable: %s", path, e)
    if not regular and font != "auto":  # selected family missing on disk
        log.warning("Notes font %s unavailable for %s, falling back to auto",
                    font, script)
        regulars, bolds = _font_candidates(script, "auto")
        for i, (path, idx) in enumerate(regulars):
            try:
                name = f"notes_auto_{script}_fb{i}"
                if not Path(path).exists():
                    continue
                pdfmetrics.registerFont(TTFont(name, path, subfontIndex=idx))
                regular = name
                break
            except Exception:  # noqa: BLE001
                continue
    if regular:
        for i, (path, idx) in enumerate(bolds):
            try:
                name = f"notes_{font}_{script}_b{i}"
                pdfmetrics.registerFont(TTFont(name, path, subfontIndex=idx))
                bold = name
                break
            except Exception:  # noqa: BLE001
                continue
        bold = bold or regular
    else:
        fallback = _CID_FALLBACK.get(script, "Helvetica")
        try:
            if fallback != "Helvetica" and not fallback.startswith("Helvetica"):
                pdfmetrics.registerFont(UnicodeCIDFont(fallback))
            regular = bold = fallback
        except Exception:  # noqa: BLE001
            regular = bold = "Helvetica"
    _font_cache[key] = (regular, bold)
    return regular, bold


# characters that exist only in Traditional / only in Simplified Chinese,
# used to tell which script the "auto" notes should follow
_TRAD_ONLY = set("個們這說實產與對後學術數據發點關係應變運讓車開問門間長飛馬語書見觀論處營圖環連選導華傳氣體頭遠檢驗時為")
_SIMP_ONLY = set("个们这说实产与对后学术数据发点关应变运让车开问门间长飞马语书见观论处营图环连选导华传气体头远检验时为")


def _detect_zh_script(text: str) -> str:
    """zh-Hant / zh-Hans from the script actually used in the transcript."""
    trad = sum(1 for ch in text if ch in _TRAD_ONLY)
    simp = sum(1 for ch in text if ch in _SIMP_ONLY)
    return "zh-Hant" if trad >= simp else "zh-Hans"


def resolve_language(notes_lang: str, transcript_lang: str,
                     sample: str = "") -> tuple[str, str]:
    """(language label for the LLM prompt, font script key)."""
    if notes_lang in NOTE_LANGS and notes_lang != "auto":
        return NOTE_LANGS[notes_lang], _SCRIPT_OF[notes_lang]
    tl = str(transcript_lang or "en").lower().split("-")[0]
    if tl in ("zh", "yue"):  # match the script the lecture itself uses
        script = _detect_zh_script(sample) if sample else _LANG_SCRIPT[tl]
        label = ("Traditional Chinese (use only Traditional characters)"
                 if script == "zh-Hant" else
                 "Simplified Chinese (use only Simplified characters)")
        return label, script
    label = _LANG_NAMES.get(tl, "English")
    return label, _LANG_SCRIPT.get(tl, "latin")


# --- Chinese script normalization -------------------------------------------
# The model sometimes slips into the other script (e.g. 学员 inside a
# otherwise Traditional note set), so every string is converted deterministically.
try:
    from opencc import OpenCC
    _CC = {"zh-Hant": OpenCC("s2t"), "zh-Hans": OpenCC("t2s")}
except Exception:  # noqa: BLE001 - opencc is optional
    _CC = None

# fallback character table (traditional, simplified) used when opencc is absent
_ZH_PAIRS = (
    "個个", "們们", "這这", "說说", "實实", "產产", "與与", "對对", "後后",
    "學学", "術术", "數数", "據据", "發发", "點点", "關关", "應应", "變变",
    "運运", "讓让", "車车", "開开", "問问", "門门", "間间", "長长", "飛飞",
    "馬马", "語语", "書书", "見见", "觀观", "論论", "處处", "營营", "圖图",
    "環环", "連连", "選选", "導导", "華华", "傳传", "氣气", "體体", "頭头",
    "遠远", "檢检", "驗验", "時时", "為为", "東东", "風风", "網网", "機机",
    "線线", "組组", "織织", "經经", "濟济", "議议", "電电", "腦脑", "決决",
    "確确", "稱称", "極极", "劇剧", "從从", "眾众", "臨临", "況况", "階阶",
    "際际", "級级", "複复", "雜杂", "響响", "態态", "顯显", "預预", "規规",
    "範范", "測测", "試试", "讀读", "覺觉", "記记", "設设", "備备", "計计",
    "訪访", "誤误", "獲获", "達达", "過过", "還还", "夢梦", "邊边", "輪轮",
    "輯辑", "觸触", "場场", "聽听", "買买", "價价", "單单", "醫医", "藥药",
    "療疗", "護护", "診诊", "統统", "質质", "穩稳", "願愿", "餘余", "乾干",
    "裡里", "臺台", "兒儿", "簡简", "專专", "業业", "員员", "斷断", "積积",
    "篩筛", "採采", "審审", "獨独", "異异", "趨趋", "勢势", "類类", "項项",
    "錄录", "雲云", "區区", "會会", "給给", "於于", "當当", "嗎吗", "無无",
    "該该", "並并", "較较", "舊旧", "雙双", "離离", "續续", "進进", "識识",
    "認认", "憶忆", "慮虑", "評评", "釋释", "義义", "擬拟", "虛虚", "頁页",
    "視视", "蓋盖", "簽签", "勸劝", "歡欢", "寶宝", "歲岁", "擊击", "舉举",
    "補补", "許许", "訴诉", "詞词", "偵侦", "優优", "聖圣", "殼壳", "違违",
    "創创", "勝胜", "薦荐", "譽誉", "戀恋", "們们", "產产", "實实",
)
_S2T = {p[0]: p[1] for p in _ZH_PAIRS}   # traditional -> simplified
_T2S = {p[1]: p[0] for p in _ZH_PAIRS}   # simplified -> traditional


# opencc's s2t maps 床 to its old variant 牀, but 床 is the customary
# Traditional form (牀 reads as a Japanese/orthographic variant), so any
# notes/subtitle text is fixed back after script conversion.
_GLYPH_FIX = str.maketrans({"牀": "床"})


def _fix_glyphs(text: str) -> str:
    return text.translate(_GLYPH_FIX)


def ensure_traditional(segments: list[dict]) -> int:
    """Rewrite Simplified Chinese transcript lines in Traditional form.

    Whisper emits Simplified Chinese for zh; the product wants Traditional
    subtitles whenever the speech is Chinese. Returns the number of lines
    changed. Safe to call repeatedly."""
    changed = 0
    for s in segments:
        t = _fix_glyphs(normalize_script(s.get("text") or "", "zh-Hant"))
        if t and t != s["text"]:
            s["text"] = t
            changed += 1
    return changed


def normalize_script(text: str, script: str) -> str:
    """Force a Chinese string into one script so the notes stay consistent."""
    if not text or script not in ("zh-Hant", "zh-Hans"):
        return text
    if _CC:
        return _fix_glyphs(_CC[script].convert(text))
    table = _T2S if script == "zh-Hant" else _S2T
    return _fix_glyphs("".join(table.get(ch, ch) for ch in text))


def _normalize_doc(doc: dict) -> dict:
    """Normalize every LLM-written string of the document to one script and
    the customary glyph forms (牀 -> 床 applies to all languages)."""
    script = doc.get("script") or ""

    def fix(x: str) -> str:
        return _fix_glyphs(normalize_script(x, script))

    doc["title"] = fix(doc.get("title", ""))
    doc["overview"] = fix(doc.get("overview", ""))
    doc["takeaways"] = [fix(x) for x in doc.get("takeaways", [])]
    for ch in doc.get("chapters", []):
        ch["title"] = fix(ch.get("title", ""))
        for topic in ch.get("topics", []):
            topic["title"] = fix(topic.get("title", ""))
            for b in topic.get("blocks", []):
                b["t"] = fix(b["t"])
    for t in doc.get("terms", []):
        t["t"] = fix(t["t"])
        t["d"] = fix(t["d"])
    return doc


def _json_obj(text: str) -> Optional[dict]:
    """Parse a JSON object from model output, recovering truncated replies."""
    if not text:
        return None
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            d = json.loads(m.group())
            if isinstance(d, dict):
                return d
        except Exception:  # noqa: BLE001
            pass
    # the reply was cut off by max_tokens: close the open string/array/object
    for cut in range(len(text), max(0, len(text) - 400), -1):
        head = text[:cut].rstrip()
        for closer in ("", '"', '"]', '"]}', '"}', "}", "]}", "]", "]"):
            try:
                d = json.loads(head + closer)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(d, dict):
                return d
    return None


def _ts(sec: float) -> str:
    sec = max(0, int(round(sec)))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _overview(chat: Callable, result: dict, label: str) -> tuple[str, list[str]]:
    titles = "; ".join((p.get("title") or "") for p in result["pages"])[:2500]
    sample = "\n".join(re.sub(r"\s+", " ", (p.get("text") or "")).strip()[:120]
                       for p in result["pages"])[:3000]
    d = _json_obj(chat(
        f"You are writing the cover page of class notes for a lecture, written "
        f"in {label}. Lecture title: {result.get('title') or ''}\n"
        f"Slide titles in order: {titles}\n"
        f"Abbreviated transcript excerpt (one chunk per slide):\n{sample}\n\n"
        'Reply with ONLY a valid JSON object: {"overview": "2-4 sentence '
        'summary of what the lecture covers", "takeaways": ["3-5 substantive '
        f'key points a student must remember, in {label}"]}}',
        max_tokens=900, temperature=0.3) or "")
    if not d:
        return "", []
    overview = str(d.get("overview") or "").strip()
    tk = d.get("takeaways")
    takeaways = [str(x).strip() for x in tk if str(x).strip()] \
        if isinstance(tk, list) else []
    return overview, takeaways[:6]


def _fallback_points(page: dict) -> list[str]:
    """Last resort when the LLM is unavailable: split the page text into
    short sentence bullets rather than one copied paragraph."""
    text = re.sub(r"\s+", " ", page.get("text") or
                  " ".join(s["text"] for s in page["segments"])).strip()
    if not text:
        return [page.get("title") or ""]
    out = []
    for s in re.split(r"(?<=[.!?。！？])\s+", text):
        s = s.strip()
        if len(s) >= 6 and not re.fullmatch(r"[嗯啊呃]+[.!？?]*", s):
            out.append(s[:240])
        if len(out) >= 5:
            break
    return out or [text[:240]]


def _fallback_topics(pages: list[dict]) -> list[dict]:
    """No LLM outline available: group consecutive slides into plain topics."""
    out, cur = [], []
    for p in pages:
        cur.append(p["index"])
        if len(cur) >= 3:
            out.append({"title": "", "focus": "", "slides": cur})
            cur = []
    if cur:
        out.append({"title": "", "focus": "", "slides": cur})
    return [{"chapter": "", "title": p.get("title") or "",
             "slides": t["slides"]}
            for t, p in zip(out, [pages[min(i * 3, len(pages) - 1)]
                                  for i in range(len(out))])]


def _structure(chat: Callable, pages: list[dict], label: str,
               result: dict) -> Optional[list[dict]]:
    """Ask the LLM for a chapter/topic outline that regroups the slides
    into a systematic knowledge structure (not slide order)."""
    by_idx = {p["index"]: p for p in pages}
    titles = "; ".join(f"{p['index']}: {p.get('title') or ''}" for p in pages)
    outline_txt = "\n".join(
        f"Slide {p['index']}: {re.sub(r'\\s+', ' ', (p.get('text') or '')).strip()[:150]}"
        for p in pages)[:5500]
    d = _json_obj(chat(
        f"You are planning the table of contents of systematic study notes "
        f"for the lecture \"{result.get('title') or ''}\", in {label}. "
        "The lecture is a sequence of slides with transcript excerpts. "
        "Regroup the material into 2-6 chapters (only give each chapter a "
        "title when the lecture has clearly separated parts; otherwise the "
        "chapter title may be empty) and each chapter into 2-8 TOPICS. "
        "Aim for one topic per 3-5 slides: several narrow topics are better "
        "than a few broad ones. A topic is a knowledge unit (concept, method, "
        "procedure, comparison, caution) that may span several non-adjacent "
        "slides; merge slides that belong to one concept and split mixed "
        "slides. Do not merely copy slide titles, and write every chapter and "
        f"topic title in {label}.\n"
        f"Slide titles: {titles[:3000]}\n{outline_txt}\n\n"
        "Reply with ONLY a valid JSON object:\n"
        '{"chapters":[{"title":"<chapter or empty>","topics":[{"title":'
        '"<knowledge-unit title, not a slide copy>","focus":"<what this '
        'topic must explain, 1 short sentence>","slides":[<slide '
        "numbers covered>]}]}]}\n"
        "Use ONLY slide numbers listed above; each slide may appear in "
        "multiple topics.",
        max_tokens=3000, temperature=0.2) or "")
    if not d or not isinstance(d.get("chapters"), list):
        return None
    out: list[dict] = []
    for ch in d["chapters"]:
        if not isinstance(ch, dict):
            continue
        ch_title = str(ch.get("title") or "").strip()
        for tp in (ch.get("topics") if isinstance(ch.get("topics"), list)
                   else []):
            if not isinstance(tp, dict):
                continue
            idx = [int(x) for x in (tp.get("slides") or [])
                   if str(x).strip().lstrip("-").isdigit()
                   and int(x) in by_idx]
            if not idx:
                continue
            out.append({"chapter": ch_title,
                        "title": str(tp.get("title") or "").strip(),
                        "focus": str(tp.get("focus") or "").strip(),
                        "slides": idx})
    if not out:
        return None
    mapped = {i for t in out for i in t["slides"]}
    missing = [p["index"] for p in pages if p["index"] not in mapped]
    for i in missing:  # keep every slide inside the notes
        best = min(range(len(out)),
                   key=lambda k: min(abs(i - s) for s in out[k]["slides"]))
        out[best]["slides"].append(i)
    return out


def _slide_block(p: dict) -> str:
    return (f"### Slide {p['index']} [{_ts(p['start'])} - {_ts(p['end'])}] "
            f"{p.get('title') or ''}\n{(p.get('text') or '')[:2200]}")


def _topic_body(chat: Callable, topic: dict, slides: list[dict],
                label: str) -> tuple[list[dict], list[dict]]:
    """Systematized knowledge body for one topic + its raw terms.

    The LLM must distill and reorganize the transcript into key points, not
    reproduce it; a character-similarity guard retries any copy-like draft.
    """
    block = "\n\n".join(_slide_block(p) for p in slides)
    target = max(6, min(24, len(block) // 700))
    instr = (
        "You are writing SYSTEMATIC study notes in "
        f"{label} for ONE knowledge topic of a lecture.\n"
        f"Topic: {topic.get('title') or ''}"
        + (f"\nFocus: {topic['focus']}" if topic.get("focus") else "") + "\n"
        "The transcript below may span several slides on this topic. Your "
        "task is to DISTILL and REORGANIZE it, never to copy it: do not "
        "transcribe, quote, or restate the slides one by one, and do not "
        "follow slide order. Synthesize the lecturer's teaching into a "
        f"structured knowledge unit in {label}: the core definition first, "
        "then principles / steps / conditions / numbers / examples / "
        "comparisons / pitfalls - in a logically ordered way a student can "
        f"review. Aim for about {target} key points so that nothing "
        "important is left out. Rules: paraphrase in your own words (never "
        "reproduce transcript sentences verbatim); ONE distilled key point "
        "per short bullet (max ~25 words); use a sub-point heading "
        '(k:"s") only when at least two bullets follow it; drop fillers, '
        "repetitions and announcements; keep technical identifiers, "
        "formulas and numbers exact; do not invent content. Reply with ONLY "
        "a valid JSON object:\n"
        '{"blocks":[{"k":"s","t":"<optional sub-point group heading>"},'
        '{"k":"b","t":"<one distilled key point>"}],\n'
        f'"terms":[{{"t":"<technical term used>","d":"<one-sentence '
        f'definition in {label}>"}}]}}\n'
        '"blocks" must start with the definition/overview bullets.\n\n' + block
    )
    mt = max(1500, min(4000, int(len(block) * 1.6)))

    def parse(d: Optional[dict]) -> tuple[list[dict], list[dict]]:
        blocks: list[dict] = []
        if d and isinstance(d.get("blocks"), list):
            for x in d["blocks"]:
                if not isinstance(x, dict):
                    continue
                kind = "s" if str(x.get("k")) == "s" else "b"
                t = str(x.get("t") or "").strip()
                if t:
                    blocks.append({"k": kind, "t": t[:400]})
        raw: list[dict] = []
        if d and isinstance(d.get("terms"), list):
            for x in d["terms"]:
                if isinstance(x, dict) and str(x.get("t", "")).strip() \
                        and str(x.get("d", "")).strip():
                    raw.append({"t": str(x["t"]).strip()[:60],
                                "d": str(x["d"]).strip()[:300]})
        # cap first, then drop sub-headings that ended up with no bullets
        blocks = blocks[:target + 8]
        blocks = [b for i, b in enumerate(blocks)
                  if b["k"] == "b"
                  or any(x["k"] == "b" for x in blocks[i + 1:i + 5])]
        return blocks, raw

    best: tuple[float, list[dict], list[dict]] = (1.0, [], [])
    for extra in ("", "\nRETRY: the previous draft reproduced transcript "
                  "sentences word-for-word. Rewrite every bullet as a "
                  "condensed paraphrase of the idea (max 25 words each)."):
        d = _json_obj(chat(instr + extra, max_tokens=mt, temperature=0.3) or "")
        blocks, raw = parse(d)
        if not blocks:
            continue
        ratio = _verbatim_ratio(blocks, block)
        if ratio < best[0]:
            best = (ratio, blocks, raw)
        if ratio <= 0.6:
            break
    return best[1], best[2]


def _verbatim_ratio(blocks: list[dict], transcript: str) -> float:
    """Similarity between the drafted bullets and the raw transcript text.

    High values mean the model copied instead of synthesizing."""
    from difflib import SequenceMatcher
    text = " ".join(b["t"] for b in blocks).lower()
    trans = re.sub(r"\s+", " ", transcript).lower()
    if not text or not trans:
        return 0.0
    return SequenceMatcher(None, text[:4000], trans[:6000]).ratio()


def _terms(chat: Callable, raw: list[dict], label: str) -> list[dict]:
    seen: dict[str, dict] = {}
    for x in raw:
        seen.setdefault(x["t"].casefold(), x)
    terms = list(seen.values())
    if len(terms) > 24:
        merged = _json_obj(chat(
            f"Below is a raw glossary extracted from lecture notes. Pick the "
            f"18 most important technical terms (merge near-duplicates), and "
            f"write a concise one-sentence definition for each in {label}. "
            "Reply with ONLY a valid JSON object "
            '{"terms":[{"t":"term","d":"definition"}]}.\n\n'
            + json.dumps({"terms": terms[:60]}, ensure_ascii=False)[:5000],
            max_tokens=1800, temperature=0.2) or "")
        if merged and isinstance(merged.get("terms"), list):
            out = [{"t": str(x.get("t", "")).strip()[:60],
                    "d": str(x.get("d", "")).strip()[:300]}
                   for x in merged["terms"]
                   if isinstance(x, dict) and str(x.get("t", "")).strip()
                   and str(x.get("d", "")).strip()]
            if out:
                return out[:18]
    return terms[:24]


# cover-page fallback used when the overview call returns nothing usable
_COVER = {
    "zh-Hant": "本講義「{title}」依主題重新組織，涵蓋：{a}。",
    "zh-Hans": "本讲义「{title}」按主题重组，涵盖：{a}。",
    "ja": "「{title}」は主題別に整理し、{a} をカバーします。",
    "ko": "\"{title}\" 강의는 주제별로 정리했으며 {a}를 다룹니다.",
    "latin": "This lecture, \"{title}\", is reorganized by topic: {a}.",
}


def generate(result: dict, notes_lang: str, chat: Callable,
             progress: Optional[Callable[[float], None]] = None) -> Optional[dict]:
    """Build the notes document (LLM calls). Returns a renderable dict."""
    prog = progress or (lambda f: None)
    pages = result.get("pages") or []
    if not pages:
        return None
    label, script = resolve_language(
        notes_lang, result.get("language") or "en",
        " ".join((p.get("text") or "") for p in pages)[:4000])
    by_idx = {p["index"]: p for p in pages}

    prog(0.05)
    overview, takeaways = _overview(chat, result, label)
    prog(0.15)
    topics = _structure(chat, pages, label, result) \
        or _fallback_topics(pages)
    raw_terms: list[dict] = []
    for ti, topic in enumerate(topics):
        slides = [by_idx[i] for i in sorted(set(topic["slides"]))]
        topic["slides"] = [p["index"] for p in slides]
        blocks, extra = _topic_body(chat, topic, slides, label)
        if blocks:
            topic["blocks"] = blocks
            raw_terms += extra
        else:  # last resort only when the LLM is unreachable: raw sentences
            topic["blocks"] = [{"k": "b", "t": x}
                               for p in slides for x in _fallback_points(p)]
        prog(0.15 + 0.7 * (ti + 1) / max(len(topics), 1))
    prog(0.88)
    terms = _terms(chat, raw_terms, label)
    prog(1.0)

    chapters: list[dict] = []
    for topic in topics:
        if not topic["blocks"]:
            continue
        if chapters and chapters[-1]["title"] == topic["chapter"]:
            chapters[-1]["topics"].append(topic)
        else:
            chapters.append({"title": topic["chapter"], "topics": [topic]})
    if not chapters:
        return None
    if not overview or not takeaways:
        names = [t["title"] for ch in chapters for t in ch["topics"] if t["title"]]
        if not overview and names:
            overview = _COVER.get(script, _COVER["latin"]).format(
                title=result.get("title") or "", a="、".join(names[:6]))
        if not takeaways:
            takeaways = names[:5]
    return _normalize_doc({
        "title": result.get("title") or "", "lang": label, "script": script,
        "overview": overview, "takeaways": takeaways, "chapters": chapters,
        "terms": terms,
        "_times": {p["index"]: (p.get("start", 0), p.get("end", 0))
                   for p in pages},
    })


_HDR = {
    "latin": {"overview": "Course overview", "takeaways": "Key takeaways",
              "notes": "Systematic notes", "toc": "Table of contents",
              "glossary": "Glossary of terms",
              "term": "Term", "definition": "Definition",
              "page": "Slide {n}", "foot": "Class notes"},
    "zh-Hans": {"overview": "课程概要", "takeaways": "要点回顾",
                "notes": "系统化笔记", "toc": "目录", "glossary": "术语注释",
                "term": "术语", "definition": "释义",
                "page": "第 {n} 页", "foot": "课堂笔记"},
    "zh-Hant": {"overview": "課程概要", "takeaways": "重點回顧",
                "notes": "系統化筆記", "toc": "目錄", "glossary": "術語註解",
                "term": "術語", "definition": "釋義",
                "page": "第 {n} 頁", "foot": "課堂筆記"},
    "ja": {"overview": "講義概要", "takeaways": "要点の整理",
           "notes": "体系ノート", "toc": "目次", "glossary": "用語注解",
           "term": "用語", "definition": "解説",
           "page": "{n} ページ目", "foot": "授業ノート"},
    "ko": {"overview": "강의 개요", "takeaways": "핵심 요약",
           "notes": "체계 노트", "toc": "목차", "glossary": "용어 풀이",
           "term": "용어", "definition": "풀이",
           "page": "{n} 페이지", "foot": "강의 노트"},
}


def render_pdf(notes: dict, meta: dict, thumbs: dict[int, Path],
               out_pdf: Path, font: str = "auto") -> None:
    """Render the notes dict into a PDF. `thumbs` maps slide index -> image."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import (Image, KeepTogether, Paragraph,
                                    SimpleDocTemplate, Spacer, Table,
                                    TableStyle)

    script = notes.get("script") or "latin"
    body, bold = _register_fonts(script, font if font in NOTE_FONTS else "auto")
    hdr = _HDR.get(script, _HDR["latin"])
    grey = colors.HexColor("#5b6b80")
    line = colors.HexColor("#d5dbe3")
    acc = colors.HexColor("#1e40af")

    def st(name, size, leading, **kw):
        kw.setdefault("fontName", body)
        kw.setdefault("textColor", colors.HexColor("#1c2733"))
        return ParagraphStyle(name, fontSize=size, leading=leading, **kw)

    s_title = st("t", 20, 26, fontName=bold, textColor=acc, spaceAfter=4)
    s_meta = st("m", 9, 13, textColor=grey)
    s_h2 = st("h2", 14.5, 19, fontName=bold, spaceBefore=14, spaceAfter=6)
    s_h3 = st("h3", 12, 16, fontName=bold, spaceBefore=8, spaceAfter=3)
    s_h4 = st("h4", 10.6, 14, fontName=bold, spaceBefore=4, spaceAfter=2)
    s_body = st("b", 10.5, 15.5)
    s_pt = st("pt", 10.2, 14.6, leftIndent=11, bulletIndent=1, spaceAfter=2)
    s_small = st("sm", 8.8, 12, textColor=grey)
    s_toc = st("toc", 9.6, 13.5)
    s_cell = st("c", 9.3, 13)
    s_cellh = st("ch", 9.5, 13, fontName=bold)

    story: list = []
    if notes.get("title"):
        story.append(Paragraph(escape(notes["title"]), s_title))
    meta_bits = [x for x in (
        f"Source: {meta.get('source')}" if meta.get("source") else "",
        f"Length: {_ts(meta['duration'])}" if meta.get("duration") else "",
        f"{meta['pages']} slides" if meta.get("pages") else "",
        f"Speech: {meta['speech_lang']}" if meta.get("speech_lang") else "",
        f"Notes language: {notes.get('lang')}" if notes.get("lang") else "",
        f"Generated: {meta['generated']}" if meta.get("generated") else "") if x]
    story.append(Paragraph(escape("  |  ".join(meta_bits)), s_meta))
    story.append(Spacer(1, 8))

    if notes.get("overview"):
        story += [Paragraph(escape(hdr["overview"]), s_h2),
                  Paragraph(escape(notes["overview"]), s_body)]
    if notes.get("takeaways"):
        story.append(Paragraph(escape(hdr["takeaways"]), s_h3))
        story += [Paragraph(escape(x), s_pt, bulletText="•")
                  for x in notes["takeaways"]]

    max_w, max_h = 16.6 * cm, 7.6 * cm
    chapters = notes.get("chapters")
    if not chapters and notes.get("sections"):  # legacy notes (page-by-page)
        story.append(Paragraph(escape(hdr["notes"]), s_h2))
        for s in notes["sections"]:
            head = Paragraph(
                escape(hdr["page"].format(n=s["i"]))
                + f'  <font size="8.5" color="#5b6b80">'
                  f'{escape(s["time"])}</font>',
                s_h3)
            blocks = [head]
            img_path = thumbs.get(s["i"])
            if img_path:
                try:
                    w0, h0 = ImageReader(str(img_path)).getSize()
                    k = min(max_w / w0, max_h / h0, 1.0)
                    blocks += [Image(str(img_path), w0 * k, h0 * k),
                               Spacer(1, 4)]
                except Exception:  # noqa: BLE001
                    log.warning("Notes thumbnail unusable: %s", img_path)
            if s.get("title"):
                blocks.append(Paragraph(escape(s["title"]),
                                        st("s" + str(s["i"]), 11.5, 15,
                                           fontName=bold)))
            story.append(KeepTogether(blocks))
            story += [Paragraph(escape(p), s_pt, bulletText="•")
                      for p in s.get("points") or []]
    else:
        multi = len(chapters) > 1 and any(c["title"] for c in chapters)
        if multi:  # table of contents for the systematic outline
            story.append(Paragraph(escape(hdr["toc"]), s_h2))
            for c in chapters:
                story.append(Paragraph(escape(c["title"]), s_h4))
                for t in c["topics"]:
                    story.append(Paragraph(
                        escape(t["title"] or " / ".join(
                            hdr["page"].format(n=i) for i in t["slides"][:3]))
                        + f'  <font size="8.5" color="#5b6b80">'
                          f'{escape(", ".join(str(i) for i in t["slides"][:8]))}'
                          f'</font>',
                        s_toc, bulletText="-"))
        story.append(Paragraph(escape(hdr["notes"]), s_h2))
        col_w = 11.4 * cm
        for ci, ch in enumerate(chapters):
            if multi:
                story.append(Paragraph(escape(ch["title"]), s_h2))
            for ti, t in enumerate(ch["topics"]):
                idxs = t["slides"]
                times = notes.get("_times") or {}
                time_rng = ""
                if idxs[0] in times and idxs[-1] in times:
                    time_rng = (f"{_ts(times[idxs[0]][0])} - "
                                f"{_ts(times[idxs[-1]][1])}")
                head = Paragraph(
                    escape(t["title"] or hdr["page"].format(n=idxs[0]))
                    + (f'  <font size="8.5" color="#5b6b80">'
                       f'{escape(time_rng)}</font>' if time_rng else ""),
                    s_h3)
                text_col: list = [head]
                for b in t["blocks"]:
                    if b["k"] == "s":
                        text_col.append(Paragraph(escape(b["t"]), s_h4))
                    else:
                        text_col.append(Paragraph(escape(b["t"]), s_pt,
                                                  bulletText="•"))
                img_col: list = []
                for i in idxs[:4]:
                    img_path = thumbs.get(i)
                    if not img_path:
                        continue
                    try:
                        w0, h0 = ImageReader(str(img_path)).getSize()
                        k = min(col_w * 0.98 / w0, 2.4 * cm / h0, 1.0)
                        img_col += [Image(str(img_path), w0 * k, h0 * k),
                                    Spacer(1, 3)]
                    except Exception:  # noqa: BLE001
                        log.warning("Notes thumbnail unusable: %s", img_path)
                if img_col:
                    img_col.append(Paragraph(
                        escape(hdr["page"].format(
                            n=", ".join(str(i) for i in idxs))),
                        s_small))
                    left, right = [text_col], [img_col]
                    table = Table([[left, right]],
                                  colWidths=[col_w, (16.6 - 11.4) * cm])
                    table.setStyle(TableStyle([
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 0),
                        ("RIGHTPADDING", (0, 0), (0, 0), 8),
                        ("RIGHTPADDING", (1, 0), (1, 0), 0),
                        ("TOPPADDING", (0, 0), (-1, -1), 0),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                    ]))
                    story.append(KeepTogether([table]))
                else:
                    story.append(KeepTogether([head] + text_col[1:]))

    if notes.get("terms"):
        story.append(Paragraph(escape(hdr["glossary"]), s_h2))
        rows = [[Paragraph(escape(hdr["term"]), s_cellh),
                 Paragraph(escape(hdr["definition"]), s_cellh)]]
        rows += [[Paragraph(escape(x["t"]), s_cell),
                  Paragraph(escape(x["d"]), s_cell)]
                 for x in notes["terms"]]
        table = Table(rows, colWidths=[4.4 * cm, 12.0 * cm], repeatRows=1)
        table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.4, line),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f8")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#fafbfd")]),
        ]))
        story.append(table)

    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setFont(body, 8)
        canvas.setFillColor(grey)
        canvas.drawCentredString(A4[0] / 2, 1.0 * cm,
                                 f"{hdr['foot']}  ·  {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(out_pdf), pagesize=A4,
                            leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.6 * cm,
                            title=notes.get("title") or hdr["foot"],
                            author="srt_gen")
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
