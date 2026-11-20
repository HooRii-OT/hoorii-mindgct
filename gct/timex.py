# Copyright 2026 HooRii Technology
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Rule-based time expressions: request time intent, unit time intervals and time features."""
import datetime as dt
import hashlib
import json
import math
import re
from collections import namedtuple

Span = namedtuple("Span", "lo hi prec start end kind")
INTENTS = ["none", "by_time", "ask_time", "latest", "order", "state_at", "action_time", "time_other"]

_MON = {m: i + 1 for i, m in enumerate("january february march april may june july august september october november december".split())}
_MON.update({m[:3]: i for m, i in list(_MON.items())}); _MON["sept"] = 9
_MONRE = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_WD_EN = {d: i for i, d in enumerate("monday tuesday wednesday thursday friday saturday sunday".split())}
_WD_EN.update({d[:3]: i for d, i in list(_WD_EN.items())}); _WD_EN.update({"tues": 1, "thur": 3, "thurs": 3})
_WD_ZH = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6, "七": 6}
_ZHNUM = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10, "半": 0.5}
_SEASON_EN = {"spring": (3, 5), "summer": (6, 8), "autumn": (9, 11), "fall": (9, 11), "winter": (12, 2)}
_SEASON_ZH = {"春": (3, 5), "夏": (6, 8), "秋": (9, 11), "冬": (12, 2)}
_NUMW = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
         "eleven": 11, "twelve": 12, "couple of": 2, "few": 3, "several": 3}

_HDR_EN = re.compile(r"\[Now:\s*(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})")
_HDR_ZH = re.compile(r"［当前时间：\s*(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})")


def _d(y, m, d):
    try:
        return dt.date(int(y), int(m), int(d))
    except ValueError:
        return None


def _month_iv(y, m):
    a = _d(y, m, 1)
    if a is None:
        return None
    b = (_d(y + (m == 12), m % 12 + 1, 1) - dt.timedelta(days=1))
    return a.toordinal(), b.toordinal(), "month"


def _year_iv(y):
    return dt.date(int(y), 1, 1).toordinal(), dt.date(int(y), 12, 31).toordinal(), "year"


def _day_iv(x):
    o = x.toordinal()
    return o, o, "day"


def _season_iv(y, rng):
    a, b = rng
    if a <= b:
        return _month_iv(y, a)[0], _month_iv(y, b)[1], "season"
    return _month_iv(y, a)[0], _month_iv(y + 1, b)[1], "season"


def _zh_int(s):
    if s is None:
        return None
    if s.isdigit():
        return int(s)
    if s in _ZHNUM:
        return _ZHNUM[s]
    if len(s) == 2 and s[0] == "十" and s[1] in _ZHNUM:
        return 10 + _ZHNUM[s[1]]
    if len(s) == 2 and s[1] == "十" and s[0] in _ZHNUM:
        return _ZHNUM[s[0]] * 10
    if len(s) == 3 and s[1] == "十":
        return _ZHNUM.get(s[0], 0) * 10 + _ZHNUM.get(s[2], 0)
    return None


def _year2(y):
    y = int(y)
    return y + 2000 if y < 50 else (y + 1900 if y < 100 else y)


def _nearest_year(m, d, anchor, past_bias=True):
    best = None
    for y in (anchor.year - 1, anchor.year, anchor.year + 1):
        x = _d(y, m, d or 1)
        if x is None:
            continue
        delta = (x - anchor).days
        score = abs(delta) + (1000 if past_bias and delta > 7 else 0)
        if best is None or score < best[0]:
            best = (score, x)
    return best[1] if best else None


def parse_now(text):
    m = _HDR_EN.search(text) or _HDR_ZH.search(text)
    return _d(*m.groups()) if m else None


def _parse_date_str(s):
    m = re.match(r"\s*(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", str(s or ""))
    return _d(*m.groups()) if m else None


_ABS = [

    (re.compile(r"(\d{4})\s*[-/.]\s*(\d{1,2})\s*[-/.]\s*(\d{1,2})"), "ymd"),
    (re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]?"), "ymd_zh"),
    (re.compile(r"(?<!\d)(\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]?"), "y2md_zh"),
    (re.compile(r"(?<!\d)(\d{1,2})\s*([./])\s*(\d{1,2})\s*\2\s*(\d{4})(?!\d)"), "dmy_num"),
    (re.compile(rf"\b({_MONRE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.I), "mdy_en"),
    (re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONRE})\.?,?\s+(\d{{4}})\b", re.I), "dmy_en"),
    (re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月(?!\s*\d{1,2}\s*[日号])"), "ym_zh"),
    (re.compile(r"(?<![\d.])(\d{4})-(\d{1,2})(?![\d-])"), "ym_num"),
    (re.compile(rf"\b({_MONRE})\.?\s+(?:of\s+)?(\d{{4}})\b", re.I), "my_en"),
    (re.compile(rf"\b({_MONRE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?!\s*,?\s*\d{{4}})", re.I), "md_en"),
    (re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONRE})\b(?!\.?,?\s+\d{{4}})", re.I), "dm_en"),
    (re.compile(r"(?<![\d年])(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]"), "md_zh"),
    (re.compile(r"\b((?i:in|during|since|until|by|of|for|through|throughout|over|around|from|early|late|mid|last|this|next))\s+(?:(?i:early|late|mid)-?\s*)?"
                r"((?i:jan(?:uary)?|feb(?:ruary)?|march|apr(?:il)?|june|july|aug(?:ust)?|sept(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)|May)"
                r"\b(?![\s,]*\d)"), "m_en"),
    (re.compile(r"(?<![\d年.\-/])(\d{1,2})\s*月份?(?!\s*\d)"), "m_zh"),
    (re.compile(r"(\d{4})\s*年(?!\s*\d{1,2}\s*月)"), "y_zh"),
    (re.compile(r"(?<![\d$.,])\b((?:19|20)\d{2})\b(?!\d|[.,]\d|%|-\d)"), "y_en"),
]


def _abs_spans(text, anchor, past_bias, hint=None):
    out, taken = [], []

    def free(a, b):
        return all(b <= x or a >= y for x, y in taken)

    for rx, kind in _ABS:
        for m in rx.finditer(text):
            a, b = m.span()
            if not free(a, b):
                continue
            g = m.groups(); iv = None
            if kind in ("ymd", "ymd_zh"):
                x = _d(g[0], g[1], g[2]); iv = _day_iv(x) if x else None
            elif kind == "y2md_zh":
                x = _d(_year2(g[0]), g[1], g[2]); iv = _day_iv(x) if x else None
            elif kind == "dmy_num":
                p, sep, q, y = int(g[0]), g[1], int(g[2]), int(g[3])
                dm, md = _d(y, q, p), _d(y, p, q)
                if dm and md and dm != md:
                    if hint is not None:
                        x = min((dm, md), key=lambda z: abs((z - hint).days))
                    else:
                        x = dm if sep == "." else md
                else:
                    x = dm or md
                iv = _day_iv(x) if x else None
            elif kind == "mdy_en":
                x = _d(g[2], _MON[g[0][:4].lower().rstrip("t")] if g[0][:4].lower().rstrip("t") in _MON else _MON[g[0][:3].lower()], g[1])
                iv = _day_iv(x) if x else None
            elif kind == "dmy_en":
                x = _d(g[2], _MON[g[1][:3].lower()], g[0]); iv = _day_iv(x) if x else None
            elif kind == "ym_zh":
                iv = _month_iv(int(g[0]), int(g[1])) if 1 <= int(g[1]) <= 12 else None
            elif kind == "ym_num":
                iv = _month_iv(int(g[0]), int(g[1])) if 1 <= int(g[1]) <= 12 and 1900 <= int(g[0]) <= 2100 else None
            elif kind == "my_en":
                iv = _month_iv(int(g[1]), _MON[g[0][:3].lower()])
            elif kind in ("md_en", "dm_en", "md_zh"):
                if anchor is None:
                    continue
                if kind == "md_en":
                    mo, dd = _MON[g[0][:3].lower()], int(g[1])
                elif kind == "dm_en":
                    mo, dd = _MON[g[1][:3].lower()], int(g[0])
                else:
                    mo, dd = int(g[0]), int(g[1])
                if not (1 <= mo <= 12 and 1 <= dd <= 31):
                    continue
                x = _nearest_year(mo, dd, anchor, past_bias)
                x = _d(x.year, mo, dd) if x else None
                iv = _day_iv(x) if x else None
            elif kind in ("y_zh", "y_en"):
                y = int(g[0]); iv = _year_iv(y) if 1900 <= y <= 2100 else None
            elif kind in ("m_en", "m_zh"):
                if anchor is None:
                    continue
                if kind == "m_en":
                    pre, mo = g[0].lower(), _MON[g[1][:3].lower()]
                else:
                    pre, mo = "", int(g[0])
                if not 1 <= mo <= 12:
                    continue
                if pre == "last":
                    y = anchor.year if mo < anchor.month else anchor.year - 1
                elif pre == "next":
                    y = anchor.year if mo > anchor.month else anchor.year + 1
                elif pre == "this" or mo == anchor.month:
                    y = anchor.year
                elif past_bias:
                    y = anchor.year if mo < anchor.month else anchor.year - 1
                else:
                    y = min((anchor.year - 1, anchor.year, anchor.year + 1), key=lambda yy: abs((dt.date(yy, mo, 15) - anchor).days))
                iv = _month_iv(y, mo)
            if iv:
                out.append(Span(iv[0], iv[1], iv[2], a, b, kind)); taken.append((a, b))
    return out, taken


def _week_start(x):
    return x - dt.timedelta(days=x.weekday())


def _rel_spans(text, anchor, taken):
    out = []
    t = text

    def free(a, b):
        return all(b <= x or a >= y for x, y in taken)

    def add(a, b, lo, hi, prec, kind):
        if free(a, b):
            out.append(Span(lo.toordinal(), hi.toordinal(), prec, a, b, kind)); taken.append((a, b))

    A = anchor
    day = dt.timedelta(days=1)

    for w, off in (("大前天", -3), ("大后天", 3), ("前天", -2), ("后天", 2), ("昨天", -1), ("昨晚", -1), ("今天", 0), ("今晚", 0),
                   ("明天", 1), ("明晚", 1)):
        for m in re.finditer(w, t):
            x = A + off * day; add(m.start(), m.end(), x, x, "day", "rel_day")

    for rx, off in ((r"\bthe day before yesterday\b", -2), (r"\bthe day after tomorrow\b", 2), (r"\byesterday\b", -1),
                    (r"\blast night\b", -1), (r"\btoday\b", 0), (r"\btonight\b", 0), (r"\btomorrow\b", 1)):
        for m in re.finditer(rx, t, re.I):
            x = A + off * day; add(m.start(), m.end(), x, x, "day", "rel_day")

    for m in re.finditer(r"(上上|上|这|本|下下|下)?\s*(?:个)?(?:周|星期|礼拜)([一二三四五六日天])", t):
        wk = {"上上": -2, "上": -1, "这": 0, "本": 0, "下": 1, "下下": 2, None: None}[m.group(1)]
        wd = _WD_ZH[m.group(2)]
        if wk is None:
            x = A - dt.timedelta(days=(A.weekday() - wd) % 7)
        else:
            x = _week_start(A) + dt.timedelta(weeks=wk, days=wd)
        add(m.start(), m.end(), x, x, "day", "rel_wd")
    for m in re.finditer(r"\b(last|next|this|on)?\s*(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tues?|wed|thur?s?|fri|sat|sun)\b",
                         t, re.I):
        if m.group(2).lower()[:3] in ("sat", "sun", "mon", "tue", "wed", "thu", "fri") and len(m.group(2)) <= 3 and not m.group(1):
            continue
        wd = _WD_EN[m.group(2).lower()] if m.group(2).lower() in _WD_EN else _WD_EN[m.group(2).lower()[:3]]
        w = (m.group(1) or "").lower()
        if w == "last":
            x = A - dt.timedelta(days=(A.weekday() - wd) % 7 or 7)
        elif w == "next":
            x = A + dt.timedelta(days=(wd - A.weekday()) % 7 or 7)
        elif w == "this":
            x = _week_start(A) + dt.timedelta(days=wd)
        else:
            x = A - dt.timedelta(days=(A.weekday() - wd) % 7)
        add(m.start(), m.end(), x, x, "day", "rel_wd")

    for m in re.finditer(r"(上上|上|这|本|下)?(?:个)?周末", t):
        wk = {"上上": -2, "上": -1, "这": 0, "本": 0, "下": 1, None: 0}[m.group(1)]
        s = _week_start(A) + dt.timedelta(weeks=wk, days=5)
        if m.group(1) is None and s > A:
            s -= dt.timedelta(weeks=1)
        add(m.start(), m.end(), s, s + day, "week", "rel_weekend")
    for m in re.finditer(r"\b(last|next|this)\s+weekend\b|\bthe weekend before last\b", t, re.I):
        if m.group(0).lower().startswith("the weekend before"):
            wk = -2
        else:
            wk = {"last": -1, "next": 1, "this": 0}[m.group(1).lower()]
        s = _week_start(A) + dt.timedelta(weeks=wk, days=5)
        add(m.start(), m.end(), s, s + day, "week", "rel_weekend")

    for m in re.finditer(r"(上上|上|这|本|下下|下)(?:个)?(?:周|星期|礼拜)(?![一二三四五六日天末])", t):
        wk = {"上上": -2, "上": -1, "这": 0, "本": 0, "下下": 2, "下": 1}[m.group(1)]
        s = _week_start(A) + dt.timedelta(weeks=wk); add(m.start(), m.end(), s, s + 6 * day, "week", "rel_week")
    for m in re.finditer(r"(上上|上|这|本|下下|下)个?月(?![\d一二三四五六七八九十]+[日号])", t):
        k = {"上上": -2, "上": -1, "这": 0, "本": 0, "下下": 2, "下": 1}[m.group(1)]
        mi = A.year * 12 + A.month - 1 + k
        lo, hi, _ = _month_iv(mi // 12, mi % 12 + 1); add(m.start(), m.end(), dt.date.fromordinal(lo), dt.date.fromordinal(hi), "month", "rel_month")
    for m in re.finditer(r"(大前年|前年|去年|今年|明年|后年)(?:的)?(\d{1,2})\s*月份?(?!\s*\d)", t):
        y = A.year + {"大前年": -3, "前年": -2, "去年": -1, "今年": 0, "明年": 1, "后年": 2}[m.group(1)]
        if 1 <= int(m.group(2)) <= 12:
            lo, hi, _ = _month_iv(y, int(m.group(2))); add(m.start(), m.end(), dt.date.fromordinal(lo), dt.date.fromordinal(hi), "month", "rel_month")
    for m in re.finditer(r"(大前年|前年|去年|今年|明年|后年)(?:的)?(春天|夏天|秋天|冬天|春|夏|秋|冬|春季|夏季|秋季|冬季)?", t):
        y = A.year + {"大前年": -3, "前年": -2, "去年": -1, "今年": 0, "明年": 1, "后年": 2}[m.group(1)]
        if m.group(2):
            lo, hi, _ = _season_iv(y, _SEASON_ZH[m.group(2)[0]]); add(m.start(), m.end(), dt.date.fromordinal(lo), dt.date.fromordinal(hi), "season", "rel_season")
        else:
            lo, hi, _ = _year_iv(y); add(m.start(), m.end(), dt.date.fromordinal(lo), dt.date.fromordinal(hi), "year", "rel_year")
    for m in re.finditer(r"\b(last|this|next)\s+(week|month|year|spring|summer|autumn|fall|winter)\b", t, re.I):
        k = {"last": -1, "this": 0, "next": 1}[m.group(1).lower()]; u = m.group(2).lower()
        if u == "week":
            s = _week_start(A) + dt.timedelta(weeks=k); add(m.start(), m.end(), s, s + 6 * day, "week", "rel_week")
        elif u == "month":
            mi = A.year * 12 + A.month - 1 + k; lo, hi, _ = _month_iv(mi // 12, mi % 12 + 1)
            add(m.start(), m.end(), dt.date.fromordinal(lo), dt.date.fromordinal(hi), "month", "rel_month")
        elif u == "year":
            lo, hi, _ = _year_iv(A.year + k); add(m.start(), m.end(), dt.date.fromordinal(lo), dt.date.fromordinal(hi), "year", "rel_year")
        else:
            rng = _SEASON_EN[u]
            y = A.year + k
            if k == -1 and (A.month - 1) // 3 > (rng[0] - 1) // 3 and rng[0] != 12:
                y = A.year
            lo, hi, _ = _season_iv(y, rng); add(m.start(), m.end(), dt.date.fromordinal(lo), dt.date.fromordinal(hi), "season", "rel_season")

    for m in re.finditer(r"([\d一二两三四五六七八九十半]+)\s*(?:个)?(天|日|周|星期|礼拜|月|年)(?:前|以前|之前)", t):
        n = _zh_int(m.group(1))
        if n is None or n > _MAX_AGO[m.group(2)]:
            continue
        u = m.group(2)
        lo, hi, prec = _ago(A, n, {"天": "day", "日": "day", "周": "week", "星期": "week", "礼拜": "week", "月": "month", "年": "year"}[u])
        add(m.start(), m.end(), lo, hi, prec, "rel_ago")
    for m in re.finditer(r"\b(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|couple of|few|several)\s+(day|week|month|year)s?\s+ago\b",
                         t, re.I):
        n = int(m.group(1)) if m.group(1).isdigit() else _NUMW[m.group(1).lower()]
        if n > _MAX_AGO[m.group(2).lower()]:
            continue
        lo, hi, prec = _ago(A, n, m.group(2).lower()); add(m.start(), m.end(), lo, hi, prec, "rel_ago")
    for m in re.finditer(r"\bin\s+(\d+|a|an|one|two|three|four|five|six)\s+(day|week|month)s?\b", t, re.I):
        n = int(m.group(1)) if m.group(1).isdigit() else _NUMW[m.group(1).lower()]
        if n > _MAX_AGO[m.group(2).lower()]:
            continue
        lo, hi, prec = _ago(A, -n, m.group(2).lower()); add(m.start(), m.end(), lo, hi, prec, "rel_in")

    for m in re.finditer(r"\b(spring|summer|autumn|fall|winter)\s+(?:of\s+)?(\d{4})\b", t, re.I):
        if not 1900 <= int(m.group(2)) <= 2100:
            continue
        lo, hi, _ = _season_iv(int(m.group(2)), _SEASON_EN[m.group(1).lower()])
        add(m.start(), m.end(), dt.date.fromordinal(lo), dt.date.fromordinal(hi), "season", "season_y")

    for m in re.finditer(r"\brecently\b|\blately\b|\bthese days\b|最近|近来|这阵子|这段时间", t, re.I):
        add(m.start(), m.end(), A - 30 * day, A, "period", "rel_recent")
    return out


_MAX_AGO = {"天": 3650, "日": 3650, "day": 3650, "周": 520, "星期": 520, "礼拜": 520, "week": 520, "月": 1200, "month": 1200,
            "年": 100, "year": 100}


def _ago(A, n, unit):
    day = dt.timedelta(days=1)
    if unit == "day":
        x = A - dt.timedelta(days=round(n)); return x, x, "day"
    if unit == "week":
        c = A - dt.timedelta(days=round(7 * n)); return c - 3 * day, c + 3 * day, "week"
    if unit == "month":
        mi = A.year * 12 + A.month - 1 - int(round(n)); lo, hi, _ = _month_iv(mi // 12, mi % 12 + 1)
        return dt.date.fromordinal(lo) - 7 * day, dt.date.fromordinal(hi) + 7 * day, "month"
    y = A.year - int(round(n)); lo, hi, _ = _year_iv(y)
    return dt.date.fromordinal(lo), dt.date.fromordinal(hi), "year"


_PERIOD = [re.compile(r"(?:Period\s+)?(\d{4})-(\d{1,2})\s+(?:to|until|–|-)\s+(\d{4})-(\d{1,2})", re.I),
           re.compile(r"(?:时间段：)?\s*(\d{4})-(\d{1,2})\s*至\s*(\d{4})-(\d{1,2})")]
_PERIOD_OPEN = [re.compile(r"(?:Period\s+)?(\d{4})-(\d{1,2})\s+(?:to|until)\s+(?:now|present|today)", re.I),
                re.compile(r"(?:时间段：)?\s*(\d{4})-(\d{1,2})\s*至\s*今"),
                re.compile(r"自\s*(\d{4})\s*年\s*(\d{1,2})\s*月\s*起"), re.compile(r"\bsince\s+(" + _MONRE + r")\.?\s+(\d{4})\b", re.I)]


_YEAR_RANGE = [re.compile(r"(?<!\d)((?:19|20)\d{2})\s*年?\s*(?:至|到|～|~|-|–|—)\s*((?:19|20)\d{2})\s*年"),
               re.compile(r"\b(?:from\s+)?((?:19|20)\d{2})\s*(?:to|until|till|through|-|–|—)\s*((?:19|20)\d{2})\b(?![-/.]\d)", re.I),
               re.compile(r"\bbetween\s+((?:19|20)\d{2})\s+and\s+((?:19|20)\d{2})\b", re.I)]


def _period_spans(text, now):
    out, taken = [], []
    for rx in _YEAR_RANGE:
        for m in rx.finditer(text):
            y1, y2 = int(m.group(1)), int(m.group(2))
            if y1 <= y2 <= y1 + 60 and not any(not (m.end() <= x or m.start() >= y) for x, y in taken):
                out.append(Span(_year_iv(y1)[0], _year_iv(y2)[1], "period", m.start(), m.end(), "period")); taken.append(m.span())
    for rx in _PERIOD:
        for m in rx.finditer(text):
            y1, m1, y2, m2 = map(int, m.groups())
            a = _month_iv(y1, m1); b = _month_iv(y2, m2)
            if a and b and a[0] <= b[1]:
                out.append(Span(a[0], b[1], "period", m.start(), m.end(), "period")); taken.append(m.span())
    for rx in _PERIOD_OPEN:
        for m in rx.finditer(text):
            if any(not (m.end() <= x or m.start() >= y) for x, y in taken):
                continue
            g = m.groups()
            if g[0].isdigit():
                y1, m1 = int(g[0]), int(g[1])
            else:
                y1, m1 = int(g[1]), _MON[g[0][:3].lower()]
            a = _month_iv(y1, m1)
            hi = (now or dt.date(2100, 1, 1)).toordinal()
            if a and a[0] <= hi:
                out.append(Span(a[0], hi, "period", m.start(), m.end(), "period_open")); taken.append(m.span())
    return out, taken


PARSE_ERRORS = {}


def find_times(text, now, anchor=None, past_bias=True, hint=None):
    try:
        return _find_times(text, now, anchor, past_bias, hint)
    except (ValueError, OverflowError, TypeError, KeyError) as e:
        PARSE_ERRORS[type(e).__name__] = PARSE_ERRORS.get(type(e).__name__, 0) + 1
        return []


def _find_times(text, now, anchor, past_bias, hint):
    anchor = anchor or now
    text = _HDR_EN.sub(lambda m: " " * len(m.group(0)), text)
    text = _HDR_ZH.sub(lambda m: " " * len(m.group(0)), text)
    per, taken = _period_spans(text, now)
    ab, taken2 = _abs_spans(_mask(text, taken), anchor, past_bias, hint)
    rel = _rel_spans(_mask(text, taken + taken2), anchor, taken + taken2) if anchor is not None else []
    return sorted(per + ab + rel, key=lambda s: s.start)


def _mask(text, spans):
    t = list(text)
    for a, b in spans:
        for i in range(a, b):
            t[i] = " "
    return "".join(t)


_ASK = re.compile(r"\bwhen\b|what time|what day|which day|what date|how long ago|how many (?:days|weeks|months|years)|how long has it been|"
                  r"什么时候|啥时候|几点|哪天|几号|哪一天|哪年|多久前|多少天|多长时间|隔了多久|多久了", re.I)
_LATEST = re.compile(r"\blast time\b|\bmost recent(?:ly)?\b|\bthe latest\b|\blast (?:visit|order|appointment|trip|booking)\b|上次|上一次|最近一次|最后一次", re.I)
_ORDER = re.compile(r"\b(?:before|after|first|earlier than|later than|prior to)\b|先.{1,20}还是|之前|之后|以前|以后|在.{1,12}前", re.I)
_STATE = re.compile(r"\b(?:live|lived|living|work|worked|working|address|job|employer|school|studied|stay(?:ed)?)\b|住在|住哪|住的|工作|上班|地址|在哪家|在哪儿上|读书|上学", re.I)
_WHERE_Q = re.compile(r"\bwhere\b|\bwhich (?:company|school|city|place|address|job)\b|\bwhat (?:was|were) my\b|\bwho was my\b|哪|什么公司|什么学校|谁", re.I)
_PAST_Q = re.compile(r"\bwhat did (?:i|we)\b|\bwho did (?:i|we)\b|\bwhere did (?:i|we)\b|\bdid (?:i|we)\b|\bwhat happened\b|\bwhat was (?:i|my)\b|"
                     r"\bwhat (?:did|have) i (?:do|done|buy|bought|eat|ate)\b|做了什么|干了啥|干了什么|做了啥|见了谁|去了哪|去哪了|买了什么|买了啥|吃了什么|吃了啥|"
                     r"发生了什么|都干|都做|那天.{0,8}(?:做|干|去|见)", re.I)
_ACTION = re.compile(r"\b(?:email|e-mail|send|book|remind(?:er)?|schedule|set (?:up|a|an|the|that|it)|call|text|message|arrange|reserve|order|plan|draft|write|"
                     r"invite|pick up|confirm|cancel|reschedule|help me|please|can you|could you|would you|i'd like you to|let'?s|find|prepare|put together|"
                     r"get|check|look up|make|convert|draft|need|planning|i'm planning|i want)\b|"
                     r"帮我|麻烦|请|给.{1,10}(?:发|打|写)|发邮件|发消息|发微信|打电话|订|预约|安排|提醒|约|买|寄|取消|改到|改约", re.I)


_ASK2 = re.compile(r"what time|what day|which day|\b(?:the )?date\b|how long ago|how many (?:days|weeks|months|years)|how long (?:has it been|"
                   r"between|since|did|was|were)|how much time|how far apart|\bapart\b|\bgap\b|time (?:passed|elapsed|between)|"
                   r"什么时候|啥时候|几点|哪天|几号|哪一天|哪年|几月|日期|日子|多久|多长时间|多少天|多少个?月|多少周|多少年|距今|相隔|隔了|间隔|过去了?多少|过了多少", re.I)
_LATEST2 = re.compile(r"\blast time\b|\bmost recent(?:ly)?\b|\bthe latest\b|\blatest\b|\b(?:my|the) last (?:visit|order|appointment|trip|booking|reservation|"
                      r"purchase|delivery|meeting|session|one)\b|\b(?:my|the) last\b(?:\s+[\w&'.-]+){0,4}?\s+(?:order|reservation|appointment|booking|visit|trip|purchase|delivery|meeting|session)s?\b|\b(?:did|do|have) (?:i|we) last\b|\b(?:i|we) last\b|上次|上一次|上回|最近一次|最近的一次|最近那次|最近那笔|最近一笔|最近的那|最后一次|最近.{0,6}(?:那次|那笔|一次|一笔|那单|一单)",
                      re.I)
_ORDER2 = re.compile(r"\bbefore or after\b|\bafter or before\b|\bwhich\b.{0,60}\b(?:first|earlier|later|sooner)\b|\bcame first\b|\bhappened first\b|"
                     r"\b(?:did|was|were|had|do)\b.{0,90}\b(?:before|after|earlier than|later than|prior to)\b.{0,90}\?|\bfirst\b.{0,6}\bor\b|\bor\b.{0,80}\bfirst\b\s*[?？]|"
                     r"哪.{0,3}先|谁先|先.{1,24}还是|哪个在前|哪个更早|哪个早|在.{1,24}(?:之前|之后|以前|以后)还是|(?:之前|之后|以前|以后)还是", re.I)
_EVENT_N = re.compile(r"\b(?:order(?:ed|s)?|purchase[ds]?|bought|buy|deliver(?:y|ed|ies)|reservations?|booking|booked|appointments?|events?|attend(?:ed)?|"
                      r"trips?|travel(?:l?ed)?|visit(?:ed|s)?|went|go|do|did|happen(?:ed)?|plans?|dinner|lunch|party|meet(?:ing)?|met)\b|"
                      r"订单|外卖|下单|下过|买了|买过|网购|购|送达|签收|预约|订座|订位|订了|位子|餐厅|活动|聚会|参加|去了|去过|做了|做过|干了|干啥|见了|见过|旅行|旅游|行程|出差|"
                      r"门诊|看病|安排|有哪些|都有|发生|\bgoing on\b|\bup to\b|\bcalendar\b|\bagenda\b|\bscheduled\b|日程|行事历", re.I)
_STATE_N = re.compile(r"\b(?:live[ds]?|living|address|home|apartment|house|city|work(?:ed|ing)?|job|employer|company|school|stud(?:y|ied|ying)|"
                      r"email|e-mail|phone|number|medications?|medicine|taking|diet|eating|habits?|routine|schedule|commute|hobby|hobbies|car|"
                      r"doctor|dentist|gym|role|title|position|team|manager|boss|partner|using|use[ds]?)\b|"
                      r"住|地址|邮箱|电话|手机|号码|工作|公司|单位|上班|学校|读书|上学|作息|习惯|饮食|吃药|用药|药|通勤|爱好|职位|岗位|领导|车|用的|在用|用什么", re.I)
_QUESTION = re.compile(r"\?|？|\b(?:what|which|where|who|whom|how|did|was|were|have i|had i|do i|remind me (?:what|which|where|who|how|of))\b|"
                       r"\b(?:recap|rundown|summary|summari[sz]e|overview|list|tell me|show me|go over|look back|walk me through)\b|"
                       r"什么|啥|哪|吗|几|谁|多少|有没有|是不是|回忆|记得|想起|想不起|来着|是怎样|怎么样|列一下|列出|汇总|总结|回顾|梳理", re.I)


def classify_intent(text, spans, now=None, recall=False):
    t = text
    if _ORDER2.search(_mask(t, [(x.start, x.end) for x in spans])):
        return "order"
    if _LATEST2.search(t):
        return "latest"
    if _ASK2.search(t) or any(not re.search(r"(?:\d{4}|,)\s*$", t[:m.start()]) for m in re.finditer(r"\bwhen\b", t, re.I)):
        return "ask_time"
    if not spans:
        return "none"
    if now is not None and not recall and not any(x.lo < now.toordinal() for x in spans):
        return "action_time"
    q = bool(_QUESTION.search(t))
    if q and _EVENT_N.search(t) and not (_STATE_N.search(t) and not re.search(r"order|purchase|reserv|book|appointment|event|trip|订单|外卖|预约|订|活动|聚会|行程", t, re.I)):
        return "by_time"
    if q and _STATE_N.search(t):
        return "state_at"
    if q:
        return "time_other"
    if _ACTION.search(t):
        return "action_time"
    return "time_other"


def request_time(text, now=None):
    now = now or parse_now(text) or None
    body = text
    m = re.search(r"Current request:\s*(.*?)(?:\nEarlier conversation|$)", text, re.S) or re.search(r"当前请求：\s*(.*?)(?:\n此前对话|$)", text, re.S)
    if m:
        body = m.group(1)
    recall = bool(_PAST_Q.search(body) or _ASK.search(body) or _LATEST.search(body) or _ASK2.search(body) or _LATEST2.search(body)
                  or re.search(r"那次|那天|上次|\bthat time\b|回忆|记得|来着|\bremind me (?:what|which|where|who|how|of)\b", body, re.I))
    planning = bool(_ACTION.search(body)) and not recall
    spans = find_times(body, now, past_bias=not planning) if now else find_times(body, None)
    intent = classify_intent(body, spans, now, recall)
    return {"now": now, "intervals": [(s.lo, s.hi, s.prec) for s in spans], "intent": intent, "spans": spans}


_LEAD_DATE = re.compile(r"^\s*\[([^\]]{4,40})\]\s*")
_REL_HINT = re.compile(r"yesterday|last (?:week|month|year|night|weekend)|\bago\b|this (?:morning|week|weekend)|tomorrow|next week|"
                       r"昨天|前天|上周|上个月|去年|今天|明天|下周|前几天", re.I)


def unit_time(f, now=None):
    val = str(f.get("value", ""))
    obs_d = _parse_date_str(f.get("observed_at"))
    lead = _LEAD_DATE.match(val)
    if lead:
        ls = find_times(lead.group(1), now or obs_d, past_bias=False)
        if ls and ls[0].prec == "day":
            obs_d = dt.date.fromordinal(ls[0].lo) if obs_d is None else obs_d
            rec_lead = dt.date.fromordinal(ls[0].lo)
        else:
            rec_lead = None
        body = val[lead.end():]
    else:
        rec_lead, body = None, val
    obs = _day_iv(obs_d) if obs_d else None
    vf, vt = _parse_date_str(f.get("valid_from")), _parse_date_str(f.get("valid_to"))
    event = None
    anchor = rec_lead or obs_d or now
    spans = find_times(body, now, anchor=anchor, past_bias=False, hint=obs_d) if (anchor or now) else find_times(body, None)
    if rec_lead is None and not str(f.get("module", "")).endswith(_SCHED):
        spans = [s for s in spans if not (s.kind == "rel_wd" and
                                          not re.match(r"(?:上|这|本|下|last|next|this)", body[s.start:s.end], re.I))]
    if event is None and spans:
        absolute = [s for s in spans if not s.kind.startswith("rel")]
        rel = [s for s in spans if s.kind.startswith("rel") and s.kind != "rel_recent"]
        s = (absolute or rel or [None])[0]
        if s is not None:
            event = (s.lo, s.hi, s.prec)
    if event is None and rec_lead is not None and not _REL_HINT.search(body):
        event = (rec_lead.toordinal(), rec_lead.toordinal(), "day")
    if event is None and (vf or vt):
        lo = (vf or dt.date(1900, 1, 1)).toordinal(); hi = (vt or now or dt.date(2100, 1, 1)).toordinal()
        event = (lo, hi, "period" if vf != vt else "day")
    return {"event": event, "obs": obs}


def overlap(a, b, slack=0):
    if a is None or b is None:
        return 0.0
    gap = max(a[0] - b[1], b[0] - a[1]) - slack
    return 1.0 if gap <= 0 else 1.0 / (1.0 + gap / 7.0)


_EVK = ("event", "appointment", "reservation", "order", "trip")
_SCHED = (".schedule", ".delivery", ".calendar", ".itinerary")
_IV_CACHE = {}


def _iso_day(s):
    try:
        return dt.date.fromisoformat(str(s)[:10]).toordinal()
    except ValueError:
        return None


def _validity(world, now):
    slots, out = {}, {}
    for f in world["facts"]:
        if f.get("observed_at"):
            slots.setdefault((f["subject"], f["module"]), []).append(f)
    nowo = now.toordinal() if now else None
    for fs in slots.values():
        if len(fs) < 2 or not any(f["state"] == "superseded" for f in fs):
            continue
        fs = sorted(fs, key=lambda f: str(f["observed_at"]))
        for a, b in zip(fs, fs[1:]):
            if a["state"] == "superseded":
                lo, hi = _iso_day(a.get("valid_from") or a["observed_at"]), _iso_day(a.get("valid_to") or b["observed_at"])
                if lo is not None and hi is not None and lo <= hi:
                    out[a["id"]] = (lo, hi)
        last = fs[-1]
        lo = _iso_day(last.get("valid_from") or last["observed_at"])
        if last["state"] != "superseded" and lo is not None and nowo is not None and lo <= nowo:
            out[last["id"]] = (lo, nowo)
    return out


def _chapters(world, now):
    out = {}
    for x in world.get("timeline") or []:
        try:
            y, m = map(int, str(x["start"]).split("-")[:2])
            lo = dt.date(y, m, 1).toordinal()
            if x.get("end"):
                y, m = map(int, str(x["end"]).split("-")[:2])
                hi = (dt.date(y + (m == 12), m % 12 + 1, 1) - dt.timedelta(days=1)).toordinal()
            elif now:
                hi = now.toordinal()
            else:
                continue
        except (ValueError, KeyError, TypeError):
            continue
        if lo <= hi and x.get("hub"):
            out[x["hub"]] = (lo, hi)
    return out


def _world_table(world):
    key = hashlib.sha1(json.dumps([world.get("world_id"), world.get("meta"), world["entities"], world.get("edges"), world["facts"],
                                   world.get("timeline")], sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()
    tab = _IV_CACHE.get(key)
    if tab is None:
        now = _parse_date_str((world.get("meta") or {}).get("now"))
        iv, obs, sched = {}, {}, {}
        valid = _validity(world, now)
        kinds = {e["id"]: e["kind"] for e in world["entities"]}
        for f in world["facts"]:
            ut = unit_time(f, now)
            u = ut["event"]
            if ut["obs"]:
                obs[f["id"]] = ut["obs"][0]
            e = (u[0], u[1]) if u else None
            if e:
                iv[f["id"]] = e
                if f["module"].endswith(_SCHED) and f.get("state") != "superseded":
                    sched.setdefault(f["subject"], []).append(e)
            elif f["id"] in valid:
                iv[f["id"]] = valid[f["id"]]
        for eid, spans in sched.items():
            if kinds.get(eid) in _EVK:
                lo, hi = min(s[0] for s in spans), max(s[1] for s in spans)
                if hi - lo <= 60:
                    iv[eid] = (lo, hi)
        for eid, ch in _chapters(world, now).items():
            iv.setdefault(eid, ch)
        grp = {e["id"]: "k:" + str(e["kind"]) for e in world["entities"]}
        for f in world["facts"]:
            sk = kinds.get(f["subject"])
            grp[f["id"]] = ("m:" + f["module"]) if sk in _EVK else ("s:" + str(f["subject"]) + "|" + f["module"])
        tab = (iv, obs, grp)
        if len(_IV_CACHE) > 256:
            _IV_CACHE.clear()
        _IV_CACHE[key] = tab
    return tab


def unit_intervals(world, unit_ids):
    iv = _world_table(world)[0]
    return [list(iv.get(u, (-1, -2))) for u in unit_ids]


def unit_time_table(world, unit_ids):
    iv, obs, grp = _world_table(world)
    gid = {}
    return [[*iv.get(u, (-1, -2)), obs.get(u, -1), gid.setdefault(grp.get(u, "?"), len(gid))] for u in unit_ids]


def iv_text(lo, hi):
    a, b = dt.date.fromordinal(int(lo)), dt.date.fromordinal(int(hi))
    if a == b:
        return a.isoformat()
    m_start, m_end = a.day == 1, (b + dt.timedelta(days=1)).day == 1
    if m_start and m_end:
        if a.month == 1 and b.month == 12:
            return str(a.year) if a.year == b.year else f"{a.year}..{b.year}"
        return f"{a.year}-{a.month:02d}" if (a.year, a.month) == (b.year, b.month) else f"{a.year}-{a.month:02d}..{b.year}-{b.month:02d}"
    return f"{a.isoformat()}..{b.isoformat()}"


TIME_FEATS = ["overlap", "inside", "cover", "recency", "recency_log", "dated", "obs_overlap", "obs_recency", "rank_recent", "rank_next"]


def _group_rank(grp, val, mask):
    import torch
    out = torch.zeros_like(val)
    idx = mask.nonzero().flatten()
    if idx.numel() == 0:
        return out
    v = val[idx].double()
    v = (v - v.min()).long()
    g = grp[idx].long() << 22
    k = g + v
    ks = torch.sort(k).values
    r = torch.searchsorted(ks, k) - torch.searchsorted(ks, g)
    out[idx] = 1.0 / (1.0 + r.to(val.dtype))
    return out


def time_request_feats(texts, t_lo, t_hi, t_ob=None, t_grp=None):
    import torch
    B, n, dev = len(texts), t_lo.shape[0], t_lo.device
    tf = torch.zeros(B, n, len(TIME_FEATS), device=dev)
    ti = torch.zeros(B, dtype=torch.long, device=dev); es = torch.zeros(B, n, device=dev)
    has = (t_hi >= t_lo).float()
    tf[:, :, 5] = has
    span = (t_hi - t_lo + 1).clamp(min=1)
    hob = (t_ob >= 0).float() if t_ob is not None else None
    for b, t in enumerate(texts):
        r = request_time(t)
        ti[b] = INTENTS.index(r["intent"])
        if r["intervals"]:
            best = torch.zeros(n, device=dev); ins = torch.zeros(n, device=dev); cov = torch.zeros(n, device=dev)
            ob = torch.zeros(n, device=dev)
            for lo, hi, _ in r["intervals"]:
                gap = torch.maximum(t_lo - hi, lo - t_hi)
                best = torch.maximum(best, torch.where(gap <= 0, torch.ones_like(gap), 1.0 / (1.0 + gap / 7.0)))
                inter = (torch.minimum(t_hi, torch.full_like(t_hi, hi)) - torch.maximum(t_lo, torch.full_like(t_lo, lo)) + 1).clamp(min=0)
                ins = torch.maximum(ins, inter / span)
                cov = torch.maximum(cov, inter / float(hi - lo + 1))
                if hob is not None:
                    go = torch.maximum(t_ob - hi, lo - t_ob)
                    ob = torch.maximum(ob, torch.where(go <= 0, torch.ones_like(go), 1.0 / (1.0 + go / 7.0)))
            tf[b, :, 0] = best * has
            tf[b, :, 1] = ins * has
            tf[b, :, 2] = cov * has
            if hob is not None:
                tf[b, :, 6] = ob * hob
            if r["intent"] == "by_time":
                es[b] = tf[b, :, 1] if hob is None else torch.where(has > 0, tf[b, :, 1], (tf[b, :, 6] >= 1).float() * hob)
            elif r["intent"] == "state_at":
                es[b] = tf[b, :, 2]
            elif r["intent"] == "time_other":
                es[b] = 0.5 * (tf[b, :, 1] + tf[b, :, 2])
        if r["now"] is not None:
            nowo = float(r["now"].toordinal())
            past = has * (t_hi <= nowo).float()
            age = (nowo - t_hi).clamp(min=0)
            tf[b, :, 3] = torch.exp(-age / 180.0) * past
            tf[b, :, 4] = (1.0 - torch.log1p(age) / math.log1p(3650.0)).clamp(min=0.0, max=1.0) * past
            if hob is not None:
                tf[b, :, 7] = torch.exp(-(nowo - t_ob).clamp(min=0) / 180.0) * hob * (t_ob <= nowo).float()
            if t_grp is not None:
                tf[b, :, 8] = _group_rank(t_grp, -t_hi, past > 0)
                tf[b, :, 9] = _group_rank(t_grp, t_lo, (has > 0) & (t_lo > nowo))
    return tf, ti, es
