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
"""Time anchoring. The extractor never sees a calendar date: it describes when a statement holds relative to the start S of
its session ("S-1d", "S+1w", "S-Tue", ...) or as a date it was told. This module turns those codes into dates with calendar
arithmetic, so the same extraction can be re-anchored and cached, and the date maths is done by a program, not by the model.

A resolved time is (start, end, precision): start / end are datetime.date, precision is "day", "month" or "year".
"""
import calendar
import datetime as dt
import re

WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
MONTHS_EN = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November",
             "December")

_REL = re.compile(r"^S\s*(?:([+-])\s*(\d{1,3})\s*([dwmy]))?$", re.I)
_WD = re.compile(r"^S\s*(?:([+-])\s*(\d{1,2})\s*w\s*:\s*|([+-])\s*)(mon|tue|wed|thu|fri|sat|sun)[a-z]*$", re.I)
_ABS = re.compile(r"^(\d{4})(?:-(\d{1,2}))?(?:-(\d{1,2}))?$")
_MD = re.compile(r"^(?:Y\s*([+-]\s*\d{1,2})?\s*:\s*)?(\d{1,2})(?:-(\d{1,2}))?$", re.I)


def parse_date(s):
    """'YYYY-MM-DD' (optionally followed by a time) -> date, else None."""
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", str(s or ""))
    if not m:
        return None
    try:
        return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def _add_months(year, month, n):
    k = year * 12 + (month - 1) + n
    return k // 12, k % 12 + 1


def _month_span(year, month):
    return dt.date(year, month, 1), dt.date(year, month, calendar.monthrange(year, month)[1]), "month"


def _year_span(year):
    return dt.date(year, 1, 1), dt.date(year, 12, 31), "year"


def resolve_when(code, start, plan=False, weekday=None):
    """Extraction time code -> (start, end, precision) or None.

    S                    the session day
    S-3d, S+2d           N days before / after the session
    S-1w, S+2w           N weeks before / after (a day, 7N days away)
    S-1m, S+1m           the calendar month N months before / after the session's month
    S-1y, S+1y           the calendar year N years before / after
    S-Tue, S+Fri         the nearest Tuesday before / Friday after the session day
    S-1w:Tue, S+1w:Fri   Tuesday of the week before / Friday of the week after the session's week (weeks start on Monday)
    2023, 2023-05, 2023-05-04   a stated year, month or day
    05-10, Y-1:05-10, Y:09, Y+1:03   month-day or month with the year relative to the session's year; a bare month-day takes
                         the nearest occurrence, the later one on a tie; for a plan the coming one, unless this year's day
                         passed at most 60 days ago ("promised on February 19 to ..." said in March); weekday (0 = Monday),
                         when the text names one, picks the year in which the day falls on it
    """
    w = str(code or "").strip()
    if not w or start is None:
        return None
    m = _REL.match(w)
    if m:
        if not m.group(1):
            return start, start, "day"
        n = int(m.group(2)) * (1 if m.group(1) == "+" else -1)
        unit = m.group(3).lower()
        if unit == "d":
            d = start + dt.timedelta(days=n)
            return d, d, "day"
        if unit == "w":
            d = start + dt.timedelta(days=7 * n)
            return d, d, "day"
        if unit == "m":
            return _month_span(*_add_months(start.year, start.month, n))
        return _year_span(start.year + n)
    m = _WD.match(w)
    if m:
        wd = WEEKDAYS.index(m.group(4).lower()[:3])
        if m.group(1):                                       # S-1w:Tue -> that weekday in the week N weeks away
            n = int(m.group(2)) * (1 if m.group(1) == "+" else -1)
            monday = start - dt.timedelta(days=start.weekday()) + dt.timedelta(days=7 * n)
            d = monday + dt.timedelta(days=wd)
        elif m.group(3) == "-":                             # S-Tue -> the nearest Tuesday strictly before the session day
            d = start - dt.timedelta(days=(start.weekday() - wd) % 7 or 7)
        else:                                                # S+Fri -> the nearest Friday strictly after the session day
            d = start + dt.timedelta(days=(wd - start.weekday()) % 7 or 7)
        return d, d, "day"
    m = _ABS.match(w)
    if m:
        y = int(m.group(1))
        try:
            if m.group(3):
                d = dt.date(y, int(m.group(2)), int(m.group(3)))
                return d, d, "day"
            if m.group(2):
                return _month_span(y, int(m.group(2)))
            return _year_span(y)
        except ValueError:
            return None
    m = _MD.match(w)
    if m and ("Y" in w.upper() or m.group(3)):
        off = int(m.group(1).replace(" ", "")) if m.group(1) else None
        mon = int(m.group(2))
        if not 1 <= mon <= 12:
            return None
        if m.group(3) is None:                               # Y-1:05 -> May of last year
            return _month_span(start.year + (off or 0), mon)
        day = int(m.group(3))
        try:
            if off is not None:
                d = dt.date(start.year + off, mon, day)
                return d, d, "day"
            cands = []
            for y in (start.year - 1, start.year, start.year + 1):
                try:
                    cands.append(dt.date(y, mon, day))
                except ValueError:
                    pass
            if not cands:
                return None
            near = [c for c in cands if abs((c - start).days) <= 183 and c.weekday() == weekday] if weekday is not None else []
            if near:                                         # the year in which the day falls on the weekday named with it
                cands = near
            if plan:                                         # a plan: the coming occurrence, or a day just past
                recent = [c for c in cands if 0 < (start - c).days <= 60]
                d = recent[-1] if recent else min((c for c in cands if c >= start), default=cands[-1])
            else:                                            # otherwise the nearest occurrence, the later one on a tie
                d = min(cands, key=lambda c: (abs((c - start).days), c < start))
            return d, d, "day"
        except ValueError:
            return None
    return None


def date_text(start, end=None, precision="day", lang="en"):
    """Human-readable date of a resolved time: '7 May 2023' / 'May 2023' / '2023', or 2023年5月7日 / 2023年5月 / 2023年."""
    if start is None:
        return ""
    if precision == "year":
        return f"{start.year}年" if lang == "zh" else str(start.year)
    if precision == "month":
        return f"{start.year}年{start.month}月" if lang == "zh" else f"{MONTHS_EN[start.month - 1]} {start.year}"
    return f"{start.year}年{start.month}月{start.day}日" if lang == "zh" else f"{start.day} {MONTHS_EN[start.month - 1]} {start.year}"


_MON = {m.lower()[:3]: i + 1 for i, m in enumerate(MONTHS_EN)}
# a month written as a whole word or a standard abbreviation, never the start of another word ("marathon", "decades")
MONTH = r"(january|february|march|april|may|june|july|august|september|october|november|december|sept|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\.?(?![a-z])"
# full weekday names in any case; abbreviations only capitalised ("Sat", not the verb "sat")
_WD_EN = re.compile(r"\b(?i:(monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b|\b(Mon|Tue|Tues|Wed|Thu|Thur|Thurs|Fri|Sat|Sun)\b")
_WD_ZH = re.compile(r"(?:周|星期|礼拜)([一二三四五六日天])")
_ZH_WD = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}
_MOVE = re.compile(r"\b(?:moved|rescheduled|postponed|pushed (?:back )?|changed|switched|brought forward)\b[^.;]*?\bto\b|改签到|改签至|改到|改至|改成|改为|"
                   r"推迟到|延期到|提前到|换到|换成", re.I)


def resolve_free(text, start, plan=False):
    """Fallback for a time the extractor wrote as English words instead of a code: 'March 20', '20 March 2023', 'March 2023',
    'yesterday', 'last Tuesday', 'last week', 'next month' ... relative to the session day."""
    w = str(text or "").strip().lower()
    if not w or start is None:
        return None
    m = re.search(r"\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b", w)
    if m:
        return resolve_when(f"{m.group(1)}-{m.group(2)}-{m.group(3)}", start)
    m = re.search(rf"\b{MONTH}\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?:,?\s*(\d{{4}}))?", w)
    if m:
        mon, day, y = _MON[m.group(1)[:3]], int(m.group(2)), m.group(3)
    else:
        m = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?{MONTH}(?:,?\s*(\d{{4}}))?", w)
        mon, day, y = (_MON[m.group(2)[:3]], int(m.group(1)), m.group(3)) if m else (None, None, None)
    if mon:
        return resolve_when(f"{y}-{mon:02d}-{day:02d}" if y else f"{mon:02d}-{day:02d}", start, plan)
    m = re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b(?:\s+(\d{4}))?", w)
    if m:
        mon = _MON[m.group(1)[:3]]
        if m.group(2):
            yr = int(m.group(2))
        elif plan:
            yr = start.year if mon >= start.month else start.year + 1
        else:
            yr = start.year if mon <= start.month else start.year - 1
        return resolve_when(f"{yr}-{mon:02d}", start)
    for pat, code in ((r"\byesterday\b", "S-1d"), (r"\btoday\b|\btonight\b|\bthis morning\b", "S"), (r"\btomorrow\b", "S+1d"),
                      (r"\blast week(end)?\b", "S-1w"), (r"\blast month\b", "S-1m"), (r"\blast year\b", "S-1y"),
                      (r"\bnext week\b", "S+1w"), (r"\bnext month\b", "S+1m")):
        if re.search(pat, w):
            return resolve_when(code, start)
    m = re.search(r"\blast (mon|tue|wed|thu|fri|sat|sun)[a-z]*\b", w)
    if m:
        return resolve_when(f"S-{m.group(1)}", start)
    return None


_ZH_MD = re.compile(r"(?:(\d{4})年)?(\d{1,2})月(\d{1,2})[日号]")
_ZH_RMD = re.compile(r"(上|下|这|本)个?月(\d{1,2})[日号]")          # 上个月25号, 下个月3号
_ADDR = "路街道巷弄村区栋楼室号"
_ANNUAL = re.compile(r"birthday|anniversary|\bturns? \d{1,3}\b|生日|纪念日|满\d{1,3}岁", re.I)
_ABBR = {"sept", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec"}
# after an abbreviated month and a day ("Jan 5"), words that may follow a date; any other word ("Jan 5 times") means it is not one
_AFTER_DATE = re.compile(r"\s+(?:at|to|and|or|until|till|through|thru|in|on|for|from|by|before|after|is|was|will|with|when|"
                         r"morning|afternoon|evening|night|am|pm)\b|\s*(?:[^\sA-Za-z]|$)", re.I)
# a day number: not part of a date already read, an address (文三路12号, 3号楼) or an ordinal (第3号)
_ZH_D = re.compile(r"(?<![月\d年第路街道巷弄村区栋楼室])(\d{1,2})[日号](?![\d线楼门床房口院厅柜座位桌窗台机码箱包车道仓店室])")
_EN_MD = re.compile(rf"\b(?:{MONTH}\s+(\d{{1,2}})(?:st|nd|rd|th)?|(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?{MONTH})\b(?:,?\s*(\d{{4}}))?", re.I)
_RANGE_AFTER = re.compile(r"\s*(?:-|–|—|to|through|thru|until)\s*(\d{1,2})(?:st|nd|rd|th)?\b(?!\s*(?:[:.]\d|am|pm|a\.m|p\.m))", re.I)
_RANGE_BEFORE = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s*(?:-|–|—|to|through|thru|until)\s*$", re.I)
_EN_D = re.compile(r"\bthe\s+(\d{1,2})(?:st|nd|rd|th)\b(?!\s+(?:floor|grade|time|place|row|year|anniversary|birthday|round|week|century))",
                  re.I)


def _weekday(text):
    """The one weekday a text names (0 = Monday), else None."""
    days = {_WD_ZH_NUM for _WD_ZH_NUM in (_ZH_WD[m.group(1)] for m in _WD_ZH.finditer(text))}
    for m in _WD_EN.finditer(text):
        days.add(WEEKDAYS.index((m.group(1) or m.group(2)).lower()[:3]))
    return days.pop() if len(days) == 1 else None


def _day_of_month(start, day, plan, weekday=None, annual=False):
    """A day number without a month: this month's, the previous or the next month's. A weekday named in the text picks among
    them; a plan or a yearly date (a birthday) takes the coming one; otherwise this month's if it is not later than the
    session day, else the nearest, the later one on a tie."""
    cands = []
    for k in (-1, 0, 1):
        y, m = _add_months(start.year, start.month, k)
        try:
            cands.append(dt.date(y, m, day))
        except ValueError:
            pass
    if weekday is not None and any(c.weekday() == weekday for c in cands):
        cands = [c for c in cands if c.weekday() == weekday]
    if not cands:
        return None
    if plan or annual:
        return min((c for c in cands if c >= start), default=cands[-1])
    this = [c for c in cands if c.month == start.month and c.year == start.year and c <= start]
    if this:
        return this[0]
    return min(cands, key=lambda c: (abs((c - start).days), c < start))


def text_dates(text, start, plan=False):
    """Calendar dates written in the fact text itself ("4月25日", "25号", "April 24", "Saturday the 28th"), resolved against the
    session day: -> (first, last, "day") or None. When the text says something was moved ("from the 24th to the 25th",
    "rescheduled to March 23", "改签到25号"), the date after the move is taken."""
    found = written_dates(text, start, plan, positions=True)
    if not found:
        return None
    mv = moved_to(text, found)
    if mv:
        return mv, mv, "day"
    out = [d for _, d in found]
    lo, hi = out[0], out[-1]
    if hi < lo or (hi - lo).days > 31:
        hi = lo
    return lo, hi, "day"


def moved_to(text, found):
    """The first written date after a word of moving ("moved to", "改签到"), else None. found: (position, date) pairs."""
    mv = _MOVE.search(str(text or ""))
    if mv:
        after = [d for pos, d in found if pos >= mv.end()]
        if after:
            return after[0]
    return None


def written_dates(text, start, plan=False, positions=False):
    """Every calendar date written in a text, in order of position (with positions: (position, date) pairs)."""
    if start is None:
        return []
    t, found = str(text or ""), []                       # (position, date)
    wd = _weekday(t)
    annual = bool(_ANNUAL.search(t))
    for m in _ZH_MD.finditer(t):
        r = resolve_when(f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}" if m.group(1) else
                         f"{int(m.group(2)):02d}-{int(m.group(3)):02d}", start, plan, wd)
        if r:
            found.append((m.start(), r[0]))
    for m in _ZH_RMD.finditer(t):
        y, mon = _add_months(start.year, start.month, {"上": -1, "下": 1}.get(m.group(1), 0))
        try:
            found.append((m.start(), dt.date(y, mon, int(m.group(2)))))
        except ValueError:
            pass
    rest = _ZH_RMD.sub(lambda m: " " * len(m.group(0)), _ZH_MD.sub(lambda m: " " * len(m.group(0)), t))
    for m in _ZH_D.finditer(rest):
        before = rest[max(0, m.start() - 4):m.start()]
        if "住" in before[-2:] or "门牌" in before or (before.rstrip() and before.rstrip()[-1] in _ADDR):
            continue                                     # 住在3号, 门牌18号, 文三路 12号: a house number
        d = _day_of_month(start, int(m.group(1)), plan, wd, annual) if 1 <= int(m.group(1)) <= 31 else None
        if d:
            found.append((m.start(), d))
    for m in _EN_MD.finditer(t):
        word = (m.group(1) or m.group(4)).lower()
        if m.group(1) and word in _ABBR and not m.group(5) and not _AFTER_DATE.match(t, m.end()):
            continue                                     # "Jan 5 times": a name and a number
        mon = _MON[word[:3]]
        day = int(m.group(2) or m.group(3))
        r = resolve_when(f"{m.group(5)}-{mon:02d}-{day:02d}" if m.group(5) else f"{mon:02d}-{day:02d}", start, plan, wd)
        if r:
            found.append((m.start(), r[0]))
            # a range in one month: "January 2-3", "March 27 to 29", "27-29 March"
            other = _RANGE_AFTER.match(t, m.end()) if m.group(2) else None
            before = None if m.group(2) else _RANGE_BEFORE.search(t[:m.start()])
            d2 = int((other or before).group(1)) if (other or before) else None
            if d2 and 1 <= d2 <= 31:
                try:
                    found.append((m.start() + (1 if other else -1), r[0].replace(day=d2)))
                except ValueError:
                    pass
    rest = _EN_MD.sub(lambda m: " " * len(m.group(0)), t)
    for m in _EN_D.finditer(rest):
        d = _day_of_month(start, int(m.group(1)), plan, wd, annual) if 1 <= int(m.group(1)) <= 31 else None
        if d:
            found.append((m.start(), d))
    found.sort()
    return found if positions else [d for _, d in found]
