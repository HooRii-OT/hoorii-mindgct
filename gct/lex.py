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
"""Lexical signatures of units and requests (hashed word and date tokens)."""
import math
import re
import zlib

import torch

LEX_V = 1 << 18
LEX_W = 16
_STOP = set("""the a an of to in on at for and or is are was were be been by with from as it its this that these those my me i you
your he she his her they them their we our who whom what which when where how why do does did have has had will would can could
should not no yes about into over under up down out so than then there here now caller current request user context previous
turn please any some all just also""".split())
_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+")


_HDR = re.compile(r"\[Now:[^\]]*\]")


REQ_CLEAN = False
_HDR_ZH = re.compile(r"［当前时间：[^］]*］")
_LABELS = re.compile(r"用户：|助理：|当前请求：|此前对话（由近到远）：|User: |Assistant: |Current request: |Earlier conversation \(most recent first\):")

DAYS = False

DATES = False

_MON = {m: i + 1 for i, m in enumerate("january february march april may june july august september october november december".split())}
_MON.update({m[:3]: i for m, i in list(_MON.items())}); _MON["sept"] = 9
_ZH_YM = re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月")
_ZH_Y = re.compile(r"(\d{4})\s*年")
_NUM = re.compile(r"\b(\d{1,4})[./-](\d{1,2})[./-](\d{1,4})\b")
_MY = re.compile(r"\b([a-z]{3,9})\.?\s+(?:\d{1,2}(?:st|nd|rd|th)?,?\s+)?(\d{4})\b")


def _date_tokens(text):
    t = text.lower(); out = []
    yr = lambda y: 1900 <= y <= 2099
    for y in re.findall(r"\b(19\d\d|20\d\d)\b", t):
        out.append(f"yr:{y}")
    for m, y in _MY.findall(t):
        if m in _MON and yr(int(y)):
            out.append(f"ym:{y}-{_MON[m]:02d}")
    for a, b, c in _NUM.findall(t):
        a, b, c = int(a), int(b), int(c)
        if yr(a):
            if 1 <= b <= 12: out.append(f"ym:{a}-{b:02d}")
        elif yr(c):
            for m in (a, b):
                if 1 <= m <= 12: out.append(f"ym:{c}-{m:02d}")
    for y, m in _ZH_YM.findall(text):
        if 1 <= int(m) <= 12: out.append(f"ym:{y}-{int(m):02d}")
    for y in _ZH_Y.findall(text):
        out.append(f"yr:{y}")
    return out


def _day_tokens(text, request):
    import datetime as _dt
    from . import timex
    now = timex.parse_now(text) if request else None
    out = []
    for sp in timex.find_times(text, now):
        if sp.hi - sp.lo <= 31:
            out += [f"ymd:{_dt.date.fromordinal(o).isoformat()}" for o in range(sp.lo, sp.hi + 1)]
    return out


def lex_tokens(text, request=False):
    raw_text = text
    if request:
        text = _HDR.sub(" ", text)
        if REQ_CLEAN:
            text = _LABELS.sub(" ", _HDR_ZH.sub(" ", text))
    toks = []
    for w in _RE.findall(text.lower()):
        if w[0] >= "\u4e00":
            toks += [w] if len(w) == 1 else [w[i:i + 2] for i in range(len(w) - 1)]
        elif len(w) >= 2 and w not in _STOP and not (w.isdigit() and len(w) < 5):
            toks.append(w)
    if DATES:
        toks = _date_tokens(text) + toks
    if DAYS:
        toks = _day_tokens(raw_text, request) + toks
    out, seen = [], set()
    for t in toks:
        if t not in seen:
            seen.add(t); out.append(t)
    return out


def lex_hash(tok):
    if LEX_V != (1 << 18):
        return zlib.crc32(("#" + tok).encode("utf-8")) % LEX_V
    return zlib.crc32(tok.encode("utf-8")) % LEX_V


def unit_lex_tensors(unit_text, device):
    toks = [lex_tokens(t)[:LEX_W] if LEX_W > 0 else lex_tokens(t) for t in unit_text]
    W = LEX_W if LEX_W > 0 else max([len(x) for x in toks] or [1])
    df = {}
    for ts in toks:
        for t in ts:
            df[t] = df.get(t, 0) + 1
    n = max(len(unit_text), 1)
    ids = torch.full((len(unit_text), W), -1, dtype=torch.long)
    wts = torch.zeros(len(unit_text), W)
    dmk = torch.zeros(len(unit_text), W)
    for u, ts in enumerate(toks):
        for j, t in enumerate(ts):
            ids[u, j] = lex_hash(t); wts[u, j] = math.log(1.0 + n / df[t]); dmk[u, j] = float(t.startswith("ym:"))
    unit_lex_tensors.last_date_mask = dmk.to(device)
    return ids.to(device), wts.to(device)


def request_lex(texts, uw, uwt):
    bits = torch.zeros(len(texts), LEX_V, device=uw.device)
    for b, t in enumerate(texts):
        ids = [lex_hash(x) for x in lex_tokens(t, request=True)]
        if ids:
            bits[b, torch.tensor(ids, device=uw.device)] = 1.0
    hit = bits[:, uw.clamp_min(0)] * uwt[None]


    return (hit.sum(-1) / math.log(1.0 + max(uw.shape[0], 1))).clamp(max=1.0)


def request_lex_feats(texts, uw, uwt, uwd):
    bits = torch.zeros(len(texts), LEX_V, device=uw.device)
    for b, t in enumerate(texts):
        ids = [lex_hash(x) for x in lex_tokens(t, request=True)]
        if ids:
            bits[b, torch.tensor(ids, device=uw.device)] = 1.0
    hb = bits[:, uw.clamp_min(0)] * (uw >= 0).float()[None]
    raw = (hb * uwt[None]).sum(-1) / math.log(1.0 + max(uw.shape[0], 1))
    date = (hb * uwd[None]).sum(-1)
    return raw, date
