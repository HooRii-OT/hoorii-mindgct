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
"""Input of a build: a case directory with case.json and sources.jsonl (one source per line; docs/BUILD_INPUTS.md).

case.json  {"world_id", "lang": "en" | "zh", "owner": "<owner's name>", "now": "YYYY-MM-DD[ HH:MM]", "timezone", "description"}

Raw material (extracted from what people said or wrote):
  chat              {"id", "type": "chat", "date", "time"?, "channel"?, "participants": [name | {"name", "kind": "person" |
                    "assistant"}], "turns": [{"speaker", "text"}]}, or with "messages": [{"role": "user" | "assistant",
                    "content"}] (user = the owner) and "assistant"?: the assistant's name
  note              {"id", "type": "note", "date", "title"?, "author"?, "text"}  paragraphs separated by blank lines
Memories another AI has already summarised (imported as second-hand facts):
  memory_list       {"id", "type": "memory_list", "date", "assistant"?, "confirmed"?, "items": [text | {"text", "date"?}]}
  memory_summaries  {"id", "type": "memory_summaries", "assistant"?, "confirmed"?, "summaries": [{"date", "text", "title"?}]}
  memory_profile    {"id", "type": "memory_profile", "date", "assistant"?, "confirmed"?, "format": "markdown" | "json",
                    "text" (markdown) | "data" (JSON object)}

A chat or a note becomes one session (a note is a session written by its author, the owner unless "author" names someone
else, with one turn per paragraph). A memory source becomes a list of dated memory items.
"""
import json
import re
from pathlib import Path

from .timeres import parse_date

_ID = re.compile(r"[A-Za-z0-9_.:-]{1,64}")


class CaseError(ValueError):
    pass


def _date(sid, v, what="date"):
    if parse_date(v) is None or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(v)):
        raise CaseError(f"source {sid}: {what} must be 'YYYY-MM-DD'")
    return str(v)


def _profile_items(fmt, s, sid):
    """Memory-profile document -> item texts ("Section > Subsection > entry")."""
    out = []
    if fmt == "markdown":
        path = []
        for raw in str(s.get("text") or "").splitlines():
            line = raw.strip()
            if not line:
                continue
            m = re.match(r"^(#{1,6})\s+(.*)$", line)
            if m:
                path = path[:len(m.group(1)) - 1] + [m.group(2).strip()]
                continue
            line = re.sub(r"^(?:[-*+]\s+|\d+[.)]\s+)", "", line).strip()
            if line:
                out.append(" > ".join(path + [line]))
    elif fmt == "json":
        def walk(x, path):
            if isinstance(x, dict):
                if x and all(not isinstance(v, (dict, list)) for v in x.values()):
                    out.append(" > ".join(path) + (": " if path else "") + "; ".join(f"{k}: {v}" for k, v in x.items()))
                else:
                    for k, v in x.items():
                        walk(v, path + [str(k)])
            elif isinstance(x, list):
                for v in x:
                    walk(v, path)
            elif x is not None and str(x).strip():
                out.append(" > ".join(path[:-1] + [f"{path[-1]}: {x}"]) if path else str(x))
        if not isinstance(s.get("data"), (dict, list)):
            raise CaseError(f"source {sid}: memory_profile with format json needs a JSON object in 'data'")
        walk(s["data"], [])
    else:
        raise CaseError(f"source {sid}: memory_profile format must be 'markdown' or 'json'")
    return out


def load_case(case_dir, sources=None):
    """-> (case, sessions, memories).
    session: {"id", "type": "chat" | "note", "date", "time", "participants": {name: "person" | "assistant"}, "turns": [{"n",
    "speaker", "text"}], "title"}; turn numbers n are 1-based positions in the source (paragraphs for notes).
    memory source: {"id", "type", "assistant", "confirmed", "items": [{"n", "text", "date"}]}; n is the 1-based item number.
    sources: another sources file to read instead of <case_dir>/sources.jsonl (a path, or relative to case_dir)."""
    d = Path(case_dir)
    case = json.loads((d / "case.json").read_text(encoding="utf-8"))
    for k in ("world_id", "lang", "owner", "now"):
        if not case.get(k):
            raise CaseError(f"case.json: '{k}' is required")
    if case["lang"] not in ("en", "zh"):
        raise CaseError("case.json: lang must be 'en' or 'zh'")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", str(case["world_id"])):
        raise CaseError("case.json: world_id must be 1-64 characters of [A-Za-z0-9_.-]")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}( \d{2}:\d{2})?", str(case["now"])) or parse_date(case["now"]) is None:
        raise CaseError("case.json: now must be 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM'")
    owner = str(case["owner"])
    sessions, memories, seen = [], [], set()
    src = Path(sources) if sources else d / "sources.jsonl"
    if sources and not src.is_absolute() and not src.exists() and (d / src).exists():
        src = d / src
    lines = src.read_text(encoding="utf-8").splitlines()
    for ln, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            s = json.loads(line)
        except ValueError as e:
            raise CaseError(f"sources.jsonl line {ln}: not JSON ({e})") from None
        sid = str(s.get("id", ""))
        if not _ID.fullmatch(sid) or sid in seen:
            raise CaseError(f"sources.jsonl line {ln}: id missing, malformed or repeated")
        seen.add(sid)
        typ = s.get("type")
        time_ = str(s.get("time") or "")
        if time_ and not re.fullmatch(r"\d{2}:\d{2}", time_):
            raise CaseError(f"source {sid}: time must be 'HH:MM'")
        if typ == "chat":
            date = _date(sid, s.get("date"))
            parts, turns = {}, []
            if s.get("messages") is not None:
                asst = str(s.get("assistant") or "Assistant")
                parts = {owner: "person", asst: "assistant"}
                for n, m in enumerate(s["messages"], 1):
                    role, text = m.get("role"), m.get("content")
                    if role not in ("user", "assistant") or not isinstance(text, str) or not text.strip():
                        continue
                    turns.append({"n": n, "speaker": owner if role == "user" else asst, "text": text.strip()})
            else:
                for p in s.get("participants") or []:
                    name, kind = (p, "person") if isinstance(p, str) else (p.get("name"), p.get("kind", "person"))
                    if not name or kind not in ("person", "assistant"):
                        raise CaseError(f"source {sid}: participants must be names or {{'name', 'kind': 'person' | 'assistant'}}")
                    parts[str(name)] = kind
                for n, t in enumerate(s.get("turns") or [], 1):
                    if t.get("speaker") not in parts:
                        raise CaseError(f"source {sid}: turn {n} speaker {t.get('speaker')!r} is not a participant")
                    if str(t.get("text") or "").strip():
                        turns.append({"n": n, "speaker": t["speaker"], "text": str(t["text"]).strip()})
            if not turns:
                raise CaseError(f"source {sid}: the chat has no text turn")
            sessions.append({"id": sid, "type": "chat", "date": date, "time": time_, "participants": parts, "turns": turns,
                             "title": s.get("channel") or ""})
        elif typ == "note":
            date = _date(sid, s.get("date"))
            author = str(s.get("author") or owner)
            paras = [p.strip() for p in re.split(r"\n\s*\n", str(s.get("text") or "")) if p.strip()]
            if not paras:
                raise CaseError(f"source {sid}: note text is empty")
            title = str(s.get("title") or "").strip()
            lab = (f"[笔记：{title}] " if case["lang"] == "zh" else f"[note: {title}] ") if title else ""
            turns = [{"n": n, "speaker": author, "text": (lab if n == 1 else "") + p} for n, p in enumerate(paras, 1)]
            sessions.append({"id": sid, "type": "note", "date": date, "time": time_, "participants": {author: "person"},
                             "turns": turns, "title": title})
        elif typ in ("memory_list", "memory_summaries", "memory_profile"):
            items = []
            if typ == "memory_list":
                date = _date(sid, s.get("date"))
                for x in s.get("items") or []:
                    text, idate = (x, None) if isinstance(x, str) else (x.get("text"), x.get("date"))
                    if str(text or "").strip():
                        items.append({"text": str(text).strip(), "date": _date(sid, idate, "item date") if idate else date})
            elif typ == "memory_summaries":
                for x in s.get("summaries") or []:
                    if str(x.get("text") or "").strip():
                        title = str(x.get("title") or "").strip()
                        items.append({"text": (f"{title}: " if title else "") + re.sub(r"\s+", " ", str(x["text"]).strip()),
                                      "date": _date(sid, x.get("date"), "summary date")})
            else:
                date = _date(sid, s.get("date"))
                items = [{"text": t, "date": date} for t in _profile_items(s.get("format"), s, sid)]
            if not items:
                raise CaseError(f"source {sid}: no memory item")
            memories.append({"id": sid, "type": typ, "assistant": str(s.get("assistant") or "AI assistant"),
                             "confirmed": s.get("confirmed") is True,
                             "items": [dict(x, n=n) for n, x in enumerate(items, 1)]})
        else:
            raise CaseError(f"source {sid}: type must be chat, note, memory_list, memory_summaries or memory_profile")
    if not sessions and not memories:
        raise CaseError("sources.jsonl holds no source")
    sessions.sort(key=lambda s: (s["date"], s["time"] or "00:00"))
    return case, sessions, memories


def turn_line(t, n, cut=None):
    s = t["text"]
    if cut and len(s) > cut:
        s = s[:cut].rstrip() + " [...]"
    return f"T{n} {t['speaker']}: {s}"


def render(session, cut=None):
    """Rendered session text and the map T-number -> source turn number."""
    lines, nums = [], {}
    for i, t in enumerate(session["turns"], 1):
        nums[i] = t["n"]
        lines.append(turn_line(t, i, cut))
    return "\n".join(lines), nums


def split_long(sessions, max_chars):
    """A session longer than max_chars is cut at turn boundaries into parts of at most max_chars (each part keeps the
    session's id, date and participants, and the original turn numbers)."""
    out = []
    for s in sessions:
        text, _ = render(s)
        if len(text) <= max_chars or len(s["turns"]) < 2:
            out.append(dict(s, part=0))
            continue
        cur, size, k = [], 0, 0
        for t in s["turns"]:
            ln = len(turn_line(t, 99)) + 1
            if cur and size + ln > max_chars:
                out.append(dict(s, turns=cur, part=k))
                cur, size, k = [], 0, k + 1
            cur.append(t)
            size += ln
        out.append(dict(s, turns=cur, part=k))
    return out


def batches(sessions, max_chars=9000, max_sessions=3):
    """Consecutive sessions grouped into extraction calls of at most max_chars rendered characters and max_sessions sessions."""
    out, cur, size = [], [], 0
    for s in sessions:
        n = len(render(s)[0])
        if cur and (size + n > max_chars or len(cur) >= max_sessions):
            out.append(cur)
            cur, size = [], 0
        cur.append(s)
        size += n
    if cur:
        out.append(cur)
    return out
