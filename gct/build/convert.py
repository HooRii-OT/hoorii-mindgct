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
"""Turn two common on-disk shapes into chat records for sources.jsonl (docs/BUILD_INPUTS.md).

  python -m gct.build.convert transcript FILE --id ID --date YYYY-MM-DD --owner NAME [--assistant NAME] [--time HH:MM]
                                        [--speakers "NAME,NAME,..."]
      a plain-text transcript, one turn per line as "Name: text" (or "Name：text"), optionally after a time stamp such as
      "[19:40]" or "2026-03-01 19:40"; a line without a speaker continues the previous turn. With --speakers only those
      names (plus the owner and the assistant) start a turn, so a line such as "Note: ..." stays text; without it, every
      "Name:" prefix is taken as a speaker and the names found are listed on stderr for checking
  python -m gct.build.convert messages FILE --id ID --date YYYY-MM-DD [--assistant NAME] [--time HH:MM]
      a JSON array of {"role", "content"} messages, or {"messages": [...]}; user = the owner, assistant = the AI assistant

The record is printed as one JSON line; append it to sources.jsonl.
"""
import argparse
import datetime as _dt
import json
import re
import sys
from pathlib import Path

_LINE = re.compile(r"^\s*([^:：]{1,40}?)\s*[:：]\s*(.+)$")
_STAMP = re.compile(r"^\s*[\[(]?\s*(?:\d{4}-\d{2}-\d{2}[ T]?)?\s*\d{1,2}:\d{2}(?::\d{2})?\s*(?:[AaPp]\.?[Mm]\.?)?\s*[\])]?\s*[-–—]?\s*")
_ID = re.compile(r"[A-Za-z0-9_.:-]{1,64}")


def check_fields(sid, date, time=None):
    """The same checks gct.build applies to sources.jsonl."""
    if not _ID.fullmatch(str(sid)):
        raise ValueError("--id must be 1-64 characters of [A-Za-z0-9_.:-]")
    try:
        ok = re.fullmatch(r"\d{4}-\d{2}-\d{2}", date) and _dt.date.fromisoformat(date)
    except ValueError:
        ok = False
    if not ok:
        raise ValueError("--date must be a real date written YYYY-MM-DD")
    if time is not None and not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", time):
        raise ValueError("--time must be HH:MM")


def transcript(text, sid, date, owner, assistant=None, time=None, speakers=None):
    """speakers: the names allowed to start a turn (the owner and the assistant are always allowed); None = any name."""
    check_fields(sid, date, time)
    allowed = None if speakers is None else {x.strip() for x in [*speakers, owner, assistant or ""] if x and x.strip()}
    turns = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        line = _STAMP.sub("", raw, count=1) if _STAMP.match(raw) and _LINE.match(_STAMP.sub("", raw, count=1)) else raw
        m = _LINE.match(line)
        if m and allowed is not None and m.group(1).strip() not in allowed:
            m = None
        if m:
            turns.append({"speaker": m.group(1).strip(), "text": m.group(2).strip()})
        elif turns:
            turns[-1]["text"] += "\n" + raw.strip()
        else:
            raise ValueError(f"the first line has no 'Name: text' speaker: {raw[:60]!r}")
    if not turns:
        raise ValueError("no turn found")
    names = list(dict.fromkeys([owner] + [t["speaker"] for t in turns]))
    parts = [{"name": n, "kind": "assistant"} if n == assistant else n for n in names]
    rec = {"id": sid, "type": "chat", "date": date, "participants": parts, "turns": turns}
    if time:
        rec["time"] = time
    return rec


def messages(obj, sid, date, assistant=None, time=None):
    check_fields(sid, date, time)
    msgs = obj.get("messages") if isinstance(obj, dict) else obj
    if not isinstance(msgs, list):
        raise ValueError("expected a JSON array of messages or {'messages': [...]}")
    keep = [{"role": m["role"], "content": m["content"]} for m in msgs
            if isinstance(m, dict) and m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str) and m["content"].strip()]
    if not keep:
        raise ValueError("no user or assistant text message found")
    rec = {"id": sid, "type": "chat", "date": date, "assistant": assistant or "Assistant", "messages": keep}
    if time:
        rec["time"] = time
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m gct.build.convert", description="Convert a transcript or a messages JSON file "
                                 "into one chat record for sources.jsonl.")
    ap.add_argument("kind", choices=("transcript", "messages"))
    ap.add_argument("file")
    ap.add_argument("--id", required=True)
    ap.add_argument("--date", required=True, help="date of the conversation, YYYY-MM-DD")
    ap.add_argument("--time", default=None, help="start time, HH:MM")
    ap.add_argument("--owner", default=None, help="the owner's name as it appears in the transcript (transcript only)")
    ap.add_argument("--assistant", default=None, help="name of the AI assistant speaker")
    ap.add_argument("--speakers", default=None, help="comma-separated names that may start a turn (transcript only; the "
                    "owner and the assistant are always allowed); other 'Name:' lines are kept as text")
    a = ap.parse_args(argv)
    try:
        text = Path(a.file).read_text(encoding="utf-8")
        if a.kind == "transcript":
            if not a.owner:
                ap.error("--owner is required for a transcript")
            sp = [x for x in a.speakers.split(",")] if a.speakers is not None else None
            rec = transcript(text, a.id, a.date, a.owner, a.assistant, a.time, sp)
            if sp is None:
                print("speakers found: " + ", ".join(p if isinstance(p, str) else p["name"] for p in rec["participants"]) +
                      " (use --speakers to restrict)", file=sys.stderr)
        else:
            rec = messages(json.loads(text), a.id, a.date, a.assistant, a.time)
    except (ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(json.dumps(rec, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
