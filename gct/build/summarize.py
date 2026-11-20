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
"""Make AI-summarised memories from a case's raw chats and notes, the way an assistant's memory feature would keep them.

    python -m gct.build.summarize --case DIR --form list|sessions|profile_md|profile_json --out FILE [--id ID] [LLM options]

The output is one source record (memory_list, memory_summaries or memory_profile) for sources.jsonl. The prompts see only the
case's sources and the owner's name; they ask for what is worth remembering, not for answers to any question. We used this to
derive the AI-memory variants of the example cases from their raw material, so that a graph built from raw material can be
compared with one built from summaries.
"""
import argparse
import json
import os
import sys
from pathlib import Path

from .case import CaseError, load_case, render
from .llm import LLM, StopBuild, address_warning

COMMON = """You are the memory feature of a personal AI assistant used by {owner}. Below are {what}, in time order, each with its date. Write down what the assistant should remember about {owner} and the people, pets, places, plans and things in {owner}'s life, so that it can help in later conversations. Keep names, dates, times, places, numbers and amounts exactly as given; when something changed or was cancelled, keep only the latest state and say what changed. Write in the language of the material ({lang}). Do not invent anything.

{material}

"""
LIST = COMMON + """Return JSON only: {{"memories": [{{"text": one memory as a short standalone sentence, "source": the id of the source it mainly comes from}}, ...]}}. Write as many memories as the material needs, one fact per memory."""
SESSIONS = COMMON + """For EACH source above write one summary paragraph (at most {words} words) of what is worth remembering from it; a source with nothing worth remembering gets an empty string.
Return JSON only: {{"summaries": [{{"source": source id, "text": the paragraph}}, ...]}} with one entry per source, in order."""
PROFILE_MD = COMMON + """Write one profile document in Markdown: a title line, then sections ("## ...") such as basics, family and people, health, work, home, routines, preferences, plans and upcoming dates, and recent events, each with short bullet points. Describe the state as of {now}.
Return JSON only: {{"markdown": the document}}"""
PROFILE_JSON = COMMON + """Write one profile as a JSON object with categories as keys (for example "basics", "people", "health", "work", "home", "routines", "preferences", "plans", "recent_events"), nested objects or lists inside, and short string values. Describe the state as of {now}.
Return JSON only: {{"profile": the object}}"""


def _material(case, sessions):
    out = []
    for s in sessions:
        text, _ = render(s)
        kind = "note" if s["type"] == "note" else "conversation"
        out.append(f"=== {s['id']} ({kind}, {s['date']}{' ' + s['time'] if s['time'] else ''}) ===\n{text}")
    return "\n\n".join(out)


def _chunks(sessions, max_chars):
    out, cur, size = [], [], 0
    for s in sessions:
        n = len(render(s)[0])
        if cur and size + n > max_chars:
            out.append(cur)
            cur, size = [], 0
        cur.append(s)
        size += n
    if cur:
        out.append(cur)
    return out


def summarize(case_dir, form, llm, sid=None, max_chars=12000, words=80, log=print):
    case, sessions, _ = load_case(case_dir)
    owner, lang = case["owner"], {"en": "English", "zh": "Chinese"}[case["lang"]]
    if not sessions:
        raise CaseError("the case has no chats or notes to summarise (memory sources are not summarised again)")
    last = max(s["date"] for s in sessions)
    sid = sid or f"memory-{form.replace('_', '-')}"
    if form in ("list", "sessions"):
        rec = {"id": sid, "type": "memory_list" if form == "list" else "memory_summaries", "assistant": "AI assistant"}
        items = []
        for chunk in _chunks(sessions, max_chars):
            dates = {s["id"]: s["date"] for s in chunk}
            what = "conversations and notes" if form == "list" else "conversations and notes (sources)"
            prompt = (LIST if form == "list" else SESSIONS).format(owner=owner, what=what, lang=lang, words=words,
                                                                    material=_material(case, chunk))
            key = "memories" if form == "list" else "summaries"

            def chk(o, key=key):
                if not isinstance(o.get(key), list):
                    raise ValueError(f"'{key}' is not a list")
            out = llm("summarize_" + form, prompt, check=chk)
            if out is None:
                raise StopBuild(f"summarising failed for sources {[s['id'] for s in chunk]} (see the call log in --cache)")
            for x in out[key]:
                if not isinstance(x, dict) or not str(x.get("text") or "").strip():
                    continue
                d = dates.get(str(x.get("source")), chunk[-1]["date"])
                items.append({"text": str(x["text"]).strip(), "date": d})
        if form == "list":
            rec.update(date=last, items=items)
        else:
            rec["summaries"] = items
        log(f"{form}: {len(items)} entries")
        return rec
    prompt = (PROFILE_MD if form == "profile_md" else PROFILE_JSON).format(
        owner=owner, what="conversations and notes", lang=lang, now=case["now"], material=_material(case, sessions))
    key = "markdown" if form == "profile_md" else "profile"

    def chk(o):
        if not (isinstance(o.get(key), str) if key == "markdown" else isinstance(o.get(key), dict)):
            raise ValueError(f"'{key}' missing")
    out = llm("summarize_" + form, prompt, check=chk)
    if out is None:
        raise StopBuild("summarising failed (see the call log in --cache)")
    rec = {"id": sid, "type": "memory_profile", "date": last, "assistant": "AI assistant"}
    rec.update({"format": "markdown", "text": out[key]} if key == "markdown" else {"format": "json", "data": out[key]})
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m gct.build.summarize", description="Make an AI-summarised memory source from a "
                                 "case's raw chats and notes (one record for sources.jsonl).")
    ap.add_argument("--case", required=True, help="case directory holding case.json and sources.jsonl (its chats and notes "
                    "are summarised)")
    ap.add_argument("--form", required=True, choices=("list", "sessions", "profile_md", "profile_json"),
                    help="list = memory_list (one short memory per line); sessions = memory_summaries (one paragraph per "
                    "chat or note); profile_md / profile_json = memory_profile (one profile document)")
    ap.add_argument("--out", required=True, help="file to write the record to (one JSON line)")
    ap.add_argument("--id", default=None, help="source id of the record (default memory-<form>)")
    ap.add_argument("--model", default="gpt-6.1-sol", help="model name (default gpt-6.1-sol)")
    ap.add_argument("--api", choices=("responses", "chat"), default=None,
                    help="wire format (default: responses for gpt-* models, else chat)")
    ap.add_argument("--base-url", default=None, help="API address (default: OPENAI_BASE_URL, else https://api.openai.com/v1)")
    ap.add_argument("--effort", default=None, help='reasoning effort (default: medium for gpt-* models, else not sent; '
                    '--effort "" sends none)')
    ap.add_argument("--extra-body", default=None, help="extra request fields as JSON")
    ap.add_argument("--cache", default=None, help="directory for cached answers and the call log")
    ap.add_argument("--save-raw", default=None, help="directory to keep every raw answer")
    ap.add_argument("--retries", type=int, default=3, help="extra attempts per call after an error or an invalid answer")
    ap.add_argument("--no-retry-unknown", action="store_true", help="do not re-send a call whose connection broke after sending")
    ap.add_argument("--timeout", type=float, default=900, help="seconds to wait for the server to start answering or to send "
                    "more of a streamed answer (default 900)")
    a = ap.parse_args(argv)
    log = lambda m: print(m, file=sys.stderr, flush=True)  # noqa: E731
    os.umask(os.umask(0o077) | 0o077)
    try:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        ap.error(f"cannot create the directory of {a.out}: {e}")
    effort = a.effort if a.effort is not None else ("medium" if a.model.startswith("gpt-") else None)
    try:
        extra = json.loads(a.extra_body) if a.extra_body else None
    except ValueError as e:
        ap.error(f"--extra-body is not JSON: {e}")
    llm = LLM(model=a.model, api=a.api, base_url=a.base_url, effort=effort or None, cache=a.cache, save_raw=a.save_raw,
              retries=a.retries, retry_unknown=not a.no_retry_unknown, timeout=a.timeout, extra_body=extra, concurrency=1)
    if address_warning(llm):
        log(address_warning(llm))
    try:
        rec = summarize(a.case, a.form, llm, a.id, log=log)
    except StopBuild as e:
        log(f"stopped: {e}")
        return 2
    except (CaseError, FileNotFoundError) as e:
        log(f"error: {e}")
        return 2
    Path(a.out).write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
    log(f"wrote {a.out}; {llm.stats['calls']} requests, {llm.stats['input_tokens']} input / {llm.stats['output_tokens']} output tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
