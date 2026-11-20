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
"""Extraction calls. Sessions are rendered with numbered turns and sent in batches; each session's result is stored on its own
(keyed by the prompt version, its participants and its text), so adding sources later only sends the new sessions. AI memories
are sent in batches of numbered items through their own prompt."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import prompts as P
from .case import batches, render, split_long


def participants_text(parts):
    out = []
    for name, kind in parts.items():
        if kind == "assistant":
            out.append(f"- {name}: an AI assistant (not a person; record nothing about it except what the person decided with its help)")
        else:
            out.append(f"- {name}: a person")
    return "\n".join(out)


def _request_tag(llm):
    """What besides the prompt decides the answer: model, wire format, effort, temperature, extra request fields, output
    limit and JSON mode (the last two only when set, so that answers cached without them keep their keys)."""
    g = lambda k: getattr(llm, k, None)  # noqa: E731
    tag = {"model": g("model"), "api": g("api"), "effort": g("effort") or None, "temperature": g("temperature"),
           "extra": g("extra_body") or {}}
    if g("max_output_tokens"):
        tag["max_output_tokens"] = g("max_output_tokens")
    if g("json_mode"):
        tag["json_mode"] = True
    return json.dumps(tag, sort_keys=True, ensure_ascii=False)


def _session_key(tag, session, text, parts, note=""):
    p = ";".join(f"{k}={v}" for k, v in sorted(parts.items()))
    return hashlib.sha256(f"{P.EXTRACT}{note}\n{tag}\n{session['type']}\n{p}\n{text}".encode()).hexdigest()[:24]


def _with_note(prompt, lang):
    note = P.LANG_NOTE.get(lang, "")
    return prompt + "\n\n" + note if note else prompt


def _check_sessions(labels):
    def chk(o):
        got = {str(x.get("id")) for x in o.get("sessions") or [] if isinstance(x, dict)}
        miss = [lab for lab in labels if lab not in got]
        if miss:
            raise ValueError(f"missing sessions {miss}")
    return chk


def extract_sessions(sessions, owner, llm, max_chars=9000, max_sessions=3, workers=2, log=print, lang="en"):
    """-> list of (session part, extraction or None, T-number -> turn number), in time order."""
    parts_all = split_long(sessions, max_chars)
    rendered, todo, res = {}, [], {}
    store = Path(llm.cache) if llm.cache else None
    for i, s in enumerate(parts_all):
        text, nums = render(s)
        parts = {owner: "person", **s["participants"]} if owner in s["participants"] else dict(s["participants"])
        key = _session_key(_request_tag(llm), s, text, parts, P.LANG_NOTE.get(lang, ""))
        rendered[i] = (text, nums, parts, key)
        hit = store / f"session_{key}.json" if store else None
        if hit and hit.exists():
            res[i] = json.loads(hit.read_text(encoding="utf-8"))
        else:
            todo.append(i)
    groups = batches([dict(parts_all[i], _i=i) for i in todo], max_chars, max_sessions)
    log(f"extract: {len(parts_all)} session parts, {len(parts_all) - len(todo)} cached, {len(groups)} calls to send")
    done = [0]

    def run(group):
        labels = [f"S{k + 1}" for k in range(len(group))]
        parts = {}
        for s in group:
            parts.update(rendered[s["_i"]][2])
        body = "\n\n".join(f"=== SESSION {lab}{' (note)' if s['type'] == 'note' else ''} ===\n{rendered[s['_i']][0]}"
                           for lab, s in zip(labels, group))
        prompt = P.EXTRACT.format(participants=participants_text(parts), kinds="/".join(P.KINDS), rels=", ".join(P.REL_LIST),
                                  kind_mods="; ".join(f"{k}: {' '.join(v)}" for k, v in P.KIND_MODS.items()), sessions=body)
        out = llm("extract", _with_note(prompt, lang), check=_check_sessions(labels))
        if out is None:
            log(f"  extraction failed for {[s['id'] for s in group]}")
            return
        by = {str(x.get("id")): x for x in out["sessions"] if isinstance(x, dict)}
        for lab, s in zip(labels, group):
            x = by.get(lab) or {}
            o = {"entities": x.get("entities") or [], "facts": x.get("facts") or [], "relations": x.get("relations") or []}
            res[s["_i"]] = o
            if store:
                target = store / f"session_{rendered[s['_i']][3]}.json"
                tmp = target.with_suffix(".tmp")
                tmp.write_text(json.dumps(o, ensure_ascii=False), encoding="utf-8")
                tmp.replace(target)
        done[0] += 1
        log(f"  extract: {done[0]} of {len(groups)} calls done")

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        list(ex.map(run, groups))
    return [(s, res.get(i), rendered[i][1]) for i, s in enumerate(parts_all)]


def extract_memories(memories, owner, llm, max_items=40, max_chars=9000, workers=2, log=print, lang="en"):
    """-> list of (memory source, items of the batch, extraction or None)."""
    jobs = []
    for m in memories:
        cur, size = [], 0
        for it in m["items"]:
            n = len(it["text"]) + 16
            if cur and (len(cur) >= max_items or size + n > max_chars):
                jobs.append((m, cur))
                cur, size = [], 0
            cur.append(it)
            size += n
        if cur:
            jobs.append((m, cur))
    log(f"memories: {sum(len(m['items']) for m in memories)} items in {len(jobs)} batches")
    done = [0]

    def run(job):
        m, chunk = job
        body = "\n".join(f"M{k + 1} [{x['date']}] {x['text']}" for k, x in enumerate(chunk))
        prompt = P.MEMORY.format(owner=owner, items=body, kinds="/".join(P.KINDS), rels=", ".join(P.REL_LIST),
                                 kind_mods="; ".join(f"{k}: {' '.join(v)}" for k, v in P.KIND_MODS.items()))

        def chk(o):
            for k in ("entities", "facts", "relations"):
                if not isinstance(o.get(k), list):
                    raise ValueError(f"'{k}' is not a list")
        out = llm("memory", _with_note(prompt, lang), check=chk)
        if out is None:
            log(f"  memory import failed for {m['id']} items {chunk[0]['n']}-{chunk[-1]['n']}")
        done[0] += 1
        log(f"  memories: {done[0]} of {len(jobs)} calls done")
        return m, chunk, out

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        return list(ex.map(run, jobs))
