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
"""build_graph: case directory -> graph in the HooRii Mind format (the input of gct.GCT.compile_graph).

  1. load      read and check case.json and sources.jsonl; order the sessions by date
  2. extract   one model call per batch of sessions (<= 9,000 characters, <= 3 sessions): entities, facts with the turns they
               come from and a time code relative to the session, relations
  3. assemble  apply the sessions in time order: entity resolution, facts, single-valued slots, edges
  4. organize  local rules to a fixed point plus one occasion-grouping call per person (gct/build/organize.py)
  5. memories  AI-summarised memories: one model call per batch of <= 40 items, merged into the graph as second-hand facts
  6. finalize  every entity reachable from the owner, the published fields only, optional history
"""
import time
from collections import defaultdict

from .assemble import Graph
from .case import load_case
from .extract import extract_memories, extract_sessions
from .organize import organize
from .prompts import NOVEL
from .text import same_value


def _time_fields(tm):
    if not tm:
        return {}
    start, end, prec = tm
    return {"valid_from": start.isoformat()} if prec == "day" else {"valid_from": start.isoformat(), "valid_to": end.isoformat()}


def finalize(case, g, keep_history=True):
    ents = {e["id"]: e for e in g.entities}
    persons = [eid for name, eid in g.spk.items() if eid != "e0" and ents[eid]["kind"] == "person"]
    linked = {e["dst"] for e in g.edges if e["src"] == "e0"} | {e["src"] for e in g.edges if e["dst"] == "e0"}
    for p in persons:                                         # a chat partner with no stated relation to the owner
        if p not in linked:
            g.add_edge("e0", "custom.talks_with", p)
    facts = [f for f in g.facts if keep_history or f["state"] != "superseded"]
    used = {"e0"} | {f["subject"] for f in facts} | {m for f in facts for m in f["mentions"]} | {e["src"] for e in g.edges} | \
        {e["dst"] for e in g.edges}
    adj = defaultdict(set)
    for e in g.edges:
        adj[e["src"]].add(e["dst"])
        adj[e["dst"]].add(e["src"])
    seen, stack = {"e0"}, ["e0"]
    while stack:
        x = stack.pop()
        for y in adj[x] - seen:
            seen.add(y)
            stack.append(y)
    orphans = 0
    for e in g.entities:                                      # every entity reachable from the owner
        if e["id"] in used and e["id"] not in seen:
            g.add_edge("e0", "custom.mentions", e["id"])
            orphans += 1
    out_facts, kept = [], {f["id"] for f in facts}
    for f in facts:
        o = {"id": f["id"], "subject": f["subject"], "module": f["module"], "value": f["value"], "state": f["state"], "vis": ["o"]}
        if f["mentions"]:
            o["mentions"] = f["mentions"]
        o["observed_at"] = f["observed_at"]
        o.update(_time_fields(f.get("time")))
        if f.get("slot"):
            o["slot"] = f["slot"]
        if f.get("superseded_by") in kept:
            o["superseded_by"] = f["superseded_by"]
        if f.get("source"):
            o["source"] = f["source"]
        o["prov"] = [f"{f['session']}#{n}" for n in f["prov"]]
        out_facts.append(o)
    out_ents = []
    for e in g.entities:
        if e["id"] not in used:
            continue
        o = {"id": e["id"], "kind": e["kind"], "name": e["name"]}
        if e.get("aliases"):
            o["aliases"] = list(e["aliases"])
        if e.get("desc"):
            o["desc"] = e["desc"]
        out_ents.append(o)
    edges = [{"src": e["src"], "rel": e["rel"], "dst": e["dst"], "state": "cur", "vis": ["o"]} for e in g.edges
             if e["src"] in used and e["dst"] in used]
    rels = {e["rel"] for e in edges}
    world = {"world_id": case["world_id"], "lang": case["lang"],
             "meta": {"owner": "e0", "now": case["now"], **({"timezone": case["timezone"]} if case.get("timezone") else {})},
             "entities": out_ents, "edges": edges, "facts": out_facts}
    nov = {k: v for k, v in NOVEL.items() if k in rels}
    if nov:
        world["novel_relations"] = nov
    return world, {"orphans_linked": orphans}


def _memory_slots(g):
    """Among facts from memories, a newer dated value of the same slot supersedes the older one, unless it gives the same
    value (text.same_value, compared without people's names)."""
    names = [e["name"] for e in g.entities if e["kind"] in ("person", "pet")]   # who it is about, not what (a place's name is the value)
    by = defaultdict(list)
    for f in g.facts:
        if f["gen"] == "memory" and f.get("slot") and f["state"] in ("confirmed", "hearsay"):
            by[(f["subject"], f["slot"])].append(f)
    n = 0
    for fs in by.values():
        fs.sort(key=lambda f: (f["observed_at"], int(f["id"][1:])))
        for old, new in zip(fs, fs[1:]):
            if old["observed_at"] < new["observed_at"] and \
                    not same_value(old["text"], new["text"], names):
                old["state"] = "superseded"
                old["superseded_by"] = new["id"]
                n += 1
    return n


def input_form(sessions, memories):
    """'raw' (chats and notes only), the memory type when all sources are memories of one type, else 'mixed'."""
    kinds = {"raw"} if sessions else set()
    kinds |= {m["type"] for m in memories}
    return kinds.pop() if len(kinds) == 1 else "mixed"


def model_label(model):
    return model.rsplit("/", 1)[-1].lower()


def build_graph(case_dir, llm, sources=None, keep_history=True, occasions=True, date_lang="en", max_chars=9000,
                max_sessions=3, workers=2, log=print):
    """Build the graph of one case. llm: a gct.build.llm.LLM; sources: another sources file instead of the case's
    sources.jsonl. Returns (graph, report)."""
    t0 = time.time()
    case, sessions, memories = load_case(case_dir, sources)
    log(f"case {case['world_id']}: {len(sessions)} sessions, {len(memories)} memory sources")
    g = Graph(case["owner"], case["lang"])
    g.date_lang = date_lang
    report = {"world_id": case["world_id"], "sessions": len(sessions), "memory_sources": len(memories), "failed": []}
    for s in sessions:                                        # every speaker is an entity before any session is applied
        for name, kind in s["participants"].items():
            g.add_speaker(name, kind)
    if sessions:
        results = extract_sessions(sessions, case["owner"], llm, max_chars, max_sessions, workers, log, lang=case["lang"])
        for s, out, nums in results:
            if out is None:
                report["failed"].append(s["id"])
                continue
            g.apply_session(s, out, nums)
        report["organize"] = organize(g, llm, occasions=occasions, log=log)
    if memories:
        for m, chunk, out in extract_memories(memories, case["owner"], llm, workers=workers, log=log, lang=case["lang"]):
            if out is None:
                report["failed"].append(f"{m['id']}#{chunk[0]['n']}-{chunk[-1]['n']}")
                continue
            g.apply_memory(m, chunk, out)
        report["memory_superseded"] = _memory_slots(g)
    world, fin = finalize(case, g, keep_history)
    world["meta"]["build"] = {"input_form": input_form(sessions, memories), "builder_model": model_label(llm.model)}
    report.update(fin)
    report["assemble"] = {"facts_dropped_unknown_subject": 0, **dict(g.stats)}   # shown also when nothing was dropped
    report["counts"] = {"entities": len(world["entities"]), "facts": len(world["facts"]), "edges": len(world["edges"]),
                        "units": len(world["entities"]) + len(world["facts"]),
                        "facts_by_state": {s: sum(f["state"] == s for f in world["facts"]) for s in
                                           sorted({f["state"] for f in world["facts"]})}}
    report["llm"] = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in llm.stats.items()}
    report["seconds"] = round(time.time() - t0, 1)
    return world, report
