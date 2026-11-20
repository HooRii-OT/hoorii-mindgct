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
"""Check a graph file before compiling it:  python -m gct.build.validate graph.json [more.json ...]

Errors (exit code 1): structure, ids (unique; entity and fact ids disjoint), references (owner, subjects, mentions, edge ends,
superseded_by), vocabulary (entity kinds, fact modules, relations; custom.* relations need a novel_relations entry), states,
vis lists, dates (YYYY-MM-DD, or a timestamp with a timezone for observed_at), valid_from <= valid_to, more than 2,048
units visible to one caller. Warnings: duplicate facts or edges, entities with the same kind and name, events, trips, appointments
or projects whose names differ only in whose they are ("Dana's quarterly review", "Ines's quarterly review"), modules unusual for the
subject's kind, facts without provenance, empty values, entities nothing points to."""
import json
import re
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path

from .prompts import KIND_MODS

OCCASIONS = ("event", "trip", "appointment", "project")

VOCAB = json.loads((Path(__file__).resolve().parents[1] / "vocab.json").read_text(encoding="utf-8"))
STATES = {"confirmed", "hearsay", "pending", "superseded", "asserted"}
CALLERS = {"o", "f", "w", "d"}
_ID = re.compile(r"[A-Za-z0-9_.:-]{1,96}")
MAX_UNITS = 2048


def _date_ok(v, stamp=False):
    if not isinstance(v, str):
        return False
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
        try:
            date.fromisoformat(v)
            return True
        except ValueError:
            return False
    if stamp and re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2}(\.\d{1,6})?)?(Z|[+-]\d{2}:\d{2})", v):
        try:
            return datetime.fromisoformat(v.replace("Z", "+00:00")).tzinfo is not None
        except ValueError:
            return False
    return False


def check(world):
    """-> (errors, warnings, summary)."""
    err, warn = [], []
    for k in ("world_id", "lang", "meta", "entities", "facts", "edges"):
        if k not in world:
            err.append(f"missing top-level field '{k}'")
    if err:
        return err, warn, {}
    if world["lang"] not in ("en", "zh"):
        err.append("lang must be 'en' or 'zh'")
    meta = world["meta"] or {}
    now = str(meta.get("now", ""))
    if not (re.fullmatch(r"\d{4}-\d{2}-\d{2}( \d{2}:\d{2})?", now) and _date_ok(now[:10])) and not _date_ok(now, stamp=True):
        err.append("meta.now must be 'YYYY-MM-DD', 'YYYY-MM-DD HH:MM' or a timestamp with a timezone")
    kinds, mods, rels = VOCAB["entity_kinds"], VOCAB["modules"], VOCAB["relations"]
    novel = world.get("novel_relations") or {}
    ents = {}
    for e in world["entities"]:
        i = e.get("id")
        if not isinstance(i, str) or not _ID.fullmatch(i):
            err.append(f"entity id {i!r} malformed")
            continue
        if i in ents:
            err.append(f"entity id {i} repeated")
        ents[i] = e
        if e.get("kind") not in kinds:
            err.append(f"entity {i}: kind {e.get('kind')!r} not in the vocabulary")
        if not str(e.get("name", "")).strip():
            err.append(f"entity {i}: empty name")
        if "aliases" in e and not (isinstance(e["aliases"], list) and all(isinstance(a, str) for a in e["aliases"])):
            err.append(f"entity {i}: aliases must be a list of strings")
    if meta.get("owner") not in ents:
        err.append(f"meta.owner {meta.get('owner')!r} is not an entity")
    facts = {}
    for f in world["facts"]:
        i = f.get("id")
        if not isinstance(i, str) or not _ID.fullmatch(i):
            err.append(f"fact id {i!r} malformed")
            continue
        if i in facts or i in ents:
            err.append(f"fact id {i} repeated or equal to an entity id")
        facts[i] = f
        if f.get("subject") not in ents:
            err.append(f"fact {i}: subject {f.get('subject')!r} is not an entity")
        if f.get("module") not in mods:
            err.append(f"fact {i}: module {f.get('module')!r} not in the vocabulary")
        elif f.get("subject") in ents:
            k = ents[f["subject"]]["kind"]
            if f["module"] != "notes.general" and f["module"] not in VOCAB["modules_by_kind"].get(k, []) and \
                    f["module"] not in KIND_MODS.get(k, []):
                warn.append(f"fact {i}: module {f['module']} is unusual for a {k}")
        v = f.get("value")
        if not isinstance(v, str) or not v.strip():
            err.append(f"fact {i}: empty value")
        elif len(v) > 8192:
            err.append(f"fact {i}: value longer than 8,192 characters")
        if f.get("state") not in STATES:
            err.append(f"fact {i}: state {f.get('state')!r} is not one of {sorted(STATES)}")
        vis = f.get("vis")
        if not isinstance(vis, list) or not vis or any(c not in CALLERS for c in vis):
            err.append(f"fact {i}: vis must be a non-empty list of caller codes o / f / w / d")
        for m in f.get("mentions") or []:
            if m not in ents:
                err.append(f"fact {i}: mention {m!r} is not an entity")
        if f.get("observed_at") not in (None, "") and not _date_ok(f["observed_at"], stamp=True):
            err.append(f"fact {i}: observed_at {f['observed_at']!r} must be YYYY-MM-DD or a timestamp with a timezone")
        for k in ("valid_from", "valid_to"):
            if f.get(k) not in (None, "") and not _date_ok(f[k]):
                err.append(f"fact {i}: {k} {f[k]!r} must be YYYY-MM-DD")
        if _date_ok(f.get("valid_from")) and _date_ok(f.get("valid_to")) and f["valid_from"] > f["valid_to"]:
            err.append(f"fact {i}: valid_from after valid_to")
        if "prov" in f and not f["prov"]:
            warn.append(f"fact {i}: no provenance (the extractor named no source turn)")
    for f in facts.values():
        if f.get("superseded_by") and f["superseded_by"] not in facts:
            err.append(f"fact {f['id']}: superseded_by {f['superseded_by']!r} is not a fact")
    ekeys = Counter()
    for n, e in enumerate(world["edges"]):
        if e.get("src") not in ents or e.get("dst") not in ents:
            err.append(f"edge {n}: endpoint not an entity ({e.get('src')!r} -> {e.get('dst')!r})")
        r = e.get("rel")
        if r not in rels and not (isinstance(r, str) and r.startswith("custom.") and r in novel):
            err.append(f"edge {n}: relation {r!r} not in the vocabulary (custom.* relations need a novel_relations entry)")
        if e.get("state") not in ("cur", "old"):
            err.append(f"edge {n}: state must be 'cur' or 'old'")
        vis = e.get("vis")
        if not isinstance(vis, list) or not vis or any(c not in CALLERS for c in vis):
            err.append(f"edge {n}: vis must be a non-empty list of caller codes")
        ekeys[(e.get("src"), r, e.get("dst"))] += 1
    for k, c in ekeys.items():
        if c > 1:
            warn.append(f"edge {k} appears {c} times")
    dup = Counter((f.get("subject"), f.get("module"), re.sub(r"\s+", " ", str(f.get("value", ""))).strip().lower()) for f in facts.values())
    for k, c in dup.items():
        if c > 1:
            warn.append(f"{c} facts with the same subject, module and value: {k[0]} {k[1]} {k[2][:60]!r}")
    names = Counter((e.get("kind"), str(e.get("name", "")).strip().lower()) for e in ents.values())
    for k, c in names.items():
        if c > 1:
            warn.append(f"{c} entities of kind {k[0]} named {k[1]!r}")
    cores = {}                                            # "Dana's quarterly review" / "Ines's quarterly review": one occasion?
    for i, e in ents.items():
        n = str(e.get("name", "")).strip().lower()
        core = re.sub(r"^the\s+", "", re.sub(r"^.*['’]s\s+", "", re.sub(r"^.*的", "", n)))
        if e.get("kind") in OCCASIONS and core and core != n:
            cores.setdefault((e.get("kind"), core), []).append(i)
    for (k, core), ids in cores.items():
        named = {str(ents[i].get("name", "")).strip().lower() for i in ids}
        if len(named) > 1:
            warn.append(f"{len(ids)} entities of kind {k} with similar names ({', '.join(sorted(named))[:120]}): the same {k}?")
    pointed = {f.get("subject") for f in facts.values()} | {m for f in facts.values() for m in f.get("mentions") or []} | \
        {e.get("src") for e in world["edges"]} | {e.get("dst") for e in world["edges"]} | {meta.get("owner")}
    for i in ents:
        if i not in pointed:
            warn.append(f"entity {i} ({ents[i].get('name')}) has no fact and no edge")
    callers = sorted({c for f in facts.values() for c in f.get("vis") or []} | {"o"})
    units = {c: len(ents) + sum(c in (f.get("vis") or []) for f in facts.values()) for c in callers}
    for c, n in units.items():
        if n > MAX_UNITS:
            err.append(f"caller {c} sees {n} units (entities + facts); a view holds at most {MAX_UNITS}")
    summary = {"entities": len(ents), "facts": len(facts), "edges": len(world["edges"]), "units": units,
               "facts_by_state": dict(Counter(f.get("state") for f in facts.values())),
               "facts_with_dates": sum(bool(f.get("valid_from")) for f in facts.values())}
    return err, warn, summary


def compile_check(world):
    """Compile the owner's view with the reference compiler (no model needed) and return its unit count."""
    from ..compile import compile_view
    return compile_view(world, "o", include_subject_name=True).n


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m gct.build.validate", description="Check graph files in the HooRii Mind format.")
    ap.add_argument("graphs", nargs="+")
    ap.add_argument("--max-warnings", type=int, default=20, help="warnings to print per file (default 20)")
    a = ap.parse_args(argv)
    bad = 0
    for p in a.graphs:
        world = json.loads(Path(p).read_text(encoding="utf-8"))
        err, warn, summary = check(world)
        if not err:
            try:
                summary["compiled_units"] = compile_check(world)
            except Exception as e:  # noqa: BLE001
                err.append(f"the reference compiler failed: {type(e).__name__}: {e}")
        print(f"{p}: {'OK' if not err else 'FAILED'} - {len(err)} errors, {len(warn)} warnings")
        print("  " + json.dumps(summary, ensure_ascii=False))
        for m in err:
            print("  error: " + m)
        for m in warn[:a.max_warnings]:
            print("  warning: " + m)
        if len(warn) > a.max_warnings:
            print(f"  ... {len(warn) - a.max_warnings} more warnings")
        bad += bool(err)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
