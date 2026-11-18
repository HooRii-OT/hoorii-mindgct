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
"""Compile a graph (entities, facts, edges) into the caller's view: unit texts, typed edges and per-unit payload;
render a request (reference time, current turn, earlier turns) into the model's input text."""
import re
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path



STATE_IDS = {"entity": 0, "confirmed": 1, "hearsay": 2, "pending": 3, "superseded": 4, "asserted": 2}
STATE_ZH = {"confirmed": "已确认", "hearsay": "传闻未确认", "pending": "待确认", "superseded": "已被更新取代的旧值", "asserted": "助手说过"}
STATE_EN = {"confirmed": "confirmed", "hearsay": "hearsay, unconfirmed", "pending": "pending confirmation", "superseded": "superseded old value",
            "asserted": "said by the assistant"}
CALLER_ZH = {"o": "本人", "f": "家庭助理", "w": "工作助理", "d": "司机助理"}
CALLER_EN = {"o": "the user", "f": "family assistant", "w": "work assistant", "d": "driver assistant"}


def world_lang(world):
    return world.get("lang") or world.get("meta", {}).get("lang") or "zh"


@lru_cache(maxsize=1)
def vocab():
    return json.loads((Path(__file__).resolve().parent / "vocab.json").read_text(encoding="utf-8"))


def module_label(m, lang="zh", extra=None):
    if extra and m in extra:
        return extra[m]
    mod = vocab()["modules"].get(m, {})
    return (mod.get("label_en") if lang == "en" else mod.get("label")) or m


def module_text(m, lang="zh", extra=None):
    mod = vocab()["modules"].get(m)
    if mod:
        return f"{m}: {mod.get('label_en', m)} / {mod.get('label', m)}"
    return f"{m}: {module_label(m, lang, extra)}"


def relation_text(rel, novel=None, lang="zh", mod_labels=None):
    v = vocab()
    en = lang == "en"
    if rel.startswith("has_fact:"):
        m = rel.split(":", 1)[1]
        return ("has record: " if en else "拥有资料：") + module_label(m, lang, mod_labels)
    if rel.startswith("fact_of:"):
        m = rel.split(":", 1)[1]
        return ("record belongs to: " if en else "资料所属对象：") + module_label(m, lang, mod_labels)
    if rel.startswith("inv:"):
        return ("inverse of " if en else "反向关系：") + relation_text(rel[4:], novel, lang, mod_labels)
    if rel == "mentioned_in":
        return "is mentioned in record" if en else "被资料提到"
    if rel == "mentions":
        return "record mentions" if en else "资料提到"
    if rel in v["relations"]:
        r = v["relations"][rel]
        return f"{rel}: {r.get('desc_en', rel)}" if en else f"{rel}：{r['desc']}"
    if novel and rel in novel:
        return f"{rel}: {novel[rel]}" if en else f"{rel}：{novel[rel]}"
    return rel


@dataclass
class CompiledGraph:
    world_id: str
    caller: str
    unit_ids: list
    unit_text: list
    unit_type: list
    unit_state: list
    unit_module: list
    fact_subject: list
    src: list
    dst: list
    rel: list
    rel_names: list
    rel_text: list
    mod_names: list
    mod_text: list
    unit_mod: list
    owner: int
    payload: dict = field(default_factory=dict)
    index: dict = field(default_factory=dict)

    @property
    def n(self):
        return len(self.unit_ids)


def entity_text(e, lang="zh"):
    if lang == "en":
        kinds = vocab().get("entity_kinds_en", {})
        s = f"{e['name']} ({kinds.get(e['kind'], e['kind'])})"
        if e.get("aliases"):
            s += ", also called " + ", ".join(e["aliases"])
        if e.get("desc"):
            s += ". " + e["desc"]
        return s
    kinds = vocab()["entity_kinds"]
    s = f"{e['name']}（{kinds.get(e['kind'], e['kind'])}）"
    if e.get("aliases"):
        s += "，别名：" + "、".join(e["aliases"])
    if e.get("desc"):
        s += "。" + e["desc"]
    return s


RENDER_PROVENANCE = "0" == "1"


RENDER_PREFIX = "0" == "1"
UNIT_SESS = False

TIME_IV = False


RENDER_TIME = "0" == "1"


RENDER_VALUE_FIRST = "0" == "1"


VAL_AT = False


GRAPH_EDGES = "all"
REWIRE_SEED = 0


RENDER_TMPL = frozenset(x for x in "".split(",") if x)
_PREP_EN = re.compile(r"\s(?:on|in|at|for|with|from|to|of|about|during|by|under|and)\s")
_LEAD_DATE = re.compile(r"^\s*[\[［【]([^\]］】]{2,40})[\]］】]\s*")


def source_text(f, lang="zh"):
    src = str(f.get("source", "") or "")
    if src.startswith("import:"):
        who = src.split(":", 1)[1] or ("another assistant" if lang == "en" else "其他助手")
        when = str(f.get("observed_at", "") or "")[:7]
        return (f"from {who}" if lang == "en" else f"来自{who}") + (f" {when}" if when else "")
    return "from user" if lang == "en" else "用户记录"


def provenance_text(f, lang="zh"):
    src = str(f.get("source", "") or "")
    if not (RENDER_PROVENANCE and src.startswith("import:")):
        return ""
    who = src.split(":", 1)[1] or "another assistant"
    when = str(f.get("observed_at", "") or "")[:7]
    if lang == "en":
        return f", imported from {who}" + (f", {when}" if when else "")
    return f"，来自 {who} 导入" + (f"，{when}" if when else "")


def short_name(name, lang="zh"):
    name = str(name or "")
    if lang == "en":
        w = name.split()
        if len(w) <= 3:
            return name, None
        m = re.match(r"^(.+?['’]s)\s+(.+)$", name)
        if m:
            head = _PREP_EN.split(" " + m.group(2) + " ")[0].split()
            return f"{m.group(1)} {head[-1] if head else m.group(2).split()[0]}", name
        return " ".join(w[:3]), name
    if len(name) <= 6:
        return name, None
    head = name.rsplit("的", 1)[-1] if "的" in name else ""
    return (head[-6:] if len(head) >= 2 else name[:6]), name


def tmpl_text(f, subject_name, lang, label):
    full = None
    if "short" in RENDER_TMPL and subject_name:
        subject_name, full = short_name(subject_name, lang)
    if "nolabel" in RENDER_TMPL:
        lab = ""
    elif "code" in RENDER_TMPL:
        if lang == "en":
            lab = str(f["module"]).split(".")[-1].replace("_", " ")
        else:
            lab = re.split(r"\s*·\s*", label)[-1][:2]
    else:
        lab = label
    st, prov = (STATE_EN if lang == "en" else STATE_ZH).get(f["state"], f["state"]), provenance_text(f, lang)
    if lang == "en":
        body = f"{lab}: {f['value']}" if lab else f"{f['value']}"
        s = f"{subject_name} | {body} ({st}{prov})" if subject_name else f"{body} ({st}{prov})"
        return s + (f" — {full}" if full else "")
    body = f"{lab}：{f['value']}" if lab else f"{f['value']}"
    s = f"{subject_name}｜{body}（{st}{prov}）" if subject_name else f"{body}（{st}{prov}）"
    return s + (f"——{full}" if full else "")


def fact_text(f, subject_name=None, lang="zh", mod_labels=None):
    label = module_label(f["module"], lang, mod_labels)
    if RENDER_TMPL and not (RENDER_VALUE_FIRST or RENDER_PREFIX):
        return tmpl_text(f, subject_name, lang, label)
    if RENDER_VALUE_FIRST:
        val, date = str(f["value"]), ""
        m = _LEAD_DATE.match(val)
        if m and m.end() < len(val):
            val, date = val[m.end():], m.group(1).strip()
        prov = provenance_text(f, lang)
        if lang == "en":
            head = " · ".join(x for x in (subject_name, label) if x)
            return f"{val} — {head} ({STATE_EN.get(f['state'], f['state'])}{', ' + date if date else ''}{prov})"
        head = "·".join(x for x in (subject_name, label) if x)
        return f"{val}——{head}（{STATE_ZH.get(f['state'], f['state'])}{'，' + date if date else ''}{prov}）"
    if RENDER_PREFIX:
        if lang == "en":
            s = f"[{STATE_EN.get(f['state'], f['state'])}; {source_text(f, lang)}] {label}: {f['value']}"
            return f"{subject_name} | {s}" if subject_name else s
        s = f"【{STATE_ZH.get(f['state'], f['state'])}；{source_text(f, lang)}】{label}：{f['value']}"
        return f"{subject_name}｜{s}" if subject_name else s
    prov = provenance_text(f, lang)
    if lang == "en":
        s = f"{label}: {f['value']} ({STATE_EN.get(f['state'], f['state'])}{prov})"
        return f"{subject_name} | {s}" if subject_name else s
    s = f"{label}：{f['value']}（{STATE_ZH.get(f['state'], f['state'])}{prov}）"
    return f"{subject_name}｜{s}" if subject_name else s


def _swap_fix(a, b, rng, forbid, rounds=20):
    n = len(a)
    key = lambda x, y: (x, y) if x <= y else (y, x)
    for _ in range(rounds):
        cnt = {}
        for k in range(n):
            cnt[key(a[k], b[k])] = cnt.get(key(a[k], b[k]), 0) + 1
        bad = [k for k in range(n) if forbid(a[k], b[k]) or cnt[key(a[k], b[k])] > 1]
        if not bad:
            return
        for k in bad:
            if not (forbid(a[k], b[k]) or cnt[key(a[k], b[k])] > 1):
                continue
            hard = forbid(a[k], b[k])
            order = list(range(n))
            rng.shuffle(order)
            for j in order:
                kk, kj = key(a[k], b[j]), key(a[j], b[k])
                if j == k or forbid(a[k], b[j]) or forbid(a[j], b[k]) or (not hard and (cnt.get(kk, 0) > 0 or cnt.get(kj, 0) > 0 or kk == kj)):
                    continue
                for x in (key(a[k], b[k]), key(a[j], b[j])):
                    cnt[x] -= 1
                b[k], b[j] = b[j], b[k]
                cnt[kk] = cnt.get(kk, 0) + 1; cnt[kj] = cnt.get(kj, 0) + 1
                break


def _rewire_edges(edges, kind, rng):
    out = [dict(e) for e in edges]
    by = {}
    for i, e in enumerate(out):
        by.setdefault(kind.get(e["dst"], ""), []).append(i)
    for k_ in sorted(by):
        ix = by[k_]
        if len(ix) < 2:
            continue
        srcs, ds = [out[i]["src"] for i in ix], [out[i]["dst"] for i in ix]
        rng.shuffle(ds)
        _swap_fix(srcs, ds, rng, lambda s, d: s == d)
        for i, d in zip(ix, ds):
            out[i]["dst"] = d
    by = {}
    for i, e in enumerate(out):
        by.setdefault((kind.get(e["src"], ""), kind.get(e["dst"], "")), []).append(i)
    for k_ in sorted(by):
        rs = [out[i]["rel"] for i in by[k_]]
        rng.shuffle(rs)
        for i, r in zip(by[k_], rs):
            out[i]["rel"] = r
    return out


def _rewire_subjects(facts, rng):
    by, subj = {}, {}
    for f in facts:
        by.setdefault(f["module"], []).append(f)
    for m in sorted(by):
        ss = [f["subject"] for f in by[m]]
        rng.shuffle(ss)
        subj.update({f["id"]: s for f, s in zip(by[m], ss)})
    return subj


def _rewire_mentions(facts, ent_ids, subj_of, rng):
    pairs = [(m, f["id"]) for f in facts for m in (f.get("mentions") or ()) if m in ent_ids and m != f["subject"]]
    fs, ms = [p[1] for p in pairs], [p[0] for p in pairs]
    rng.shuffle(ms)
    _swap_fix(fs, ms, rng, lambda fid, m: m == subj_of[fid])
    men = {f["id"]: [] for f in facts}
    for fid, m in zip(fs, ms):
        men[fid].append(m)
    return men


OBJ_ROUTE_ONLY = False


def compile_view(world, caller, include_subject_name=False, include_old_edges=False):
    lang = world_lang(world)
    novel = world.get("novel_relations", {})
    mod_labels = world.get("module_labels") or {}
    ents = world["entities"]
    facts = [f for f in world["facts"] if caller in f["vis"]]
    edges = [e for e in world["edges"] if caller in e["vis"] and (include_old_edges or e["state"] == "cur")]

    unit_ids, text, utype, ustate, umod, fsubj = [], [], [], [], [], []
    for e in ents:
        unit_ids.append(e["id"]); text.append(entity_text(e, lang)); utype.append(0)
        ustate.append(0); umod.append(e["kind"]); fsubj.append(-1)
    index = {u: i for i, u in enumerate(unit_ids)}
    if OBJ_ROUTE_ONLY:
        payload_ro = [i for i, e in enumerate(ents) if e.get("v03") == "C"]
    names = {e["id"]: e["name"] for e in ents}
    payload = {}
    if OBJ_ROUTE_ONLY:
        payload["route_only"] = payload_ro
    if GRAPH_EDGES not in ("all", "none", "star", "rewire"):
        raise ValueError(f"GRAPH_EDGES {GRAPH_EDGES!r}")
    subj_of, men_of = {f["id"]: f["subject"] for f in facts}, None
    if GRAPH_EDGES == "rewire":
        import random
        rng = random.Random(f"{world['world_id']}|{caller}|{REWIRE_SEED}")
        edges = _rewire_edges(edges, {e["id"]: e["kind"] for e in ents}, rng)
        subj_of = _rewire_subjects(facts, rng)
        men_of = _rewire_mentions(facts, {e["id"] for e in ents}, subj_of, rng)
    for f in facts:
        index[f["id"]] = len(unit_ids)
        unit_ids.append(f["id"])
        sname = names.get(f.get("v03_prev_subject", f["subject"]) if OBJ_ROUTE_ONLY else f["subject"])
        text.append(fact_text(f, sname if include_subject_name else None, lang, mod_labels))
        utype.append(1); ustate.append(STATE_IDS.get(f["state"], 1)); umod.append(f["module"])
        fsubj.append(index[subj_of[f["id"]]])
        payload[f["id"]] = {"subject": f["subject"], "module": f["module"], "value": f["value"],
                            "state": f["state"], "source": f.get("source", "")}
        if UNIT_SESS:
            payload[f["id"]]["session"] = f.get("prov_session") or ""

    rel_names, rel_idx = [], {}

    def rid(name):
        if name not in rel_idx:
            rel_idx[name] = len(rel_names)
            rel_names.append(name)
        return rel_idx[name]

    rels = vocab()["relations"]
    src, dst, rel = [], [], []
    for e in (edges if GRAPH_EDGES in ("all", "rewire") else ()):
        s, d = index[e["src"]], index[e["dst"]]
        inv = rels.get(e["rel"], {}).get("inverse") or f"inv:{e['rel']}"
        for a, b, r in ((s, d, e["rel"]), (d, s, inv)):
            src.append(a); dst.append(b); rel.append(rid(r))
    for f in (facts if GRAPH_EDGES != "none" else ()):
        s, d = index[subj_of[f["id"]]], index[f["id"]]
        src.append(s); dst.append(d); rel.append(rid(f"has_fact:{f['module']}"))
        src.append(d); dst.append(s); rel.append(rid(f"fact_of:{f['module']}"))
        if GRAPH_EDGES == "star":
            continue
        for m in (men_of[f["id"]] if men_of is not None else f.get("mentions") or ()):
            if m in index and m != subj_of[f["id"]]:
                src.append(index[m]); dst.append(d); rel.append(rid("mentioned_in"))
                src.append(d); dst.append(index[m]); rel.append(rid("mentions"))

    if TIME_IV:
        from .timex import unit_time_table
        payload["t_iv"] = unit_time_table(world, unit_ids)
    if RENDER_TIME:
        from .timex import iv_text, unit_intervals
        for i, (lo, hi) in enumerate(unit_intervals(world, unit_ids)):
            if hi >= lo:
                text[i] = f"[{iv_text(lo, hi)}] {text[i]}"
    if VAL_AT:
        payload["val_at"] = [0 if ty == 0 else max(0, t.rfind(str(payload[u]["value"]))) for u, t, ty in zip(unit_ids, text, utype)]
    mod_names, mod_idx, unit_mod = [], {}, []
    for ty, m in zip(utype, umod):
        if ty == 0:
            unit_mod.append(-1)
            continue
        if m not in mod_idx:
            mod_idx[m] = len(mod_names)
            mod_names.append(m)
        unit_mod.append(mod_idx[m])
    mod_text = [module_text(m, lang, mod_labels) for m in mod_names]
    return CompiledGraph(world_id=world["world_id"], caller=caller, unit_ids=unit_ids, unit_text=text,
                         unit_type=utype, unit_state=ustate, unit_module=umod, fact_subject=fsubj,
                         src=src, dst=dst, rel=rel, rel_names=rel_names,
                         rel_text=[relation_text(r, novel, lang, mod_labels) for r in rel_names],
                         mod_names=mod_names, mod_text=mod_text, unit_mod=unit_mod,
                         owner=index[world["meta"].get("owner", "e0")] if world["meta"].get("owner", "e0") in index else 0,
                         payload=payload, index=index)


CTX_RECENT_FIRST = False

CTX_MAX_MSGS = None
CTX_RANDOM_WINDOWS = None


def query_text(world, q):
    now = world["meta"].get("now") or world.get("seed", {}).get("now", "")
    ctx = list(q.get("ctx") or [])
    if CTX_RANDOM_WINDOWS:
        import zlib
        k = CTX_RANDOM_WINDOWS[zlib.crc32(f"{world.get('world_id', '')}/{q.get('id', '')}".encode()) % len(CTX_RANDOM_WINDOWS)]
        ctx = ctx if k < 0 else (ctx[-k:] if k > 0 else [])
    elif CTX_MAX_MSGS is not None:
        ctx = ctx[-CTX_MAX_MSGS:] if CTX_MAX_MSGS > 0 else []
    en = world_lang(world) == "en"
    head = f"[Now: {now} | Caller: {CALLER_EN.get(q['caller'], q['caller'])}]" if en else f"［当前时间：{now}｜调用方：{CALLER_ZH.get(q['caller'], q['caller'])}］"
    U, A, CUR, EARLIER = ("User: ", "Assistant: ", "Current request: ", "Earlier conversation (most recent first):") if en else ("用户：", "助理：", "当前请求：", "此前对话（由近到远）：")
    if CTX_RECENT_FIRST:
        lines = [head, CUR + q["text"]]
        if ctx:
            lines.append(EARLIER)
            lines += [(U if role == "u" else A) + t for role, t in reversed(ctx)]
        return "\n".join(lines)
    lines = [head] + [(U if role == "u" else A) + t for role, t in ctx]
    lines.append(CUR + q["text"])
    return "\n".join(lines)
