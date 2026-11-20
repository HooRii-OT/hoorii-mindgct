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
"""Assembly: extracted sessions -> one graph, applied in time order (the same code adds new sessions to an existing build).

Entities resolve against what the graph already holds (proper names, aliases, unique roles such as "mother of X", and
descriptive names such as "Sam's car"); facts are appended with stable ids and keep the turns they come from; a newer value of
the same single-valued slot marks the older fact "superseded" (it is never deleted); edges are deduplicated."""
import re
from collections import Counter

from .prompts import KINDS, KIND_MODS, NOVEL, REL_LIST, STATES
from .text import CJK, GENERIC, UNIQUE_ROLES, is_proper, is_title, keep_alias, norm, overlap, role_groups, roles_compatible, same_value
from .timeres import _WD as _WDCODE, _weekday, moved_to, parse_date, resolve_free, resolve_when, text_dates, written_dates

WEAK_ROLES = {"person", "friend", "acquaintance", "contact", "someone", "other", "mention", "mentioned", "participant",
              "朋友", "熟人", "联系人", "人"}
KIND_GROUPS = [{"organization", "group", "service", "place"}, {"event", "trip", "project"}, {"asset", "document", "vehicle"}]


TITLES = {"dr", "mr", "mrs", "ms", "miss", "mx", "prof", "professor", "coach", "sir", "madam", "dame", "rev", "father", "sister",
          "brother", "aunt", "auntie", "uncle", "grandma", "grandpa", "granny", "nana", "teacher", "doctor", "nurse", "officer"}
UNIQUE_RELS = {"spouse_of"}                              # relations with a single target per source


def name_tokens(name):
    """Words of a person's name without a leading title: "Dr. Ruth Fenn" -> ["ruth", "fenn"]."""
    toks = norm(name).split()
    while toks and toks[0] in TITLES:
        toks = toks[1:]
    return toks


_CJK_NAME = re.compile(rf"[{CJK}]{{2,4}}")
_CJK_TITLED = re.compile(rf"^([{CJK}])([{CJK}]{{0,2}}?)(老师|医生|大夫|阿姨|师傅|叔叔|主任|校长|经理|总|教练|护士|律师)$")


def same_person_name(a, b, first_names=True):
    """Could two person names name the same person? Equal names; a title with the surname of a full name ("Dr. Huang",
    "Dr. Ana Huang"; 王老师, 王丽老师); a Chinese given name and the full name it ends ("子航", "周子航"); and, with first_names,
    a first name and a full name ("Sam", "Sam Rivera"). Two full names with different surnames or first names cannot."""
    a, b = str(a).strip(), str(b).strip()
    if _CJK_NAME.fullmatch(a) and _CJK_NAME.fullmatch(b):
        if a == b:
            return True
        ta, tb = _CJK_TITLED.match(a), _CJK_TITLED.match(b)
        if ta and tb:
            return ta.group(1) == tb.group(1) and ta.group(3) == tb.group(3) and (not ta.group(2) or not tb.group(2))
        if ta or tb:
            return False
        s, l = sorted((a, b), key=len)
        return len(s) == 2 and len(l) == 3 and l.endswith(s)
    ta, tb = name_tokens(a), name_tokens(b)
    if not ta or not tb:
        return False
    if ta == tb:
        return True
    for x, tx, ty in ((a, ta, tb), (b, tb, ta)):            # "Dr. Huang": a title and a surname
        words = norm(x).split()
        if words and words[0] in TITLES and len(tx) == 1 and len(ty) >= 2 and tx[0] == ty[-1]:
            return True
    if not first_names or ta[0] != tb[0]:
        return False
    return len(ta) == 1 or len(tb) == 1


def fuller_name(a, b):
    """Is a a fuller form of the person name b ("周子航" of "子航", "Dr. Ana Huang" of "Dr. Huang")?"""
    if _CJK_NAME.fullmatch(a) and _CJK_NAME.fullmatch(b):
        return len(a) > len(b) and same_person_name(a, b, first_names=False)
    return len(name_tokens(a)) > len(name_tokens(b)) and same_person_name(a, b, first_names=False)


def compatible(a, b):
    return a == b or any(a in g and b in g for g in KIND_GROUPS)


def kind_group(k):
    return next((min(g) for g in KIND_GROUPS if k in g), k)


class Graph:
    """A graph under construction. Entities: {"id", "kind", "name", "aliases", "desc"}; facts carry build-only fields
    (text, slot, prov, session, speakers, time) that finalize() turns into the published form."""

    def __init__(self, owner, lang):
        self.lang = lang
        self.entities, self.facts, self.edges = [], [], []
        self.edge_keys = set()
        self.owner_name = owner
        self.spk = {}                       # speaker name -> entity id
        self.keys = {}                      # resolution key -> entity id (the first holder)
        self.key_all = {}                   # resolution key -> every entity id holding it
        self.slots = {}                     # (subject, slot) -> latest fact id
        self.roles = {}                     # entity id -> roles the extraction gave it ("plumber", "manager", ...)
        self.stats = Counter()
        self.add_speaker(owner, "person")

    # ---------------------------------------------------------------- primitives
    def add_entity(self, kind, name, aliases=(), desc="", **extra):
        eid = f"e{len(self.entities)}"
        self.entities.append({"id": eid, "kind": kind, "name": name, "aliases": list(aliases), "desc": desc, **extra})
        return eid

    def ent(self, eid):
        return self.entities[int(eid[1:])]

    def add_edge(self, src, rel, dst):
        if src == dst or (src, rel, dst) in self.edge_keys:
            return False
        self.edge_keys.add((src, rel, dst))
        self.edges.append({"src": src, "rel": rel, "dst": dst, "state": "cur"})
        return True

    def add_speaker(self, name, kind):
        if name in self.spk:
            return self.spk[name]
        if not self.entities:
            eid = self.add_entity("person", name)
        else:
            eid = self.add_entity("service" if kind == "assistant" else "person", "AI assistant" if kind == "assistant" and
                                  name.lower() in ("assistant", "ai", "ai assistant") else name)
            if kind == "assistant":
                self.add_edge("e0", "uses", eid)
        self.spk[name] = eid
        self.keys[("name", norm(name))] = eid
        self.key_all.setdefault(("name", norm(name)), []).append(eid)
        return eid

    def _ent_keys(self, e, of_id):
        ks = []
        for nm in [e.get("name", "")] + [a for a in e.get("aliases") or [] if isinstance(a, str)]:
            if is_title(nm):
                ks.append(("title", of_id or "", norm(nm)))
            elif is_proper(nm) and len(norm(nm)) >= 2:
                ks.append(("name", norm(nm)))
        role = norm(e.get("role", ""))
        if role in UNIQUE_ROLES and of_id:
            ks.append(("role", of_id, UNIQUE_ROLES[role]))
        if e.get("name") and not is_proper(e["name"]):
            n = norm(e["name"])                              # a bare role word ("Dad", 外婆) is keyed together with whose it is
            ks.append(("desc", kind_group(e.get("kind")), n) + ((of_id,) if n in GENERIC else ()))
        return ks

    def _other_person(self, key, eid, role):
        """A bare first name ("Marcus") that matches a known person with a different role (a colleague, then a plumber) is
        taken to be someone else."""
        if key[0] != "name" or " " in key[1] or not key[1].isascii() or eid in self.spk.values():
            return False
        return self.ent(eid)["kind"] == "person" and not roles_compatible(self.roles.get(eid), role_groups(role))

    def _name_variant(self, name, role):
        """The one known person (a participant or the owner included) whose name this one can be by a title and surname or a
        Chinese given name ("Dr. Huang" / "Dr. Ana Huang", "子航" / "周子航"), with a compatible role; or the participant whose
        first name it is, when no other known person has that first name."""
        mine, spk = role_groups(role), set(self.spk.values())
        hit = [e["id"] for e in self.entities if e["kind"] == "person" and roles_compatible(mine, self.roles.get(e["id"]))
               and any(same_person_name(name, x, first_names=False) for x in [e["name"], *e.get("aliases", [])] if is_proper(x))]
        if not hit and len(name_tokens(name)) == 1:          # a participant's first name ("Rachel" for the owner Rachel Moreno),
            same = [e["id"] for e in self.entities if e["kind"] == "person"   # when no one else has that first name
                    and any(same_person_name(name, x) for x in [e["name"], *e.get("aliases", [])] if is_proper(x))]
            hit = same if len(same) == 1 and same[0] in spk else []
        return hit if len(hit) == 1 else []

    def _named_differently(self, e, eid):
        """Both the new entity and the known one have proper names, and the names differ: a role ("partner", "mother of X")
        or a description does not make them one."""
        name, E = str(e.get("name", "")).strip(), self.ent(eid)
        if not (is_proper(name) and is_proper(E["name"])):
            return False
        known = {norm(x) for x in [E["name"], *E.get("aliases", [])]}
        return norm(name) not in known and not (E["kind"] == "person" and same_person_name(name, E["name"]))

    def _ref(self, ref, local):
        ref = str(ref or "").strip()
        if not ref:
            return None
        if ref in local:
            return local[ref]
        if ref in self.spk:
            return self.spk[ref]
        hit = self.keys.get(("name", norm(ref)))
        if hit:
            return hit
        hit = self._by_name(ref, getattr(self, "_cur_ents", []), local)   # an entity of the session or the graph by its exact
        if hit or not is_proper(ref):                                     # name or alias ("Leo's dad")
            return hit
        # a person or pet named by its first name only ("Juni", "Sam"): the one known entity with that first name
        cands = [e["id"] for e in self.entities if e["kind"] in ("person", "pet") and is_proper(e["name"])
                 and same_person_name(ref, e["name"]) and len(name_tokens(ref)) == 1]
        return cands[0] if len(cands) == 1 else None

    def restates(self, old, new):
        """Same value (text.same_value), or the newer fact names no thing (other than people) that the older one does not
        already name."""
        names = [e["name"] for e in self.entities if e["kind"] == "person" and is_proper(e["name"])]
        names += [self.ent(f["subject"])["name"] for f in (old, new) if self.ent(f["subject"])["kind"] in ("person", "pet")]
        if same_value(old["text"], new["text"], names):
            return True
        things = lambda f: {x for x in [f["subject"], *f["mentions"]] if self.ent(x)["kind"] != "person"}  # noqa: E731
        a, b = things(old), things(new)
        return bool(b) and b <= a

    def _time(self, text, code, start, plan):
        """Time of a fact from the extractor's time code and the dates written in its text. A written date replaces the code
        only when the code says no more than "the day it was said" (S, or the session / memory day itself), when the text
        says the thing was moved ("rescheduled to the 20th": the date after the move), when the code is a weekday and a written
        date falls on that weekday ("Monday the 30th"), or when the code is a month or year and the written date lies in
        it. Otherwise the code is kept: "the week before December 21", "from Monday to Sunday the 30th", "4月9日说上周…" keep
        their codes. A written date that replaces a contradicting code is counted (date_code_overridden_by_text)."""
        wd = _weekday(text)
        tc = resolve_when(code, start, plan, wd) or resolve_free(code, start, plan)
        found = written_dates(text, start, plan, positions=True)
        c = str(code or "").strip()
        if not found:
            self.stats["when_resolved" if tc else ("when_unresolved" if c else "when_empty")] += 1
            return tc
        td = text_dates(text, start, plan)
        mv = moved_to(text, found)
        use = None
        if tc is None:
            use = td
        elif mv:
            use = (mv, mv, "day")
        elif c.upper() in ("S", "D") or (tc[2] == "day" and tc[0] == start):
            use = td
        elif _WDCODE.match(c):
            same = [d for _, d in found if d.weekday() == tc[0].weekday() and abs((d - tc[0]).days) <= 31]
            if same:
                d = min(same, key=lambda x: abs((x - tc[0]).days))
                use = (d, d, "day")
        elif tc[2] in ("month", "year") and tc[0] <= td[0] <= tc[1]:
            use = td
        elif abs((tc[0] - start).days) > 366 and all(abs((d - start).days) <= 366 for _, d in found):
            use = td                                         # a code years away beside a month-day written without a year
                                                             # ("Y+06:05-09" for 5月9日): the year of the code is wrong
        if use is None:
            self.stats["when_resolved"] += 1
            return tc
        self.stats["date_from_text"] += 1
        if tc and c.upper() not in ("S", "D") and tc[0] != start and not (tc[0] <= use[0] <= tc[1] or use[0] <= tc[0] <= use[1]):
            self.stats["date_code_overridden_by_text"] += 1
        return use

    # ---------------------------------------------------------------- one session
    def apply_session(self, sess, out, nums):
        """sess: session (case.load_case); out: its extraction {"entities", "facts", "relations"}; nums: T-number -> source
        turn number. Returns the ids of the new facts."""
        start = parse_date(sess["date"])
        for name, kind in sess["participants"].items():
            self.add_speaker(name, kind)
        speakers = {t["speaker"] for t in sess["turns"]}
        turn_speaker = {t["n"]: t["speaker"] for t in sess["turns"]}
        local, pending_edges, new = {}, [], []
        ents = [e for e in out.get("entities") or [] if isinstance(e, dict) and str(e.get("name", "")).strip()]
        self._cur_ents = ents
        ids = [str(x.get("id", "")) for x in ents]
        for rnd in range(2):                                 # second pass: "of" may name an entity defined later in the list
            for e in ents:
                lid = str(e.get("id", ""))
                if lid in local:
                    continue
                of_id = self._ref(e.get("of"), local)
                if of_id is None and str(e.get("of", "")).strip() in ids and rnd == 0:
                    continue
                kind = e.get("kind") if e.get("kind") in KINDS else "project"
                ks = self._ent_keys(e, of_id)
                role = norm(e.get("role", ""))
                hit = []                                     # every entity holding one of the keys, not only the first
                for k in ks:
                    for h in self.key_all.get(k, []):
                        if h not in hit and not self._other_person(k, h, role) \
                                and not (k[0] in ("role", "desc") and self._named_differently(e, h)):
                            hit.append(h)
                hit = [h for h in hit if compatible(self.ent(h)["kind"], kind)]
                if not hit and kind == "person" and is_proper(e["name"]):
                    hit = self._name_variant(str(e["name"]).strip(), role)
                hit.sort(key=lambda h: self.ent(h)["kind"] != kind)
                eid = hit[0] if hit else None
                if eid in self.spk.values() and kind != "person":
                    eid = None
                name = str(e["name"]).strip()[:80]
                if eid is None:
                    eid = self.add_entity(kind, name, desc=str(e.get("desc", "") or "")[:120])
                    self.stats["entities_new"] += 1
                else:
                    self.stats["entities_resolved"] += 1
                    E = self.ent(eid)
                    if eid not in self.spk.values() and is_proper(name) and (not is_proper(E["name"]) or (
                            E["kind"] == "person" and fuller_name(name, E["name"]))):   # keep the fuller name ("周子航")
                        E["aliases"] = list(dict.fromkeys(E["aliases"] + [E["name"]]))
                        E["name"] = name
                E = self.ent(eid)
                if eid not in self.spk.values():
                    for a in [name] + [x for x in e.get("aliases") or [] if isinstance(x, str)]:
                        a = str(a).strip()[:60]
                        if keep_alias(a) and a != E["name"] and a not in E["aliases"] and len(E["aliases"]) < 8:
                            E["aliases"].append(a)
                for k in ks:
                    self.keys.setdefault(k, eid)
                    if eid not in self.key_all.setdefault(k, []):
                        self.key_all[k].append(eid)
                if role_groups(role):
                    self.roles.setdefault(eid, set()).update(role_groups(role))
                local[lid] = eid
                rel = e.get("rel") if e.get("rel") in REL_LIST else None
                pending_edges.append((of_id, rel, eid))
        for of_id, rel, eid in pending_edges:
            src = of_id if of_id and of_id != eid else None
            if src is None:
                src = next((self.spk[t["speaker"]] for t in sess["turns"] if t["speaker"] in self.spk), "e0")
                rel = "custom.mentions"
            self.add_edge(src, rel or "custom.mentions", eid)
        for f in out.get("facts") or []:
            if not isinstance(f, dict) or not str(f.get("text", "")).strip():
                self.stats["fact_empty"] += 1
                continue
            subj = self._ref(f.get("about"), local)
            if subj is None:                                 # the extractor named a subject that is neither a participant nor
                self.stats["facts_dropped_unknown_subject"] += 1   # an entity of the session: the fact is left out
                continue
            kind = self.ent(subj)["kind"]
            mods = KIND_MODS.get(kind, ["notes.general"])
            mod = f.get("module") if f.get("module") in mods else "notes.general"
            self.stats["module_fixed"] += f.get("module") not in mods
            prov = []
            for x in f.get("turns") or []:
                try:
                    n = nums.get(int(re.sub(r"\D", "", str(x)) or -1))
                except ValueError:
                    n = None
                if n and n not in prov:
                    prov.append(n)
            if not prov:
                self.stats["fact_no_turns"] += 1
            st = f.get("state") if f.get("state") in STATES else "confirmed"
            ment = []
            for m in f.get("mentions") or []:
                x = self._ref(m, local)
                if x and x != subj and x not in self.spk.values():
                    ment.append(x)
            when = str(f.get("when", "") or "")[:20]
            text = re.sub(r"\s+", " ", str(f["text"]).strip())
            tm = self._time(text, when, start, st == "pending")
            sl = norm(f.get("slot", ""))[:40]
            fid = f"f{len(self.facts) + 1}"
            self.facts.append({"id": fid, "subject": subj, "module": mod, "text": text, "state": st, "mentions": sorted(set(ment),
                               key=lambda u: int(u[1:])), "slot": sl, "when": when, "time": tm, "session": sess["id"],
                               "observed_at": sess["date"], "prov": prov, "speakers": sorted({turn_speaker[n] for n in prov}),
                               "gen": "fact"})
            new.append(fid)
            self.stats["facts"] += 1
            if sl and st == "confirmed":                    # a newer value of a single-valued slot supersedes the older fact
                old = self.slots.get((subj, sl))
                if old:
                    of = self.facts[int(old[1:]) - 1]
                    if self.restates(of, self.facts[-1]):   # a restatement of the same value: both stay confirmed
                        self.stats["restated"] += 1
                    elif of["state"] != "superseded":
                        of["state"] = "superseded"
                        of["superseded_by"] = fid
                        self.stats["superseded"] += 1
                self.slots[(subj, sl)] = fid
        for r in out.get("relations") or []:
            if not isinstance(r, dict) or r.get("rel") not in REL_LIST:
                continue
            a, b = self._ref(r.get("src"), local), self._ref(r.get("dst"), local)
            if a and b and a != b:
                self.add_edge(a, r["rel"], b)
                self.stats["relations"] += 1
        return new


    # ---------------------------------------------------------------- memories another AI summarised
    def resolve(self, name, kind, aliases=(), of=None, rel=None, role=None):
        """Entity of an imported memory, or None for a new one: owner words; a unique exact name or alias; for a person, the
        unique known person whose name it can be (a first name of a full name, never two full names with different surnames,
        never a title taken for a first name); the known holder of a unique role ("mother of X") or the target of a unique
        relation (spouse_of). Two different proper names are never joined by a role or a relation."""
        n = norm(name)
        if n in self.owner_keys:
            return "e0"
        cands = [e for e in self.entities if compatible(e["kind"], kind)]
        word = lambda x: x and keep_alias(x) and norm(x) not in GENERIC  # noqa: E731  ("哥哥", "Mom": not names, never matched)
        keys = {norm(x) for x in [name, *aliases] if word(x)}
        hit = [e for e in cands if keys & {norm(x) for x in [e["name"], *e.get("aliases", [])] if word(x)}]
        if len(hit) > 1:                                     # an ambiguous alias: the one whose own name it is
            hit = [e for e in hit if norm(e["name"]) == n]
        if len(hit) == 1:
            return hit[0]["id"]
        if len(hit) > 1:
            return None
        proper = is_proper(name)
        if kind == "person" and proper:
            mine = role_groups(role)                         # a vet "Dr. Ana Huang" is not Tom's mother "Ana"
            hit = [e for e in cands if e["kind"] == "person" and e["id"] != "e0" and is_proper(e["name"])
                   and any(same_person_name(name, x) for x in [e["name"], *e.get("aliases", [])] if is_proper(x))
                   and roles_compatible(mine, self.roles.get(e["id"]))]
            if len(hit) == 1:
                return hit[0]["id"]
        target = None
        r = norm(role or "")
        if of and r in UNIQUE_ROLES:
            target = self.keys.get(("role", of, UNIQUE_ROLES[r]))
        if target is None and of and rel in UNIQUE_RELS:
            ts = [e["dst"] for e in self.edges if e["src"] == of and e["rel"] == rel and compatible(self.ent(e["dst"])["kind"], kind)]
            target = ts[0] if len(ts) == 1 else None
        if target is not None:
            T = self.ent(target)
            if proper and is_proper(T["name"]) and norm(name) not in {norm(x) for x in [T["name"], *T.get("aliases", [])]} \
                    and not (T["kind"] == "person" and same_person_name(name, T["name"])):
                return None
            return target
        return None

    def _by_name(self, name, ents, local):
        """A subject given by name instead of id: the one entity of the batch, else of the graph, with that exact name or
        alias; None when there is none or more than one."""
        n = norm(name)
        hit = {local[str(e.get("id", ""))] for e in ents if str(e.get("id", "")) in local
               and n in {norm(x) for x in [e.get("name", ""), *(e.get("aliases") or [])] if isinstance(x, str)}}
        if not hit:
            hit = {e["id"] for e in self.entities if n in {norm(x) for x in [e["name"], *e.get("aliases", [])]}}
        return hit.pop() if len(hit) == 1 else None

    def apply_memory(self, src, chunk, out):
        """One batch of memory items and its extraction. Facts are second-hand ("hearsay") unless the source says the user
        confirmed them; their value starts with the memory's date in brackets."""
        self.owner_keys = {norm(x) for x in ("owner", "user", "the user", "me", "myself", "用户", self.owner_name)}
        local = {"OWNER": "e0"}
        ents = [e for e in out.get("entities") or [] if isinstance(e, dict) and str(e.get("name", "")).strip()]
        for rnd in range(2):
            for e in ents:
                lid = str(e.get("id", ""))
                if lid in local:
                    continue
                of = local.get(str(e.get("of", "OWNER")))
                if of is None and rnd == 0:
                    continue
                of = of or "e0"
                kind = e.get("kind") if e.get("kind") in KINDS else "project"
                rel = e.get("rel") if e.get("rel") in REL_LIST else None
                aliases = [str(a) for a in e.get("aliases") or [] if isinstance(a, str) and keep_alias(a)]
                role = norm(e.get("role", ""))
                eid = self.resolve(e["name"], kind, aliases, of, rel, role)
                name = str(e["name"]).strip()[:80]
                if eid is None:
                    eid = self.add_entity(kind, name, [a[:60] for a in aliases if a != name][:8], str(e.get("desc", "") or "")[:120])
                    self.stats["memory_entities_new"] += 1
                else:
                    self.stats["memory_entities_resolved"] += 1
                    E = self.ent(eid)
                    if eid != "e0" and E["kind"] == "person" and fuller_name(name, E["name"]):   # keep the fuller name
                        E["aliases"] = list(dict.fromkeys(E["aliases"] + [E["name"]]))[:8]
                        E["name"] = name
                    if eid != "e0":
                        for a in [name, *aliases]:
                            if a and a != E["name"] and a not in E["aliases"] and len(E["aliases"]) < 8:
                                E["aliases"].append(a[:60])
                if role in UNIQUE_ROLES and of != eid:
                    self.keys.setdefault(("role", of, UNIQUE_ROLES[role]), eid)
                if role_groups(role) and eid != "e0":
                    self.roles.setdefault(eid, set()).update(role_groups(role))
                if rel and of != eid:
                    self.add_edge(of, rel, eid)
                local[lid] = eid
        seen = {(f["subject"], f["module"], f["text"], f["state"]) for f in self.facts if f["gen"] == "memory"}
        for f in out.get("facts") or []:
            if not isinstance(f, dict) or not str(f.get("text", "")).strip():
                continue
            nums = []
            for x in f.get("items") or []:
                try:
                    nums.append(int(re.sub(r"\D", "", str(x)) or 0))
                except ValueError:
                    pass
            ev = [chunk[k - 1] for k in nums if 1 <= k <= len(chunk)]
            if not ev:
                self.stats["memory_fact_no_item"] += 1
                continue
            about = str(f.get("about", "OWNER") or "OWNER").strip()
            subj = local.get(about) or ("e0" if norm(about) in self.owner_keys else self._by_name(about, ents, local))
            if subj is None:                                 # a subject the batch does not define (by id or by name): left out
                self.stats["facts_dropped_unknown_subject"] += 1
                continue
            kind = self.ent(subj)["kind"]
            mod = f.get("module") if f.get("module") in KIND_MODS.get(kind, ["notes.general"]) else "notes.general"
            d = max(x["date"] for x in ev)
            st = "pending" if f.get("state") == "pending" else ("confirmed" if src["confirmed"] else "hearsay")
            text = re.sub(r"\s+", " ", str(f["text"]).strip())
            if (subj, mod, text, st) in seen:
                self.stats["memory_fact_duplicate"] += 1
                continue
            seen.add((subj, mod, text, st))
            when = str(f.get("when", "") or "")[:20]
            code = "S" + when[1:] if when.startswith("D") else when
            start = parse_date(d)
            tm = self._time(text, code, start, st == "pending")
            ment = sorted({local[m] for m in f.get("mentions") or [] if m in local} - {subj}, key=lambda u: int(u[1:]))
            fid = f"f{len(self.facts) + 1}"
            self.facts.append({"id": fid, "subject": subj, "module": mod, "text": text, "value": f"[{d}] {text}", "state": st,
                               "mentions": ment, "slot": norm(f.get("slot", ""))[:40], "when": when, "time": tm,
                               "session": src["id"], "observed_at": d, "prov": [x["n"] for x in ev], "speakers": [],
                               "gen": "memory", "source": "import:" + src["assistant"]})
            self.stats["memory_facts"] += 1
        for r in out.get("relations") or []:
            if not isinstance(r, dict):
                continue
            a, b = local.get(str(r.get("src"))), local.get(str(r.get("dst")))
            if a and b and a != b and r.get("rel") in REL_LIST:
                self.add_edge(a, r["rel"], b)
