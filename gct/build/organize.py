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
"""Organisation: local rules over the assembled graph, iterated to a fixed point, that give it an event-centred structure.

  photos       photo / picture assets are dissolved; their facts move to the owner they hang from
  coreference  entities of compatible kind whose core names or aliases coincide are merged
  topic        a fact on a person that mentions a more specific entity (event / trip > project > possession > pet > place /
               organisation) moves there; the person becomes a mention
  occasions    facts on a person that describe the same occasion are grouped into an event / trip / appointment / project
               entity (one model call per person, the only call of this stage)
  split        an occasion with more than 15 facts is split into one occasion per date, each part_of the original
  attendance   an occasion with no attendee gets the participants its facts mention
  aliases      "<X>'s thing" gets the aliases "thing" and "the <head noun>" (X的东西 -> 东西)
  time         every fact value gets its date in brackets at the end
  supersession among confirmed facts of the same subject and slot the newer value supersedes the older
  retraction   a fact that cancels, removes, completes or stops something supersedes the earlier facts about it; an occasion
               that is called off loses its earlier plans, one that is moved its earlier (different) times
  schedule     every occasion gets a dated schedule fact
coreference, topic, split, attendance and aliases repeat until nothing changes; nothing is deleted except merged entities.
"""
import re
from collections import Counter, defaultdict

from . import prompts as P
from .assemble import kind_group
from .text import CJK, STOP, is_proper, is_title, norm, numbers, overlap, roles_compatible, same_value, tokens
from .timeres import _weekday, date_text, parse_date, resolve_when, text_dates, written_dates

CAP = 15
OCC_KINDS = {"event", "trip", "appointment", "project"}
TOPIC_RANK = {"event": 0, "trip": 0, "project": 1, "asset": 2, "document": 2, "vehicle": 2, "pet": 3, "place": 4, "organization": 4,
              "group": 4, "service": 4}
SCHEDULE = {"event": "event.schedule", "trip": "trip.calendar", "appointment": "appointment.schedule", "project": "project.schedule"}
PHOTO = re.compile(r"\b(photo|photos|photograph|picture|pictures|pic|pics|selfie|snapshot|image|screenshot)\b|照片|相片|图片|截图|自拍",
                   re.I)
RETRACT_EN = re.compile(r"\b(?:used to|no longer|not anymore|anymore|(?:has |had )?stopped|quit|dropped|gave up|got rid of|called off|call off|removed?|"
                        r"removing|deleted?|cancel+(?:ed|led|s|ing)?|cancellation|crossed off|took (?:it )?off|completed?|finished|"
                        r"is done with|done with|changed (?:their|his|her) mind|rescheduled|moved .{0,40} to|updated .{0,60} to|"
                        r"switched (?:from|to)|replaced|instead of|lost interest)\b", re.I)
RETRACT_ZH = re.compile(r"(不再|已经不|没再|取消|作废|退掉|退订|删掉|删除|去掉|划掉|改到|改成|改为|改期|推迟到|提前到|换成|放弃|戒掉|戒了|"
                        r"停掉|停了|不干了|已完成|完成了|做完了|办完了|结束了)")
RET_STOP = {"user", "longer", "anymore", "stopped", "quit", "dropped", "gave", "rid", "removed", "remove", "removal", "deleted",
            "delete", "deletion", "cancelled", "canceled", "cancel", "cancels", "cancellation", "crossed", "took", "taken", "completed",
            "complete", "finished", "done", "changed", "mind", "rescheduled", "moved", "updated", "switched", "replaced", "instead",
            "grew", "lost", "interest", "interested", "now", "need", "needs", "needed", "track", "tracking", "list", "todo", "do",
            "requested", "request", "wants", "want", "already", "just", "has", "had", "their", "they", "them", "its", "item", "items",
            "task", "tasks", "keep", "keeping", "considers", "thinks", "finds", "because", "since", "after", "new", "must", "be",
            "been", "will", "would", "plan", "plans", "planned"}
RESCHEDULE = re.compile(r"\b(?:rescheduled|re-scheduled|moved (?:it |the \w+ )?to|postponed|pushed (?:it )?back|brought forward|"
                        r"changed (?:the date |the time )?to)\b|改到|改至|改成|改为|改期|推迟到|推迟|提前到|延期到|延期", re.I)
COMPLETE = re.compile(r"\b(?:finished|completed?|is done with|done with|wrapped up)\b|完成了|已完成|做完了|办完了|结束了", re.I)
OTHER_RETRACT = re.compile(r"\b(?:no longer|not anymore|anymore|stopped|quit|dropped|gave up|got rid of|removed?|deleted?|cancel\w*|"
                           r"called off|changed (?:their|his|her) mind|instead of|switched|replaced|lost interest)\b|不再|取消|放弃|戒|停", re.I)
ONGOING = re.compile(r"\b(?:is|are|am|was|were|keeps?|been|still)\s+\w+ing\b|\b(?:underway|in progress|ongoing|working on)\b|正在|进行中|还在",
                     re.I)
# phrases with a retraction word that do not retract anything: "stopped by the bakery", "dropped her daughter off at school",
# "took the day off"
NOT_RETRACT = re.compile(r"\bstop(?:ped|s)? (?:by|in|at)\b|\bdrop(?:ped|s)?\s+(?:\w+\s+){0,3}off\b|\bdrop-?off\b|"
                         r"\btook\s+(?:the\s+|a\s+|some\s+|two\s+|\w+\s+)?(?:day|days|week|weeks|morning|afternoon|evening|time|weekend)\s+off\b|"
                         r"\btook off (?:work|from)\b", re.I)
TIMEWORD = re.compile(r"^(?:last|next|this|week|weeks|weekend|month|months|year|years|day|days|today|tomorrow|yesterday|tonight|morning|"
                      r"afternoon|evening|night|ago|recently|soon|earlier|later|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
                      r"january|february|march|april|may|june|july|august|september|october|november|december|"
                      r"上周|下周|本周|这周|昨天|今天|明天|前天|后天|上个|下个|个月|去年|今年|明年|周末|早上|晚上|下午|上午|中午|星期|礼拜)$")
TIME_HEAD = re.compile(r"\s*(?:(?:on|for|until|till|from)\s+)?(?:\d|the\s+\d|next\b|tomorrow|later\b|earlier\b|"
                       r"(?:mon|tues|wednes|thurs|fri|satur|sun)day|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b)|"
                       r"\s*(?:\d|下|明天|后天|周|星期|礼拜|本周|这周|月底|月初)", re.I)
HEDGE = re.compile(r"\b(?:might|(?-i:may)|could|would|thinking (?:about|of)|considering|consider|plans? to|wants? to|hopes? to|"
                   r"asked (?:whether|if)|whether|if)\b|可能|也许|或许|打算|考虑|想要|想把|要不要|是否|如果", re.I)
CANCEL = re.compile(r"\b(?:cancel+ed|called off|scrapped)\b|(?<!可以)(?<!免费)(?<!能)取消了|已取消|被取消|作废|不去了|泡汤", re.I)
OBJ_EN = [r"\b(?:instead of|switched from|replaced) (?P<o>.+?)(?: with | by |$)",
          r"\bno longer (?:wants to|plans(?: to)?|enjoys|likes|loves|is (?:interested in|into)|needs)?\s*(?P<o>.+)$",
          r"\b(?:removing|removes?|removed|deleting|deleted?|cancel+(?:ed|led|ing|s)?|crossed off|took off)\s+(?P<o>.+?)(?: from |$)",
          r"\b(?:finished|completed|is done with|done with|stopped|quit|gave up|got rid of|lost interest in|dropped)\s+(?P<o>.+)$"]


class World:
    def __init__(self, g):
        self.g = g
        self.E = {e["id"]: e for e in g.entities}
        self.F = {f["id"]: f for f in g.facts}
        self.part = [g.spk[n] for n in g.spk if self.E[g.spk[n]]["kind"] == "person"]
        self.dead = set()
        self.new_e = 0
        self.role = {}
        self.ca_alias = set()
        self.stats = Counter()
        self.first = "e0"

    def kind(self, eid):
        return self.E[eid]["kind"]

    def facts_of(self, eid):
        return [f for f in self.F.values() if f["subject"] == eid]

    def alive(self):
        return [e for e in list(self.E.values()) if e["id"] not in self.dead]

    def add_entity(self, kind, name, desc="", **extra):
        eid = f"x{self.new_e}"
        self.new_e += 1
        self.E[eid] = {"id": eid, "kind": kind, "name": name, "aliases": [], "desc": desc, **extra}
        return eid

    def add_edge(self, src, rel, dst):
        return self.g.add_edge(src, rel, dst)

    def move_fact(self, f, dst, why):
        if f["subject"] == dst:
            return False
        old = f["subject"]
        f["subject"] = dst
        m = [x for x in f.get("mentions") or [] if x != dst]
        if old in self.part or self.kind(old) == "person":
            m.append(old)
        f["mentions"] = sorted(set(m), key=str)
        f["module"] = f["module"] if f["module"] in P.KIND_MODS.get(self.kind(dst), []) else "notes.general"
        self.stats[why] += 1
        return True

    def repoint(self, old, new):
        for f in self.F.values():
            if f["subject"] == old:
                f["subject"] = new
            if old in (f.get("mentions") or []):
                f["mentions"] = sorted(set(new if x == old else x for x in f["mentions"]) - {f["subject"]}, key=str)
        edges, keys = [], set()
        for e in self.g.edges:
            e["src"] = new if e["src"] == old else e["src"]
            e["dst"] = new if e["dst"] == old else e["dst"]
            k = (e["src"], e["rel"], e["dst"])
            if e["src"] != e["dst"] and k not in keys:
                keys.add(k)
                edges.append(e)
        self.g.edges, self.g.edge_keys = edges, keys

    def kill(self, eid):
        self.dead.add(eid)
        self.g.edges = [e for e in self.g.edges if eid not in (e["src"], e["dst"])]
        self.g.edge_keys = {(e["src"], e["rel"], e["dst"]) for e in self.g.edges}
        for f in self.F.values():
            if eid in (f.get("mentions") or []):
                f["mentions"] = [x for x in f["mentions"] if x != eid]


def _ord(eid):
    return (eid[0] != "e", int(eid[1:]))


def core_name(name, owners):
    """(possessor index or -1, name without the possessive owner prefix): "Maya's pottery class" -> (0, "pottery class")."""
    n = norm(name)
    for i, o in enumerate(owners):
        no = norm(o)
        for p in (no + "s ", no + "的"):
            if n.startswith(p) and len(n) > len(p):
                return i, n[len(p):].strip()
    return -1, n


def r_photos(W):
    n = 0
    for e in W.alive():
        if e["kind"] != "asset" or e["id"] in W.part or not PHOTO.search(e["name"] + " " + e.get("desc", "")):
            continue
        who, _ = core_name(e["name"], [W.E[p]["name"] for p in W.part])
        owners = ([W.part[who]] if who >= 0 else []) + [x["src"] for x in W.g.edges if x["dst"] == e["id"] and x["src"] not in W.dead]
        dst = next((o for o in owners if W.kind(o) != "asset"), W.first)
        for f in W.facts_of(e["id"]):
            W.move_fact(f, dst, "photo_fact_moved")
        W.kill(e["id"])
        n += 1
    W.stats["photos"] += n
    return n


def r_coref(W):
    owners = [W.E[p]["name"] for p in W.part]
    cand = [e for e in W.alive() if e["id"] not in W.part]
    parent = {e["id"]: e["id"] for e in cand}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def grp(k):
        return kind_group(k) if k != "person" else "person"

    def ok(c):
        return (len(c) >= 3 or (re.search(rf"[{CJK}]", c) and len(c) >= 2)) and c not in STOP

    names = defaultdict(list)
    for e in cand:
        who, c = core_name(e["name"], owners)
        if ok(c) and (who >= 0 or is_proper(e["name"])) and not is_title(c):
            names[(grp(e["kind"]), who, c)].append(e["id"])
    links = [(ids, len(k[2].split()) == 1 and k[2].isascii() and k[1] < 0) for k, ids in names.items()]
    for e in cand:
        who0, _ = core_name(e["name"], owners)
        for al in e.get("aliases") or []:
            if (e["id"], al) in W.ca_alias:
                continue
            who, c = core_name(al, owners)
            if who < 0 and re.match(r"^(my|our)\s|^(我的|我们的|我)", al.strip().lower()):
                who = who0
                c = norm(al)
            if who < 0 and not is_proper(al):
                continue
            hit = names.get((grp(e["kind"]), who, c))
            if hit and ok(c) and not is_title(c):
                links.append(([e["id"]] + hit, len(c.split()) == 1 and c.isascii() and who < 0))
    roles = getattr(W.g, "roles", {})

    def clash(a, b):                                         # two people with incompatible known roles are not merged
        return W.kind(a) == "person" and W.kind(b) == "person" and not roles_compatible(roles.get(a), roles.get(b))
    for ids, bare in links:                                 # bare: matched on a single first name only
        for x in ids[1:]:
            if bare and all(clash(y, x) for y in ids if y != x):   # compatible with any member of the group is enough
                W.stats["merge_blocked_roles"] += 1
                continue
            a, b = find(ids[0]), find(x)
            if a != b:
                parent[max(a, b, key=_ord)] = min(a, b, key=_ord)
    groups = defaultdict(list)
    for e in cand:
        groups[find(e["id"])].append(e["id"])
    n = 0
    for root, ids in groups.items():
        if len(ids) < 2:
            continue
        for x in ids:
            if x != root:
                merge_into(W, root, x)
                n += 1
    W.stats["merged"] += n
    return n


def merge_into(W, root, x):
    R, X = W.E[root], W.E[x]
    for a in [X["name"]] + list(X.get("aliases") or []):
        if a and a != R["name"] and a not in R["aliases"] and len(R["aliases"]) < 10:
            R["aliases"].append(a)
    if len(X.get("desc", "")) > len(R.get("desc", "")):
        R["desc"] = X["desc"]
    if X.get("date") and not R.get("date"):
        R["date"] = X["date"]
    roles = getattr(W.g, "roles", None)
    if roles is not None and x in roles:
        roles.setdefault(root, set()).update(roles.pop(x))
    W.repoint(x, root)
    W.dead.add(x)


def r_topic(W):
    n = 0
    for f in list(W.F.values()):
        s = f["subject"]
        if W.kind(s) != "person" or f["gen"] != "fact":
            continue
        cands = [x for x in f.get("mentions") or [] if x in W.E and x not in W.dead and W.kind(x) in TOPIC_RANK]
        if not cands:
            continue
        top = min(cands, key=lambda x: (TOPIC_RANK[W.kind(x)], (f.get("mentions") or []).index(x)))
        n += W.move_fact(f, top, "topic_moved")
    return n


def r_aliases(W):
    owners = [W.E[p]["name"] for p in W.part]
    n = 0
    for e in W.alive():
        if e["id"] in W.part or e["kind"] == "person" or e.get("split"):
            continue
        who, c = core_name(e["name"], owners)
        if who < 0 or not c:
            continue
        al = [c]
        if not re.search(rf"[{CJK}]", c):
            head = re.split(r"\s(?:on|in|at|for|with|from|to|of|about|during|by)\s", " " + c + " ")[0].split()
            if head and head[-1].isalpha() and len(head[-1]) >= 3 and head[-1] not in STOP:
                al.append("the " + head[-1])
        for a in al:
            if a.lower() != e["name"].lower() and a not in e["aliases"] and len(e["aliases"]) < 10:
                e["aliases"].append(a)
                W.ca_alias.add((e["id"], a))
                n += 1
    W.stats["aliases"] += n
    return n


def r_attend(W):
    has = {e["dst"] for e in W.g.edges if e["rel"] in ("attends", "leads", "organizes")}
    n = 0
    for e in W.alive():
        if e["kind"] not in OCC_KINDS or e["id"] in has:
            continue
        ps = {m for f in W.facts_of(e["id"]) for m in f.get("mentions") or [] if m in W.part}
        for p in sorted(ps):
            n += W.add_edge(p, "leads" if e["kind"] == "project" else "attends", e["id"])
    W.stats["attend_edges"] += n
    return n


def r_split(W):
    n = 0
    for e in W.alive():
        if e["kind"] not in OCC_KINDS or e.get("split") or e.get("split_parent"):
            continue
        fs = [f for f in W.facts_of(e["id"]) if f["gen"] == "fact"]
        if len(fs) <= CAP:
            continue
        by = defaultdict(list)
        for f in fs:
            by[(f["time"][0].isoformat() if f.get("time") else f["observed_at"])[:10]].append(f)
        if len(by) < 2:
            continue
        e["split_parent"] = True
        for d, group in sorted(by.items()):
            dd = parse_date(d)
            nm = f"{e['name']}: {date_text(dd, lang=W.g.date_lang) or d}"
            occ = W.add_entity("event" if e["kind"] == "project" else e["kind"], nm[:90], desc=f"one occasion of {e['name']}"[:120],
                               date=d, split=True)
            W.add_edge(occ, "part_of", e["id"])
            for p in sorted({p for f in group for p in (f.get("mentions") or []) if p in W.part} or {W.first}):
                W.add_edge(p, "attends", occ)
            for f in group:
                W.move_fact(f, occ, "split_moved")
            n += 1
    W.stats["split_occasions"] += n
    return n


def r_occasions(W, pid, llm):
    """Group the facts on subject pid into occasions with one model call (chunks of 160 facts)."""
    fs = sorted([f for f in W.facts_of(pid) if f["gen"] == "fact"], key=lambda f: int(f["id"][1:]))
    if not fs:
        return
    ex = sorted([e for e in W.alive() if e["kind"] in OCC_KINDS], key=lambda e: e["name"])
    xlab = {f"X{j}": e["id"] for j, e in enumerate(ex, 1)}
    existing = "\n".join(f"{k} {W.E[v]['name']} ({W.E[v]['kind']})" for k, v in xlab.items()) or "(none)"
    groups = []
    for lo in range(0, len(fs), 160):
        sub = {f"F{k + 1}": f["id"] for k, f in enumerate(fs[lo:lo + 160])}
        lines = [f"{lab} [{(W.F[fid]['time'][0].isoformat() if W.F[fid].get('time') else W.F[fid]['observed_at'])}] {W.F[fid]['text']}"
                 for lab, fid in sub.items()]
        prompt = P.GROUP.format(who=W.E[pid]["name"], facts="\n".join(lines), existing=existing)
        if P.LANG_NOTE.get(W.g.lang):
            prompt += "\n\n" + P.LANG_NOTE[W.g.lang]

        def chk(o):
            if not isinstance(o.get("groups"), list):
                raise ValueError("'groups' is not a list")
        out = llm("group", prompt, check=chk)
        if out is None:
            W.stats["group_failed"] += 1
            continue
        W.stats["group_calls"] += 1
        for same in out.get("same") or []:                   # existing occasions that are one occasion
            ids = sorted({xlab[x] for x in same if isinstance(x, str) and x in xlab and xlab[x] not in W.dead
                          and W.kind(xlab[x]) in OCC_KINDS}, key=_ord) if isinstance(same, list) else []
            for x in ids[1:]:
                merge_into(W, ids[0], x)
                W.stats["occasions_merged"] += 1
            for lab in list(xlab):
                if xlab[lab] in ids[1:]:
                    xlab[lab] = ids[0]
        for gr in out["groups"]:
            if not isinstance(gr, dict):
                continue
            fl = gr.get("facts") or {}
            roles = fl if isinstance(fl, dict) else {x: "" for x in fl if isinstance(x, str)}
            ids = [(sub[x], str(r)) for x, r in roles.items() if x in sub]
            if ids:
                groups.append({**gr, "facts": ids, "existing": xlab.get(str(gr.get("existing", "")), "")})
    kind = W.kind(pid)
    used = set()
    for gr in groups:
        ids = [(x, r) for x, r in gr["facts"] if x not in used and x in W.F and W.F[x]["subject"] == pid]
        if not ids:
            continue
        used.update(x for x, _ in ids)
        if gr.get("existing") and gr["existing"] in W.E and gr["existing"] not in W.dead:
            occ = gr["existing"]
            W.stats["to_existing_occasion"] += 1
        else:
            k = gr.get("kind") if gr.get("kind") in OCC_KINDS else "event"
            nm = str(gr.get("name") or "").strip()[:80] or (f"{W.E[pid]['name']}'s " + " ".join(sorted(
                tokens(W.F[ids[0][0]]["text"]), key=len, reverse=True)[:3]))
            d = str(gr.get("date") or "")
            d = d if re.fullmatch(r"\d{4}(-\d{2}(-\d{2})?)?", d) else ""
            occ = W.add_entity(k, nm, desc=f"occasion grouped from {len(ids)} fact(s)", date=d)
            W.stats["new_occasions"] += 1
        people = {pid} if kind in ("person", "group", "household", "pet") else set()
        people |= {m for x, _ in ids for m in W.F[x].get("mentions") or [] if m in W.part}
        for q in sorted(people or {W.first}):
            W.add_edge(q, "leads" if W.kind(occ) == "project" and W.kind(q) == "person" else "attends", occ)
        if kind == "place":
            W.add_edge(occ, "located_at", pid)
        elif kind not in ("person", "group", "household", "pet"):
            W.add_edge(occ, "custom.mentions", pid)
        for x, _ in ids:
            W.move_fact(W.F[x], occ, "fact_to_occasion")


def _span_text(d, lang):
    """'2023-05-07' / '2023-05' / '2023' -> date text."""
    m = re.fullmatch(r"(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?", d or "")
    if not m:
        return ""
    t = resolve_when(d, parse_date("2000-01-01"))
    return date_text(*t, lang=lang) if t else ""


def r_values(W):
    """value = text + ' (date)', the date being the time the fact refers to, else the date of its source."""
    lang = W.g.date_lang
    for f in W.F.values():
        if f["gen"] != "fact":
            continue
        d = date_text(*f["time"], lang=lang) if f.get("time") else date_text(parse_date(f["observed_at"]), lang=lang)
        f["value"] = f"{f['text'].rstrip('.')} ({d})" if d else f["text"]


def r_slots(W):
    """Every occasion gets a dated schedule fact (unless it has one): the occasion's date from the grouping, else the latest
    still-valid schedule fact, else a plan, else any dated fact, else the date of its first source."""
    lang = W.g.date_lang
    n = 0
    for e in W.alive():
        if e["kind"] not in OCC_KINDS:
            continue
        mod = SCHEDULE[e["kind"]]
        if any(f["module"] == mod for f in W.facts_of(e["id"])):
            continue
        fs = sorted([f for f in W.facts_of(e["id"]) if f["gen"] == "fact"], key=lambda f: f["observed_at"])
        if not fs:
            continue
        cancels = [f for f in fs if f["state"] == "confirmed" and _cancels(W, f, e)]
        if cancels:                                          # called off, and not planned again afterwards: no current date
            last = max(cancels, key=lambda f: (f["observed_at"], int(re.sub(r"\D", "", f["id"]) or 0)))
            if not any(f.get("time") and f["state"] != "superseded" and not CANCEL.search(f["text"]) and _earlier(last, f)
                       for f in fs):
                W.stats["schedule_skipped_cancelled"] += 1
                continue
            fs = [f for f in fs if f not in cancels]

        def iso(f):
            return f["time"][0].isoformat()[:{"day": 10, "month": 7}.get(f["time"][2], 4)]
        dated = [f for f in fs if f.get("time") and f["state"] != "superseded"] or [f for f in fs if f.get("time")]
        sched = [f for f in dated if f["module"].split(".")[-1] in ("schedule", "calendar", "itinerary")][::-1]
        plans = [f for f in dated if f["state"] == "pending"]
        d = e.get("date") if not sched else ""
        d = d or next((iso(f) for f in sched + plans + dated), "") or fs[0]["observed_at"]
        val = e["name"] if e.get("split") else f"{e['name']}: {_span_text(d, lang) or d}"
        st = fs[0]["state"]
        if st == "superseded":
            st = next((f["state"] for f in fs if f["state"] != "superseded"), "confirmed")
        fid = f"y{len(W.F)}"
        W.F[fid] = {"id": fid, "subject": e["id"], "module": mod, "value": val, "text": val, "state": st, "mentions": [],
                    "slot": "date", "when": "", "time": resolve_when(d, parse_date("2000-01-01")), "session": fs[0]["session"],
                    "observed_at": fs[0]["observed_at"], "prov": list(fs[0]["prov"]), "speakers": [], "gen": "slot"}
        n += 1
    W.stats["schedule_facts"] += n


def r_supersede(W):
    by = defaultdict(list)
    for f in W.F.values():
        if f.get("slot") and f["gen"] == "fact" and f["state"] in ("confirmed", "superseded"):
            by[(f["subject"], f["slot"])].append(f)
    n = 0
    for fs in by.values():
        fs.sort(key=lambda f: (f["observed_at"], int(f["id"][1:])))
        names = [e["name"] for e in W.alive() if e["kind"] == "person" and is_proper(e["name"])]
        if W.kind(fs[0]["subject"]) in ("person", "pet"):             # a place's or thing's own name is part of the value
            names.append(W.E[fs[0]["subject"]]["name"])
        for old, new in zip(fs, fs[1:]):
            things = lambda f: {x for x in [f["subject"], *f.get("mentions", [])] if W.kind(x) != "person"}  # noqa: E731
            same = same_value(old["text"], new["text"], names) or (things(new) and things(new) <= things(old))
            if not same and old["state"] == "confirmed":
                old["state"] = "superseded"
                old["superseded_by"] = new["id"]
                n += 1
    W.stats["superseded_slot"] += n


def _retract_object(text):
    if CANCEL.search(text):                                  # "X's Friday train for his Toronto trip was cancelled": the train
        o = _cancel_object(text)
        if o:
            return o
    if re.search(rf"[{CJK}]", text):
        m = RETRACT_ZH.search(text)
        after = re.split(r"[，。；,;！!？?]", text[m.end():])[0] if m else ""
        before = re.split(r"[，。；,;！!？?]", text[:m.start()])[-1] if m else ""
        return after if len(after) >= 2 else before
    t = text
    for pat in OBJ_EN:
        m = re.search(pat, t, re.I)
        if m and m.group("o").strip():
            return re.sub(r"\s+(?:because|since|as|so|and now|but now|but|which|while|after|now that)\s+.*$", "",
                          m.group("o").strip(), flags=re.I)
    return t


_CUT_EN = re.compile(r"\s+(?:for|of|at|from|with|on|in|to|by|because|since|due|and|but|after|so|until|as|which|that)\b|[,.;:!?]", re.I)
_LEAD_EN = re.compile(r"^\s*(?:the|a|an|my|our|his|her|their|its|this|that)\s+", re.I)
_AUX_EN = re.compile(r"\s+(?:was|were|is|are|has been|have been|had been|got|has|have|had|been|be)\s*$", re.I)


def _cancel_object(text):
    """What a cancellation names, as its head noun phrase: the object after the verb, or with nothing after the verb the
    subject before it ("The party was called off"), cut at the first preposition or clause word and without a possessor:
    "cancelled the hotel breakfast add-on for the trip" -> "hotel breakfast add-on", "Mateo's Friday train for his Toronto trip
    was cancelled" -> "Friday train". In Chinese the head after the last 的: 取消了厦门酒店的接机服务 -> 接机服务,
    去多伦多的周五火车取消了 -> 周五火车."""
    m = CANCEL.search(text)
    if not m:
        return ""
    if re.search(rf"[{CJK}]", m.group(0)):
        after = re.split(r"[，。；,;！!？?了]", text[m.end():])[0].strip()
        if len(after) >= 2:
            return after.split("的")[-1]
        before = re.split(r"[，。；,;！!？?]", text[:m.start()])[-1].strip()
        return re.sub(r"(已经|已|被|又|也|都|就)$", "", before).split("的")[-1]

    def head(phrase):
        phrase = _CUT_EN.split(phrase, maxsplit=1)[0]
        phrase = re.sub(r"^.*['’]s\s+", "", _LEAD_EN.sub("", phrase).strip())       # "Mateo Varga's Friday train" -> "Friday train"
        return _LEAD_EN.sub("", phrase).strip()
    after = head(text[m.end():])
    if after and after.lower() not in ("it", "them", "everything", "all"):
        return after
    before = re.split(r"[,.;:!?]", text[:m.start()])[-1]
    return head(_AUX_EN.sub("", before))


def _cancels(W, f, e):
    """A confirmed fact calls off the occasion e itself, not a part of it: what it cancels shares a word with the occasion's
    name ("Grace called off the bridal shower"), unlike "cancelled the hotel breakfast add-on for the Vancouver trip"."""
    if not CANCEL.search(f["text"]):
        return False
    owners = [W.E[p]["name"] for p in W.part]
    _, core = core_name(re.sub(r":\s*[^:]*$", "", e["name"]), owners)
    stop = set()
    for o in owners:
        stop |= tokens(o)
    obj = tokens(_cancel_object(f["text"])) - stop - RET_STOP
    return bool(obj & (tokens(core) - stop))


def _time_move(txt):
    """The fact moves something in time ("rescheduled", "moved to March 27", "改到4月28日"), not to another place."""
    m = RESCHEDULE.search(txt)
    if not m:
        return False
    if re.search(r"rescheduled|re-scheduled|postponed|pushed|brought forward|改期|推迟|提前|延期", m.group(0), re.I):
        return True
    return bool(TIME_HEAD.match(txt, m.end()))


_CLOCK = re.compile(r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)|\b\d{1,2}:\d{2}\b|\d{1,2}点", re.I)


def _says_when(f):
    """The fact's own words say when (a date, a weekday or a clock time), whatever its module: "is on March 18 at 8 am"."""
    return bool(written_dates(f["text"], parse_date(f["observed_at"])) or _weekday(f["text"]) is not None or _CLOCK.search(f["text"]))


def _stated_time(f):
    """The fact names a time of its own (a time code other than the day it was said, or a date in its text)."""
    return str(f.get("when") or "").strip() not in ("", "S") or bool(text_dates(f["text"], parse_date(f["observed_at"])))


def _earlier(t, r):
    """Did fact t come before fact r? An earlier source, or an earlier turn of the same source, or (same day, two sources)
    the source applied first."""
    if t["observed_at"] != r["observed_at"]:
        return t["observed_at"] < r["observed_at"]
    if t["session"] == r["session"]:
        return bool(t.get("prov") and r.get("prov")) and max(t["prov"]) < min(r["prov"])
    return int(re.sub(r"\D", "", t["id"]) or 0) < int(re.sub(r"\D", "", r["id"]) or 0)


def r_retract(W, thr=0.6):
    owner_words = set()
    for p in W.part:
        owner_words |= tokens(W.E[p]["name"])
    facts = sorted([f for f in W.F.values() if f["gen"] == "fact"], key=lambda f: (f["observed_at"], int(f["id"][1:])))
    n_ret = n_sup = 0
    for i, r in enumerate(facts):
        if r["state"] != "confirmed":                        # only something that did happen retracts: not a plan or a rumour
            continue
        txt = NOT_RETRACT.sub(" ", r["text"])
        if not (RETRACT_EN.search(txt) or RETRACT_ZH.search(txt)):
            continue
        obj = {t for t in tokens(_retract_object(txt)) if t not in RET_STOP and t not in owner_words and not t.isdigit()
               and not re.fullmatch(rf"[{CJK}]", t) and not TIMEWORD.match(t)}
        if not obj:
            continue
        n_ret += 1
        need = 1 if len(obj) == 1 else 2
        completion = bool(COMPLETE.search(txt)) and not OTHER_RETRACT.search(txt)   # "finished X" ends plans and ongoing work only
        moved = _time_move(txt) and not OTHER_RETRACT.search(txt)   # a new date makes only statements of a date obsolete
        cancel_day = written_dates(r["text"], parse_date(r["observed_at"])) if CANCEL.search(txt) else []
        scope = {r["subject"]} | set(r.get("mentions") or [])
        for t in facts[:i]:
            if t["observed_at"] >= r["observed_at"] or t["state"] not in ("confirmed", "pending"):
                continue
            if t["subject"] not in scope and r["subject"] not in set(t.get("mentions") or []):
                continue
            if RETRACT_EN.search(t["text"]) or RETRACT_ZH.search(t["text"]):
                continue
            if (t.get("time") and t["time"] == r.get("time")) or (numbers(t["text"]) and numbers(t["text"]) == numbers(r["text"])):
                continue                                     # the same date or the same numbers: a restatement, not a retraction
            if completion and t["state"] != "pending" and not ONGOING.search(t["text"]):
                continue
            if moved and not (t.get("time") and _stated_time(t)):
                continue
            if cancel_day and t.get("time") and _stated_time(t) and not any(t["time"][0] <= d <= t["time"][1] for d in cancel_day):
                continue                                     # "the flight on the 24th was cancelled" is not the flight on April 2
            share = obj & tokens(t["text"])
            if len(share) >= need and len(share) / len(obj) >= thr:
                t["state"] = "superseded"
                t["superseded_by"] = r["id"]
                n_sup += 1
    for r in facts:                                          # an occasion that is called off: its earlier plans no longer hold
        if W.kind(r["subject"]) not in OCC_KINDS or r["state"] != "confirmed" or not _cancels(W, r, W.E[r["subject"]]):
            continue
        day = parse_date(r["observed_at"])
        for t in W.facts_of(r["subject"]):
            if t is r or t["gen"] not in ("fact", "slot") or CANCEL.search(t["text"]) or not _earlier(t, r):
                continue
            future = t["state"] == "confirmed" and t.get("time") and day and t["time"][1] >= day   # an arrangement still ahead
            if t["state"] == "pending" or future:
                t["state"] = "superseded"
                t["superseded_by"] = r["id"]
                n_sup += 1
    for r in facts:                                          # an occasion moved to another time: its earlier times no longer hold
        if W.kind(r["subject"]) not in OCC_KINDS or not RESCHEDULE.search(r["text"]) or not r.get("time"):
            continue
        if r["state"] not in ("confirmed", "pending") or HEDGE.search(r["text"]):   # a move ahead is pending; "might move" is none
            continue
        for t in W.facts_of(r["subject"]):
            if t is not r and t["gen"] == "fact" and t["state"] in ("confirmed", "pending") and _earlier(t, r) \
                    and t.get("time") and t["time"] != r["time"] and _stated_time(t) \
                    and not (t["time"][0] <= r["time"][0] and r["time"][1] <= t["time"][1]) \
                    and (t["module"].split(".")[-1] in ("schedule", "calendar", "itinerary", "status") or _says_when(t)):
                t["state"] = "superseded"
                t["superseded_by"] = r["id"]
                n_sup += 1
    W.stats["retractions"] += n_ret
    W.stats["superseded_retract"] += n_sup


def organize(g, llm, occasions=True, max_iter=8, log=print):
    """Organise the assembled graph g in place; returns the rule statistics."""
    W = World(g)
    r_photos(W)
    r_coref(W)
    r_topic(W)
    if occasions:
        for p in sorted(W.part, key=lambda x: W.E[x]["name"]):
            r_occasions(W, p, llm)
        hubs = sorted([e["id"] for e in W.alive() if e["id"] not in W.part and e["kind"] not in OCC_KINDS
                       and sum(f["gen"] == "fact" for f in W.facts_of(e["id"])) > CAP], key=lambda x: W.E[x]["name"])
        for h in hubs:
            r_occasions(W, h, llm)
    it = 0
    for it in range(1, max_iter + 1):
        if r_coref(W) + r_topic(W) + r_split(W) + r_attend(W) + r_aliases(W) == 0:
            break
    W.stats["fixpoint_iterations"] = it
    r_values(W)
    r_supersede(W)
    r_retract(W)
    r_slots(W)
    # write back: alive entities (new ones get e-ids after the existing ones), facts sorted by time
    ents = [e for e in g.entities if e["id"] not in W.dead] + [e for e in W.E.values() if e["id"][0] == "x" and e["id"] not in W.dead]
    emap = {e["id"]: f"e{i}" for i, e in enumerate(ents)}
    facts = sorted([f for f in W.F.values() if f["subject"] not in W.dead],
                   key=lambda f: (f["observed_at"], f["id"][0] != "f", int(f["id"][1:])))
    fmap = {f["id"]: f"f{i + 1}" for i, f in enumerate(facts)}
    for e in ents:
        e["id"] = emap[e["id"]]
        for k in ("date", "split", "split_parent"):
            e.pop(k, None)
    for f in facts:
        f["id"] = fmap[f["id"]]
        f["subject"] = emap[f["subject"]]
        f["mentions"] = sorted({emap[x] for x in f.get("mentions") or [] if x in emap} - {f["subject"]}, key=lambda u: int(u[1:]))
        if f.get("superseded_by"):
            f["superseded_by"] = fmap.get(f["superseded_by"])
            if not f["superseded_by"]:
                f.pop("superseded_by")
    edges, keys = [], set()
    for e in g.edges:
        if e["src"] in emap and e["dst"] in emap:
            k = (emap[e["src"]], e["rel"], emap[e["dst"]])
            if k[0] != k[2] and k not in keys:
                keys.add(k)
                edges.append({"src": k[0], "rel": k[1], "dst": k[2], "state": "cur"})
    g.entities, g.facts, g.edges, g.edge_keys = ents, facts, edges, keys
    g.spk = {n: emap[e] for n, e in g.spk.items() if e in emap}
    log("organize: " + ", ".join(f"{k} {v}" for k, v in sorted(W.stats.items()) if v))
    return dict(W.stats)
