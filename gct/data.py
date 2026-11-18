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
"""Request-independent tensors of a compiled view (unit vectors, token memories, keys, lexical and time features)."""
import hashlib
import json
import re
from collections import OrderedDict
from pathlib import Path

import torch

from .compile import compile_view, query_text, vocab

STATUS = ["resolved", "partial", "missing", "ambiguous", "no_relevant"]


def module_list():
    return list(vocab()["modules"].keys())


def module_texts():
    v = vocab()["modules"]
    return [f"资料：{v[m]['label']}" for m in module_list()]


_VIEW_CACHE, _VIEW_BYTES = OrderedDict(), [0]


def graph_tensors(cg, cache, rel_cache, device, tokp=None):
    budget = int(float("0") * (1 << 30))
    if budget <= 0:
        return _graph_tensors(cg, cache, rel_cache, device, tokp)
    key = (id(cache), id(rel_cache), cg.world_id, cg.caller, cg.n, hash(tuple(cg.unit_text)), hash(tuple(cg.rel_text)),
           hash(tuple(cg.mod_text)), len(cg.src))
    st = _VIEW_CACHE.get(key)
    if st is None:
        G = _graph_tensors(cg, cache, rel_cache, device, None)
        st = {k: (v.half().cpu() if k == "E" else v.cpu()) if torch.is_tensor(v) else v for k, v in G.items()}
        _VIEW_CACHE[key] = st
        _VIEW_BYTES[0] += sum(v.numel() * v.element_size() for v in st.values() if torch.is_tensor(v))
        while len(_VIEW_CACHE) > 1 and _VIEW_BYTES[0] > budget:
            _, old = _VIEW_CACHE.popitem(last=False)
            _VIEW_BYTES[0] -= sum(v.numel() * v.element_size() for v in old.values() if torch.is_tensor(v))
    else:
        _VIEW_CACHE.move_to_end(key)
    G = {k: (v.to(device).float() if k == "E" else v.to(device)) if torch.is_tensor(v) else v for k, v in st.items()}
    if tokp is not None:
        G["TOK"], G["TOKM"] = tokp.get(cg)
        if getattr(tokp, "chunks", 0) > 0:
            G["CHK"], G["CHKM"] = tokp.get_chunks(cg)
        if getattr(tokp, "xprov", None) is not None:
            G["TOKX"], G["TOKXM"] = tokp.xprov.get(cg)
        if getattr(tokp, "rcaprov", None) is not None:
            if getattr(tokp.rcaprov, "lazy", False):
                G["RCAP"] = (tokp.rcaprov, cg)
            else:
                v_ = tokp.rcaprov.get(cg)
                G["RCAS"], G["RCASM"] = v_[:2]
                if len(v_) > 2:
                    G["RCAX"], G["RCAXM"], G["RCAXI"] = v_[2:]
    return G


MENTION_ROOT_OFF = False


REL_SCALE = {}


UNIT_SESS = False
UNIT_TEXT = False
KEY_CACHE = None


def _graph_tensors(cg, cache, rel_cache, device, tokp=None):
    E = cache.get(cg.world_id, cg.unit_text).to(device)
    C = rel_cache.get("relations", cg.rel_text).to(device) if cg.rel_text else torch.zeros(1, E.shape[1], device=device)
    src = torch.tensor(cg.src, dtype=torch.long, device=device)
    dst = torch.tensor(cg.dst, dtype=torch.long, device=device)
    rel = torch.tensor(cg.rel, dtype=torch.long, device=device)
    indeg = torch.zeros(cg.n, device=device).index_add_(0, dst, torch.ones_like(dst, dtype=torch.float)) if len(cg.dst) else torch.zeros(cg.n, device=device)
    is_fact = torch.tensor(cg.unit_type, dtype=torch.float, device=device)

    Mod = rel_cache.get("modules_all", cg.mod_text).to(device) if cg.mod_text else torch.zeros(0, E.shape[1], device=device)
    Mod = torch.cat([Mod, torch.zeros(1, E.shape[1], device=device)], 0)
    unit_mod = torch.tensor([m if m >= 0 else len(cg.mod_names) for m in cg.unit_mod], dtype=torch.long, device=device)
    G = {"E": E, "C": C, "src": src, "dst": dst, "rel": rel, "indeg_norm": (1 + indeg).rsqrt(),
         **({"EK": KEY_CACHE.get(cg.world_id, cg.unit_text).to(device)} if KEY_CACHE is not None else {}),
         **({"unit_text": cg.unit_text} if UNIT_TEXT else {}),
         "state": torch.tensor(cg.unit_state, dtype=torch.long, device=device),
         "is_fact": is_fact, "is_entity": 1 - is_fact, "owner": cg.owner,
         "Mod": Mod, "unit_mod": unit_mod, "n_mod": len(cg.mod_names)}
    gmi = {m: i for i, m in enumerate(module_list())}
    G["unit_gmod"] = torch.tensor([gmi.get(cg.mod_names[m], -1) if m >= 0 else -1 for m in cg.unit_mod], dtype=torch.long, device=device)
    gen = torch.Generator(device="cpu").manual_seed(len(cg.src) + cg.n)
    G["src_shuffled"] = src[torch.randperm(len(cg.src), generator=gen).to(device)] if len(cg.src) else src

    G["subj"] = torch.tensor([fs if fs >= 0 else u for u, fs in enumerate(cg.fact_subject)], dtype=torch.long, device=device)
    if UNIT_SESS:
        sid = {}
        G["unit_sess"] = torch.tensor([sid.setdefault(v, len(sid)) if v else -1 for v in
                                       ((cg.payload.get(u) or {}).get("session", "") if isinstance(cg.payload.get(u), dict) else "" for u in cg.unit_ids)],
                                      dtype=torch.long, device=device)
    if REL_SCALE:
        G["rel_scale"] = torch.tensor([REL_SCALE.get(r[4:] if r.startswith("inv:") else r, 1.0) for r in cg.rel_names] or [1.0],
                                      dtype=torch.float, device=device)
    if "mentioned_in" in getattr(cg, "rel_names", ()):
        m = rel == cg.rel_names.index("mentioned_in")
        if MENTION_ROOT_OFF:
            m = m & (src != cg.owner)
        G["men_src"], G["men_dst"] = src[m], dst[m]


    if len(cg.dst):
        perm = torch.argsort(dst, stable=True)
        ds = dst[perm]
        ar = torch.arange(cg.n, device=device)
        G["dst_perm"] = perm
        G["seg_start"] = torch.searchsorted(ds, ar, right=False)
        G["seg_end"] = torch.searchsorted(ds, ar, right=True) - 1
    else:
        G["dst_perm"] = src; G["seg_start"] = torch.zeros(cg.n, dtype=torch.long, device=device); G["seg_end"] = torch.full((cg.n,), -1, dtype=torch.long, device=device)
    if "t_iv" in cg.payload:
        tv = torch.tensor(cg.payload["t_iv"], dtype=torch.float32)
        G["t_lo"], G["t_hi"] = tv[:, 0].to(device), tv[:, 1].to(device)
        if tv.shape[1] >= 4:
            G["t_ob"], G["t_grp"] = tv[:, 2].to(device), tv[:, 3].long().to(device)
    if "uw" not in cg.payload:
        from .lex import unit_lex_tensors
        cg.payload["uw"] = tuple(x.cpu() for x in unit_lex_tensors(cg.unit_text, "cpu")) + (unit_lex_tensors.last_date_mask.cpu(),)
    G["uw"], G["uwt"] = (x.to(device) for x in cg.payload["uw"][:2])
    G["uwd"] = cg.payload["uw"][2].to(device) if len(cg.payload["uw"]) > 2 else torch.zeros_like(G["uwt"])
    if tokp is not None:
        G["TOK"], G["TOKM"] = tokp.get(cg)
        if getattr(tokp, "chunks", 0) > 0:
            G["CHK"], G["CHKM"] = tokp.get_chunks(cg)
        if getattr(tokp, "xprov", None) is not None:
            G["TOKX"], G["TOKXM"] = tokp.xprov.get(cg)
        if getattr(tokp, "rcaprov", None) is not None:
            if getattr(tokp.rcaprov, "lazy", False):
                G["RCAP"] = (tokp.rcaprov, cg)
            else:
                v_ = tokp.rcaprov.get(cg)
                G["RCAS"], G["RCASM"] = v_[:2]
                if len(v_) > 2:
                    G["RCAX"], G["RCAXM"], G["RCAXI"] = v_[2:]
    return G


