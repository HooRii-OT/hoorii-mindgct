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
"""GCT network: a frozen Qwen3-0.6B backbone with graph layers inserted after chosen decoder layers, a read-out over the
graph units and a close-read block that re-reads the top candidates. One forward pass scores every unit of a graph view."""
NO_SCHEMA_MODES = ("v3", "full", "bsym_c", "vec_gnn")
from dataclasses import dataclass, field

import math
import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lex import request_lex, request_lex_feats

QWEN = None
TOKENIZER = None


def deg_norm_scale(G, mode, q, n):
    deg = torch.zeros(n, device=G["src"].device).index_add_(0, G["src"], torch.ones_like(G["src"], dtype=torch.float))
    ent = (G["is_fact"] < 0.5) & (deg > 0)
    one = torch.ones_like(deg)
    if mode == "sqrt":
        return torch.where(ent, deg.clamp_min(1.0).rsqrt(), one)
    if mode == "hub":
        if not bool(ent.any()):
            return one
        thr = torch.quantile(deg[ent], q)
        return torch.where(ent & (deg > thr), (thr / deg.clamp_min(1.0)).sqrt(), one)
    raise ValueError(f"deg_norm {mode}")


def gmemo(G, owner, key, fn):
    if torch.is_grad_enabled() or None:
        return fn()
    m = G.setdefault("_memo", {})
    k = (id(owner), key)
    if k not in m:
        m[k] = fn()
    return m[k]


def tok_proj(lin, TOK, chunk=131072):
    if torch.is_grad_enabled() or TOK.shape[0] <= chunk:
        return lin(TOK.float())
    return torch.cat([lin(TOK[i:i + chunk].float()) for i in range(0, TOK.shape[0], chunk)])


TIME_INTENTS = ["none", "by_time", "ask_time", "latest", "order", "state_at", "action_time", "time_other"]
TIME_NF = 10


def tok_maxsim_mask(G, c):
    m = G["TOKM"]
    if not c.content_skip0 or m.shape[1] < 2:
        return m
    m = m.clone()
    m[:, 0] &= ~m[:, 1]
    return m


def tok_wb_mask(G, c):
    m = G["TOKM"]
    if not c.wb_skip0 or m.shape[1] < 2:
        return m
    m = m.clone()
    m[:, 0] &= ~m[:, 1]
    return m


def fanout_topm_share(ge, src, m_src, n, tau, scale=None):
    B, M, E = ge.shape
    with torch.no_grad():
        dev = ge.device
        k = m_src if torch.is_tensor(m_src) and m_src.dim() > 0 else torch.full((E,), float(m_src), device=dev)
        k = k.round().clamp_min(1).long()
        srt, perm = torch.sort(src, stable=True)
        first = torch.searchsorted(srt, srt, right=False)
        rank = torch.arange(E, device=dev) - first
        kk = torch.zeros(n, dtype=torch.long, device=dev).scatter_(0, src, k)
        k_pos = kk[srt]
        o1 = torch.argsort(-ge.detach(), dim=-1, stable=True)
        o2 = torch.argsort(src[o1], dim=-1, stable=True)
        order = o1.gather(-1, o2)
        gs = ge.detach().gather(-1, order)
        sel = (rank == k_pos) & (rank > 0)
        pos = torch.nonzero(sel).flatten()
        theta = torch.full((B, M, n), float("-inf"), device=dev, dtype=ge.dtype)
        if pos.numel():
            val = 0.5 * (gs[..., pos] + gs[..., pos - 1])
            theta.scatter_(-1, srt[pos][None, None].expand(B, M, -1), val)
        th_e = theta.gather(-1, src[None, None].expand(B, M, E))
        if scale == "src_std":
            ex = src[None, None].expand(B, M, E)
            g0 = ge.detach()
            cnt = torch.zeros(n, device=dev, dtype=ge.dtype).index_add_(0, src, torch.ones(E, device=dev, dtype=ge.dtype)).clamp_min(1)
            mu = torch.zeros(B, M, n, device=dev, dtype=ge.dtype).scatter_add_(-1, ex, g0) / cnt
            var = torch.zeros(B, M, n, device=dev, dtype=ge.dtype).scatter_add_(-1, ex, g0 * g0) / cnt - mu * mu
            sd = var.clamp_min(0).sqrt().gather(-1, ex).clamp_min(1e-3)
        else:
            sd = 1.0
    return torch.sigmoid((ge - th_e) / (tau * sd))


@dataclass
class Cfg:
    mode: str = "full"
    graph_after: tuple = (8, 14, 20)
    n_lm_layers: int = 20
    d: int = 1024
    d_e: int = 1024
    d_k: int = 128
    d_r: int = 64
    d_h: int = 256
    M: int = 8
    K: int = 32
    tau: float = 0.5
    soft_temp: float = 0.05
    fact_topology: bool = True
    hard: bool = False
    schema: bool = None
    tok_mem: bool = False
    elbo: bool = False
    eps_prior: float = 0.02
    prior_mode: str = "graph"
    kl_norm: str = "per_unit"
    free_bits: float = 0.0
    prior_detach: bool = False
    hard_subject: bool = False
    anchor_clamp: bool = False
    act_topm: int = 0
    fanout_m: float = 0.0
    lex_feat: bool = False
    lex_anchor: float = 0.0
    anchor_teach: float = 0.0
    lex_phi: bool = False
    lex_q_init: float = 0.0
    lex_ent_only: bool = False
    lex_fact_reach: bool = False
    hard_reach: bool = False
    gate_cos: bool = False


    read_ledger: bool = False

    mod_comp: bool = False


    subj_factor: bool = False


    turn_gate: bool = False


    dlg_state: bool = False


    read_budget: bool = False

    read_lex2: bool = False
    read_lex: bool = False
    path_teach: bool = False
    route_gate: bool = False


    act_norm: float = 0.0
    read_norm: float = 0.0
    read_comp: bool = False
    read_comp_init: float = -4.0
    prov_feat: bool = False
    state_prior: bool = False
    tok_readout: bool = False
    rerank_top: int = 0


    time_feat: bool = False


    time_entry: int = 0

    time_ro_gain: float = 1.0

    time_entry_init: float = -9.0

    fact_entry: int = 0


    fact_entry_score: str = "lex"


    mention_reach: bool = False

    fanout_deg: bool = False

    probe_mask_tmpl: bool = False


    content_cos: bool = False


    probe_mask_ws: bool = False


    content_skip0: bool = False


    need_pool: bool = False


    need_pool_carry: bool = False
    fanout_topm: bool = False


    fanout_topm_tau: float = 1.0
    fanout_topm_scale: str = "src_std"


    need_match: bool = False


    wb_skip0: bool = False


    read_any: bool = False


    wb_unit: bool = False


    wb_aux: float = 0.0

    slot_inhibit: bool = False


    hub_damp: bool = False
    deg_norm: str = ""


    deg_norm_q: float = 0.9
    unit_roles: bool = False


    refractory: int = 0
    accept_thr: float = 0.5


    role_send: bool = False


    ent_entry: int = 0


    wide_entry: int = 0


    wide_entry_score: str = "hyb"

    de_dim: int = 256
    de_keys: bool = False

    de_key_dim: int = 0
    de_qpool: str = "mean"


    de_distill: float = 0.0

    wide_entry_layer: int = 0

    wide_entry_ents: int = 0

    entry_only: bool = False


    de_req: str = "own"

    de_read: bool = False
    de_head: int = 0

    de_head_scale: bool = False


    de_qtf: int = 0


    de_qtf_init: str = "random"

    de_qtf_pool: str = "head"

    de_rank_distill: float = 0.0

    de_rd_tau: float = 0.05
    tok_nl: int = 1


    tok_mix_ref: int = 0
    tok_mix_init: float = 3.0
    tok_adapt: int = 0

    tok_adapt_attn: bool = False

    tok_subj: str = ""


    reader: str = ""


    reader_n: int = 16
    reader_lora: int = 16
    reader_q_tok: int = 400
    reader_doc_tok: int = 384
    gumbel_temp: float = 0.5
    steps: int = 2
    hop_persist: str = "off"


    hop_kmax: tuple = None
    hop_train_k: tuple = None
    hop_halt: str = "none"

    hop_shared_noise: bool = False
    read_path: bool = False

    switch_theta: float = 0.0


    switch_from: int = -1


    switch_none: float = 0.0

    n_sit: int = 64
    setro_m: int = 0


    setro_layers: int = 2
    setro_dim: int = 512
    setro_heads: int = 8
    px_m: int = 0


    px_layers: int = 2
    px_dim: int = 512
    px_heads: int = 8
    px_req_tok: int = 64
    pair_sel: int = 0


    pair_sel_k: int = 5
    pair_sel_rank: int = 256
    pair_sel_agg: str = "last"
    pair_sel_grad: bool = False
    pair_sel_gate: bool = False

    rca_m: int = 0


    rca_l0: int = 8
    rca_t: int = 49
    rca_read_layer: int = 20
    rca_lora: int = 16
    rca_suffix: str = "tmpl"
    rca_head_init: str = "yesno"
    rca_ckpt: bool = True
    rca_alpha_init: float = -3.0
    rca_detach_kv: bool = True
    rca_p_swap: float = 0.5
    rca_seg: int = 1


    rca_seg_max: int = 0

    rca_seg_merge: bool = False

    rca_fuse_grad: bool = False

    setro_conf: bool = False


    extra: dict = field(default_factory=dict)


def seg_sum(x, G, dim=2):
    perm, st, en = G["dst_perm"], G["seg_start"], G["seg_end"]
    xi = x.index_select(dim, perm)
    xs = xi.cumsum(dim)
    has = (en >= st)
    end_v = xs.index_select(dim, en.clamp_min(0))
    start_v = xs.index_select(dim, (st - 1).clamp_min(0))
    shape = [1] * x.dim(); shape[dim] = -1
    first = (st <= 0).view(shape)
    out = end_v - torch.where(first, torch.zeros_like(start_v), start_v)
    return (out * has.view(shape).to(out.dtype)).to(x.dtype)


def slot_groups(G):
    n = G["is_fact"].shape[0]
    nm = int(G["unit_mod"].max()) + 1 if n else 1
    own = -(torch.arange(n, device=G["subj"].device) + 1)
    key = torch.where(G["is_fact"] > 0, G["subj"] * nm + G["unit_mod"], own)
    return torch.unique(key, return_inverse=True)[1]


def slot_lse(x, gid):
    ng = int(gid.max()) + 1
    gi = gid[None, None].expand_as(x)
    mx = torch.full(x.shape[:-1] + (ng,), -1e30, device=x.device, dtype=x.dtype).scatter_reduce(2, gi, x.detach(), "amax", include_self=True)
    mu = mx.gather(2, gi)
    sm = torch.zeros(x.shape[:-1] + (ng,), device=x.device, dtype=x.dtype).scatter_add(2, gi, torch.exp(x - mu))
    return mu + torch.log(sm.gather(2, gi))


ROLE_NF = 5


def seg_sum_sparse(vals, dst, n):
    order = torch.argsort(dst, stable=True)
    d = dst[order]
    cs = vals[..., order].double().cumsum(-1)
    uniq, counts = torch.unique_consecutive(d, return_counts=True)
    ends = counts.cumsum(0) - 1
    tot = cs[..., ends] - torch.cat([torch.zeros_like(cs[..., :1]), cs[..., ends[:-1]]], -1)
    out = torch.zeros(vals.shape[:-1] + (n,), dtype=torch.float64, device=vals.device)
    out[..., uniq] = tot
    return out.to(vals.dtype)


class GraphLayer(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.c = c
        d, de, dk, dr = c.d, c.d_e, c.d_k, c.d_r
        self.ln = nn.LayerNorm(d)
        self.probe = nn.Parameter(torch.randn(c.M, dk) * 0.02)
        self.Wpk = nn.Linear(d, dk, bias=False)
        self.Wq = nn.Linear(d, dk, bias=False)
        self.Wk = nn.Linear(de, dk, bias=False)
        self.Wg = nn.Linear(d, dr, bias=c.gate_cos)
        self.Wc = nn.Linear(de, dr, bias=False)
        self.gate_b = nn.Parameter(torch.tensor(0.5))
        self.state_emb = nn.Embedding(5, 4)
        self.phi = nn.Sequential(nn.Linear(3 + 4, 32), nn.GELU(), nn.Linear(32, 1))
        self.kappa = nn.Parameter(torch.tensor(10.0))
        self.tau_p = nn.Parameter(torch.tensor(0.2))
        if c.schema:
            self.Ws = nn.Linear(d, dr, bias=False)
            self.Wsit = nn.Linear(de, dr, bias=False)
            self.sit_b = nn.Parameter(torch.tensor(-1.0))
            self.U = nn.Parameter(torch.randn(c.n_sit, dr) * 0.02)
            self.V = nn.Parameter(torch.randn(c.n_sit, dr) * 0.02)
            self.Wmk = nn.Linear(de, dr, bias=False)
            self.Wq2 = nn.Linear(d, dr, bias=False)
            self.need_b = nn.Parameter(torch.tensor(1.0))
            self.kappa_m = nn.Parameter(torch.tensor(10.0))
            self.tau_m = nn.Parameter(torch.tensor(0.3))
            self.phi = nn.Sequential(nn.Linear(3 + 4 + 1, 32), nn.GELU(), nn.Linear(32, 1))

        self.Wtq = nn.Linear(d, dk, bias=False)
        self.Wuk = nn.Linear(de, dk, bias=False)
        self.Wv = nn.Linear(de, d, bias=False)
        nn.init.normal_(self.Wv.weight, std=0.002)
        if c.tok_mem:
            self.Wtk = nn.Linear(de, dk, bias=False)
            self.Wtv = nn.Linear(de, d, bias=False)
            nn.init.normal_(self.Wtv.weight, std=0.002)
        self.null_logit = nn.Parameter(torch.tensor(0.0))
        self.wb_gate = nn.Parameter(torch.tensor(-2.0))
        if c.content_cos:
            self.cc_tau = nn.Parameter(torch.tensor(10.0))

    def threshold(self, s):
        hard = (s > self.c.tau).to(s.dtype)
        if not self.training:
            return s * hard
        soft = torch.sigmoid((s - self.c.tau) / self.c.soft_temp)
        if self.c.hard:
            return s * (hard + soft - soft.detach())
        return s * soft

    def forward(self, H, tok_mask, G, s_prev, write=True, sit_E=None):
        c = self.c
        Hf = self.ln(H.float())
        B = Hf.shape[0]

        lg = torch.einsum("mk,btk->bmt", self.probe, self.Wpk(Hf)) / c.d_k ** 0.5
        pmask = G["_pmask"] if (c.probe_mask_tmpl and "_pmask" in G) else tok_mask
        lg = lg.masked_fill(~pmask[:, None, :], -1e4)
        P = torch.softmax(lg, -1) @ Hf
        Pn = F.normalize(P, dim=-1)
        sim_pp = Pn @ Pn.transpose(1, 2)
        div = (sim_pp - torch.eye(c.M, device=P.device)).pow(2).mean()

        if c.tok_mem and "TOK" in G:
            tk = tok_proj(self.Wtk, G["TOK"])
            if c.content_cos:
                at = torch.einsum("bmk,ntk->bmnt", F.normalize(self.Wq(P), dim=-1), F.normalize(tk, dim=-1)) * self.cc_tau
            else:
                at = torch.einsum("bmk,ntk->bmnt", self.Wq(P), tk) / c.d_k ** 0.5
            a = at.masked_fill(~tok_maxsim_mask(G, c)[None, None], -1e4).max(-1).values
        else:
            a = torch.einsum("bmk,nk->bmn", self.Wq(P), gmemo(G, self, "Wk", lambda: self.Wk(G["E"]))) / c.d_k ** 0.5
        if c.mode == "no_content":
            a = a * G["is_entity"]

        sit = need = need_unit = None
        gq = self.Wg(P)
        if c.schema:
            sit = torch.sigmoid(torch.einsum("bmr,sr->bms", self.Ws(P), self.Wsit(sit_E)) / c.d_r ** 0.5 + self.sit_b)
            Modk = self.Wmk(G["Mod"])
            need = torch.sigmoid(torch.einsum("bmr,jr->bmj", self.Wq2(P) + sit @ self.U, Modk) / c.d_r ** 0.5 + self.need_b)
            need_unit = need[:, :, G["unit_mod"]]
            gq = gq + sit @ self.V

        st = self.state_emb(G["state"])[None, None].expand(B, c.M, -1, -1)
        s_cur, bp = s_prev, None
        s_any = torch.zeros_like(a)
        prov_rel = torch.zeros(B, c.M, a.shape[-1], c.d_r, device=a.device) if c.prov_feat else None
        for _step in range(c.steps):
            if c.mode != "no_prop" and G["src"].numel():
                Cr = self.Wc(G["C"])
                g = torch.sigmoid(torch.einsum("bmr,qr->bmq", gq, Cr) + self.gate_b)
                sp = s_cur[:, :, G["src"]]
                msg = sp * g[:, :, G["rel"]]
                if prov_rel is not None:
                    prov_rel = prov_rel + seg_sum(msg[..., None] * Cr[G["rel"]][None, None], G, dim=2)


                act_in = seg_sum((sp > 0).to(sp.dtype) if not self.training else sp, G)
                p = seg_sum(msg, G) * (1 + act_in).rsqrt()
                if not self.training:
                    mx = torch.full_like(a, -1.0).scatter_reduce_(2, G["dst"].expand_as(msg), msg, "amax")
                    eidx = torch.arange(msg.shape[-1], device=msg.device).expand_as(msg)
                    hit = (msg == mx.gather(2, G["dst"].expand_as(msg))) & (msg > 0)
                    bp = torch.full(a.shape, -1, dtype=torch.long, device=a.device).scatter_reduce_(
                        2, G["dst"].expand_as(msg), torch.where(hit, eidx, torch.full_like(eidx, -1)), "amax")
            else:
                p = torch.zeros_like(a)
            feats = [a[..., None], p[..., None], s_cur[..., None], st]
            if c.schema:
                feats.append(need_unit[..., None])
            s = torch.sigmoid(self.phi(torch.cat(feats, -1)).squeeze(-1))
            if c.fact_topology and c.mode != "no_prop":
                p_any = p.max(1, keepdim=True).values.expand_as(p)
                s = s * (1 - G["is_fact"] + G["is_fact"] * torch.sigmoid(self.kappa * (p_any - self.tau_p)))
            if c.schema:
                s = s * (1 - G["is_fact"] + G["is_fact"] * torch.sigmoid(self.kappa_m * (need_unit - self.tau_m)))
            s_cur = self.threshold(s)
            s_any = torch.maximum(s_any, s_cur)
        s_hat = s_cur

        prov = torch.where(p > self.tau_p, 1, torch.where(s_prev > 0, 2, 0)) if not self.training else None

        delta, overflow = None, None
        if write and c.mode != "no_write":
            u = s_hat.max(1).values
            K = min(c.K, u.shape[1])
            overflow = (u > 0).sum(1) > K
            uv, ui = torch.topk(u, K, dim=1)
            bias = torch.where(uv > 0, torch.log(uv.clamp_min(1e-6)), torch.full_like(uv, float("-inf")))
            if c.tok_mem and "TOK" in G:
                Tu = G["TOK"].shape[1]
                tokk = G["TOK"][ui].float()
                keys = self.Wtk(tokk).reshape(B, K * Tu, -1)
                vals = self.Wtv(tokk).reshape(B, K * Tu, -1)
                tb = (bias[:, :, None] + torch.where(G["TOKM"][ui], 0.0, float("-inf"))[:, :, :]).reshape(B, K * Tu)
                lgw = torch.einsum("btk,bjk->btj", self.Wtq(Hf), keys) / c.d_k ** 0.5 + tb[:, None, :]
                lgw = torch.cat([lgw, self.null_logit.expand(B, lgw.shape[1], 1)], -1)
                att = torch.softmax(lgw, -1)[..., :K * Tu]
                delta = torch.sigmoid(self.wb_gate) * (att @ vals)
            else:
                Eu = G["E"][ui]
                lgw = torch.einsum("btk,bjk->btj", self.Wtq(Hf), self.Wuk(Eu)) / c.d_k ** 0.5 + bias[:, None, :]
                lgw = torch.cat([lgw, self.null_logit.expand(B, lgw.shape[1], 1)], -1)
                att = torch.softmax(lgw, -1)[..., :K]
                delta = torch.sigmoid(self.wb_gate) * (att @ self.Wv(Eu))
        return {"s": s_hat, "s_any": s_any, "a": a, "p": p, "delta": delta, "div": div, "bp": bp, "prov": prov, "overflow": overflow,
                "prov_rel": prov_rel.max(1).values if prov_rel is not None else None,
                "sit": sit, "need": need, "need_unit": need_unit}


class ElboGraphLayer(GraphLayer):
    li = 0
    _hop_on = True

    def __init__(self, c: Cfg):
        super().__init__(c)
        self.phi = nn.Sequential(nn.Linear(3 + 4 + 1 + 1 + (1 if c.lex_phi else 0), 32), nn.GELU(), nn.Linear(32, 1))
        if c.lex_feat:
            self.lex_w = nn.Parameter(torch.full((1,), 8.0))
            self.lex_q = nn.Parameter(torch.full((1,), float(c.lex_q_init)))
        if c.dlg_state:
            self.prev_q = nn.Parameter(torch.full((1,), 0.1))
            self.prev_a = nn.Parameter(torch.zeros(1))
        if c.gate_cos:
            self.gate_tau = nn.Parameter(torch.tensor(5.0))
            self.register_buffer("rel_mean", torch.zeros(c.d_e))
        if c.fact_entry > 0:
            self.fe_b = nn.Parameter(torch.full((1,), -9.0))
            self.fe_w = nn.Parameter(torch.zeros(1))
        if c.time_feat:
            self.t_W = nn.Parameter(torch.zeros(len(TIME_INTENTS), TIME_NF))


        if (c.time_entry or 0) > 0:
            self.te_b = nn.Parameter(torch.full((1,), -9.0 if c.time_entry_init is None else float(c.time_entry_init)))
            self.te_w = nn.Parameter(torch.zeros(1))
        if c.fanout_deg:
            self.fd_b = nn.Parameter(torch.zeros(1))
            self.fd_w = nn.Parameter(torch.zeros(1))
        if c.need_match:
            self.nm_b = nn.Parameter(torch.full((1,), -6.0))
            self.nm_wl = nn.Parameter(torch.zeros(1))
            self.nm_wa = nn.Parameter(torch.zeros(1))
        if c.wb_unit:
            de = self.Wuk.in_features
            self.Wwk_u = nn.Linear(de, c.d_k, bias=False)
            self.Wwv_u = nn.Linear(de, self.Wv.out_features, bias=False)
            nn.init.zeros_(self.Wwv_u.weight)
        if c.slot_inhibit:
            self.si_l = nn.Parameter(torch.zeros(1))
        if c.hub_damp:
            self.hd_b = nn.Parameter(torch.zeros(1))
        if c.unit_roles:
            self.role_q = nn.Linear(ROLE_NF, 1, bias=False)
            nn.init.zeros_(self.role_q.weight)
        if c.role_send:
            self.role_s = nn.Linear(ROLE_NF, 1, bias=False)
            nn.init.zeros_(self.role_s.weight)
        if c.ent_entry > 0:
            self.ee_b = nn.Parameter(torch.full((1,), -9.0))
            self.ee_w = nn.Parameter(torch.zeros(1))
        self._dbg = None
        self._dbg_edge = None
        self._dbg_share = None

    def _n_hops(self):
        c = self.c
        if not self._hop_on:
            return c.steps
        if self.training and c.hop_train_k is not None:
            lo, hi = c.hop_train_k[self.li]
            return random.randint(lo, hi)
        if c.hop_kmax is not None:
            return c.hop_kmax[self.li]
        return c.steps


    def forward(self, H, tok_mask, G, z_prev, write=True, sit_E=None):
        c = self.c
        Hf = self.ln(H.float())
        B = Hf.shape[0]
        lg = torch.einsum("mk,btk->bmt", self.probe, self.Wpk(Hf)) / c.d_k ** 0.5
        pmask = G["_pmask"] if (c.probe_mask_tmpl and "_pmask" in G) else tok_mask
        lg = lg.masked_fill(~pmask[:, None, :], -1e4)
        P = torch.softmax(lg, -1) @ Hf
        Pn = F.normalize(P, dim=-1)
        div = ((Pn @ Pn.transpose(1, 2)) - torch.eye(c.M, device=P.device)).pow(2).mean()
        sparse = False
        if sparse:
            a = None
        elif c.tok_mem and "TOK" in G:
            if c.content_cos:
                tk = gmemo(G, self, "Wtk_n", lambda: F.normalize(tok_proj(self.Wtk, G["TOK"]), dim=-1))
                at = torch.einsum("bmk,ntk->bmnt", F.normalize(self.Wq(P), dim=-1), tk) * self.cc_tau
            else:
                tk = gmemo(G, self, "Wtk", lambda: tok_proj(self.Wtk, G["TOK"]))
                at = torch.einsum("bmk,ntk->bmnt", self.Wq(P), tk) / c.d_k ** 0.5
            a = at.masked_fill(~tok_maxsim_mask(G, c)[None, None], -1e4).max(-1).values
        else:
            a = torch.einsum("bmk,nk->bmn", self.Wq(P), gmemo(G, self, "Wk", lambda: self.Wk(G["E"]))) / c.d_k ** 0.5
        if c.mode == "no_content" and a is not None:
            a = a * G["is_entity"]
        sit = torch.sigmoid(torch.einsum("bmr,sr->bms", self.Ws(P), self.Wsit(sit_E)) / c.d_r ** 0.5 + self.sit_b)
        Modk = self.Wmk(G["Mod"])
        need = torch.sigmoid(torch.einsum("bmr,jr->bmj", self.Wq2(P) + sit @ self.U, Modk) / c.d_r ** 0.5 + self.need_b)
        if c.need_pool:
            needp = need.max(1, keepdim=True).values
            if c.need_pool_carry:
                if self.li == 0 or "_needp" not in G:
                    G["_needp"] = needp
                else:
                    needp = torch.maximum(needp, G["_needp"])
                    G["_needp"] = needp
            need_unit = needp[:, :, G["unit_mod"]].expand(-1, need.shape[1], -1)
        else:
            need_unit = need[:, :, G["unit_mod"]]
        if c.need_match and a is not None:
            lx_u = G["lex"][:, None, :].to(need_unit.dtype) if "lex" in G else torch.zeros_like(need_unit)
            with torch.no_grad():
                med = a.median(-1, keepdim=True).values
                mad = (a - med).abs().median(-1, keepdim=True).values
            a_hat = ((a - med) / (1.4826 * mad + 1e-3)).clamp(-5.0, 5.0)
            m_u = torch.sigmoid(self.nm_b + self.nm_wl * lx_u + self.nm_wa * a_hat)
            need_unit = 1 - (1 - need_unit) * (1 - m_u)
        gq = self.Wg(P) + sit @ self.V
        st = self.state_emb(G["state"])[None, None].expand(B, c.M, -1, -1)
        eps = c.eps_prior
        if sparse:
            return self._sparse_hops(Hf, P, G, z_prev, sit, need, need_unit, gq, st, div, write, B)
        z_cur, kl_total, bp = z_prev, torch.zeros((), device=a.device), None
        g_c = None
        if c.fact_entry > 0:


            lexs = G.get("lex_raw", G.get("lex"))
            if c.fact_entry_score == "lex" and lexs is not None:
                sc = lexs[:, None, :].float().expand_as(a)
                ok = (sc > 0).to(a.dtype)
            else:
                sc, ok = a, torch.ones_like(a)
            kM = min(c.fact_entry, a.shape[-1])
            af = sc.masked_fill(G["is_fact"][None, None] == 0, float("-inf"))
            ca = torch.zeros_like(a).scatter_(-1, af.topk(kM, dim=-1).indices, 1.0) * G["is_fact"][None, None] * ok
            g_c = torch.sigmoid(self.fe_b + self.fe_w * sc) * ca
        g_e = None
        if c.ent_entry > 0 and ("lex_raw" in G or "lex" in G):

            se = G.get("lex_raw", G.get("lex")).float()
            kE = min(c.ent_entry, se.shape[-1])
            ae = se.masked_fill(G["is_fact"][None] > 0, float("-inf"))
            ce = torch.zeros_like(se).scatter_(-1, ae.topk(kE, dim=-1).indices, 1.0) * (1 - G["is_fact"])[None] * (se > 0).to(se.dtype)
            g_e = (torch.sigmoid(self.ee_b + self.ee_w * se) * ce)[:, None, :].expand_as(a)
        if (c.time_entry or 0) > 0 and "t_es" in G:


            sct = G["t_es"]
            kT = min(c.time_entry, sct.shape[-1])
            ct = torch.zeros_like(sct).scatter_(-1, sct.topk(kT, dim=-1).indices, 1.0) * (sct > 0).to(sct.dtype)
            g_t = (torch.sigmoid(self.te_b + self.te_w * sct) * ct)[:, None, :].expand_as(a)
            g_c = g_t if g_c is None else 1 - (1 - g_c) * (1 - g_t)
        wc = None
        if c.wide_entry > 0 and G.get("_wide") is not None:
            wc = G["_wide"][:, None, :].to(a.dtype).expand_as(a)
        out_deg = None
        if c.fanout_deg and G["src"].numel():
            out_deg = torch.zeros(a.shape[-1], device=a.device).index_add_(0, G["src"], torch.ones_like(G["src"], dtype=torch.float))
        gate_mass = torch.zeros((), device=a.device)
        t_edge = None
        q_h1 = None
        z_any, s_any = torch.zeros_like(a), torch.zeros_like(a)
        prov_rel = torch.zeros(B, c.M, a.shape[-1], c.d_r, device=a.device) if c.prov_feat else None
        hop = self._hop_on
        persist = hop and c.hop_persist in ("layer", "global")
        halt = hop and c.hop_halt == "fixpoint" and not self.training
        K, n_run = self._n_hops(), 0
        if persist:


            Vin = (z_prev > 0).to(a.dtype) if c.hop_persist == "global" else torch.zeros_like(a)
            zm = Vin.clone()
            q_acc = z_prev if c.hop_persist == "global" else torch.zeros_like(a)
        u_shared = torch.rand_like(a).clamp(1e-6, 1 - 1e-6) if (self.training and hop and c.hop_shared_noise) else None
        path, pm = None, self.__dict__.get("_pm")
        if c.read_path and pm is not None:
            if self.li == 0 or "_path" not in G:
                G["_path"] = {"pi": pm.pi0[None, None].expand(B, a.shape[-1], -1).clone(), "tau": torch.zeros(B, a.shape[-1], device=a.device), "hop": 0}
            path = G["_path"]
            WbC = pm.Wb(G["C"])
        if path is not None or halt:
            prev_z = z_prev > 0
            done = torch.zeros(B, dtype=torch.bool, device=a.device)
        roles_on = c.unit_roles or c.refractory > 0 or c.role_send
        si_gid = None
        if c.slot_inhibit:
            si_gid = G.get("_si_gid")
            if si_gid is None:
                si_gid = G["_si_gid"] = slot_groups(G)
        dn_s = None
        if c.deg_norm and G["src"].numel():
            dn_s = G.get("_dn_s_" + c.deg_norm + str(c.deg_norm_q))
            if dn_s is None:
                dn_s = G["_dn_s_" + c.deg_norm + str(c.deg_norm_q)] = deg_norm_scale(G, c.deg_norm, c.deg_norm_q, a.shape[-1])
        hd_deg = None
        if c.hub_damp and G["src"].numel():
            hd_deg = torch.zeros(a.shape[-1], device=a.device).index_add_(0, G["src"], torch.ones_like(G["src"], dtype=torch.float))
        if roles_on:
            dwell = (z_prev > 0).to(a.dtype)
            propd = torch.zeros_like(a)
            first = torch.zeros_like(a)
            anc_r = torch.zeros_like(a); anc_r[:, :, G["owner"]] = 1.0
            if "lex" in G and c.lex_anchor > 0:
                anc_r = torch.maximum(anc_r, (G["lex"] >= c.lex_anchor).to(a.dtype)[:, None, :].expand_as(a))
        for _step in range(K):
            n_run += 1
            msg = None
            z_in = z_cur

            if c.mode != "no_prop" and G["src"].numel():
                if c.gate_cos:
                    Cr = gmemo(G, self, "WcC", lambda: self.Wc(G["C"] - self.rel_mean))
                    g = torch.sigmoid(self.gate_tau * torch.einsum("bmr,qr->bmq", F.normalize(gq, dim=-1), F.normalize(Cr, dim=-1)) + self.gate_b)
                else:
                    Cr = self.Wc(G["C"])
                    g = torch.sigmoid(torch.einsum("bmr,qr->bmq", gq, Cr) + self.gate_b)
                if "rel_scale" in G:
                    g = g * G["rel_scale"][None, None, :]
                gate_mass = gate_mass + g[:, :, G["rel"]].mean()
                src_idx = G["src_shuffled"] if (c.prior_mode == "shuffled" and "src_shuffled" in G) else G["src"]
                zj = z_cur[:, :, src_idx]
                if c.refractory > 0:
                    zj = zj * (dwell[:, :, src_idx] < c.refractory).to(zj.dtype)
                msg = g[:, :, G["rel"]] * zj
                if c.fanout_m > 0:


                    gr = g[:, :, G["rel"]].clamp(1e-6, 1 - 1e-6)
                    ge = torch.log(gr) - torch.log1p(-gr) + a[:, :, G["dst"]]
                    ge = ge + G["is_fact"][G["dst"]] * torch.log(need_unit[:, :, G["dst"]].clamp_min(1e-4))
                    if c.lex_feat and "lex" in G:
                        ge = ge + self.lex_w * G["lex"][:, None, G["dst"]]
                    src_e = src_idx[None, None].expand_as(ge)
                    mx = torch.full_like(a, -1e4).scatter_reduce(2, src_e, ge, "amax", include_self=True)
                    ex = torch.exp(ge - mx.gather(2, src_e))
                    den = torch.zeros_like(a).scatter_add(2, src_e, ex)
                    m_src = c.fanout_m
                    if out_deg is not None:
                        m_src = c.fanout_m * torch.exp(self.fd_b + self.fd_w * torch.log1p(out_deg[src_idx]))
                    if c.fanout_topm:
                        share = fanout_topm_share(ge, src_idx, m_src, a.shape[-1], c.fanout_topm_tau, c.fanout_topm_scale)
                    else:
                        share = (m_src * ex / den.gather(2, src_e).clamp_min(1e-6)).clamp(max=1.0)
                    msg = msg * share
                    if self._dbg_share is not None:
                        self._dbg_share.append((share.detach(), zj.detach()))
                if dn_s is not None:
                    msg = msg * dn_s[src_idx]
                if hd_deg is not None:
                    msg = msg * torch.exp(-self.hd_b * torch.log1p(hd_deg[src_idx]))
                if c.role_send:
                    rs = torch.stack([anc_r, (dwell / 8.0).clamp(max=1.0), propd, (dwell > 0).to(a.dtype) * (1 - propd), first / 8.0], -1)
                    msg = msg * torch.exp(self.role_s(rs).squeeze(-1).clamp(-6.0, 6.0))[:, :, src_idx]
                if _step == 0 and self.training and c.path_teach:
                    t_edge = g[:, :, G["rel"]] * (share if c.fanout_m > 0 else 1.0)
                if self._dbg_edge is not None:
                    self._dbg_edge.append((g[:, :, G["rel"]].detach(), msg.detach()))
                if prov_rel is not None:
                    prov_rel = prov_rel + seg_sum(msg[..., None] * Cr[G["rel"]][None, None], G, dim=2)
                log1m = seg_sum(torch.log1p(-msg.clamp(max=1 - 1e-5)), G)
                p_reach = 1 - (1 - eps) * torch.exp(log1m)
                p = p_reach
                if not self.training:
                    mx = torch.full_like(a, -1.0).scatter_reduce_(2, G["dst"].expand_as(msg), msg, "amax")
                    eidx = torch.arange(msg.shape[-1], device=msg.device).expand_as(msg)
                    hit = (msg == mx.gather(2, G["dst"].expand_as(msg))) & (msg > 0)
                    bp_new = torch.full(a.shape, -1, dtype=torch.long, device=a.device).scatter_reduce_(
                        2, G["dst"].expand_as(msg), torch.where(hit, eidx, torch.full_like(eidx, -1)), "amax")
                    bp = bp_new if (bp is None or not persist) else torch.where(Vin > 0, bp, bp_new)
            else:
                p_reach = torch.full_like(a, eps)
                p = p_reach
            if g_c is not None:
                p_reach = 1 - (1 - p_reach) * (1 - g_c)
                p = p_reach
            if g_e is not None:
                p_reach = 1 - (1 - p_reach) * (1 - g_e)
                p = p_reach
            if wc is not None:
                p_reach = 1 - (1 - p_reach) * (1 - (1 - eps) * wc)
                p = p_reach
            prior = (1 - G["is_fact"]) * p_reach + G["is_fact"] * p_reach * need_unit
            if c.state_prior:
                sup_mask = (G["state"] == 4).to(prior.dtype)
                prior = prior * (1 - sup_mask) + eps * sup_mask
            if c.prior_mode == "uniform":
                prior = torch.full_like(prior, 1.0) * prior.mean(-1, keepdim=True).detach()
            prior = prior.clamp(1e-5, 1 - 1e-5)
            prior_logit = torch.log(prior) - torch.log1p(-prior)

            feats = torch.cat([a[..., None], p[..., None], z_cur[..., None], st, need_unit[..., None], prior_logit[..., None]], -1)
            if c.lex_phi:
                lx = G.get("lex_q", G["lex"])[:, None, :, None].expand(feats.shape[0], feats.shape[1], -1, 1) if "lex" in G else torch.zeros_like(feats[..., :1])
                if c.lex_fact_reach and "lex" in G:
                    lx = lx * ((1 - G["is_fact"]) + G["is_fact"] * (p > eps + 1e-3).to(lx.dtype))[..., None]
                feats = torch.cat([feats, lx], -1)
            q_logit = self.phi(feats).squeeze(-1)
            if c.unit_roles:
                term = (dwell > 0).to(a.dtype) * (1 - propd)
                rf = torch.stack([anc_r, (dwell / 8.0).clamp(max=1.0), propd, term, first / 8.0], -1)
                q_logit = q_logit + self.role_q(rf).squeeze(-1)
            if c.lex_feat and "lex" in G:
                lexq = G.get("lex_q", G["lex"])[:, None, :].expand_as(q_logit)
                if c.lex_fact_reach:
                    reach = (p > eps + 1e-3).to(lexq.dtype)
                    lexq = lexq * ((1 - G["is_fact"]) + G["is_fact"] * reach)
                q_logit = q_logit + 10.0 * self.lex_q * lexq
            if c.dlg_state and "prev" in G:
                q_logit = q_logit + 10.0 * self.prev_q * G["prev"][:, None, :]
            if c.time_feat and "t_f" in G:
                q_logit = q_logit + 10.0 * (G["t_f"] * self.t_W[G["t_int"]][:, None, :]).sum(-1)[:, None, :]
            q = torch.sigmoid(q_logit).clamp(1e-5, 1 - 1e-5)
            if c.route_gate:
                reach = ((p - eps) / (1 - eps)).clamp(0.0, 1.0)
                anc = torch.zeros_like(q); anc[:, :, G["owner"]] = 1.0
                if "lex" in G:
                    anc = torch.maximum(anc, (G["lex"] >= max(c.lex_anchor, 1e-6)).to(q.dtype)[:, None, :].expand_as(q))
                if c.dlg_state and "prev" in G:
                    anc = torch.maximum(anc, torch.sigmoid(self.prev_a) * G["prev"][:, None, :].to(q.dtype))
                ent = (1 - G["is_fact"])[None, None]
                q = (q * (ent * (anc + (1 - anc) * reach) + (1 - ent))).clamp(1e-5, 1 - 1e-5)
                q_logit = torch.log(q) - torch.log1p(-q)
            if c.hard_reach:
                rch = (1 - G["is_fact"]) + G["is_fact"] * (p > eps + 1e-3).to(q.dtype)
                q = (rch * q + (1 - rch) * 1e-4).clamp(1e-5, 1 - 1e-5)
                q_logit = torch.log(q) - torch.log1p(-q)
            if c.hard_subject:
                sf = z_in[:, :, G["subj"]] if not c.entry_only else torch.zeros_like(z_in)
                if c.mention_reach and "men_src" in G and not c.entry_only:
                    sf = torch.maximum(sf, torch.zeros_like(sf).scatter_reduce(2, G["men_dst"][None, None].expand(sf.shape[0], sf.shape[1], -1),
                                                                               z_in[:, :, G["men_src"]], "amax", include_self=True))
                if g_c is not None:
                    sf = torch.maximum(sf, g_c)
                if wc is not None:
                    sf = torch.maximum(sf, wc)
                sup = (1 - G["is_fact"]) + G["is_fact"] * sf
                q = (sup * q + (1 - sup) * 1e-4).clamp(1e-5, 1 - 1e-5)
                q_logit = torch.log(q) - torch.log1p(-q)
            if c.act_topm > 0 and q.shape[-1] > c.act_topm:
                keep = torch.zeros_like(q)
                keep.scatter_(-1, q.topk(c.act_topm, dim=-1).indices, 1.0)
                keep[:, :, G["owner"]] = 1.0
                q = (keep * q + (1 - keep) * (1e-4 + q - q.detach())).clamp(1e-5, 1 - 1e-5)
                q_logit = torch.log(q) - torch.log1p(-q)
            if c.act_norm > 0:
                scale = (c.act_norm / q.sum(-1, keepdim=True).clamp_min(1e-6)).clamp(max=1.0)
                q = (q * scale).clamp(1e-5, 1 - 1e-5)
                q_logit = torch.log(q) - torch.log1p(-q)
            if si_gid is not None:
                ql_new = q_logit - self.si_l * (slot_lse(q_logit, si_gid) - q_logit)
                q = (q + (torch.sigmoid(ql_new) - torch.sigmoid(q_logit))).clamp(1e-5, 1 - 1e-5)
                q_logit = ql_new
            own = None
            if c.anchor_clamp:
                own = torch.zeros_like(q); own[:, :, G["owner"]] = 1.0
            if c.lex_anchor > 0 and c.anchor_teach <= 0 and "lex" in G:
                anc = (G["lex"] >= c.lex_anchor).to(q.dtype)[:, None, :].expand_as(q)
                own = anc if own is None else torch.maximum(own, anc)
            if own is not None:
                q = (own * (1 - 1e-5) + (1 - own) * q).clamp(1e-5, 1 - 1e-5)
                q_logit = torch.log(q) - torch.log1p(-q)
            if _step == 0:
                q_h1 = q
            pk = prior.detach() if c.prior_detach else prior
            kl = q * (torch.log(q) - torch.log(pk)) + (1 - q) * (torch.log1p(-q) - torch.log1p(-pk))
            if own is not None:
                kl = kl * (1 - own)
            if wc is not None:
                kl = kl * (1 - wc)
            if persist:
                kl = kl * (1 - Vin)
            if c.free_bits > 0:
                kl = (kl - c.free_bits).clamp_min(0.0)
            kl_total = kl_total + (kl.mean() if c.kl_norm == "per_unit" else kl.sum(-1).mean() / 100.0)

            if self.training:
                u = u_shared if u_shared is not None else torch.rand_like(q).clamp(1e-6, 1 - 1e-6)
                noise = torch.log(u) - torch.log1p(-u)
                soft = torch.sigmoid((q_logit + noise) / c.gumbel_temp)
                hard = (soft > 0.5).to(soft.dtype)
                z_cur = hard + soft - soft.detach()
            else:
                z_cur = (q > c.accept_thr).to(q.dtype)
            if wc is not None:
                z_cur = wc + (1 - wc) * z_cur
            if persist:
                z_new = z_cur
                acc = (z_new > 0.5) & (Vin <= 0)
                if self.training:
                    z_cur = Vin * z_in + (1 - Vin) * z_new
                    zm = Vin * zm + (1 - Vin) * z_new
                else:
                    z_cur = torch.where(Vin > 0, z_in, z_new)
                    zm = torch.where(Vin > 0, zm, z_new)
                q_acc = torch.where(acc, q, q_acc)
                newm, changed = acc, acc.flatten(1).any(1)
                Vin = torch.maximum(Vin, acc.to(a.dtype))
                z_any = torch.maximum(z_any, zm); s_any = torch.maximum(s_any, zm * q_acc)
            else:
                z_any = torch.maximum(z_any, z_cur); s_any = torch.maximum(s_any, z_cur * q)
                if path is not None or halt:
                    cur_z = z_cur > 0
                    newm, changed = cur_z & ~prev_z, (cur_z ^ prev_z).flatten(1).any(1)
                    prev_z = cur_z
            if roles_on:
                act = (zm if persist else z_cur).detach() > 0.5
                newly = act & ~(dwell > 0)
                first = torch.where(newly, torch.full_like(first, float(n_run)), first)
                dwell = dwell + act.to(a.dtype)
                if msg is not None:
                    m_d = msg.detach()
                    dstE = G["dst"][None, None].expand_as(m_d)
                    mxi = torch.full_like(a, -1.0).scatter_reduce(2, dstE, m_d, "amax", include_self=True)
                    hit = ((m_d == mxi.gather(2, dstE)) & (m_d > 0) & newly.gather(2, dstE)).to(a.dtype)
                    propd = torch.maximum(propd, torch.zeros_like(a).scatter_reduce(2, src_idx[None, None].expand_as(m_d), hit, "amax", include_self=True))
            if path is not None and msg is not None:
                ma = msg.detach().max(1).values
                dst_e = G["dst"][None].expand_as(ma)
                mxa = torch.full((B, a.shape[-1]), -1.0, device=a.device).scatter_reduce(1, dst_e, ma, "amax")
                hit = (ma == mxa.gather(1, dst_e)) & (ma > 0)
                eidx = torch.arange(ma.shape[-1], device=a.device)[None].expand_as(ma)
                best = torch.full((B, a.shape[-1]), -1, dtype=torch.long, device=a.device).scatter_reduce(
                    1, dst_e, torch.where(hit, eidx, torch.full_like(eidx, -1)), "amax")
                newly = newm.any(1) & (best >= 0)
                bi = best.clamp_min(0)
                par, rl = G["src"][bi], G["rel"][bi]
                pi = path["pi"]
                pi_new = torch.tanh(pm.Wa(pi.gather(1, par[..., None].expand(-1, -1, pi.shape[-1]))) + WbC[rl])
                path["hop"] += 1
                path["pi"] = torch.where(newly[..., None], pi_new, pi)
                path["tau"] = torch.where(newly, torch.full_like(path["tau"], float(path["hop"])), path["tau"])
            if self._dbg is not None:
                self._dbg.append((zm if persist else z_cur).detach())
            if halt:
                done = done | ((~changed) & torch.tensor(n_run >= 2 or persist, device=a.device))
                if bool(done.all()):
                    break
        if path is not None:
            path["P"] = P.mean(1)
        z = z_cur
        if persist:
            z = zm
            q = torch.where(Vin > 0, q_acc, q)
        prov = torch.where(p > eps + 1e-3, 1, torch.where(z_prev > 0, 2, 0)) if not self.training else None
        delta, overflow = None, None
        if write and c.mode != "no_write":
            u = (z * q).max(1).values
            K = min(c.K, u.shape[1])
            overflow = (u > 0).sum(1) > K
            uv, ui = torch.topk(u, K, dim=1)
            bias = torch.where(uv > 0, torch.log(uv.clamp_min(1e-6)), torch.full_like(uv, float("-inf")))
            wb_u = None
            if c.wb_unit:
                Eu = G["E"][ui].float()
                lgw = torch.einsum("btk,bjk->btj", self.Wtq(Hf), self.Wwk_u(Eu)) / c.d_k ** 0.5 + bias[:, None, :]
                lgw = torch.cat([lgw, self.null_logit.expand(B, lgw.shape[1], 1)], -1)
                att = torch.softmax(lgw, -1)[..., :K]
                delta = torch.sigmoid(self.wb_gate) * (att @ self.Wwv_u(Eu))
                wb_u = att
            elif c.tok_mem and "TOK" in G:
                Tu = G["TOK"].shape[1]
                tokk = G["TOK"][ui].float()
                keys = self.Wtk(tokk).reshape(B, K * Tu, -1)
                vals = self.Wtv(tokk).reshape(B, K * Tu, -1)
                tb = (bias[:, :, None] + torch.where(tok_wb_mask(G, c)[ui], 0.0, float("-inf"))).reshape(B, K * Tu)
                lgw = torch.einsum("btk,bjk->btj", self.Wtq(Hf), keys) / c.d_k ** 0.5 + tb[:, None, :]
                lgw = torch.cat([lgw, self.null_logit.expand(B, lgw.shape[1], 1)], -1)
                att = torch.softmax(lgw, -1)[..., :K * Tu]
                delta = torch.sigmoid(self.wb_gate) * (att @ vals)
                if c.wb_aux > 0 and self.training:
                    wb_u = att.reshape(B, att.shape[1], K, Tu).sum(-1)
            else:
                Eu = G["E"][ui]
                lgw = torch.einsum("btk,bjk->btj", self.Wtq(Hf), self.Wuk(Eu)) / c.d_k ** 0.5 + bias[:, None, :]
                lgw = torch.cat([lgw, self.null_logit.expand(B, lgw.shape[1], 1)], -1)
                delta = torch.sigmoid(self.wb_gate) * (torch.softmax(lgw, -1)[..., :K] @ self.Wv(Eu))
        aux = {}
        if c.unit_roles:
            term = (dwell > 0).to(a.dtype) * (1 - propd)
            aux["roles"] = torch.stack([anc_r, (dwell / 8.0).clamp(max=1.0), propd, term, first / 8.0], -1).max(1).values
        if c.wb_aux > 0 and self.training and delta is not None and wb_u is not None:
            aux = {"wb_att_u": wb_u, "wb_ui": ui, "wb_sel": uv > 0, "wb_tmask": tok_mask}
        return {"s": z * q, "s_any": s_any, "z": z, "z_any": z_any, "q": q, "q_h1": q_h1, "a": a, "p": p, "delta": delta, "div": div, "bp": bp, "prov": prov,
                "prov_rel": prov_rel.max(1).values if prov_rel is not None else None,
                "overflow": overflow, "sit": sit, "need": need, "need_unit": need_unit, "kl": kl_total, "gate": gate_mass / max(1, n_run), "t_edge": t_edge,
                **aux}


    def _sparse_hops(self, Hf, P, G, z_prev, sit, need, need_unit, gq, st, div, write, B):
        c, eps, n = self.c, self.c.eps_prior, G["E"].shape[0]
        dev = Hf.device
        tk = gmemo(G, self, "Wtk", lambda: tok_proj(self.Wtk, G["TOK"]))
        qP = self.Wq(P)
        Cr = gmemo(G, self, "WcC", lambda: self.Wc(G["C"] - self.rel_mean))
        g = torch.sigmoid(self.gate_tau * torch.einsum("bmr,qr->bmq", F.normalize(gq, dim=-1), F.normalize(Cr, dim=-1)) + self.gate_b)
        if "rel_scale" in G:
            g = g * G["rel_scale"][None, None, :]
        is_fact = G["is_fact"]
        lexq_all = G.get("lex_q", G["lex"]) if "lex" in G else None
        anc_ent = torch.zeros(n, dtype=torch.bool, device=dev); anc_ent[G["owner"]] = True
        if "lex" in G:
            anc_ent |= (G["lex"] >= max(c.lex_anchor, 1e-6)).any(0)
        if c.dlg_state and "prev" in G:
            anc_ent |= (G["prev"] > 0).any(0)
        anc_ent &= (is_fact == 0)
        z_cur = z_prev
        z_any = torch.zeros(B, c.M, n, device=dev); s_any = torch.zeros_like(z_any)
        q_full = p_full = None
        a_full = torch.zeros(B, c.M, n, device=dev)
        bp = torch.full((B, c.M, n), -1, dtype=torch.long, device=dev)
        const_q = torch.where(is_fact > 0, torch.full_like(is_fact, 1e-4), torch.full_like(is_fact, 1e-5)).float()
        for _step in range(c.steps):
            z_in = z_cur
            act = (z_in > 0).any(1).any(0)
            emask = act[G["src"]]
            eid = emask.nonzero().flatten()
            src, dst, rel = G["src"][eid], G["dst"][eid], G["rel"][eid]
            cand = anc_ent.clone()
            cand[dst] = True
            cand |= (is_fact > 0) & act[G["subj"]]
            cidx = cand.nonzero().flatten()
            pos = torch.full((n,), -1, dtype=torch.long, device=dev); pos[cidx] = torch.arange(cidx.numel(), device=dev)
            at = torch.einsum("bmk,ntk->bmnt", qP, tk[cidx]) / c.d_k ** 0.5
            a_c = at.masked_fill(~G["TOKM"][cidx][None, None], -1e4).max(-1).values
            a_full[:, :, cidx] = a_c

            log1m = torch.zeros(B, c.M, n, device=dev)
            if eid.numel():
                gr = g[:, :, rel]
                msg = gr * z_in[:, :, src]
                grc = gr.clamp(1e-6, 1 - 1e-6)
                ge = torch.log(grc) - torch.log1p(-grc) + a_c[:, :, pos[dst]]
                ge = ge + is_fact[dst] * torch.log(need_unit[:, :, dst].clamp_min(1e-4))
                if c.lex_feat and "lex" in G:
                    ge = ge + self.lex_w * G["lex"][:, None, dst]
                src_e = src[None, None].expand_as(ge)
                mx = torch.full((B, c.M, n), -1e4, device=dev).scatter_reduce(2, src_e, ge, "amax", include_self=True)
                ex = torch.exp(ge - mx.gather(2, src_e))
                den = torch.zeros(B, c.M, n, device=dev).scatter_add(2, src_e, ex)
                share = (c.fanout_m * ex / den.gather(2, src_e).clamp_min(1e-6)).clamp(max=1.0)
                msg = msg * share
                dst_e = dst[None, None].expand_as(msg)
                log1m = seg_sum_sparse(torch.log1p(-msg.clamp(max=1 - 1e-5)), dst, n)
                mxm = torch.full((B, c.M, n), -1.0, device=dev).scatter_reduce_(2, dst_e, msg, "amax")
                hit = (msg == mxm.gather(2, dst_e)) & (msg > 0)
                bp = torch.full((B, c.M, n), -1, dtype=torch.long, device=dev).scatter_reduce_(
                    2, dst_e, torch.where(hit, eid[None, None].expand_as(msg), torch.full_like(dst_e, -1)), "amax")
            p_full = 1 - (1 - eps) * torch.exp(log1m)

            p = p_full[:, :, cidx]
            prior = ((1 - is_fact[cidx]) * p + is_fact[cidx] * p * need_unit[:, :, cidx]).clamp(1e-5, 1 - 1e-5)
            prior_logit = torch.log(prior) - torch.log1p(-prior)
            feats = torch.cat([a_c[..., None], p[..., None], z_in[:, :, cidx][..., None], st[:, :, cidx], need_unit[:, :, cidx][..., None],
                               prior_logit[..., None]], -1)
            if c.lex_phi:
                lx = lexq_all[:, None, cidx, None].expand(B, c.M, -1, 1) if lexq_all is not None else torch.zeros_like(feats[..., :1])
                if c.lex_fact_reach and lexq_all is not None:
                    lx = lx * ((1 - is_fact[cidx]) + is_fact[cidx] * (p > eps + 1e-3).to(lx.dtype))[..., None]
                feats = torch.cat([feats, lx], -1)
            q_logit = self.phi(feats).squeeze(-1)
            if c.lex_feat and lexq_all is not None:
                lexq = lexq_all[:, None, cidx].expand_as(q_logit)
                if c.lex_fact_reach:
                    lexq = lexq * ((1 - is_fact[cidx]) + is_fact[cidx] * (p > eps + 1e-3).to(lexq.dtype))
                q_logit = q_logit + 10.0 * self.lex_q * lexq
            if c.dlg_state and "prev" in G:
                q_logit = q_logit + 10.0 * self.prev_q * G["prev"][:, None, cidx]
            q = torch.sigmoid(q_logit).clamp(1e-5, 1 - 1e-5)
            reach = ((p - eps) / (1 - eps)).clamp(0.0, 1.0)
            anc = (cidx == G["owner"]).float()[None, None].expand_as(q).clone()
            if "lex" in G:
                anc = torch.maximum(anc, (G["lex"][:, cidx] >= max(c.lex_anchor, 1e-6)).to(q.dtype)[:, None, :].expand_as(q))
            if c.dlg_state and "prev" in G:
                anc = torch.maximum(anc, torch.sigmoid(self.prev_a) * G["prev"][:, None, cidx].to(q.dtype))
            ent = (1 - is_fact[cidx])[None, None]
            q = (q * (ent * (anc + (1 - anc) * reach) + (1 - ent))).clamp(1e-5, 1 - 1e-5)
            sup = (1 - is_fact[cidx]) + is_fact[cidx] * z_in[:, :, G["subj"][cidx]]
            q = (sup * q + (1 - sup) * 1e-4).clamp(1e-5, 1 - 1e-5)
            own = (cidx == G["owner"]).float()[None, None]
            if c.lex_anchor > 0 and c.anchor_teach <= 0 and "lex" in G:
                own = torch.maximum(own, (G["lex"][:, cidx] >= c.lex_anchor).to(q.dtype)[:, None, :])
            q = (own * (1 - 1e-5) + (1 - own) * q).clamp(1e-5, 1 - 1e-5)
            q_full = const_q[None, None].expand(B, c.M, n).clone()
            q_full[:, :, cidx] = q
            z_cur = (q_full > c.accept_thr).to(q_full.dtype)
            z_any = torch.maximum(z_any, z_cur); s_any = torch.maximum(s_any, z_cur * q_full)
        z, q = z_cur, q_full
        prov = torch.where(p_full > eps + 1e-3, 1, torch.where(z_prev > 0, 2, 0))
        delta, overflow = None, None
        if write and c.mode != "no_write":
            u = (z * q).max(1).values
            K = min(c.K, u.shape[1])
            overflow = (u > 0).sum(1) > K
            uv, ui = torch.topk(u, K, dim=1)
            bias = torch.where(uv > 0, torch.log(uv.clamp_min(1e-6)), torch.full_like(uv, float("-inf")))
            Tu = G["TOK"].shape[1]
            tokk = G["TOK"][ui].float()
            keys = self.Wtk(tokk).reshape(B, K * Tu, -1)
            vals = self.Wtv(tokk).reshape(B, K * Tu, -1)
            tb = (bias[:, :, None] + torch.where(G["TOKM"][ui], 0.0, float("-inf"))).reshape(B, K * Tu)
            lgw = torch.einsum("btk,bjk->btj", self.Wtq(Hf), keys) / c.d_k ** 0.5 + tb[:, None, :]
            lgw = torch.cat([lgw, self.null_logit.expand(B, lgw.shape[1], 1)], -1)
            delta = torch.sigmoid(self.wb_gate) * (torch.softmax(lgw, -1)[..., :K * Tu] @ vals)
        def a_fn(idx, qP=qP, tk=tk):
            return (torch.einsum("bmk,ntk->bmnt", qP, tk[idx]) / c.d_k ** 0.5).masked_fill(~G["TOKM"][idx][None, None], -1e4).max(-1).values
        return {"a_fn": a_fn, "s": z * q, "s_any": s_any, "z": z, "z_any": z_any, "q": q, "q_h1": None, "a": a_full, "p": p_full, "delta": delta, "div": div,
                "bp": bp, "prov": prov, "prov_rel": None, "overflow": overflow, "sit": sit, "need": need, "need_unit": need_unit,
                "kl": torch.zeros((), device=dev), "gate": torch.zeros((), device=dev), "t_edge": None}


class VecGraphLayer(nn.Module):

    def __init__(self, c: Cfg, d_g=128):
        super().__init__()
        self.c, self.d_g = c, d_g
        d, de = c.d, c.d_e
        self.ln = nn.LayerNorm(d)
        self.init = nn.Linear(de, d_g)
        self.nq = nn.Linear(d_g, d_g, bias=False)
        self.tk = nn.Linear(d, d_g, bias=False)
        self.tv = nn.Linear(d, d_g, bias=False)
        self.msg = nn.Linear(d_g, d_g, bias=False)
        self.relg = nn.Linear(de, d_g)
        self.upd = nn.Sequential(nn.Linear(3 * d_g, 2 * d_g), nn.GELU(), nn.Linear(2 * d_g, d_g))
        self.lnh = nn.LayerNorm(d_g)
        self.score = nn.Linear(d_g, 1)
        self.Wtq = nn.Linear(d, d_g, bias=False)
        self.Wv = nn.Linear(d_g, d, bias=False)
        nn.init.normal_(self.Wv.weight, std=0.002)
        self.null_logit = nn.Parameter(torch.tensor(0.0))
        self.wb_gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, H, tok_mask, G, h_prev, write=True):
        c = self.c
        Hf = self.ln(H.float())
        B, n = Hf.shape[0], G["E"].shape[0]
        h = h_prev if h_prev is not None else self.init(G["E"])[None].expand(B, -1, -1)
        att = torch.einsum("bnk,btk->bnt", self.nq(h), self.tk(Hf)) / self.d_g ** 0.5
        att = att.masked_fill(~tok_mask[:, None, :], -1e4)
        a = att.max(-1).values
        m_tok = torch.softmax(att, -1) @ self.tv(Hf)
        if G["src"].numel():
            gate = torch.sigmoid(self.relg(G["C"]))[G["rel"]]
            msg = self.msg(h)[:, G["src"]] * gate[None]
            deg = torch.zeros(n, device=h.device).index_add_(0, G["dst"], torch.ones_like(G["dst"], dtype=torch.float))
            m_g = torch.zeros_like(h).index_add_(1, G["dst"], msg) * (1 + deg).rsqrt()[None, :, None]
        else:
            m_g = torch.zeros_like(h)
        h = self.lnh(h + self.upd(torch.cat([h, m_tok, m_g], -1)))
        sc = torch.sigmoid(self.score(h).squeeze(-1))
        delta = None
        if write:
            K = min(c.K, n)
            uv, ui = torch.topk(sc, K, dim=1)
            hu = torch.gather(h, 1, ui[..., None].expand(-1, -1, self.d_g))
            lgw = torch.einsum("btk,bjk->btj", self.Wtq(Hf), hu) / self.d_g ** 0.5 + torch.log(uv + 1e-6)[:, None, :]
            lgw = torch.cat([lgw, self.null_logit.expand(B, lgw.shape[1], 1)], -1)
            delta = torch.sigmoid(self.wb_gate) * (torch.softmax(lgw, -1)[..., :K] @ self.Wv(hu))
        s_rep = sc[:, None, :]
        return {"s": s_rep, "a": a[:, None, :], "p": m_g.norm(dim=-1)[:, None, :], "delta": delta, "h": h,
                "div": torch.zeros((), device=h.device), "bp": None, "prov": None, "overflow": None}


class PathMod(nn.Module):
    DP = 32

    def __init__(self, d_e, d, dp=DP):
        super().__init__()
        self.dp = dp
        self.pi0 = nn.Parameter(torch.zeros(dp))
        self.Wa = nn.Linear(dp, dp, bias=False)
        self.Wb = nn.Linear(d_e, dp)
        self.Wpq = nn.Linear(d, dp)


class SetReadout(nn.Module):
    TMAX, CMAX = 64, 8

    def __init__(self, c):
        super().__init__()
        D, h = c.setro_dim, c.setro_heads
        assert D % h == 0, (D, h)
        self.h = h
        self.q_in = nn.Linear(c.d, D)
        self.t_in = nn.Linear(c.d_e, D)
        self.e_in = nn.Linear(c.d_e, D)
        self.f_in = nn.Linear(5, D)
        self.pos = nn.Embedding(1 + self.TMAX + self.CMAX, D)
        self.blocks = nn.ModuleList()
        for _ in range(c.setro_layers):
            self.blocks.append(nn.ModuleDict({
                "ln1": nn.LayerNorm(D), "sa": nn.Linear(D, 3 * D), "sa_o": nn.Linear(D, D),
                "ln2": nn.LayerNorm(D), "ca_q": nn.Linear(D, D), "ca_kv": nn.Linear(D, 2 * D), "ca_o": nn.Linear(D, D),
                "ln3": nn.LayerNorm(D), "ua": nn.Linear(D, 3 * D), "ua_o": nn.Linear(D, D),
                "ln4": nn.LayerNorm(D), "ff": nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))}))
        self.sib = nn.Parameter(torch.zeros(c.setro_layers, h))
        self.ln_o = nn.LayerNorm(D)
        self.out = nn.Linear(D, 2)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        if c.setro_conf:
            self.conf_u = nn.Linear(D, 1)
            self.conf_s = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, 2))

    def _heads(self, x):
        return x.unflatten(-1, (self.h, -1)).transpose(-3, -2)

    def forward(self, Hr, mask, G, top, valid, lg):
        B, N = top.shape
        tok = G["TOK"][:, :self.TMAX]; tm = G["TOKM"][:, :self.TMAX]
        parts = [self.t_in(tok[top].float())]
        masks = [tm[top]]
        if "CHK" in G:
            parts.append(self.t_in(G["CHK"][:, :self.CMAX][top].float()))
            masks.append(G["CHKM"][:, :self.CMAX][top])
        T = parts[0].shape[2]
        cls = self.e_in(G["E"][top].float()) + self.f_in(torch.cat([lg.float().clamp(-20, 20) / 5, G["is_fact"][top][..., None].float()], -1))
        x = torch.cat([cls[:, :, None]] + parts, 2)
        S = x.shape[2]
        pid = torch.cat([torch.arange(1 + T, device=x.device), 1 + self.TMAX + torch.arange(S - 1 - T, device=x.device)])
        x = x + self.pos(pid)
        km = torch.cat([torch.ones(B, N, 1, dtype=torch.bool, device=x.device)] + masks, 2)
        r = self.q_in(Hr.float())
        rmask = mask[:, None, None, :]
        same = (G["subj"][top][:, :, None] == G["subj"][top][:, None, :]).float()
        eye = torch.eye(N, dtype=torch.bool, device=x.device)
        ok = (valid[:, None, :] | eye)[:, None]
        for li, blk in enumerate(self.blocks):
            q, k, v = blk["sa"](blk["ln1"](x).reshape(B * N, S, -1)).chunk(3, -1)
            o = F.scaled_dot_product_attention(self._heads(q), self._heads(k), self._heads(v), attn_mask=km.reshape(B * N, 1, 1, S))
            x = x + blk["sa_o"](o.transpose(-3, -2).flatten(-2)).reshape(B, N, S, -1)
            q = self._heads(blk["ca_q"](blk["ln2"](x)).reshape(B, N * S, -1))
            k, v = blk["ca_kv"](r).chunk(2, -1)
            o = F.scaled_dot_product_attention(q, self._heads(k), self._heads(v), attn_mask=rmask)
            x = x + blk["ca_o"](o.transpose(-3, -2).flatten(-2)).reshape(B, N, S, -1)
            c0 = x[:, :, 0]
            q, k, v = blk["ua"](blk["ln3"](c0)).chunk(3, -1)
            bias = (self.sib[li][None, :, None, None] * same[:, None]).masked_fill(~ok, float("-inf"))
            o = F.scaled_dot_product_attention(self._heads(q), self._heads(k), self._heads(v), attn_mask=bias)
            x = torch.cat([(c0 + blk["ua_o"](o.transpose(-3, -2).flatten(-2)))[:, :, None], x[:, :, 1:]], 2)
            x = x + blk["ff"](blk["ln4"](x))
        h = self.ln_o(x[:, :, 0])
        conf = None
        if hasattr(self, "conf_u"):
            vm = valid[..., None].to(h.dtype)
            pool = torch.cat([(h * vm).sum(1) / vm.sum(1).clamp_min(1), h.masked_fill(~valid[..., None], -1e4).amax(1).clamp_min(-1e3)], -1)
            conf = (self.conf_u(h).squeeze(-1), self.conf_s(pool))
        return self.out(h) * valid[..., None].to(x.dtype), conf


class CrossReadout(nn.Module):
    UMAX = 128

    def __init__(self, c):
        super().__init__()
        D, h = c.px_dim, c.px_heads
        assert D % h == 0, (D, h)
        self.h, self.rq = h, c.px_req_tok
        self.q_in = nn.Linear(c.d, D)
        self.t_in = nn.Linear(c.d_e, D)
        self.e_in = nn.Linear(c.d_e, D)
        self.f_in = nn.Linear(5, D)
        self.pos = nn.Embedding(1 + self.UMAX + c.px_req_tok, D)
        self.typ = nn.Embedding(3, D)
        self.blocks = nn.ModuleList([nn.ModuleDict({"ln1": nn.LayerNorm(D), "qkv": nn.Linear(D, 3 * D), "o": nn.Linear(D, D),
                                                    "ln2": nn.LayerNorm(D), "ff": nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))})
                                     for _ in range(c.px_layers)])
        self.ln_o = nn.LayerNorm(D)
        self.out = nn.Linear(D, 2)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, Hr, mask, G, top, valid, lg):
        B, N = top.shape
        if "TOKX" in G:
            tok, tm = G["TOKX"][:, :self.UMAX], G["TOKXM"][:, :self.UMAX]
        else:
            tok, tm = G["TOK"][:, :self.UMAX], G["TOKM"][:, :self.UMAX]
        T = tok.shape[1]
        ut = self.t_in(tok[top].float())
        um = tm[top]
        L = mask.shape[1]; R = min(self.rq, L)
        order = torch.sort((~mask).to(torch.int8), dim=1, stable=True).indices[:, :R]
        rt = self.q_in(Hr.float().gather(1, order[..., None].expand(-1, -1, Hr.shape[-1])))
        rm = mask.gather(1, order)
        cls = self.e_in(G["E"][top].float()) + self.f_in(torch.cat([lg.float().clamp(-20, 20) / 5, G["is_fact"][top][..., None].float()], -1))
        x = torch.cat([cls[:, :, None], ut, rt[:, None].expand(-1, N, -1, -1)], 2)
        S = x.shape[2]
        dev = x.device
        pid = torch.cat([torch.arange(1 + T, device=dev), 1 + self.UMAX + torch.arange(R, device=dev)])
        tid = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), torch.ones(T, dtype=torch.long, device=dev), torch.full((R,), 2, dtype=torch.long, device=dev)])
        x = x + self.pos(pid) + self.typ(tid)
        km = torch.cat([torch.ones(B, N, 1, dtype=torch.bool, device=dev), um, rm[:, None].expand(-1, N, -1)], 2).reshape(B * N, 1, 1, S)
        x = x.reshape(B * N, S, -1)
        for blk in self.blocks:
            q, k, v = blk["qkv"](blk["ln1"](x)).chunk(3, -1)
            hd = lambda t: t.unflatten(-1, (self.h, -1)).transpose(1, 2)
            o = F.scaled_dot_product_attention(hd(q), hd(k), hd(v), attn_mask=km)
            x = x + blk["o"](o.transpose(1, 2).flatten(-2))
            x = x + blk["ff"](blk["ln2"](x))
        return self.out(self.ln_o(x[:, 0])).reshape(B, N, 2) * valid[..., None].to(x.dtype)


class PairSelector(nn.Module):
    D = 512

    def __init__(self, c):
        super().__init__()
        R = c.pair_sel_rank
        din = 256 + c.d_e
        self.trunk = nn.Sequential(nn.LayerNorm(din), nn.Linear(din, self.D), nn.GELU())
        self.A = nn.Linear(self.D, R)
        self.B = nn.Linear(self.D, R, bias=False)
        self.H = nn.Linear(c.d_h, R)
        nn.init.zeros_(self.B.weight)
        nn.init.zeros_(self.H.weight); nn.init.ones_(self.H.bias)
        self.bias = nn.Parameter(torch.zeros(3))
        if c.pair_sel_gate:
            self.gate = nn.Linear(c.d_h + 8, 1)
            nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)
        self.scale = R ** -0.5

    def forward(self, hid, E, q, subj, uid):
        x = self.trunk(torch.cat([hid.float(), E.float()], -1))
        a = self.A(x) * self.H(q.float())[:, None]
        P = torch.einsum("bir,bjr->bij", a, self.B(x)) * self.scale
        rel = torch.stack([subj[:, :, None] == subj[:, None, :], subj[:, None, :] == uid[:, :, None], subj[:, :, None] == uid[:, None, :]], -1)
        return P + (rel.to(P.dtype) * self.bias).sum(-1)


def pair_greedy(s, P, valid, k, agg="last"):
    B, N = s.shape
    base = s.float().masked_fill(~valid, -1e8)
    chosen = torch.zeros(B, N, dtype=torch.bool, device=s.device)
    key = torch.arange(N, device=s.device, dtype=torch.float32)[None].expand(B, -1) + k
    acc = torch.zeros(B, N, device=s.device, dtype=torch.float32)
    for t in range(k):
        j = (base + acc).masked_fill(chosen, -1e9).argmax(1, keepdim=True)
        chosen = chosen.scatter(1, j, True)
        key = key.scatter(1, j, torch.where(valid.gather(1, j), float(t), key.gather(1, j)))
        row = P.gather(1, j[..., None].expand(-1, 1, N)).squeeze(1).float()
        acc = row if (agg == "last" or t == 0) else (torch.maximum(acc, row) if agg == "max" else acc + (row - acc) / (t + 1))
    return key.argsort(1)


def _merge_rows(x, y, idx):
    if torch.is_tensor(x) and torch.is_tensor(y) and x.dim() > 0 and x.shape[1:] == y.shape[1:] and y.shape[0] == idx.numel():
        return x.index_copy(0, idx, y.to(x.dtype))
    return x


class LoRALinear(nn.Module):

    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.randn(r, base.in_features) * (1 / r))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + ((x.float() @ self.A.t()) @ self.B.t() * self.scale).to(x.dtype)


class GraphFusedModel(nn.Module):
    def __init__(self, c: Cfg, device="cuda", sit_E=None):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.c = c
        if c.mode == "full":
            c.mode = "v3"
        if c.schema is None:
            c.schema = c.mode not in NO_SCHEMA_MODES
        if c.schema:
            from .situations import N_LATENT
            base = sit_E.float() if sit_E is not None else torch.randn(c.n_sit - N_LATENT, c.d_e) * 0.02
            self.sit_E = nn.Parameter(torch.cat([base, torch.randn(N_LATENT, c.d_e, device=base.device) * 0.02], 0))
            c.n_sit = self.sit_E.shape[0]
        self.tok = AutoTokenizer.from_pretrained(TOKENIZER or QWEN)
        lm = AutoModelForCausalLM.from_pretrained(QWEN, dtype=torch.bfloat16, attn_implementation="sdpa").model
        lm.layers = lm.layers[:c.n_lm_layers]
        for prm in lm.parameters():
            prm.requires_grad_(False)
        self.lm = lm
        self._compile_graph = False
        if c.extra.get("lora", True):
            start = min(c.graph_after) if c.graph_after else 0
            for li in range(start, c.n_lm_layers):
                att = getattr(lm.layers[li], "self_attn", None)
                if att is None or not hasattr(att, "q_proj"):
                    continue
                att.q_proj = LoRALinear(att.q_proj)
                att.v_proj = LoRALinear(att.v_proj)
        after = tuple(c.n_lm_layers for _ in c.graph_after) if c.mode == "late" else c.graph_after
        self.after = after if c.mode != "bsym_c" else ()
        Layer = VecGraphLayer if c.mode == "vec_gnn" else (ElboGraphLayer if c.elbo else GraphLayer)
        self.layers = nn.ModuleList([Layer(c) for _ in self.after]) if self.after else nn.ModuleList()
        if c.mode == "bsym_c":
            self.content = GraphLayer(Cfg(**{**c.__dict__, "mode": "no_prop"}))
        self.L = max(1, len(self.after))

        self.Wr = nn.Linear(c.d_e, c.d_h)
        self.Wrq = nn.Linear(c.d, c.d_h)
        self.state_emb = nn.Embedding(5, 4)

        self.ln_r = nn.LayerNorm(c.d)
        self.Wxu = nn.Linear(c.d_e, c.d_h, bias=False)
        self.Wxt = nn.Linear(c.d, c.d_h, bias=False)
        self.Wxv = nn.Linear(c.d, c.d_h, bias=False)
        n_stats = (4 if c.schema else 3) * self.L + 4 + 1 + 1 + (2 if c.tok_readout else 0) + (2 if c.read_lex else 0) + (4 if c.read_lex2 else 0) + (4 if c.dlg_state else 0)
        if c.mode == "vec_gnn":
            self.Wh = nn.Linear(128, c.d_h)
        if c.prov_feat:
            self.Wprov = nn.Linear(c.d_r, c.d_h)
        if c.mod_comp:
            self.mc_a = nn.Parameter(torch.full((2,), -3.0))
        if c.turn_gate:
            self.turn_head = nn.Sequential(nn.Linear(c.d_h + 4 + 2, 64), nn.GELU(), nn.Linear(64, 1))
            nn.init.constant_(self.turn_head[2].bias, 3.0)
        self.unit_head = nn.Sequential(nn.Linear(2 * c.d_h + n_stats + (2 * c.d_h if c.prov_feat else 0) + (PathMod.DP + 2 if c.read_path else 0), 256), nn.GELU(), nn.Linear(256, 4))
        if c.time_feat:
            self.ro_tW = nn.Parameter(torch.zeros(len(TIME_INTENTS), TIME_NF))
        if c.unit_roles:
            self.role_ro = nn.Linear(ROLE_NF, 4)
            nn.init.zeros_(self.role_ro.weight); nn.init.zeros_(self.role_ro.bias)
        if c.read_path:
            with torch.no_grad():
                self.unit_head[0].weight[:, -(PathMod.DP + 2):].zero_()
        self.comp_alpha = nn.Parameter(torch.full((1,), float(c.read_comp_init)))
        if c.read_ledger:
            self.ledger_g = nn.Parameter(torch.full((1,), 1.0))
            self.ledger_a = nn.Parameter(torch.full((1,), -2.0))
        if c.read_budget:
            self.bud_head = nn.Sequential(nn.Linear(c.d_h + 8, 64), nn.GELU(), nn.Linear(64, 1))
            nn.init.constant_(self.bud_head[2].bias, 1.0)
            self.bud_T = nn.Parameter(torch.full((1,), 1.0))
        self.comp_bias = nn.Parameter(torch.zeros(1))
        if c.setro_m > 0:
            self.setro = SetReadout(c)
        if c.px_m > 0:
            self.px = CrossReadout(c)
        if c.rca_m > 0:
            from .rca import RCABlock
            assert c.rca_read_layer <= c.n_lm_layers and c.rca_l0 < c.rca_read_layer, (c.rca_l0, c.rca_read_layer, c.n_lm_layers)
            self.rca = RCABlock(lm, c)
            self.rca_alpha = nn.Parameter(torch.full((2,), float(c.rca_alpha_init)))
        if c.pair_sel > 0:
            self.pairsel = PairSelector(c)
        if c.rerank_top > 0:
            dr = 128
            self.rr_in = nn.Sequential(nn.Linear(2 * c.d_h + 3 + 3 + 4 + 1, dr), nn.GELU(), nn.Linear(dr, dr))
            self.rr_q = nn.Linear(dr, dr); self.rr_k = nn.Linear(dr, dr); self.rr_v = nn.Linear(dr, dr)
            self.rr_sib = nn.Parameter(torch.zeros(1))
            self.rr_ff = nn.Sequential(nn.LayerNorm(dr), nn.Linear(dr, dr), nn.GELU(), nn.Linear(dr, dr))
            self.rr_out = nn.Linear(dr, 2)
            nn.init.zeros_(self.rr_out.weight); nn.init.zeros_(self.rr_out.bias)
        self.status_head = nn.Sequential(nn.Linear(c.d_h + 4, 128), nn.GELU(), nn.Linear(128, 5))
        self.need_e = nn.Linear(c.d_h, c.d_h)
        self.need_m = nn.Linear(c.d_e, c.d_h)
        if c.read_path:
            self.pathmod = PathMod(c.d_e, c.d)
        for i, l in enumerate(self.layers):
            l.li = i
            if c.read_path:
                l.__dict__["_pm"] = self.pathmod
        for name, v in (("hop_kmax", c.hop_kmax), ("hop_train_k", c.hop_train_k)):
            assert v is None or len(v) == len(self.layers), f"{name} needs one entry per graph layer ({len(self.layers)}): {v}"
        if c.wide_entry > 0:
            assert c.wide_entry_score in ("lex", "dense", "hyb"), c.wide_entry_score
            assert c.switch_theta <= 0, "wide_entry: per-request switching (suffix re-run) is not supported"
            if c.wide_entry_score in ("dense", "hyb"):
                assert c.de_req in ("own", "emb") and (c.de_req == "own" or c.de_keys), (c.de_req, c.de_keys)
                if c.de_req == "own":
                    self.de_ln = nn.LayerNorm(c.d)
                    kq = (c.de_key_dim or c.d_e) if c.de_keys else c.de_dim
                    if c.de_qpool == "attn":
                        self.de_att = nn.Linear(c.d, 4)
                        self.de_mlp = nn.Sequential(nn.Linear(5 * c.d, 2048), nn.GELU(), nn.Linear(2048, kq))
                    elif c.de_qpool == "split":
                        self.de_att = nn.Linear(c.d, 4)
                        self.de_att_h = nn.Linear(c.d, 4)
                        self.de_mlp = nn.Sequential(nn.Linear(10 * c.d + 1, 2048), nn.GELU(), nn.Linear(2048, kq))
                    else:
                        self.de_q = nn.Linear(c.d, kq)
                    if c.de_qtf > 0:
                        self._make_qtf(lm, c)
                if not c.de_keys:
                    self.de_u = nn.Linear(c.d_e, c.de_dim)
                self.de_tau = nn.Parameter(torch.tensor(3.0))
                if c.de_head > 0:
                    kd = (c.de_key_dim or c.d_e) if c.de_keys else c.de_dim
                    self.dh_w1 = nn.Linear(kd, c.de_head)
                    self.dh_w2 = nn.Linear(c.de_head, 2)
                    nn.init.zeros_(self.dh_w2.weight); nn.init.zeros_(self.dh_w2.bias)
                    if c.de_head_scale:
                        self.dh_ls = nn.Parameter(torch.tensor(math.log(kd)))
                if c.de_read:
                    self.ro_de = nn.Parameter(torch.zeros(2))
        if c.tok_nl > 1:
            self.tok_mix = nn.Parameter(torch.zeros(c.tok_nl))
            with torch.no_grad():
                self.tok_mix[c.tok_mix_ref] = c.tok_mix_init
        if c.tok_adapt > 0 or c.tok_adapt_attn:
            w = max(c.tok_adapt, 64)
            if c.tok_adapt > 0:
                self.ta_down = nn.Linear(c.d_e, c.tok_adapt)
                self.ta_up = nn.Linear(c.tok_adapt, c.d_e)
                nn.init.zeros_(self.ta_up.weight); nn.init.zeros_(self.ta_up.bias)
            if c.tok_adapt_attn:
                self.ta_qkv = nn.Linear(c.d_e, 3 * w)
                self.ta_o = nn.Linear(w, c.d_e)
                nn.init.zeros_(self.ta_o.weight); nn.init.zeros_(self.ta_o.bias)
        if c.tok_subj == "pool":
            self.ts_g = nn.Parameter(torch.tensor(1.0))
        elif c.tok_subj == "add":
            self.ts_add = nn.Linear(c.d_e, c.d_e, bias=False)
            nn.init.zeros_(self.ts_add.weight)
        if c.reader:
            raise NotImplementedError("the cross-reading read-out is not part of this release")
            self.reader = CrossReader(c.reader, LoRALinear, r=c.reader_lora, q_tok=c.reader_q_tok, doc_tok=c.reader_doc_tok)
            self.rd_a = nn.Parameter(torch.tensor(0.5))
            self.rd_w = nn.Parameter(torch.tensor(1.0))
            self.rd_b = nn.Parameter(torch.tensor(0.0))
        self._hooks = []
        self.to(device)
        self.device = device


    def encode_queries(self, texts):
        _pb = int("0")
        b = self.tok(texts, return_tensors='pt', padding=True, truncation=True, max_length=384, **{'pad_to_multiple_of': _pb} if _pb else {})
        if _pb and bool(b["attention_mask"].all()):
            pad = torch.full((b["input_ids"].shape[0], _pb), self.tok.pad_token_id, dtype=b["input_ids"].dtype)
            b = {"input_ids": torch.cat([b["input_ids"], pad], 1), "attention_mask": torch.cat([b["attention_mask"], torch.zeros_like(pad)], 1)}
        return b["input_ids"].to(self.device), b["attention_mask"].to(self.device).bool()


    def _request_inputs(self, texts, G, enc=None):
        c = self.c
        ids, mask = enc if enc is not None else self.encode_queries(texts)
        inp = {"ids": ids, "mask": mask}
        if c.probe_mask_tmpl:
            inp["pmask"] = self._template_mask(texts, mask)
        if (c.time_feat or (c.time_entry or 0) > 0) and "t_lo" in G:
            from .timex import time_request_feats
            inp["t_f"], inp["t_int"], inp["t_es"] = time_request_feats(texts, G["t_lo"], G["t_hi"], G.get("t_ob"), G.get("t_grp"))
        if c.lex_feat and "uw" in G:
            inp["lex"] = request_lex(texts, G["uw"], G["uwt"])
            if c.read_lex2 and "uwd" in G:
                inp["lex_raw"], inp["lex_date"] = request_lex_feats(texts, G["uw"], G["uwt"], G["uwd"])
            if c.dlg_state:
                cur = [t.rsplit("Current request: ", 1)[-1].rsplit("当前请求：", 1)[-1].split("\nEarlier conversation", 1)[0].split("\n此前对话", 1)[0] for t in texts]
                inp["lex_cur"] = request_lex(cur, G["uw"], G["uwt"])
        if c.dlg_state and "prev" in G:
            inp["prev"] = G["prev"]
        if c.wide_entry > 0 and c.de_req == "own" and c.de_qpool == "split":
            inp["qcur"], inp["qhist"] = self._turn_masks(texts, mask)
        if c.reader:
            inp["texts"] = list(texts)
        if (c.wide_entry > 0 and c.wide_entry_score in ("dense", "hyb") and c.de_req == "own" and (c.de_distill > 0 or c.de_rank_distill > 0) and self.training
                and self.__dict__.get("_kenc") is not None):
            with torch.no_grad():
                inp["qkey_t"] = self.__dict__["_kenc"].encode_query(texts).to(self.device)
        if c.wide_entry > 0 and c.wide_entry_score in ("dense", "hyb") and c.de_req == "emb":
            inp["qkey"] = self.__dict__["_kenc"].encode_query(texts).to(self.device)
        return inp

    def _turn_masks(self, texts, mask):
        import re as _re
        from . import lex as _lx
        enc = self.tok(texts, return_offsets_mapping=True, padding=True, truncation=True, max_length=384)
        cur, hist = torch.zeros_like(mask), torch.zeros_like(mask)
        for b, (t, offs) in enumerate(zip(texts, enc["offset_mapping"])):
            tmpl = [m.span() for pat in (_lx._HDR, _lx._HDR_ZH, _lx._LABELS) for m in pat.finditer(t)]
            m = _re.search(r"(?:Current request: |当前请求：)(.*?)(?=\n(?:Earlier conversation|此前对话)|$)", t, _re.S)
            cs, ce = m.span(1) if m else (0, len(t))
            for j, (s0, e0) in enumerate(offs[:mask.shape[1]]):
                chars = [i for i in range(s0, e0) if not t[i].isspace()]
                if any(cs <= i < ce for i in chars):
                    cur[b, j] = True
                elif any(not any(a <= i < z for a, z in tmpl) for i in chars):
                    hist[b, j] = True
        cur, hist = cur & mask, hist & mask
        empty = ~cur.any(1)
        cur[empty] = mask[empty]
        hist[empty] = False
        return cur, hist

    def _template_mask(self, texts, mask):
        from . import lex as _lx
        enc = self.tok(texts, return_offsets_mapping=True, padding=True, truncation=True, max_length=384)
        out = torch.zeros_like(mask)
        for b, (t, offs) in enumerate(zip(texts, enc["offset_mapping"])):
            spans = [m.span() for pat in (_lx._HDR, _lx._HDR_ZH, _lx._LABELS) for m in pat.finditer(t)]
            for j, (s0, e0) in enumerate(offs[:mask.shape[1]]):
                if e0 > s0 and not any(a <= s0 and e0 <= z for a, z in spans):
                    if self.c.probe_mask_ws and not any(ch.isalnum() for ch in t[s0:e0]):
                        continue
                    out[b, j] = True
        out = out & mask
        empty = ~out.any(1)
        out[empty] = mask[empty]
        return out

    def _bind_inputs(self, inp, G):
        c = self.c
        if "pmask" in inp:
            G["_pmask"] = inp["pmask"]
        for k in ("lex", "lex_raw", "lex_date", "lex_cur", "prev", "t_f", "t_int", "t_es"):
            if k in inp:
                G[k] = inp[k]
        if "qkey" in inp:
            G["_qkey"] = inp["qkey"]
        if "qcur" in inp:
            G["_qcur"], G["_qhist"] = inp["qcur"], inp["qhist"]
        G["_qkey_t"] = inp.get("qkey_t")
        if "texts" in inp:
            G["_texts"] = inp["texts"]
        if "lex" in inp:
            G["lex_q"] = inp["lex"] * (1 - G["is_fact"])[None] if c.lex_ent_only else inp["lex"]

    def _make_qtf(self, lm, c):
        import copy
        L0 = c.graph_after[c.wide_entry_layer]
        if c.de_qtf_init == "emb":
            from transformers import AutoModel
            src = AutoModel.from_pretrained(c.extra["de_keys_path"], dtype=torch.float32)
            assert L0 + c.de_qtf <= len(src.layers), (L0, c.de_qtf, len(src.layers))
            layers, norm = [copy.deepcopy(src.layers[i]) for i in range(L0, L0 + c.de_qtf)], copy.deepcopy(src.norm)
            del src
        else:
            assert c.de_qtf_init in ("random", "lm"), c.de_qtf_init
            assert L0 + c.de_qtf <= len(lm.layers) or c.de_qtf_init == "random", (L0, c.de_qtf, len(lm.layers))
            layers = [copy.deepcopy(lm.layers[min(i, len(lm.layers) - 1)]) for i in range(L0, L0 + c.de_qtf)]
            norm = copy.deepcopy(lm.norm)
            if c.de_qtf_init == "random":
                for ly in layers:
                    for mod in ly.modules():
                        if isinstance(mod, nn.Linear):
                            nn.init.normal_(mod.weight, std=0.02)
                            if mod.bias is not None:
                                nn.init.zeros_(mod.bias)
                        elif hasattr(mod, "weight") and isinstance(getattr(mod, "weight"), nn.Parameter) and mod.weight.dim() == 1:
                            nn.init.ones_(mod.weight)
                        if isinstance(mod, LoRALinear):
                            nn.init.zeros_(mod.B)
        self.de_qtf_layers = nn.ModuleList(layers).float()
        self.de_qtf_norm = norm.float()
        for p_ in list(self.de_qtf_layers.parameters()) + list(self.de_qtf_norm.parameters()):
            p_.requires_grad_(True)

    def _qtf(self, H, mask):
        from transformers.masking_utils import create_causal_mask
        x = H.float()
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        cm = create_causal_mask(config=self.lm.config, inputs_embeds=x, attention_mask=mask.long(), past_key_values=None, position_ids=pos)
        pe = self.lm.rotary_emb(x, pos)
        for ly in self.de_qtf_layers:
            x = ly(x, attention_mask=cm, position_embeddings=pe, position_ids=pos, past_key_values=None, use_cache=False)
            x = x[0] if isinstance(x, tuple) else x
        return x

    def _wide_entry(self, H, mask, G):
        c = self.c
        isf = G["is_fact"] > 0
        k = G.get("_wide_k")
        if k is None:
            k = G["_wide_k"] = min(c.wide_entry, int(isf.sum()))
        wide = torch.zeros(H.shape[0], isf.shape[0], device=H.device)
        de = None
        if c.wide_entry_score in ("dense", "hyb"):
            if c.de_req == "emb":
                qv = G["_qkey"].float()
            elif c.de_qtf > 0 and c.de_qtf_pool == "last":
                Hq = self.de_qtf_norm(self._qtf(H, mask))
                last = (mask.long() * torch.arange(mask.shape[1], device=mask.device)[None]).argmax(1)
                qv = F.normalize(Hq[torch.arange(Hq.shape[0], device=Hq.device), last], dim=-1)
            else:
                m = mask.float()[..., None]
                hq = self.de_ln((self._qtf(H, mask) if c.de_qtf > 0 else H).float())
                mean = (hq * m).sum(1) / m.sum(1).clamp_min(1.0)
                if c.de_qpool == "attn":
                    al = self.de_att(hq).masked_fill(~mask[..., None], -1e4)
                    pooled = torch.einsum("btk,btd->bkd", torch.softmax(al, 1), hq)
                    qv = F.normalize(self.de_mlp(torch.cat([pooled.flatten(1), mean], -1)), dim=-1)
                elif c.de_qpool == "split":
                    parts = []
                    for att, mk in ((self.de_att, G["_qcur"]), (self.de_att_h, G["_qhist"])):
                        mf = mk.float()[..., None]
                        al = att(hq).masked_fill(~mk[..., None], -1e4)
                        pooled = torch.einsum("btk,btd->bkd", torch.softmax(al, 1), hq) * mk.any(1).float()[:, None, None]
                        parts += [pooled.flatten(1), (hq * mf).sum(1) / mf.sum(1).clamp_min(1.0)]
                    parts.append(G["_qhist"].any(1).float()[:, None])
                    qv = F.normalize(self.de_mlp(torch.cat(parts, -1)), dim=-1)
                else:
                    qv = F.normalize(self.de_q(mean), dim=-1)
            if c.de_keys:
                uv = G["EK"].float()
            else:
                uv = gmemo(G, self, "de_u", lambda: F.normalize(self.de_u(G["E"].float()), dim=-1))
            de = self.de_tau.exp() * (qv @ uv.T)
            G["_de_qv"], G["_de_uv"] = qv, uv
            if c.de_rank_distill > 0 and self.training and G.get("_qkey_t") is not None:
                G["_de_rd"] = (qv @ uv.T, G["_qkey_t"].float() @ uv.T)
            if k > 0:
                wide.scatter_(1, de.detach().masked_fill(~isf[None], float("-inf")).topk(k, dim=1).indices, 1.0)
        if c.wide_entry_score in ("lex", "hyb"):
            lx = G.get("lex_raw", G.get("lex"))
            if lx is not None and k > 0:
                lx = lx.float()
                hit = torch.zeros_like(wide).scatter_(1, lx.masked_fill(~isf[None], float("-inf")).topk(k, dim=1).indices, 1.0)
                wide = torch.maximum(wide, hit * (lx > 0).to(wide.dtype))
        G["_wide"] = wide * isf[None].to(wide.dtype)
        if c.wide_entry_ents > 0:
            ise = ~isf
            ke = G.get("_wide_ke")
            if ke is None:
                ke = G["_wide_ke"] = min(c.wide_entry_ents, int(ise.sum()))
            went = torch.zeros_like(wide)
            if de is not None and ke > 0:
                went.scatter_(1, de.detach().masked_fill(isf[None], float("-inf")).topk(ke, dim=1).indices, 1.0)
            if c.wide_entry_score in ("lex", "hyb"):
                lx = G.get("lex_raw", G.get("lex"))
                if lx is not None and ke > 0:
                    lx = lx.float()
                    hit = torch.zeros_like(went).scatter_(1, lx.masked_fill(isf[None], float("-inf")).topk(ke, dim=1).indices, 1.0)
                    went = torch.maximum(went, hit * (lx > 0).to(went.dtype))
            G["_wide"] = G["_wide"] + went * ise[None].to(went.dtype)
        G["_de_sc"] = de


    def forward(self, texts, G, cand_mask=None):
        c = self.c
        if c.switch_theta > 0 and c.hop_persist != "off" and not self.training and cand_mask is None and len(self.layers):
            return self._switch_forward(texts, G)
        return self._forward(texts, G, cand_mask)

    def _set_hop(self, on):
        for l in self.layers:
            l._hop_on = on

    def _switch_forward(self, texts, G):
        c = self.c
        self._set_hop(False)
        self._sw_cache_at, self._sw_cache = (c.switch_from if c.switch_from >= 0 else None), None
        try:
            o = self._forward(texts, G, None)
        finally:
            self._set_hop(True)
            self._sw_cache_at = None
        low = torch.sigmoid(o["unit"][..., 0].float()).amax(1) < c.switch_theta
        if c.switch_none > 0:
            low = low & (torch.softmax(o["status"].float(), -1)[:, 4] < c.switch_none)
        o["switched"] = low
        st = self.__dict__.setdefault("switch_stats", [0, 0])
        st[0] += len(texts); st[1] += int(low.sum())
        if not bool(low.any()):
            self._sw_cache = None
            return o
        idx = low.nonzero().flatten()
        keep = {k: G[k] for k in ("lex", "lex_raw", "lex_date", "lex_cur", "lex_q", "prev", "t_f", "t_int", "t_es") if k in G}
        try:
            for k, v in keep.items():
                G[k] = v.index_select(0, idx)
            if c.switch_from >= 0:
                o2 = self._suffix_forward(idx, o["rec"], G)
            else:
                o2 = self._forward([texts[i] for i in idx.tolist()], G, None)
        finally:
            G.update(keep)
            G.pop("_path", None)
            self._sw_cache = None
        for k in ("unit", "status", "q", "act", "overflow", "turn_gate", "budget", "need"):
            o[k] = _merge_rows(o.get(k), o2.get(k), idx)
        for k in ("layer_s", "layer_any", "layer_a", "layer_q_h1"):
            o[k] = [_merge_rows(x, y, idx) for x, y in zip(o[k], o2[k])]
        o["rec"] = [{kk: _merge_rows(v, r2.get(kk), idx) for kk, v in r1.items()} for r1, r2 in zip(o["rec"], o2["rec"])]
        return o

    def _suffix_forward(self, idx, rec1, G):
        c, j, cache = self.c, self.c.switch_from, self._sw_cache
        assert cache is not None and c.mode not in ("vec_gnn", "bsym_c"), "suffix re-run: no cached inputs for this configuration"
        sel = lambda x: x.index_select(0, idx) if torch.is_tensor(x) and x.dim() > 0 and x.shape[0] == rec1[0]["s"].shape[0] else x
        H, mask, prev = sel(cache["H"]), sel(cache["mask"]), sel(cache["prev"])
        if cache["path"] is not None:
            G["_path"] = {k: sel(v) for k, v in cache["path"].items()}
        rec = [{k: sel(v) for k, v in r.items()} for r in rec1[:j]]
        kw = {"sit_E": self.sit_E} if c.schema else {}
        la0 = self.after[j]

        def run(gi, H, prev):
            r = self.layers[gi](H, mask, G, prev, write=True, **kw)
            rec.append(r)
            return (H + r["delta"].to(H.dtype) if r["delta"] is not None else H), r["s"]

        for gi in range(j, len(self.after)):
            if self.after[gi] == la0:
                H, prev = run(gi, H, prev)
        later = [gi for gi in range(j, len(self.after)) if self.after[gi] != la0]
        st = {"s": prev}
        hooks = []

        def make_hook(gis):
            def hook(_mod, _inp, out):
                tup = isinstance(out, tuple)
                Hh = out[0] if tup else out
                for gi in gis:
                    Hh, st["s"] = run(gi, Hh, st["s"])
                return (Hh,) + tuple(out[1:]) if tup else Hh
            return hook

        by_layer = {}
        for gi in later:
            by_layer.setdefault(self.after[gi], []).append(gi)
        for la, gis in by_layer.items():
            hooks.append(self.lm.layers[la - 1].register_forward_hook(make_hook(gis)))
        try:
            H = self._lm_suffix(H, mask, la0)
        finally:
            for h in hooks:
                h.remove()
        return self.readout(H, mask, G, rec, None)

    def _lm_suffix(self, H, mask, start):
        lm = self.lm
        if start < len(lm.layers):
            assert type(lm).__name__ == "Qwen3Model", f"suffix re-run through LM layers is implemented for Qwen3Model, not {type(lm).__name__}"
            from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
            pos = torch.arange(H.shape[1], device=H.device).unsqueeze(0)
            kw = {"config": lm.config, "inputs_embeds": H, "attention_mask": mask.long(), "past_key_values": None, "position_ids": pos}
            cm = {"full_attention": create_causal_mask(**kw)}
            if lm.has_sliding_layers:
                cm["sliding_attention"] = create_sliding_window_causal_mask(**kw)
            pe = lm.rotary_emb(H, pos)
            for i in range(start, len(lm.layers)):
                H = lm.layers[i](H, attention_mask=cm[lm.config.layer_types[i]], position_embeddings=pe, position_ids=pos,
                                 past_key_values=None, use_cache=False)
                H = H[0] if isinstance(H, tuple) else H
        return lm.norm(H)

    def _forward(self, texts, G, cand_mask=None):
        c = self.c
        plain = not (c.read_path or c.hop_persist != "off" or c.hop_kmax is not None)
        PROF = None
        _t = None
        if PROF:
            torch.cuda.synchronize(); _t = [time.perf_counter()]; self._prof = {"graph_ms": 0.0}
        inp = self._request_inputs(texts, G)
        self._bind_inputs(inp, G)
        if PROF:
            torch.cuda.synchronize(); _t.append(time.perf_counter())
        return self._core(inp["ids"], inp["mask"], G, cand_mask, _t)

    def _tok_unit_on(self):
        c = self.c
        return c.tok_nl > 1 or c.tok_adapt > 0 or c.tok_adapt_attn or bool(c.tok_subj)

    def _tok_unit(self, X, M, G=None):
        c = self.c
        if c.tok_nl > 1:
            X = torch.einsum("ntkd,k->ntd", X.view(X.shape[0], X.shape[1], c.tok_nl, -1), torch.softmax(self.tok_mix, 0).to(X.dtype))
        if c.tok_adapt_attn:
            xb = X.to(torch.bfloat16)
            q, k, v = F.linear(xb, self.ta_qkv.weight.to(xb.dtype), self.ta_qkv.bias.to(xb.dtype)).chunk(3, -1)
            am = M[:, None, :].expand(-1, M.shape[1], -1)
            o = F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None], attn_mask=am[:, None])[:, 0]
            X = X + F.linear(o, self.ta_o.weight.to(o.dtype), self.ta_o.bias.to(o.dtype)).to(X.dtype) * M[..., None].to(X.dtype)
        if c.tok_adapt > 0:
            xb = X.to(torch.bfloat16)
            h = F.gelu(F.linear(xb, self.ta_down.weight.to(xb.dtype), self.ta_down.bias.to(xb.dtype)))
            X = X + F.linear(h, self.ta_up.weight.to(h.dtype), self.ta_up.bias.to(h.dtype)).to(X.dtype) * M[..., None].to(X.dtype)
        if c.tok_subj:
            isf, subj = G["is_fact"] > 0, G["subj"]
            if c.tok_subj == "add":
                X = X + (self.ts_add(G["E"][subj].float()) * isf[:, None]).to(X.dtype)[:, None, :] * M[..., None].to(X.dtype)
            elif c.tok_subj == "pool":
                X = torch.cat([X, (G["E"][subj].float() * self.ts_g).to(X.dtype)[:, None, :]], 1)
                M = torch.cat([M, isf[:, None]], 1)
            else:
                k = int(c.tok_subj.split(":")[1])
                X = torch.cat([X, X[subj, :k]], 1)
                M = torch.cat([M, M[subj, :k] & isf[:, None]], 1)
        return X, M

    def _core(self, ids, mask, G, cand_mask, _t):
        c = self.c
        if "TOK" in G and self._tok_unit_on():
            raw = G.setdefault("_TOK_RAW", (G["TOK"], G["TOKM"]))
            G["TOK"], G["TOKM"] = gmemo(G, self, "tok_unit", lambda: self._tok_unit(raw[0], raw[1], G))
        PROF = _t is not None
        B, n = ids.shape[0], G["E"].shape[0]
        st = {"s": torch.zeros(B, c.M, n, device=self.device), "rec": [], "i": 0}
        st["s"][:, :, G["owner"]] = 1.0
        hooks = []
        by_layer = {}
        for gi, la in enumerate(self.after):
            by_layer.setdefault(la, []).append(gi)

        def make_hook(gis):
            def hook(_mod, _inp, out):
                tup = isinstance(out, tuple)
                H = out[0] if tup else out
                for gi in gis:
                    prev = st.get("h") if c.mode == "vec_gnn" else st["s"]
                    if gi == 0 and c.wide_entry > 0 and c.wide_entry_layer > 0:
                        G["_wide"] = G["_de_sc"] = G["_de_qv"] = None
                    if gi == c.wide_entry_layer and c.wide_entry > 0:
                        self._wide_entry(H, mask, G)
                    if getattr(self, "_sw_cache_at", None) == gi:
                        self._sw_cache = {"H": H, "mask": mask, "prev": prev, "path": dict(G["_path"]) if "_path" in G else None}
                    if PROF:
                        torch.cuda.synchronize(); _g0 = time.perf_counter()
                    r = self.layers[gi](H, mask, G, prev, write=True, **({"sit_E": self.sit_E} if c.schema else {}))
                    if PROF:
                        torch.cuda.synchronize(); self._prof["graph_ms"] += 1000 * (time.perf_counter() - _g0)
                    st["s"] = r["s"]
                    if "h" in r:
                        st["h"] = r["h"]
                    st["rec"].append(r)
                    if r["delta"] is not None:
                        H = H + r["delta"].to(H.dtype)
                return (H,) + tuple(out[1:]) if tup else H
            return hook

        if getattr(self, "_compile_graph", False) and not getattr(self, "_graph_compiled", False):
            for _gl in self.layers:
                _gl.forward = torch.compile(_gl.forward, dynamic=True)
            self._graph_compiled = True
        for la, gis in by_layer.items():
            hooks.append(self.lm.layers[la - 1].register_forward_hook(make_hook(gis)))
        try:
            if c.rca_m > 0:
                from transformers import DynamicCache
                G["_rca_kv"] = DynamicCache()
                H = self.lm(input_ids=ids, attention_mask=mask.long(), past_key_values=G["_rca_kv"], use_cache=True).last_hidden_state
            else:
                H = self.lm(input_ids=ids, attention_mask=mask.long()).last_hidden_state
        finally:
            for h in hooks:
                h.remove()
        if c.mode == "bsym_c":
            st["rec"].append(self.content(H, mask, G, st["s"], write=False))
        if PROF:
            torch.cuda.synchronize(); _t.append(time.perf_counter())
        out = self.readout(H, mask, G, st["rec"], cand_mask)
        if PROF:
            torch.cuda.synchronize(); _t.append(time.perf_counter())
            self._prof.update({"tok_lex_ms": 1000 * (_t[1] - _t[0]), "lm_plus_graph_ms": 1000 * (_t[2] - _t[1]), "readout_ms": 1000 * (_t[3] - _t[2])})
            self._prof["lm_ms"] = self._prof["lm_plus_graph_ms"] - self._prof["graph_ms"]
            act = [int((r.get("s_any", r["s"]).amax(1) > 0).any(0).sum()) for r in st["rec"]]
            self._prof["active_units"] = act
            if G["src"].numel():
                self._prof["active_edges"] = [int((r.get("s_any", r["s"]).amax(1) > 0).any(0)[G["src"]].sum()) for r in st["rec"]]
                self._prof["n_edges"] = int(G["src"].numel())
        return out

    def _read(self, logits, h, act, G):
        c = self.c
        B, n = h.shape
        N = min(int(G.get("_read_n", c.reader_n)), n)
        key = h.detach().float().masked_fill(~act, -1e4)
        if self.training and G.get("_read_gold") is not None:
            key = key + 1e3 * ((G["_read_gold"] > 0) & act).to(key.dtype)
        top = key.topk(N, dim=1).indices
        valid = act.gather(1, top)
        rows = G.get("_read_rows")
        if rows is not None:
            rm = torch.zeros(B, dtype=torch.bool, device=h.device); rm[rows] = True
            valid = valid & rm[:, None]
        vb, vj = valid.nonzero(as_tuple=True)
        ut = G["unit_text"]
        tl = top.tolist()
        s = torch.zeros(B, N, device=h.device)
        if vb.numel():
            sc = self.reader(G["_texts"], [(b, ut[tl[b][j]]) for b, j in zip(vb.tolist(), vj.tolist())])
            s = s.index_put((vb, vj), sc.to(s.dtype))
        fused = self.rd_a * h.float().gather(1, top) + self.rd_w * s + self.rd_b
        if self.training:
            return logits, {"idx": top, "score": fused, "valid": valid}
        fused = fused.clamp(min=-29.0)
        req = logits[..., 0].float()
        low = fused.masked_fill(~valid, 1e4).min(1, keepdim=True).values
        notread = act.clone(); notread.scatter_(1, top, valid.logical_not() & act.gather(1, top))
        req = torch.where(notread & valid.any(1, keepdim=True), torch.minimum(req, low - 1.0), req)
        req = req.scatter(1, top, torch.where(valid, fused, req.gather(1, top)))
        return torch.cat([req[..., None].to(logits.dtype), logits[..., 1:]], -1), None

    def _rerank(self, logits, Hr, mask, G, rec, ue, xctx):
        c = self.c; B, n = logits.shape[:2]; N = min(c.rerank_top, n)
        if rec and "z" in rec[0]:
            act = torch.stack([r.get("z_any", r["z"]).max(1).values for r in rec], 0).amax(0) > 0
        else:
            act = torch.stack([r.get("s_any", r["s"]).max(1).values for r in rec], 0).amax(0) > 0
        score = torch.maximum(logits[..., 0], logits[..., 1]).detach().masked_fill(~act, -1e4)
        top = score.topk(N, dim=1).indices
        valid = act.gather(1, top)
        qt = self.Wxt(Hr)
        def tokfeat(idx):
            tm = G["TOKM"][idx]
            tk = self.Wxu(G["TOK"][idx].float())
            xt = torch.einsum("bnth,bqh->bnqt", tk, qt) / c.d_h ** 0.5
            xt = xt.masked_fill(~mask[:, None, :, None], -1e4)
            per = xt.amax(2).clamp(-5, 5)
            tmf = tm.float(); den = tmf.sum(-1).clamp_min(1)
            mx = per.masked_fill(~tm, -5).amax(-1)
            return torch.stack([mx, (per * tmf).sum(-1) / den, ((per > 0.5).float() * tmf).sum(-1) / den], -1)
        subj = G["subj"][top]
        lg = logits.gather(1, top[..., None].expand(-1, -1, logits.shape[-1]))
        ue_c = ue[top]; xc = xctx.gather(1, top[..., None].expand(-1, -1, xctx.shape[-1]))
        x = torch.cat([ue_c * xc, ue_c, tokfeat(top), tokfeat(subj), lg, G["is_fact"][top][..., None]], -1)
        h = self.rr_in(x)
        att = torch.einsum("bnd,bmd->bnm", self.rr_q(h), self.rr_k(h)) / h.shape[-1] ** 0.5
        att = att + self.rr_sib * (subj[:, :, None] == subj[:, None, :]).to(att.dtype)
        att = att.masked_fill(~valid[:, None, :], -1e4)
        h = h + torch.softmax(att, -1) @ self.rr_v(h)
        h = h + self.rr_ff(h)
        d = self.rr_out(h) * valid[..., None].to(h.dtype)
        upd = torch.zeros(B, n, 2, device=logits.device, dtype=logits.dtype).scatter(1, top[..., None].expand(-1, -1, 2), d.to(logits.dtype))
        return torch.cat([logits[..., :2] + upd, logits[..., 2:]], -1)

    def _setro_apply(self, logits, Hr, mask, G, act, uid=None):
        B, n = logits.shape[:2]; N = min(self.c.setro_m, n)
        score = torch.maximum(logits[..., 0], logits[..., 1]).detach().masked_fill(~act, -1e4)
        tops = score.topk(N, dim=1).indices
        valid = act.gather(1, tops)
        top = tops if uid is None else uid[tops]
        lg = logits.gather(1, tops[..., None].expand(-1, -1, logits.shape[-1])).detach()
        d, conf = self.setro(Hr, mask, G, top, valid, lg)
        self._setro_out = {"idx": top, "valid": valid, "u": conf[0], "set": conf[1]} if conf is not None else None
        upd = torch.zeros(B, n, 2, device=logits.device, dtype=logits.dtype).scatter(1, tops[..., None].expand(-1, -1, 2), d.to(logits.dtype))
        return torch.cat([logits[..., :2] + upd, logits[..., 2:]], -1)

    def _px_apply(self, logits, Hr, mask, G, act, uid=None):
        B, n = logits.shape[:2]; N = min(self.c.px_m, n)
        score = torch.maximum(logits[..., 0], logits[..., 1]).detach().masked_fill(~act, -1e4)
        tops = score.topk(N, dim=1).indices
        valid = act.gather(1, tops)
        top = tops if uid is None else uid[tops]
        lg = logits.gather(1, tops[..., None].expand(-1, -1, logits.shape[-1])).detach()
        d = self.px(Hr, mask, G, top, valid, lg)
        upd = torch.zeros(B, n, 2, device=logits.device, dtype=logits.dtype).scatter(1, tops[..., None].expand(-1, -1, 2), d.to(logits.dtype))
        return torch.cat([logits[..., :2] + upd, logits[..., 2:]], -1)

    def _rca_apply(self, logits, mask, G, act, uid=None):
        from .rca import rca_candidates
        c = self.c
        B, n = logits.shape[:2]; M = min(c.rca_m, n)
        assert ("RCAS" in G or "RCAP" in G) and G.get("_rca_kv") is not None, "rca_m needs G['RCAS'] / G['RCAP'] (train.py --rca_m) and the routing K/V"
        gold = G.get("_rca_gold") if self.training else None
        if gold is not None and uid is not None:
            gold = gold[:, uid]
        tops, valid = rca_candidates(logits[..., 0].detach(), act, M, gold=gold, p_swap=c.rca_p_swap if self.training else 0.0)
        top = tops if uid is None else uid[tops]
        kv = G["_rca_kv"]
        req_kv = []
        for j in range(c.rca_l0, c.rca_read_layer):
            k, v = kv.layers[j].keys, kv.layers[j].values
            req_kv.append((k.detach(), v.detach()) if c.rca_detach_kv else (k, v))
        pos0 = mask.sum(1) if self.tok.padding_side == "right" else torch.full((B,), mask.shape[1], device=mask.device, dtype=torch.long)
        top_b, Gb = top, G
        if "RCAP" in G:
            prov, cg_ = G["RCAP"]
            uq, inv = torch.unique(top, return_inverse=True)
            v_ = prov.get_units(cg_, uq)
            Gb = {"RCAS": v_[0], "RCASM": v_[1]}
            if len(v_) > 2:
                Gb.update(RCAX=v_[2], RCAXM=v_[3], RCAXI=v_[4])
            top_b = inv.view_as(top)
        if c.rca_seg > 1 and Gb.get("RCAXI") is not None and Gb["RCAXI"].shape[0] > 0:
            s = self._rca_seg_scores(req_kv, mask, Gb, top_b, valid, pos0)
        else:
            s = self.rca(req_kv, mask, Gb["RCAS"][top_b], Gb["RCASM"][top_b], pos0)
        a = F.softplus(self.rca_alpha).to(logits.dtype)
        lg = logits.gather(1, tops[..., None].expand(-1, -1, logits.shape[-1]))
        fused = lg[..., :2] + a * (s if c.rca_fuse_grad else s.detach()).to(logits.dtype)
        if not self.training and ('1' or None):


            v = [float(x) for x in '1'.split(",")]
            a = torch.tensor(v if len(v) == 2 else v * 2, device=logits.device, dtype=logits.dtype)
            fused = lg[..., :2] + a * s.detach().to(logits.dtype)
        self._rca_out = {"idx": tops, "uid": top, "valid": valid, "s": s,
                         "none_logit": getattr(self.rca, "none_logit", None)}
        out = logits[..., :2].scatter(1, tops[..., None].expand(-1, -1, 2), torch.where(valid[..., None], fused, lg[..., :2]))
        if not self.training:
            low = fused.masked_fill(~valid[..., None], 1e4).amin(1, keepdim=True)
            inM = torch.zeros(B, n, dtype=torch.bool, device=logits.device).scatter(1, tops, valid)
            other = (act & ~inM & valid.any(1, keepdim=True))[..., None]
            out = torch.where(other, torch.minimum(out, low - 1.0), out)
        return torch.cat([out, logits[..., 2:]], -1)

    def _rca_seg_scores(self, req_kv, mask, G, top, valid, pos0):
        B, M = top.shape
        xi = G["RCAXI"][top]
        ok = (xi >= 0) & valid[..., None]
        cnt = ok.flatten(1).sum(1)
        E = int(cnt.max()) if B else 0
        if self.c.rca_seg_max > 0:
            E = min(E, self.c.rca_seg_max)
        st, sm = G["RCAS"][top], G["RCASM"][top]
        if E == 0:
            return self.rca(req_kv, mask, st, sm, pos0)
        T, d = st.shape[2], st.shape[3]
        own = torch.arange(M, device=top.device)[None, :, None].expand(B, M, xi.shape[2]).flatten(1)
        flat_ok, flat_xi = ok.flatten(1), xi.flatten(1)
        order = flat_ok.long().argsort(dim=1, descending=True, stable=True)[:, :E]
        e_ok = flat_ok.gather(1, order); e_xi = flat_xi.gather(1, order).clamp(min=0); e_own = own.gather(1, order)
        xs = G["RCAX"][e_xi]; xm = G["RCAXM"][e_xi] & e_ok[..., None]
        if self.c.rca_seg_merge:
            s_all = self.rca(req_kv, mask, torch.cat([st, xs.to(st.dtype)], 1), torch.cat([sm, xm], 1), pos0)
            s, se = s_all[:, :M], s_all[:, M:].masked_fill(~e_ok[..., None], float("-inf"))
        else:
            s = self.rca(req_kv, mask, st, sm, pos0)
            se = self.rca(req_kv, mask, xs.to(st.dtype), xm, pos0).masked_fill(~e_ok[..., None], float("-inf"))
        return s.scatter_reduce(1, e_own[..., None].expand(-1, -1, s.shape[-1]), se, reduce="amax", include_self=True)

    def _pair_sel_apply(self, logits, feats, q, G, act, uid=None):
        B, n = logits.shape[:2]; N = min(self.c.pair_sel, n); k = min(self.c.pair_sel_k, N)
        g = self.c.pair_sel_grad
        s_all = logits[..., 0] if g else logits[..., 0].detach()
        tops = s_all.detach().masked_fill(~act, -1e4).topk(N, dim=1).indices
        valid = act.gather(1, tops)
        top = tops if uid is None else uid[tops]
        s = s_all.gather(1, tops)
        hid = self.unit_head[1](self.unit_head[0](feats.gather(1, tops[..., None].expand(-1, -1, feats.shape[-1]))))
        P = self.pairsel(hid if g else hid.detach(), G["E"][top], q if g else q.detach(), G["subj"][top], top)
        gt = None
        if self.c.pair_sel_gate:
            s8 = F.pad(s.detach().float().masked_fill(~valid, -10.0)[:, :8], (0, max(0, 8 - N)), value=-10.0).clamp(-20, 20) / 5
            gt = torch.sigmoid(self.pairsel.gate(torch.cat([q.detach().float(), s8], -1))).squeeze(-1)
            P = P * gt[:, None, None]
        self._pair_gate = gt
        if self.training:
            self._pair_out = {"idx": top, "valid": valid, "s": s, "P": P, "g": gt}
            return logits
        order = pair_greedy(s, P, valid, k, self.c.pair_sel_agg)
        v = torch.empty_like(s).scatter(1, order, s)
        return torch.cat([logits[..., 0].scatter(1, tops, v)[..., None], logits[..., 1:]], -1)

    def readout(self, H, mask, G, rec, cand_mask):
        Hf = H.float()
        m = mask.float()[..., None]
        q = self.Wrq((Hf * m).sum(1) / m.sum(1))
        stats = []
        if any(r.get("a_fn") is not None for r in rec):
            ever = torch.stack([r.get("s_any", r["s"]).amax(1) for r in rec], 0).amax(0).gt(0).any(0).nonzero().flatten()
            if ever.numel():
                for r in rec:
                    if r.get("a_fn") is not None:
                        r["a"][:, :, ever] = r["a_fn"](ever)
        for r in rec:
            s_r = r["s_any"] if (self.c.read_any and r.get("s_any") is not None) else r["s"]
            stats += [s_r.max(1).values, r["a"].max(1).values, r["p"].max(1).values]
            if r.get("need_unit") is not None:
                stats.append(r["need_unit"].max(1).values)
        st = torch.stack(stats, -1)
        B, n = st.shape[:2]
        ue = gmemo(G, self, "Wr", lambda: self.Wr(G["E"]))
        Hr = self.ln_r(Hf)
        xl = torch.einsum("nh,bth->bnt", gmemo(G, self, "Wxu", lambda: self.Wxu(G["E"])), self.Wxt(Hr)) / self.c.d_h ** 0.5
        xl = xl.masked_fill(~mask[:, None, :], -1e4)
        xmax = xl.max(-1).values
        xctx = torch.softmax(xl, -1) @ self.Wxv(Hr)
        first = ue[None] * q[:, None]
        if "h" in rec[-1]:
            first = first + self.Wh(rec[-1]["h"])
        feats = [first, ue[None] * xctx, st, xmax[..., None],
                 self.state_emb(G["state"])[None].expand(B, -1, -1),
                 G["is_fact"][None, :, None].expand(B, -1, 1)]
        if self.c.read_lex:
            lx = G["lex"] if "lex" in G else torch.zeros(B, n, device=Hf.device)
            feats += [lx[..., None], lx[:, G["subj"]][..., None]]
        if self.c.read_lex2:
            lr = torch.log1p(G["lex_raw"]) if "lex_raw" in G else torch.zeros(B, n, device=Hf.device)
            ld = G["lex_date"] if "lex_date" in G else torch.zeros(B, n, device=Hf.device)
            feats += [lr[..., None], lr[:, G["subj"]][..., None], ld[..., None], ld[:, G["subj"]][..., None]]
        if self.c.dlg_state:
            pv = G["prev"] if "prev" in G else torch.zeros(B, n, device=Hf.device)
            lc = G["lex_cur"] if "lex_cur" in G else torch.zeros(B, n, device=Hf.device)
            feats += [pv[..., None], pv[:, G["subj"]][..., None], lc[..., None], lc[:, G["subj"]][..., None]]
        if self.c.tok_readout and "TOK" in G:


            tk = tok_proj(self.Wxu, G["TOK"])
            xq = self.Wxt(Hr)

            def _maxsim(tkc, xq_, tm):
                xt = torch.einsum("nth,bqh->bnqt", tkc, xq_) / self.c.d_h ** 0.5
                xt = xt.masked_fill(~mask[:, None, :, None], -1e4).masked_fill(~tm[None, :, None, :], -1e4)
                return xt.amax(2)


            Bq, tq, Tu = xq.shape[0], xq.shape[1], tk.shape[1]
            cn = max(1, int("2048") * 2 ** 20 // max(1, 4 * Bq * tq * Tu))
            TM = G["TOKM"]
            if n <= cn:
                per_tok = _maxsim(tk, xq, TM)
            elif torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                per_tok = torch.cat([checkpoint(_maxsim, tk[i:i + cn], xq, TM[i:i + cn], use_reentrant=False) for i in range(0, n, cn)], 1)
            else:
                per_tok = torch.cat([_maxsim(tk[i:i + cn], xq, TM[i:i + cn]) for i in range(0, n, cn)], 1)
            xmax_tok = per_tok.amax(-1)
            frac_tok = ((per_tok > 0.5).float() * G["TOKM"][None].float()).sum(-1) / G["TOKM"].float().sum(-1).clamp_min(1)[None]
            feats += [xmax_tok[..., None], frac_tok[..., None]]
        if self.c.prov_feat:
            prs = [r["prov_rel"] for r in rec if r.get("prov_rel") is not None]
            pr = torch.stack(prs, 0).sum(0) if prs else torch.zeros(B, n, self.c.d_r, device=Hf.device)
            pv = self.Wprov(pr)
            feats += [pv * q[:, None], pv]
        if self.c.read_path:
            pt, pm = G.pop("_path", None), self.pathmod
            if pt is not None and "P" in pt:
                feats += [pt["pi"] * pm.Wpq(pt["P"])[:, None, :], (pt["tau"] / 8.0)[..., None], (pt["tau"] > 0).to(pt["pi"].dtype)[..., None]]
            else:
                feats.append(torch.zeros(B, n, pm.dp + 2, device=Hf.device))
        feats = torch.cat(feats, -1)
        logits = self.unit_head(feats)
        if self.c.de_read and self.c.wide_entry > 0 and G.get("_de_sc") is not None:
            dsc = G["_de_sc"].to(logits.dtype)
            logits = logits + torch.stack([self.ro_de[0] * dsc, self.ro_de[1] * dsc, torch.zeros_like(dsc), torch.zeros_like(dsc)], -1)
        if self.c.de_head > 0 and self.c.wide_entry > 0 and G.get("_de_qv") is not None:
            w1q = G["_de_qv"].float()[:, :, None] * self.dh_w1.weight.T[None]
            if self.c.de_head_scale:
                w1q = w1q * self.dh_ls.exp()
            hh = F.gelu(torch.einsum("nk,bkh->bnh", G["_de_uv"].float(), w1q) + self.dh_w1.bias)
            logits = logits + F.pad(self.dh_w2(hh), (0, 2)).to(logits.dtype)
        if self.c.time_feat and "t_f" in G:
            dtl = (G["t_f"] * self.ro_tW[G["t_int"]][:, None, :]).sum(-1)
            if self.c.time_ro_gain not in (None, 1.0):
                dtl = self.c.time_ro_gain * dtl
            logits = logits + torch.stack([dtl, dtl, torch.zeros_like(dtl), torch.zeros_like(dtl)], -1)
        if self.c.unit_roles:
            rr = [r["roles"] for r in rec if r.get("roles") is not None]
            if rr:
                logits = logits + self.role_ro(torch.stack(rr, 0).amax(0))
        if self.c.rerank_top > 0 and "TOK" in G:
            logits = self._rerank(logits, Hr, mask, G, rec, ue, xctx)
        self._setro_out = None
        if self.c.setro_m > 0:
            assert "TOK" in G, "setro_m needs the unit token memories (--tok_mem)"
            key = "z" if rec and "z" in rec[0] else "s"
            act = torch.stack([r.get(key + "_any", r[key]).max(1).values for r in rec], 0).amax(0) > 0
            logits = self._setro_apply(logits, Hr, mask, G, act)
        if self.c.px_m > 0:
            assert "TOK" in G, "px_m needs the unit token memories (--tok_mem)"
            key = "z" if rec and "z" in rec[0] else "s"
            act = torch.stack([r.get(key + "_any", r[key]).max(1).values for r in rec], 0).amax(0) > 0
            logits = self._px_apply(logits, Hr, mask, G, act)
        self._rca_out = None
        if self.c.rca_m > 0:
            key = "z" if rec and "z" in rec[0] else "s"
            act = torch.stack([r.get(key + "_any", r[key]).max(1).values for r in rec], 0).amax(0) > 0
            logits = self._rca_apply(logits, mask, G, act)
        self._pair_out = None
        if self.c.pair_sel > 0:
            assert not (self.c.read_ledger or self.c.read_budget or self.c.mod_comp or self.c.subj_factor), "pair_sel: later re-scoring would reorder the picks"
            key = "z" if rec and "z" in rec[0] else "s"
            act = torch.stack([r.get(key + "_any", r[key]).max(1).values for r in rec], 0).amax(0) > 0
            logits = self._pair_sel_apply(logits, feats, q, G, act)
        self._budget = None
        if self.c.read_ledger and "module_E" in G and "unit_gmod" in G:


            gm = G["unit_gmod"]; fidx = (gm >= 0).nonzero().flatten()
            if len(fidx):
                h0 = logits[..., 0]
                subj = G["subj"][fidx]; mods = gm[fidx]
                qe = self.need_e(q[:, None, :] * ue[subj][None])
                need = (qe * self.need_m(G["module_E"].float())[mods][None]).sum(-1)
                hf = h0[:, fidx]
                _, sid = torch.unique(subj * (int(gm.max()) + 1) + mods, return_inverse=True)
                nsl = int(sid.max()) + 1; sidb = sid[None].expand(B, -1)
                mx = torch.full((B, nsl), -1e4, device=hf.device, dtype=hf.dtype).scatter_reduce(1, sidb, hf, "amax", include_self=True)
                den = torch.zeros(B, nsl, device=hf.device, dtype=hf.dtype).scatter_add(1, sidb, torch.exp(hf - mx.gather(1, sidb)))
                comp = (mx + torch.log(den.clamp_min(1e-12))).gather(1, sidb) - hf
                hf2 = hf + F.softplus(self.ledger_g) * F.logsigmoid(need) - F.softplus(self.ledger_a) * comp
                h0 = h0.clone(); h0[:, fidx] = hf2
                logits = torch.cat([h0[..., None], logits[..., 1:]], -1)
        if self.c.read_budget and self.c.elbo and rec and "z" in rec[0]:


            act = torch.stack([r.get("z_any", r["z"]).max(1).values for r in rec], 0).amax(0) > 0
            v = logits[..., 0] / F.softplus(self.bud_T)
            top = torch.sigmoid(logits[..., 0]).masked_fill(~act, 0).topk(min(8, v.shape[1]), dim=1).values
            if top.shape[1] < 8:
                top = F.pad(top, (0, 8 - top.shape[1]))
            Bq = F.softplus(self.bud_head(torch.cat([q, top], -1))).squeeze(-1) + 0.1
            Bq = torch.minimum(Bq, act.sum(1).to(Bq.dtype).clamp_min(1e-3))
            vv = v.detach().masked_fill(~act, -1e4)
            lo = vv.masked_fill(~act, 1e4).min(1).values - 1.0; hi = vv.max(1).values
            for _ in range(40):
                mid = (lo + hi) / 2
                ssum = ((vv - mid[:, None]).clamp(0, 1) * act).sum(1)
                lo = torch.where(ssum > Bq.detach(), mid, lo); hi = torch.where(ssum > Bq.detach(), hi, mid)
            t0 = (lo + hi) / 2
            xs = vv - t0[:, None]
            inS = (xs > 0) & (xs < 1) & act; sat = (xs >= 1) & act
            nS = inS.sum(1)
            t = ((v * inS).sum(1) - (Bq - sat.sum(1).to(Bq.dtype))) / nS.clamp_min(1).to(Bq.dtype)
            t = torch.where(nS > 0, t, t0)
            x = ((v - t[:, None]).clamp(0, 1) * act).clamp(1e-4, 1 - 1e-4)
            logits = torch.cat([(torch.log(x) - torch.log1p(-x))[..., None], logits[..., 1:]], -1)
            self._budget = Bq
        rd_h = rd_act = None
        if self.c.reader and rec and "z" in rec[0] and G.get("_texts") is not None and G.get("unit_text") is not None:
            rd_h = logits[..., 0]
            rd_act = torch.stack([r.get("z_any", r["z"]).max(1).values for r in rec], 0).amax(0) > 0
        if cand_mask is None and self.c.elbo and rec and "z" in rec[0]:

            z_any = torch.stack([r.get("z_any", r["z"]).max(1).values for r in rec], 0).amax(0)
            if self.c.read_comp:


                h0 = logits[..., 0]
                lse = torch.logsumexp(h0.masked_fill(z_any <= 0, -1e4), dim=1, keepdim=True)
                h0 = h0 - torch.nn.functional.softplus(self.comp_alpha) * lse + self.comp_bias
                logits = torch.cat([h0[..., None], logits[..., 1:]], -1)
            if self.c.read_norm > 0:


                h0 = logits[..., 0].masked_fill(z_any <= 0, -1e4)
                p0 = (torch.log_softmax(h0, dim=1) + math.log(self.c.read_norm)).exp().clamp(1e-6, 1 - 1e-6)
                logits = torch.cat([(torch.log(p0) - torch.log1p(-p0))[..., None], logits[..., 1:]], -1)
            if self.training:


                eps_l = 1e-3
                pz = z_any[..., None] * torch.sigmoid(logits[..., :2]) + (1 - z_any[..., None]) * eps_l
                pz = pz.clamp(1e-6, 1 - 1e-6)
                logits = torch.cat([torch.log(pz) - torch.log1p(-pz), logits[..., 2:]], -1)
            else:


                cand_mask = z_any > 0
        elif cand_mask is None and self.c.mode != "bsym_c" and self.c.extra.get("readout_gate", True):
            s_all = torch.stack([r.get("s_any", r["s"]).max(1).values for r in rec], 0).amax(0)
            if self.training:

                soft = torch.log(torch.sigmoid((s_all - self.c.tau) / 0.05) + 1e-4)
                logits = torch.cat([logits[..., :2] + soft[..., None], logits[..., 2:]], -1)
            else:

                cand_mask = s_all > 0
        if cand_mask is not None:
            logits = logits.masked_fill(~cand_mask[..., None], -30.0)
        read = None
        if rd_h is not None:
            logits, read = self._read(logits, rd_h, rd_act, G)
        smax = rec[-1]["s"].max(1).values
        summ = torch.stack([smax.max(1).values, smax.sum(1) / 10, (smax > self.c.tau).float().sum(1) / 10,
                            torch.sigmoid(logits[..., 0]).max(1).values], -1)
        status = self.status_head(torch.cat([q, summ], -1))
        if self.c.mod_comp and "unit_gmod" in G:
            gm = G["unit_gmod"]
            isf_m = (gm >= 0)
            nm = int(gm.max().item()) + 1 if isf_m.any() else 0
            if nm > 0:
                idx = gm.clamp_min(0)[None].expand(B, -1)
                outs = []
                for h in range(2):
                    l = logits[..., h].masked_fill(~isf_m[None], -1e4)
                    M1 = torch.full((B, nm), -1e4, device=l.device, dtype=l.dtype).scatter_reduce(1, idx, l, reduce="amax")
                    is_top = l >= M1.gather(1, idx) - 1e-6
                    M2 = torch.full((B, nm), -1e4, device=l.device, dtype=l.dtype).scatter_reduce(1, idx, l.masked_fill(is_top, -1e4), reduce="amax")
                    other = torch.where(is_top, M2.gather(1, idx), M1.gather(1, idx))
                    inhib = F.softplus(other - l) * isf_m[None].to(l.dtype)
                    outs.append(logits[..., h] - F.softplus(self.mc_a[h]) * inhib)
                logits = torch.cat([torch.stack(outs, -1), logits[..., 2:]], -1)
        if self.c.subj_factor:
            isf = G["is_fact"][None, :, None]
            ls = F.logsigmoid(logits[..., 2])[:, G["subj"]]
            lp = (F.logsigmoid(logits[..., :2]) + ls[..., None]).clamp(max=-1e-6)
            fl = lp - torch.log(-torch.expm1(lp))
            logits = torch.cat([isf * fl + (1 - isf) * logits[..., :2], logits[..., 2:]], -1)
        tg = None
        if self.c.turn_gate:
            pr0 = torch.sigmoid(logits[..., 0])
            pv = G["prev"] if "prev" in G else torch.zeros_like(pr0)
            lc = G["lex_cur"] if "lex_cur" in G else torch.zeros_like(pr0)
            ov_prev = (pr0 * pv).sum(1) / (pr0.sum(1) + 1.0)
            tg = self.turn_head(torch.cat([q, summ, ov_prev[:, None], lc.max(1).values[:, None]], -1)).squeeze(-1)
            lgate = F.logsigmoid(tg)[:, None]
            lp = (F.logsigmoid(logits[..., :2]) + lgate[..., None]).clamp(max=-1e-6)
            logits = torch.cat([lp - torch.log(-torch.expm1(lp)), logits[..., 2:]], -1)
        ov = [r["overflow"] for r in rec if r.get("overflow") is not None]
        needs = [r["need"] for r in rec if r.get("need") is not None]
        kls = [r["kl"] for r in rec if r.get("kl") is not None]
        gms = [r["gate"] for r in rec if r.get("gate") is not None]
        return {"turn_gate": tg, "unit": logits, "status": status, "q": q, "ue": ue, "act": smax, "rec": rec, "budget": self._budget,
                "de_sc": G.get("_de_sc") if self.c.wide_entry > 0 else None,
                "read": read,
                "setro": self._setro_out,
                "rca": self._rca_out,
                "pair_sel": self._pair_out,
                "pair_gate": getattr(self, "_pair_gate", None) if self.c.pair_sel > 0 else None,
                "de_rd": G.get("_de_rd") if self.c.de_rank_distill > 0 else None,
                "de_q": G.get("_de_qv") if self.c.de_distill > 0 else None,
                "de_qt": G.get("_qkey_t") if self.c.de_distill > 0 else None,
                "kl": sum(kls) if kls else None,
                "gate": sum(gms) / len(gms) if gms else None,
                "need": torch.stack([x.max(1).values for x in needs]).amax(0) if needs else None,
                "layer_s": [r["s"] for r in rec],
                "layer_any": [r.get("s_any", r["s"]) for r in rec],
                "layer_a": [r["a"] for r in rec],
                "layer_q_h1": [r.get("q_h1") for r in rec],
                "layer_t_edge": [r.get("t_edge") for r in rec],
                "div": sum(r["div"] for r in rec) / len(rec),
                "overflow": torch.stack(ov).any(0) if ov else torch.zeros(B, dtype=torch.bool, device=q.device)}


