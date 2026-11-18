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
"""Graph layers: exact Torch rewrite of the inference forward for this model configuration."""
import types


def _check(c):
    want = dict(mode='v4', tok_mem=True, content_cos=False, content_skip0=False, need_pool=False, need_match=False,
                fact_entry=32, fact_entry_score='lex', ent_entry=0, wide_entry=32, fanout_deg=True, fanout_topm=False,
                gate_cos=True, prior_mode='graph', state_prior=False, prov_feat=False, lex_phi=True, lex_fact_reach=True,
                lex_feat=True, dlg_state=True, time_feat=True, route_gate=True, hard_reach=False, hard_subject=True,
                mention_reach=True, entry_only=False, act_topm=0, act_norm=0.0, slot_inhibit=False, anchor_clamp=True,
                unit_roles=False, refractory=0, role_send=False, deg_norm='', hub_damp=False, hop_persist='global',
                hop_halt='fixpoint', wb_unit=False, wb_skip0=False, probe_mask_tmpl=False, read_path=True)
    bad = {k: (getattr(c, k, None), v) for k, v in want.items() if (getattr(c, k, None) or None) != (v or None)}
    if (c.time_entry or 0) > 0:
        bad['time_entry'] = c.time_entry
    if c.lex_anchor <= 0 or c.anchor_teach <= 0:
        bad['lex_anchor/anchor_teach'] = (c.lex_anchor, c.anchor_teach)
    return bad


def path_update(path, pm, WbC, G, ma, newm_any, B, n):
    torch = __import__('torch')
    dst_e = G["dst"][None].expand_as(ma)
    mxa = torch.full((B, n), -1.0, device=ma.device).scatter_reduce(1, dst_e, ma, "amax")
    hit = (ma == mxa.gather(1, dst_e)) & (ma > 0)
    eidx = torch.arange(ma.shape[-1], device=ma.device)[None].expand_as(ma)
    best = torch.full((B, n), -1, dtype=torch.long, device=ma.device).scatter_reduce(
        1, dst_e, torch.where(hit, eidx, torch.full_like(eidx, -1)), "amax")
    newly = newm_any & (best >= 0)
    bi = best.clamp_min(0)
    par, rl = G["src"][bi], G["rel"][bi]
    pi = path["pi"]
    pi_new = torch.tanh(pm.Wa(pi.gather(1, par[..., None].expand(-1, -1, pi.shape[-1]))) + WbC[rl])
    path["hop"] += 1
    path["pi"] = torch.where(newly[..., None], pi_new, pi)
    path["tau"] = torch.where(newly, torch.full_like(path["tau"], float(path["hop"])), path["tau"])


def memo_const(G, owner, key, fn):
    memo = G.setdefault('_edge_const', {})
    k = (id(owner), key)
    if k not in memo:
        memo[k] = fn()
    return memo[k]


def _own_terms(torch, q_like, G):
    own = torch.zeros_like(q_like)
    own[:, :, G["owner"]] = 1.0
    return own, own * (1 - 1e-5), 1 - own


def make_forward(module_globals, table, native=None):
    torch = module_globals['torch']
    F = module_globals['F']
    gmemo = module_globals['gmemo']
    tok_proj = module_globals['tok_proj']
    tok_maxsim_mask = module_globals['tok_maxsim_mask']
    tok_wb_mask = module_globals['tok_wb_mask']
    seg_sum = module_globals['seg_sum']
    original_forward = module_globals['ElboGraphLayer'].forward
    fuse = {'P': None, 'H': None}

    def fused_cat(layer, names, x, key):
        ws = fuse[key]
        if ws is None:
            return [getattr(layer, nm)(x) for nm in names]
        w, sizes = ws[id(layer)]
        return list(torch.nn.functional.linear(x, w).split(sizes, -1))

    def forward(self, H, tok_mask, G, z_prev, write=True, sit_E=None):
        if self.training or torch.is_grad_enabled() or not self._hop_on or self._dbg is not None or self._dbg_edge is not None \
                or self._dbg_share is not None or not ("TOK" in G):
            return original_forward(self, H, tok_mask, G, z_prev, write, sit_E)
        c = self.c
        Hf = self.ln(H.float())
        B = Hf.shape[0]
        pk_h, tq_h = fused_cat(self, ('Wpk', 'Wtq'), Hf, 'H')
        lg = torch.einsum("mk,btk->bmt", self.probe, pk_h) / c.d_k ** 0.5
        pmask = tok_mask
        lg = lg.masked_fill(~pmask[:, None, :], -1e4)
        P = torch.softmax(lg, -1) @ Hf
        div = memo_const(G, self, 'zero', lambda: torch.zeros(()))
        q_p, s_p, q2_p = fused_cat(self, ('Wq', 'Ws', 'Wq2'), P, 'P')
        tk = gmemo(G, self, "Wtk", lambda: tok_proj(self.Wtk, G["TOK"]))
        at = torch.einsum("bmk,ntk->bmnt", q_p, tk) / c.d_k ** 0.5
        a = at.masked_fill(~tok_maxsim_mask(G, c)[None, None], -1e4).max(-1).values
        sit = torch.sigmoid(torch.einsum("bmr,sr->bms", s_p, memo_const(G, self, "Wsit", lambda: self.Wsit(sit_E))) / c.d_r ** 0.5 + self.sit_b)
        Modk = memo_const(G, self, "Wmk", lambda: self.Wmk(G["Mod"]))
        need = torch.sigmoid(torch.einsum("bmr,jr->bmj", q2_p + sit @ self.U, Modk) / c.d_r ** 0.5 + self.need_b)
        need_unit = need[:, :, G["unit_mod"]]
        gq = self.Wg(P) + sit @ self.V
        st = memo_const(G, self, "state", lambda: self.state_emb(G["state"]))[None, None].expand(B, c.M, -1, -1)
        eps = c.eps_prior
        z_cur = z_prev
        g_c = None
        lexs = G.get("lex_raw", G.get("lex"))
        if lexs is not None:
            sc = lexs[:, None, :].float().expand_as(a)
            ok = (sc > 0).to(a.dtype)
        else:
            sc, ok = a, torch.ones_like(a)
        kM = min(c.fact_entry, a.shape[-1])
        af = sc.masked_fill(G["is_fact"][None, None] == 0, float("-inf"))
        ca = torch.zeros_like(a).scatter_(-1, af.topk(kM, dim=-1).indices, 1.0) * G["is_fact"][None, None] * ok
        g_c = torch.sigmoid(self.fe_b + self.fe_w * sc) * ca
        wc = None
        if G.get("_wide") is not None:
            wc = G["_wide"][:, None, :].to(a.dtype).expand_as(a)
        has_edges = bool(G["src"].numel())
        out_deg = None
        if has_edges:
            out_deg = memo_const(G, self, "out_deg", lambda: torch.zeros(a.shape[-1], device=a.device).index_add_(0, G["src"], torch.ones_like(G["src"], dtype=torch.float)))
        z_any, s_any = torch.zeros_like(a), torch.zeros_like(a)
        K, n_run = self._n_hops(), 0
        Vin = (z_prev > 0).to(a.dtype)
        zm = Vin.clone()
        q_acc = z_prev
        path, pm = None, self.__dict__.get("_pm")
        if c.read_path and pm is not None:
            if self.li == 0 or "_path" not in G:
                G["_path"] = {"pi": pm.pi0[None, None].expand(B, a.shape[-1], -1).clone(), "tau": torch.zeros(B, a.shape[-1], device=a.device), "hop": 0}
            path = G["_path"]
            WbC = gmemo(G, pm, "edge_WbC", lambda: pm.Wb(G["C"]))
        done = torch.zeros(B, dtype=torch.bool, device=a.device)
        isf = G["is_fact"]

        if has_edges:
            Cr = gmemo(G, self, "WcC", lambda: self.Wc(G["C"] - self.rel_mean))
            g = torch.sigmoid(self.gate_tau * torch.einsum("bmr,qr->bmq", F.normalize(gq, dim=-1), memo_const(G, self, "Crn", lambda: F.normalize(Cr, dim=-1))) + self.gate_b)
            if "rel_scale" in G:
                g = g * G["rel_scale"][None, None, :]
            src_idx = G["src"]
            g_rel = g[:, :, G["rel"]]
            gr = g[:, :, G["rel"]].clamp(1e-6, 1 - 1e-6)
            ge = torch.log(gr) - torch.log1p(-gr) + a[:, :, G["dst"]]
            ge = ge + memo_const(G, self, "isf_dst", lambda: G["is_fact"][G["dst"]]) * torch.log(need_unit[:, :, G["dst"]].clamp_min(1e-4))
            if "lex" in G:
                ge = ge + self.lex_w * G["lex"][:, None, G["dst"]]
            src_e = src_idx[None, None].expand_as(ge)
            mx = torch.full_like(a, -1e4).scatter_reduce(2, src_e, ge, "amax", include_self=True)
            ex = torch.exp(ge - mx.gather(2, src_e))
            den = torch.zeros_like(a).scatter_add(2, src_e, ex)
            m_src = memo_const(G, self, "m_src", lambda: c.fanout_m * torch.exp(self.fd_b + self.fd_w * torch.log1p(out_deg[src_idx])))
            share = (m_src * ex / den.gather(2, src_e).clamp_min(1e-6)).clamp(max=1.0)
        one_minus_gc = 1 - g_c
        wc_term = (1 - (1 - eps) * wc) if wc is not None else None
        not_fact = 1 - G["is_fact"]
        has_lex = "lex" in G
        if has_lex:
            lexq_base = G.get("lex_q", G["lex"])
            lx_base = lexq_base[:, None, :, None]
            lexq_rows = lexq_base[:, None, :]
            lex_gain = 10.0 * self.lex_q
        prev_term = 10.0 * self.prev_q * G["prev"][:, None, :] if "prev" in G else None
        time_term = 10.0 * (G["t_f"] * self.t_W[G["t_int"]][:, None, :]).sum(-1)[:, None, :] if "t_f" in G else None
        anc = None
        q_like = a
        anc = torch.zeros_like(q_like)
        anc[:, :, G["owner"]] = 1.0
        if has_lex:
            anc = torch.maximum(anc, (G["lex"] >= max(c.lex_anchor, 1e-6)).to(q_like.dtype)[:, None, :].expand_as(q_like))
        if "prev" in G:
            anc = torch.maximum(anc, torch.sigmoid(self.prev_a) * G["prev"][:, None, :].to(q_like.dtype))
        ent = (1 - G["is_fact"])[None, None]
        one_minus_anc = 1 - anc
        one_minus_ent = 1 - ent
        own, own_hi, one_minus_own = memo_const(G, self, ("own", tuple(q_like.shape)), lambda: _own_terms(torch, q_like, G))
        one_minus_wc = (1 - wc) if wc is not None else None
        men = "men_src" in G
        if men:
            men_dst = G["men_dst"][None, None].expand(q_like.shape[0], q_like.shape[1], -1)
        p = None
        z = None
        if native is not None and native.enabled and B == 1:
            nat = native(self, G, a, need_unit, g_c, wc, z_prev, anc, g_rel if has_edges else None, share if has_edges else None,
                         st, lx_base if has_lex else None, lex_gain if has_lex else None, prev_term, time_term, own, K, has_edges)
            z, q, z_any, s_any, p, n_run, ma_h, newm_h = nat
            if path is not None and has_edges:
                for h in range(n_run):
                    path_update(path, pm, WbC, G, ma_h[h][None], newm_h[h][None], B, a.shape[-1])
        for _step in range(K if z is None else 0):
            n_run += 1
            z_in = z_cur
            if has_edges:
                zj = z_cur[:, :, src_idx]
                msg = g_rel * zj
                msg = msg * share
                log1m = seg_sum(torch.log1p(-msg.clamp(max=1 - 1e-5)), G)
                p_reach = 1 - (1 - eps) * torch.exp(log1m)
                p = p_reach
            else:
                p_reach = torch.full_like(a, eps)
                p = p_reach
            p_reach = 1 - (1 - p_reach) * one_minus_gc
            p = p_reach
            if wc is not None:
                p_reach = 1 - (1 - p_reach) * wc_term
                p = p_reach
            prior = not_fact * p_reach + G["is_fact"] * p_reach * need_unit
            prior = prior.clamp(1e-5, 1 - 1e-5)
            prior_logit = torch.log(prior) - torch.log1p(-prior)
            feats = torch.cat([a[..., None], p[..., None], z_cur[..., None], st, need_unit[..., None], prior_logit[..., None]], -1)
            if has_lex:
                lx = lx_base.expand(feats.shape[0], feats.shape[1], -1, 1)
                lx = lx * ((1 - G["is_fact"]) + G["is_fact"] * (p > eps + 1e-3).to(lx.dtype))[..., None]
            else:
                lx = torch.zeros_like(feats[..., :1])
            feats = torch.cat([feats, lx], -1)
            q_logit = self.phi(feats).squeeze(-1)
            if has_lex:
                lexq = lexq_rows.expand_as(q_logit)
                reach = (p > eps + 1e-3).to(lexq.dtype)
                lexq = lexq * ((1 - G["is_fact"]) + G["is_fact"] * reach)
                q_logit = q_logit + lex_gain * lexq
            if prev_term is not None:
                q_logit = q_logit + prev_term
            if time_term is not None:
                q_logit = q_logit + time_term
            q = torch.sigmoid(q_logit).clamp(1e-5, 1 - 1e-5)
            reach = ((p - eps) / (1 - eps)).clamp(0.0, 1.0)
            q = (q * (ent * (anc + one_minus_anc * reach) + one_minus_ent)).clamp(1e-5, 1 - 1e-5)
            sf = z_in[:, :, G["subj"]]
            if men:
                sf = torch.maximum(sf, torch.zeros_like(sf).scatter_reduce(2, men_dst, z_in[:, :, G["men_src"]], "amax", include_self=True))
            sf = torch.maximum(sf, g_c)
            if wc is not None:
                sf = torch.maximum(sf, wc)
            sup = not_fact + isf * sf
            q = (sup * q + (1 - sup) * 1e-4).clamp(1e-5, 1 - 1e-5)
            q = (own_hi + one_minus_own * q).clamp(1e-5, 1 - 1e-5)
            z_cur = (q > c.accept_thr).to(q.dtype)
            if wc is not None:
                z_cur = wc + one_minus_wc * z_cur
            z_new = z_cur
            acc = (z_new > 0.5) & (Vin <= 0)
            z_cur = torch.where(Vin > 0, z_in, z_new)
            zm = torch.where(Vin > 0, zm, z_new)
            q_acc = torch.where(acc, q, q_acc)
            changed = acc.flatten(1).any(1)
            Vin = torch.maximum(Vin, acc.to(a.dtype))
            z_any = torch.maximum(z_any, zm)
            s_any = torch.maximum(s_any, zm * q_acc)
            if path is not None and has_edges:
                path_update(path, pm, WbC, G, msg.detach().max(1).values, acc.any(1), B, a.shape[-1])
            done = done | ((~changed) & torch.tensor(True, device=a.device))
            if bool(done.all()):
                break
        if path is not None:
            path["P"] = P.mean(1)
        if z is None:
            z = zm
            q = torch.where(Vin > 0, q_acc, q)
        delta, overflow = None, None
        if write:
            u = (z * q).max(1).values
            Kw = min(c.K, u.shape[1])
            overflow = (u > 0).sum(1) > Kw
            uv, ui = torch.topk(u, Kw, dim=1)
            bias = torch.where(uv > 0, torch.log(uv.clamp_min(1e-6)), torch.full_like(uv, float("-inf")))
            Tu = G["TOK"].shape[1]
            keys = table(G, self, "Wtk", ui, None).reshape(B, Kw * Tu, -1)
            vals = table(G, self, "Wtv", ui, None).reshape(B, Kw * Tu, -1)
            tb = (bias[:, :, None] + memo_const(G, self, "wbmask", lambda: torch.where(tok_wb_mask(G, c), 0.0, float("-inf")))[ui]).reshape(B, Kw * Tu)
            lgw = torch.einsum("btk,bjk->btj", tq_h, keys) / c.d_k ** 0.5 + tb[:, None, :]
            lgw = torch.cat([lgw, self.null_logit.expand(B, lgw.shape[1], 1)], -1)
            att = torch.softmax(lgw, -1)[..., :Kw * Tu]
            delta = torch.sigmoid(self.wb_gate) * (att @ vals)
        return {"s": z * q, "s_any": s_any, "z": z, "z_any": z_any, "q": q, "q_h1": None, "a": a, "p": p, "delta": delta, "div": div,
                "bp": None, "prov": None, "prov_rel": None, "overflow": overflow, "sit": sit, "need": need, "need_unit": need_unit,
                "kl": None, "gate": None, "t_edge": None, "n_run": n_run}
    forward._edge_fuse = fuse
    return forward


def install(runtime, tables, enabled=True, native_library=None):
    import inspect
    model = runtime.model
    layers = list(model.layers)
    cls = type(layers[0])
    bad = _check(model.c)
    if bad:
        raise ValueError('graph_fast_unsupported_config:%s' % bad)
    if native_library:
        raise ValueError('the native hop loop is not part of this package')
    native = None
    fast = make_forward(vars(inspect.getmodule(cls)), tables.table, native)
    state = {'fused_P': False, 'fused_H': False}
    import torch
    g = torch.Generator().manual_seed(0)
    for key, names, rows_list in (('P', ('Wq', 'Ws', 'Wq2'), (8,)), ('H', ('Wpk', 'Wtq'), (9, 17, 31, 32, 33, 34, 37, 48, 61, 64, 128, 200, 384))):
        ws, ok = {}, True
        for layer in layers:
            mods = [getattr(layer, nm) for nm in names]
            if any(m.bias is not None for m in mods):
                ok = False
                break
            w = torch.cat([m.weight.detach() for m in mods], 0).contiguous()
            sizes = [m.out_features for m in mods]
            for rows in rows_list:
                x = torch.randn(1, rows, mods[0].in_features, generator=g) * 3
                fused = torch.nn.functional.linear(x, w).split(sizes, -1)
                if not all(torch.equal(f, m(x)) for f, m in zip(fused, mods)):
                    ok = False
            ws[id(layer)] = (w, sizes)
        if ok:
            fast._edge_fuse[key] = ws
        state['fused_' + key] = ok
    state.update({'enabled': enabled, 'hops': []})
    for layer in layers:
        assert type(layer) is cls and cls.__name__ == 'ElboGraphLayer'
        wb_forward = layer.forward

        def forward(self, H, tok_mask, G, z_prev, write=True, sit_E=None, _wb=wb_forward):
            if state['enabled']:
                return fast(self, H, tok_mask, G, z_prev, write, sit_E)
            return _wb(H, tok_mask, G, z_prev, write, sit_E)
        layer.forward = types.MethodType(forward, layer)
    state['native'] = native
    runtime.info.update(edge_graph_fast=True, edge_graph_native=bool(native))
    return state


