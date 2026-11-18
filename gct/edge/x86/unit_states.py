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
"""Close-read block states of the candidate units with the fixed block prefix computed once."""
from __future__ import annotations

import torch

from .kernels import ext, make


class PrefixEncoder:
    def __init__(self, lm, tok, l0):
        from ... import rca
        from transformers import DynamicCache
        self.lm, self.tok, self.l0, self.E = lm, tok, l0, ext()
        cfg = lm.config
        self.hq, self.hkv, self.hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.rep = self.hq // self.hkv
        mk = make('fp32')
        f = lambda m: m.weight.detach().float().contiguous()
        self.layers = []
        for li in range(l0):
            L = lm.layers[li]
            at, mlp = L.self_attn, L.mlp
            assert type(at.q_proj).__name__ == 'Linear', 'layers below l0 must be plain Linear (no LoRA)'
            self.layers.append(dict(ln1=f(L.input_layernorm), ln2=f(L.post_attention_layernorm), qn=f(at.q_norm), kn=f(at.k_norm),
                                    eps=float(L.input_layernorm.variance_epsilon), qeps=float(at.q_norm.variance_epsilon),
                                    qkv=mk(torch.cat([f(at.q_proj), f(at.k_proj), f(at.v_proj)], 0)), o=mk(f(at.o_proj)),
                                    gu=mk(torch.cat([f(mlp.gate_proj), f(mlp.up_proj)], 0)), inter=mlp.gate_proj.weight.shape[0],
                                    down=mk(f(mlp.down_proj)), scaling=float(at.scaling)))
        pre = tok(rca.PREFIX, add_special_tokens=False)['input_ids']
        self.P = len(pre)
        with torch.no_grad():
            ids = torch.tensor(pre)[None]
            H = lm.embed_tokens(ids).float()
            pos = torch.arange(self.P)[None]
            pe = lm.rotary_emb(H, pos)
            cache = DynamicCache()
            for li in range(l0):
                H = lm.layers[li](H, attention_mask=None, position_ids=pos, past_key_values=cache, use_cache=True, position_embeddings=pe)
            self.pk = [cache.layers[li].keys[0].float().contiguous() for li in range(l0)]
            self.pv = [cache.layers[li].values[0].float().contiguous() for li in range(l0)]

    @torch.no_grad()
    def blocks(self, id_lists, T, chunk=256):
        E, hq, hkv, hd, rep = self.E, self.hq, self.hkv, self.hd, self.rep
        d = self.lm.config.hidden_size
        n = len(id_lists)
        S = torch.zeros(n, T, d)
        Mk = torch.zeros(n, T, dtype=torch.bool)
        for i in range(0, n, chunk):
            part = id_lists[i:i + chunk]
            B = len(part)
            blk = torch.zeros(B, T, dtype=torch.long)
            um = torch.zeros(B, T, dtype=torch.bool)
            for j, x in enumerate(part):
                blk[j, :len(x)] = torch.tensor(x); um[j, :len(x)] = True
            rows = um.reshape(-1).nonzero().squeeze(1)
            R = rows.numel()
            x = self.lm.embed_tokens(blk.reshape(-1).index_select(0, rows)).float().contiguous()
            b_, t_ = (rows // T).contiguous(), (rows % T).contiguous()
            cos, sin = self.lm.rotary_emb(x[None], (self.P + t_)[None])
            cos, sin = cos[0].float().contiguous(), sin[0].float().contiguous()
            reqm = torch.ones(self.P, dtype=torch.bool)
            umc = um.contiguous()
            for li, ly in enumerate(self.layers):
                h = E.rms_norm(x, ly['ln1'], ly['eps'])
                qkv = ly['qkv'](h).contiguous()
                q_req, k_pad, v_pad = E.qkv_prep2(qkv, ly['qn'], ly['kn'], cos, sin, ly['qeps'], hq, hkv, b_, t_, B, T)
                s_req = torch.bmm(q_req.view(hkv, R * rep, hd), self.pk[li].transpose(1, 2))
                o = E.attn_own(s_req, q_req, k_pad, v_pad, reqm, umc, b_, t_, ly['scaling'])
                E.add_req(o, torch.bmm(s_req, self.pv[li]).view(hkv, R, rep, hd))
                x = ly['o'](o, res=x)
                h = E.rms_norm(x, ly['ln2'], ly['eps'])
                x = ly['down'](E.silu_mul(ly['gu'](h).contiguous(), ly['inter']), res=x)
            S[i:i + B].reshape(B * T, d).index_copy_(0, rows, x)
            Mk[i:i + B] = um
        return S, Mk


_ENC = {}


def encode_units_fast(lm, tok, texts, l0, T, suffix="tmpl", chunk=256, seg=1):
    from ... import rca
    key = (id(lm), l0)
    if key not in _ENC:
        _ENC[key] = PrefixEncoder(lm, tok, l0)
    enc = _ENC[key]
    suf = tok(rca.SUFFIX[suffix], add_special_tokens=False)['input_ids']
    keep = T - len(suf)
    full = tok(list(texts), add_special_tokens=False)['input_ids'] if len(texts) else []
    st, mk = enc.blocks([ids[:keep] + suf for ids in full], T, chunk)
    if seg <= 1:
        return st, mk
    xi = torch.full((len(texts), seg - 1), -1, dtype=torch.long)
    pieces = []
    for i, ids in enumerate(full):
        if len(ids) <= keep:
            continue
        need = -(-len(ids) // keep)
        ext_ = [ids[j * keep:(j + 1) * keep] for j in range(1, min(need, seg))]
        if need > seg:
            ext_[-1] = ids[-keep:]
        for j, p in enumerate(ext_):
            xi[i, j] = len(pieces); pieces.append(p)
    if not pieces:
        return st, mk, st.new_zeros(0, T, st.shape[-1]), mk.new_zeros(0, T), xi
    xs, xm = enc.blocks([p + suf for p in pieces], T, chunk)
    return st, mk, xs, xm, xi


def install():
    from ... import rca
    original = rca.encode_units
    rca.encode_units = encode_units_fast
    return original
