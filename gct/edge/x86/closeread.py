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
"""Close-read block on packed real rows with the request K/V shared across candidates (W8A8 or fp32)."""
from __future__ import annotations

import types

import torch

from .common import STATE, effective
from .kernels import ext, make


def install_rca(block, kind, last_row=True):
    lm = block.lm
    mk = make(kind)
    E = ext()
    cfg = lm.config
    hq, hkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    rep = hq // hkv
    layers = []
    for li in range(block.l0, block.read):
        L = lm.layers[li]
        at, mlp = L.self_attn, L.mlp
        lo = block.lora[li - block.l0]
        shared = block.__dict__.get('base')
        sh = shared.lora[li - block.l0] if shared is not None else None

        def t(name, _lo=lo, _sh=sh):
            return ([_sh[name]] if _sh is not None and len(_sh) else []) + ([_lo[name]] if len(_lo) else [])
        wq, wk, wv = effective(at.q_proj, t('q')), effective(at.k_proj, t('k')), effective(at.v_proj, t('v'))
        wg, wu = effective(mlp.gate_proj, t('gate')), effective(mlp.up_proj, t('up'))
        f = lambda m: m.weight.detach().float().contiguous()
        layers.append(dict(ln1=f(L.input_layernorm), ln2=f(L.post_attention_layernorm), qn=f(at.q_norm), kn=f(at.k_norm),
                           eps=float(L.input_layernorm.variance_epsilon), qeps=float(at.q_norm.variance_epsilon),
                           qkv=mk(torch.cat([wq, wk, wv], 0)), o=mk(effective(at.o_proj, t('o'))),
                           gu=mk(torch.cat([wg, wu], 0)), inter=wg.shape[0],
                           down=mk(effective(mlp.down_proj, t('down'))), scaling=float(at.scaling)))
    head_w = block.head.weight.detach().float(); head_b = block.head.bias.detach().float()
    norm_w = block.norm.weight.detach().float().contiguous(); norm_eps = float(block.norm.eps or 1e-6)
    nl = len(layers)

    def forward(self, req_kv, req_mask, unit_states, unit_mask, pos0):
        B, M, T, d = unit_states.shape
        outs = []
        for b in range(B):
            um = unit_mask[b].reshape(M, T).contiguous()
            rows = um.reshape(-1).nonzero().squeeze(1)
            R = rows.numel()
            out = torch.zeros(M, head_w.shape[0])
            if R == 0:
                outs.append(out); continue
            counts = um.sum(-1)
            live = counts > 0
            ends = (counts.cumsum(0) - 1)[live]
            x = unit_states[b].reshape(M * T, d).index_select(0, rows).float().contiguous()
            blk, tpos = (rows // T).contiguous(), (rows % T).contiguous()
            cos, sin = lm.rotary_emb(x[None], (pos0[b] + tpos)[None])
            cos, sin = cos[0].float().contiguous(), sin[0].float().contiguous()
            reqm = req_mask[b].contiguous()
            for j, ly in enumerate(layers):
                Kr, Vr = req_kv[j]
                Kr, Vr = Kr[b].float(), Vr[b].float()
                h = E.rms_norm(x, ly['ln1'], ly['eps'])
                qkv = ly['qkv'](h).contiguous()
                q_req, k_pad, v_pad = E.qkv_prep2(qkv, ly['qn'], ly['kn'], cos, sin, ly['qeps'], hq, hkv, blk, tpos, M, T)
                if last_row and j == nl - 1:
                    q_req = q_req.index_select(1, ends).contiguous()
                    b_l, t_l = blk.index_select(0, ends).contiguous(), tpos.index_select(0, ends).contiguous()
                    x = x.index_select(0, ends).contiguous()
                else:
                    b_l, t_l = blk, tpos
                Rq = q_req.shape[1]
                s_req = torch.bmm(q_req.view(hkv, Rq * rep, hd), Kr.transpose(1, 2))
                o = E.attn_own(s_req, q_req, k_pad, v_pad, reqm, um, b_l, t_l, ly['scaling'])
                E.add_req(o, torch.bmm(s_req, Vr).view(hkv, Rq, rep, hd))
                x = ly['o'](o, res=x)
                h = E.rms_norm(x, ly['ln2'], ly['eps'])
                x = ly['down'](E.silu_mul(ly['gu'](h).contiguous(), ly['inter']), res=x)
            hidden = x if last_row else x.index_select(0, ends)
            hn = E.rms_norm(hidden.contiguous(), norm_w, norm_eps)
            out[live] = hn @ head_w.t() + head_b
            outs.append(out)
        return torch.stack(outs)

    block.forward = types.MethodType(forward, block)
    return {'rca_layers': nl, 'layout': 'v3: packed rows, shared request K/V, fused C++, last layer last rows only' if last_row
            else 'v3 without last-row'}


