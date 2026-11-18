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
"""Fused request decoder layers with exact reuse of the request-header prefix in the layers before the first graph layer."""
from __future__ import annotations

from collections import OrderedDict
import re
import types

import torch
import torch.nn.functional as F

from .common import STATE, effective
from .kernels import ext, make

HEADER = re.compile(r'^(\[Now: [^\n]*\]\n(?:Current request:)?|［当前时间：[^\n]*］\n(?:当前请求：)?)')


class FusedRequest:
    def __init__(self, model, kind, prefix=True, capacity=8):
        self.model, self.E = model, ext()
        lm = model.lm
        cfg = lm.config
        self.hq, self.hkv, self.hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.rep = self.hq // self.hkv
        self.prefix_layers = min(min(model.c.graph_after), model.rca.l0) if prefix else 0
        self.capacity = capacity
        self.entries = OrderedDict()
        self.head_ids = {}
        self.stats = {'hits': 0, 'misses': 0, 'unmatched': 0, 'fallback': 0}
        self.state = None
        mk = make(kind)
        f = lambda m: m.weight.detach().float().contiguous()
        self.specs = []
        for li, L in enumerate(lm.layers):
            at, mlp = L.self_attn, L.mlp
            wq, wk, wv = effective(at.q_proj), effective(at.k_proj), effective(at.v_proj)
            wg, wu = effective(mlp.gate_proj), effective(mlp.up_proj)
            self.specs.append(dict(li=li, ln1=f(L.input_layernorm), ln2=f(L.post_attention_layernorm), qn=f(at.q_norm),
                                   kn=f(at.k_norm), eps=float(L.input_layernorm.variance_epsilon),
                                   qeps=float(at.q_norm.variance_epsilon), qkv=mk(torch.cat([wq, wk, wv], 0)),
                                   o=mk(effective(at.o_proj)), gu=mk(torch.cat([wg, wu], 0)), inter=wg.shape[0],
                                   down=mk(effective(mlp.down_proj)), scaling=float(at.scaling)))
            original = L.forward

            def forward(self_, hidden_states, attention_mask=None, position_ids=None, past_key_values=None, use_cache=False,
                        position_embeddings=None, _orig=original, _spec=self.specs[-1], **kw):
                if STATE['on']:
                    out = self.run(_spec, hidden_states, attention_mask, past_key_values, position_embeddings)
                    if out is not None:
                        return out
                    self.stats['fallback'] += 1
                return _orig(hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                             past_key_values=past_key_values, use_cache=use_cache,
                             position_embeddings=position_embeddings, **kw)
            L.forward = types.MethodType(forward, L)
        fused = self
        model_forward = model.forward

        def mforward(self_, texts, G, cand_mask=None, _orig=model_forward):
            fused.state = {'text': texts[0] if len(texts) == 1 else None}
            try:
                return _orig(texts, G, cand_mask)
            finally:
                fused.state = None
        model.forward = types.MethodType(mforward, model)
        lm_forward = lm.forward

        def lforward(*a, _orig=lm_forward, **kw):
            fused.begin(kw.get('input_ids'))
            return _orig(*a, **kw)
        lm.forward = lforward


    def begin(self, ids):
        st = self.state
        if st is None:
            return
        st['mode'] = 'off'
        if not self.prefix_layers or ids is None or ids.shape[0] != 1 or st['text'] is None:
            return
        m = HEADER.match(st['text'])
        if not m:
            self.stats['unmatched'] += 1
            return
        head = self.head_ids.get(m.group(1))
        if head is None:
            head = self.head_ids[m.group(1)] = tuple(self.model.tok(m.group(1), add_special_tokens=False)['input_ids'])
            while len(self.head_ids) > 64:
                self.head_ids.pop(next(iter(self.head_ids)))
        p = len(head)
        if ids.shape[1] <= p or tuple(ids[0, :p].tolist()) != head:
            self.stats['unmatched'] += 1
            return
        if head in self.entries:
            self.entries.move_to_end(head)
            st.update(mode='hit', p=p, entry=self.entries[head])
            self.stats['hits'] += 1
        else:
            st.update(mode='miss', p=p, head=head, entry={'out': {}, 'k': {}, 'v': {}})
            self.stats['misses'] += 1


    def _plain_causal(self, mask, t):
        if mask is None:
            return True
        if mask.dim() != 4 or mask.shape[0] != 1 or mask.shape[-1] != t or mask.shape[-2] != t:
            return False
        m = mask[0, 0]
        allowed = m if m.dtype == torch.bool else (m == 0)
        return bool(torch.equal(allowed, torch.ones(t, t, dtype=torch.bool).tril()))

    def run(self, s, hidden, mask, cache, pe):
        if hidden.shape[0] != 1 or hidden.dtype != torch.float32 or torch.is_grad_enabled() or pe is None:
            return None
        T = hidden.shape[1]
        if not self._plain_causal(mask, T):
            return None
        E, hq, hkv, hd, rep = self.E, self.hq, self.hkv, self.hd, self.rep
        li = s['li']
        st = self.state or {}
        mode = st.get('mode', 'off') if li < self.prefix_layers else 'off'
        p = st.get('p', 0) if mode == 'hit' else 0
        x = hidden[0, p:].contiguous()
        n = T - p
        cos, sin = pe
        cos, sin = cos[0, p:].float().contiguous(), sin[0, p:].float().contiguous()
        h = E.rms_norm(x, s['ln1'], s['eps'])
        qkv = s['qkv'](h).contiguous()
        zeros = torch.zeros(n, dtype=torch.long)
        q_req, k_pad, v_pad = E.qkv_prep2(qkv, s['qn'], s['kn'], cos, sin, s['qeps'], hq, hkv, zeros,
                                          torch.arange(n), 1, n)
        q = q_req.permute(0, 2, 1, 3).reshape(1, hq, n, hd)
        k, v = k_pad, v_pad
        if mode == 'hit':
            entry = st['entry']
            kk = torch.cat([entry['k'][li], k], 2)
            vv = torch.cat([entry['v'][li], v], 2)
            allowed = torch.cat([torch.ones(n, p, dtype=torch.bool), torch.ones(n, n, dtype=torch.bool).tril()], 1)
            att = F.scaled_dot_product_attention(q, kk, vv, attn_mask=allowed, scale=s['scaling'], enable_gqa=True)
        else:
            kk, vv = k, v
            att = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=s['scaling'], enable_gqa=True)
        o = att[0].transpose(0, 1).reshape(n, hq * hd).contiguous()
        x = s['o'](o, res=x)
        h = E.rms_norm(x, s['ln2'], s['eps'])
        x = s['down'](E.silu_mul(s['gu'](h).contiguous(), s['inter']), res=x)
        if cache is not None:
            cache.update(kk.contiguous(), vv.contiguous(), li)
        if mode == 'miss':
            p0 = st['p']
            entry = st['entry']
            entry['out'][li] = x[:p0].clone()
            entry['k'][li] = k[:, :, :p0].clone()
            entry['v'][li] = v[:, :, :p0].clone()
            if li == self.prefix_layers - 1:
                self.entries[st['head']] = entry
                while len(self.entries) > self.capacity:
                    self.entries.popitem(last=False)
        if mode == 'hit':
            x = torch.cat([st['entry']['out'][li], x], 0)
        return x[None]


def install(model, kind, prefix=True):
    fr = FusedRequest(model, kind, prefix)
    return fr
