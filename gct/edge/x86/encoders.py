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
"""Compile-time encoders on the CPU: fp32 weights and length-sorted batches (outputs in the original order)."""
from __future__ import annotations

import torch


def _perm(lengths):
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    inv = [0] * len(order)
    for pos, i in enumerate(order):
        inv[i] = pos
    return order, torch.tensor(inv, dtype=torch.long)


def install(R, fp32=True, sort=True):
    from ... import embed as E
    info = {'fp32': fp32, 'sort': sort}
    if fp32:
        R.enc.m.float(); R.kenc.m.float()
    if not sort:
        return info
    enc, kenc = R.enc, R.kenc
    enc_encode, tok_at, k_encode = enc.encode, enc.encode_tokens_at, kenc.encode

    def encode(texts, bs=256):
        texts = list(texts)
        if len(texts) < 2:
            return enc_encode(texts, bs)
        lens = [len(x) for x in enc.tok(texts, truncation=True, max_length=E.EMB_MAXLEN)['input_ids']]
        order, inv = _perm(lens)
        return enc_encode([texts[i] for i in order], bs)[inv]

    def encode_tokens_at(texts, starts, T_u, head=0, layers=None, bs=128):
        texts = list(texts)
        if len(texts) < 2:
            return tok_at(texts, starts, T_u, head, layers, bs)
        lens = [len(x) for x in enc.tok(texts, truncation=True, max_length=E.TOK_CAP)['input_ids']]
        order, inv = _perm(lens)
        out = tok_at([texts[i] for i in order], [starts[i] for i in order], T_u, head, layers, bs)
        if isinstance(out, tuple):
            return tuple(t[inv] if torch.is_tensor(t) and t.shape[:1] == (len(texts),) else t for t in out)
        return out[inv]

    def k_enc(texts, bs=64):
        texts = list(texts)
        if len(texts) < 2:
            return k_encode(texts, bs)
        kenc.tok.truncation_side = 'right'
        lens = [len(x) for x in kenc.tok(texts, truncation=True, max_length=kenc.max_len)['input_ids']]
        order, inv = _perm(lens)
        return k_encode([texts[i] for i in order], bs)[inv]
    enc.encode, enc.encode_tokens_at, kenc.encode = encode, encode_tokens_at, k_enc
    return info
