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
"""Frozen encoders and caches: pooled unit vectors and token memories from the first layers of Qwen3-0.6B,
close-read block states of the candidate units."""
import os
import time
import hashlib
from pathlib import Path

import torch

QWEN = None
TOKENIZER = None
EMB_MAXLEN = 96
TOK_CAP = 512
EMB_LAYER = 14


TOK_SINK = "0" == "1"


class TextEncoder:
    def __init__(self, device="cuda", layer=EMB_LAYER, dtype=torch.bfloat16):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(TOKENIZER or QWEN)
        m = AutoModelForCausalLM.from_pretrained(QWEN, dtype=dtype, attn_implementation="sdpa").model
        m.layers = m.layers[:layer]
        self.m = m.to(device).eval()
        self.device = device

    @torch.no_grad()
    def encode(self, texts, bs=256):
        out = []
        for i in range(0, len(texts), bs):
            b = self.tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=EMB_MAXLEN).to(self.device)
            h = self.m(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state.float()
            mask = b["attention_mask"].unsqueeze(-1).float()
            out.append(((h * mask).sum(1) / mask.sum(1)).cpu())
        return torch.cat(out) if out else torch.zeros(0, 1024)


    @torch.no_grad()
    def encode_tokens_at(self, texts, starts, T_u, head=0, layers=None, bs=128):
        layers = list(layers) if layers else [len(self.m.layers)]
        assert max(layers) <= len(self.m.layers), f"layers {layers} beyond the encoder's {len(self.m.layers)}"
        d, K = self.m.norm.weight.shape[0], len(layers)
        out = torch.zeros(len(texts), T_u, K * d, dtype=torch.float16, device=self.device)
        mask = torch.zeros(len(texts), T_u, dtype=torch.bool, device=self.device)
        cap = {}
        hooks = [self.m.layers[l - 1].register_forward_hook(lambda _m, _i, o, l=l: cap.__setitem__(l, o[0] if isinstance(o, tuple) else o))
                 for l in layers if l != len(self.m.layers)]
        try:
            for i in range(0, len(texts), bs):
                b = self.tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=TOK_CAP).to(self.device)
                cap.clear()
                last = self.m(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
                hs = [last if l == len(self.m.layers) else self.m.norm(cap[l]) for l in layers]
                L = b["attention_mask"].sum(1).tolist()
                idx = torch.zeros(len(L), T_u, dtype=torch.long)
                for r, (n_tok, s0) in enumerate(zip(L, starts[i:i + bs])):
                    s = max(int(s0), head)
                    pos = list(range(min(head, n_tok))) + list(range(s, min(n_tok, s + T_u - head)))
                    if not pos:
                        pos = list(range(min(T_u, n_tok)))
                    idx[r, :len(pos)] = torch.tensor(pos); mask[i + r, :len(pos)] = True
                idx = idx.to(self.device)
                for k, h in enumerate(hs):
                    out[i:i + len(L), :, k * d:(k + 1) * d] = h.gather(1, idx[..., None].expand(-1, -1, h.shape[-1])).to(torch.float16)
                out[i:i + len(L)] *= mask[i:i + len(L), :, None]
                del hs, last
        finally:
            for h in hooks:
                h.remove()
        return out, mask

    @torch.no_grad()
    def encode_tokens(self, texts, T_u=16, bs=256):
        if T_u <= 0:
            T_u = max([len(x) for x in self.tok(list(texts), add_special_tokens=True)["input_ids"]] or [1])
            T_u = max(1, min(T_u, TOK_CAP))
        out = mask = None
        for i in range(0, len(texts), bs):
            b = self.tok(texts[i:i + bs], return_tensors="pt", padding="max_length", truncation=True, max_length=T_u).to(self.device)
            if TOK_SINK:
                ids, am = b["input_ids"], b["attention_mask"]
                sid = self.tok.convert_tokens_to_ids("<|im_start|>")
                h = self.m(input_ids=torch.cat([torch.full_like(ids[:, :1], sid), ids], 1),
                           attention_mask=torch.cat([torch.ones_like(am[:, :1]), am], 1)).last_hidden_state[:, 1:]
            else:
                h = self.m(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
            if out is None:
                out = torch.empty(len(texts), T_u, h.shape[-1], dtype=torch.float16, device=self.device)
                mask = torch.empty(len(texts), T_u, dtype=torch.bool, device=self.device)
            out[i:i + h.shape[0]] = h.to(torch.float16); mask[i:i + h.shape[0]] = b["attention_mask"].bool()
            del h
        return (out, mask) if out is not None else (torch.zeros(0, T_u, 1024, dtype=torch.float16, device=self.device),
                                                     torch.zeros(0, T_u, dtype=torch.bool, device=self.device))


class TokenProvider:

    def __init__(self, encoder, T_u=16, cap=512, span="head", head=0, layers=None):
        from collections import OrderedDict
        self.enc, self.T_u, self.cap = encoder, T_u, cap


        self.span, self.head, self.layers = span, head, list(layers) if layers else [EMB_LAYER]
        self.custom = span != "head" or self.layers != [EMB_LAYER]
        if self.custom:
            assert T_u > 0 and not TOK_SINK, "tok_span / tok_layers need a fixed T_u and no GRAPH_TOK_SINK"
            self.tenc = encoder if max(self.layers) == EMB_LAYER else TextEncoder(encoder.device, layer=max(self.layers))
        self.lru = OrderedDict()
        self.host, self.host_bytes = OrderedDict(), 0
        self.table, self.table_stats = None, {"views": 0, "units": 0, "units_encoded": 0, "views_fallback": 0}


        self.chunks, self.chunk_len, self.chunk_start = 0, 16, 0


        self.xprov, self.lru_gb, self.host_gb = None, None, None
        self.clru = OrderedDict()


    def _encode_custom(self, cg):
        starts = [0] * cg.n
        if self.span == "value":
            va = cg.payload.get("val_at")
            assert va is not None, "tok_span value needs compile.VAL_AT (train.py sets it with --tok_span value)"
            offs = self.enc.tok(list(cg.unit_text), add_special_tokens=True, return_offsets_mapping=True)["offset_mapping"]
            for i, (a, o) in enumerate(zip(va, offs)):
                if a > 0:
                    starts[i] = next((k for k, (x, y) in enumerate(o) if y > a), len(o))
        return self.tenc.encode_tokens_at(list(cg.unit_text), starts, self.T_u, self.head,
                                          None if self.layers == [len(self.tenc.m.layers)] else self.layers)

    def _to_host(self, key, val):
        hb = int(float(self.host_gb if self.host_gb is not None else "0") * (1 << 30))
        sz = sum(t.numel() * t.element_size() for t in val)
        if hb <= 0 or sz > hb:
            return
        self.host[key] = tuple(t.to("cpu") for t in val)
        self.host_bytes += sz
        while self.host_bytes > hb:
            _, old = self.host.popitem(last=False)
            self.host_bytes -= sum(t.numel() * t.element_size() for t in old)

    def get(self, cg):

        key = (cg.world_id, cg.caller, cg.n, hash(tuple(cg.unit_text)))
        if key in self.lru:
            self.lru.move_to_end(key)
            return self.lru[key]
        hit = self.host.pop(key, None)
        if hit is not None:
            self.host_bytes -= sum(t.numel() * t.element_size() for t in hit)


        budget = int(float(self.lru_gb if self.lru_gb is not None else "6") * (1 << 30))
        est = cg.n * max(self.T_u, 1) * 1024 * 2 * len(self.layers)
        total = sum(t.numel() * t.element_size() for t, _ in self.lru.values())
        while self.lru and total + est > budget:
            k_old, v_old = self.lru.popitem(last=False)
            total -= v_old[0].numel() * v_old[0].element_size()
            self._to_host(k_old, v_old)
        got = None
        if hit is not None:
            got = tuple(t.to(self.enc.device) for t in hit)
        elif self.custom:
            got = self._encode_custom(cg)
        tok, mask = got if got is not None else self.enc.encode_tokens(cg.unit_text, self.T_u)
        self.lru[key] = (tok, mask)
        total = sum(t.numel() * t.element_size() for t, _ in self.lru.values())
        while len(self.lru) > 1 and (len(self.lru) > self.cap or total > budget):
            k_old, v_old = self.lru.popitem(last=False)
            total -= v_old[0].numel() * v_old[0].element_size()
            self._to_host(k_old, v_old)
        return tok, mask


    def get_chunks(self, cg):
        key = (cg.world_id, cg.caller, cg.n, hash(tuple(cg.unit_text)))
        if key in self.clru:
            self.clru.move_to_end(key)
            return self.clru[key]
        from types import SimpleNamespace
        span, C, Lc = self.chunk_start + self.chunks * self.chunk_len, self.chunks, self.chunk_len
        texts, outs, oms = list(cg.unit_text), [], []
        for i in range(0, len(texts), 4096):
            got = None
            tok, mask = got if got is not None else self.enc.encode_tokens(texts[i:i + 4096], span)
            k, d = tok.shape[0], tok.shape[-1]
            x = tok[:, self.chunk_start:span].float().view(k, C, Lc, d)
            m = mask[:, self.chunk_start:span].view(k, C, Lc)
            cnt = m.sum(-1)
            outs.append(((x * m[..., None]).sum(2) / cnt.clamp_min(1)[..., None]).to(torch.float16)); oms.append(cnt > 0)
            del tok, mask, x
        dev = self.enc.device
        val = ((torch.cat(outs), torch.cat(oms)) if outs else
               (torch.zeros(0, C, 1024, dtype=torch.float16, device=dev), torch.zeros(0, C, dtype=torch.bool, device=dev)))
        budget = int(float("2") * (1 << 30))
        self.clru[key] = val
        total = sum(t.numel() * t.element_size() for t, _ in self.clru.values())
        while len(self.clru) > 1 and (len(self.clru) > self.cap or total > budget):
            _, v_old = self.clru.popitem(last=False)
            total -= v_old[0].numel() * v_old[0].element_size()
        return val


class RCAStateProvider:

    def __init__(self, lm, tok, l0, T, suffix, seg=1):
        from collections import OrderedDict
        self.lm, self.tok, self.l0, self.T, self.suffix = lm, tok, l0, T, suffix
        self.seg = seg
        self.lru, self.host, self.host_bytes = OrderedDict(), OrderedDict(), 0

    def _bytes(self, v):
        return sum(t.numel() * t.element_size() for t in v)

    def _prof(self, kind, n_enc=0, secs=0.0):
        return
        st = self.__dict__.setdefault("_st", {"calls": 0, "gpu": 0, "host": 0, "miss": 0, "units": 0, "secs": 0.0, "t0": time.time()})
        st["calls"] += 1; st[kind] += 1; st["units"] += n_enc; st["secs"] += secs
        if st["calls"] % 200 == 0:
            print(f"rca_prof calls {st['calls']} gpu_hit {st['gpu']} host_hit {st['host']} miss {st['miss']} units_enc {st['units']} "
                  f"enc_s {st['secs']:.1f} wall_s {time.time() - st['t0']:.1f}", flush=True)

    def build_table(self, cg, chunk=512):
        from .rca import encode_units
        vkey = (cg.world_id, cg.caller, cg.n, hash(tuple(cg.unit_text)))
        if vkey in self.__dict__.setdefault("tables", {}):
            return
        self.tables = {}
        if cg.n == 0:
            return
        S = M = None
        X, XM, XI, r = [], [], [], 0
        with torch.no_grad():
            for i in range(0, cg.n, chunk):
                out = encode_units(self.lm, self.tok, list(cg.unit_text[i:i + chunk]), self.l0, self.T, self.suffix, seg=self.seg)
                if S is None:
                    S = out[0].new_empty((cg.n,) + tuple(out[0].shape[1:])); M = out[1].new_empty((cg.n,) + tuple(out[1].shape[1:]))
                S[i:i + out[0].shape[0]] = out[0]; M[i:i + out[1].shape[0]] = out[1]
                if self.seg > 1:
                    xi = out[4].clone(); xi[xi >= 0] += r; r += out[2].shape[0]
                    X.append(out[2]); XM.append(out[3]); XI.append(xi)
                del out
        tb = {"S": S, "M": M}
        if self.seg > 1:
            tb.update(X=torch.cat(X), XM=torch.cat(XM), XI=torch.cat(XI))
        self.tables = {vkey: tb}
        cg._rca_vkey = vkey

    def get_units(self, cg, uidx):
        from collections import OrderedDict
        tb = self.__dict__.get("tables", {}).get(getattr(cg, "_rca_vkey", None))
        if tb is not None:
            if self.seg <= 1:
                return tb["S"][uidx], tb["M"][uidx]
            return tb["S"][uidx], tb["M"][uidx], tb["X"], tb["XM"], tb["XI"][uidx]
        if not hasattr(self, "ucache"):
            self.ucache, self.ubytes = OrderedDict(), 0
        vkey = getattr(cg, "_rca_vkey", None)
        if vkey is None:
            vkey = (cg.world_id, cg.caller, cg.n, hash(tuple(cg.unit_text)))
            try:
                cg._rca_vkey = vkey
            except Exception:
                pass
        ids = uidx.tolist()
        for i in ids:
            if (vkey, i) in self.ucache:
                self.ucache.move_to_end((vkey, i))
        need = [i for i in ids if (vkey, i) not in self.ucache]
        t0 = time.time()
        if need:
            from .rca import encode_units
            with torch.no_grad():
                out = encode_units(self.lm, self.tok, [cg.unit_text[i] for i in need], self.l0, self.T, self.suffix, seg=self.seg)
            for j, i in enumerate(need):
                if self.seg > 1:
                    st, mk, xs, xm, xi = out
                    rows = xi[j][xi[j] >= 0]
                    val = (st[j], mk[j], xs[rows], xm[rows])
                else:
                    val = (out[0][j], out[1][j])
                self.ucache[(vkey, i)] = val; self.ubytes += self._bytes(val)
        vals = [self.ucache[(vkey, i)] for i in ids]
        budget = int(float("12") * (1 << 30))
        while self.ubytes > budget and len(self.ucache) > len(ids):
            _, old = self.ucache.popitem(last=False); self.ubytes -= self._bytes(old)
        self._prof("miss" if need else "gpu", len(need), time.time() - t0 if need else 0.0)
        S = torch.stack([v[0] for v in vals]); M = torch.stack([v[1] for v in vals])
        if self.seg <= 1:
            return S, M
        XI = torch.full((len(ids), self.seg - 1), -1, dtype=torch.long, device=S.device); xs, xm, r = [], [], 0
        for j, v in enumerate(vals):
            k = v[2].shape[0]
            if k:
                xs.append(v[2]); xm.append(v[3]); XI[j, :k] = torch.arange(r, r + k, device=S.device); r += k
        X = torch.cat(xs) if xs else S.new_zeros(0, self.T, S.shape[-1]); XM = torch.cat(xm) if xm else M.new_zeros(0, self.T)
        return S, M, X, XM, XI

    def get(self, cg):
        key = (cg.world_id, cg.caller, cg.n, hash(tuple(cg.unit_text)))
        if key in self.lru:
            self.lru.move_to_end(key)
            self._prof("gpu")
            return self.lru[key]
        dev = next(self.lm.parameters()).device
        hit = self.host.pop(key, None)
        t0 = time.time()
        if hit is not None:
            self.host_bytes -= self._bytes(hit)
            val = tuple(t.to(dev) for t in hit)
            self._prof("host")
        else:
            from .rca import encode_units
            texts, outs, oms = list(cg.unit_text), [], []
            with torch.no_grad():
                if self.seg > 1:
                    val = encode_units(self.lm, self.tok, texts, self.l0, self.T, self.suffix, seg=self.seg) if texts else None
                else:
                    for i in range(0, len(texts), 2048):
                        st, m = encode_units(self.lm, self.tok, texts[i:i + 2048], self.l0, self.T, self.suffix)
                        outs.append(st); oms.append(m)
                    val = (torch.cat(outs), torch.cat(oms)) if outs else None
            self._prof("miss", len(texts), time.time() - t0)
            if val is None:
                z, zm = (torch.zeros(0, self.T, self.lm.config.hidden_size, dtype=torch.bfloat16, device=dev), torch.zeros(0, self.T, dtype=torch.bool, device=dev))
                val = (z, zm) if self.seg <= 1 else (z, zm, z.clone(), zm.clone(), torch.zeros(0, self.seg - 1, dtype=torch.long, device=dev))
        budget = int(float("12") * (1 << 30))
        hb = int(float("0") * (1 << 30))
        self.lru[key] = val
        total = sum(self._bytes(v) for v in self.lru.values())
        while len(self.lru) > 1 and total > budget:
            k_old, v_old = self.lru.popitem(last=False)
            total -= self._bytes(v_old)
            if 0 < self._bytes(v_old) <= hb:
                self.host[k_old] = tuple(t.to("cpu") for t in v_old); self.host_bytes += self._bytes(v_old)
                while self.host_bytes > hb:
                    _, o = self.host.popitem(last=False); self.host_bytes -= self._bytes(o)
        return val


class EmbCache:

    def __init__(self, root, encoder=None):
        self.root = Path(root) if root is not None else None
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)
        self.enc = encoder
        self.mem = {}

    def get(self, key, texts):
        path = self.root / f"{key}.pt" if self.root is not None else None
        if key not in self.mem and path is not None and path.exists():
            try:
                self.mem[key] = torch.load(path)
            except Exception:
                self.mem[key] = {}
        table = self.mem.setdefault(key, {})
        missing = [t for t in dict.fromkeys(texts) if self._h(t) not in table]
        if missing:
            if self.enc is None:
                raise RuntimeError(f"{len(missing)} texts not cached for {key} and no encoder loaded")
            vecs = self.enc.encode(missing)
            for t, v in zip(missing, vecs):
                table[self._h(t)] = v.to(torch.float16)
            if path is not None:
                tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
                torch.save(table, tmp)
                os.replace(tmp, path)
        return torch.stack([table[self._h(t)] for t in texts]).float() if texts else torch.zeros(0, 1024)

    @staticmethod
    def _h(t):
        return hashlib.sha1((t if EMB_MAXLEN == 96 else f"<len{EMB_MAXLEN}>{t}").encode("utf-8")).hexdigest()
