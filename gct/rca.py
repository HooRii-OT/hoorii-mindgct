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
"""Close-read block: candidate unit text and request read together by the backbone's upper layers."""
import torch
import torch.nn as nn
import torch.nn.functional as F


PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. "
          "Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n<Instruct>: Given a user's request to their "
          "personal assistant (with the conversation so far), judge whether this record from the user's personal knowledge base is "
          "needed to handle the request\n<Document>: ")
SUFFIX = {"tmpl": "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", "eos": "<|im_end|>"}


def _cfg(c, name, default):
    return getattr(c, name, None) if getattr(c, name, None) is not None else default


@torch.no_grad()
def encode_units(lm, tok, texts, l0, T, suffix="tmpl", chunk=256, seg=1):
    if seg > 1:
        return _encode_seg(lm, tok, texts, l0, T, suffix, chunk, seg)
    for li in range(l0):
        att = lm.layers[li].self_attn
        assert not any(p.requires_grad for p in lm.layers[li].parameters()), f"rca: layer {li} < l0 = {l0} is trainable"
        assert type(att.q_proj).__name__ == "Linear", f"rca: layer {li} < l0 = {l0} carries a LoRA (l0 must be <= the first graph layer)"
    dev = lm.embed_tokens.weight.device
    pre = tok(PREFIX, add_special_tokens=False)["input_ids"]
    suf = tok(SUFFIX[suffix], add_special_tokens=False)["input_ids"]
    keep = T - len(suf)
    assert keep >= 1, f"rca: T = {T} leaves no room for the unit after the {len(suf)}-token suffix"
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    S, M = [], []
    for i in range(0, len(texts), chunk):
        ids = [tok(t, add_special_tokens=False)["input_ids"][:keep] + suf for t in texts[i:i + chunk]]
        blk = torch.full((len(ids), T), pad, dtype=torch.long)
        msk = torch.zeros(len(ids), T, dtype=torch.bool)
        for j, x in enumerate(ids):
            blk[j, :len(x)] = torch.tensor(x); msk[j, :len(x)] = True
        X = torch.cat([torch.tensor(pre)[None].expand(len(ids), -1), blk], 1).to(dev)
        H = lm.embed_tokens(X)
        pos = torch.arange(X.shape[1], device=dev)[None].expand(len(ids), -1)
        pe = lm.rotary_emb(H, pos)
        for li in range(l0):
            H = lm.layers[li](H, attention_mask=None, position_ids=pos, position_embeddings=pe)
        S.append(H[:, len(pre):].detach()); M.append(msk.to(dev))
    return torch.cat(S), torch.cat(M)


def _encode_seg(lm, tok, texts, l0, T, suffix, chunk, seg):
    suf = tok(SUFFIX[suffix], add_special_tokens=False)["input_ids"]
    keep = T - len(suf)
    full = tok(list(texts), add_special_tokens=False)["input_ids"] if len(texts) else []
    st, mk = encode_units(lm, tok, texts, l0, T, suffix, chunk)
    xi = torch.full((len(texts), seg - 1), -1, dtype=torch.long)
    pieces = []
    for i, ids in enumerate(full):
        if len(ids) <= keep:
            continue
        need = -(-len(ids) // keep)
        ext = [ids[j * keep:(j + 1) * keep] for j in range(1, min(need, seg))]
        if need > seg:
            ext[-1] = ids[-keep:]
        for j, p in enumerate(ext):
            xi[i, j] = len(pieces); pieces.append(p)
    dev = st.device
    if not pieces:
        return st, mk, st.new_zeros(0, T, st.shape[-1]), mk.new_zeros(0, T), xi.to(dev)
    xs, xm = _encode_ids(lm, tok, pieces, l0, T, suf, chunk)
    return st, mk, xs, xm, xi.to(dev)


def _encode_ids(lm, tok, pieces, l0, T, suf, chunk):
    dev = lm.embed_tokens.weight.device
    pre = tok(PREFIX, add_special_tokens=False)["input_ids"]
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    S, M = [], []
    for i in range(0, len(pieces), chunk):
        ids = [p + suf for p in pieces[i:i + chunk]]
        blk = torch.full((len(ids), T), pad, dtype=torch.long)
        msk = torch.zeros(len(ids), T, dtype=torch.bool)
        for j, x in enumerate(ids):
            blk[j, :len(x)] = torch.tensor(x); msk[j, :len(x)] = True
        X = torch.cat([torch.tensor(pre)[None].expand(len(ids), -1), blk], 1).to(dev)
        H = lm.embed_tokens(X)
        pos = torch.arange(X.shape[1], device=dev)[None].expand(len(ids), -1)
        pe = lm.rotary_emb(H, pos)
        for li in range(l0):
            H = lm.layers[li](H, attention_mask=None, position_ids=pos, position_embeddings=pe)
        S.append(H[:, len(pre):].detach()); M.append(msk.to(dev))
    return torch.cat(S), torch.cat(M)


class _LoRA(nn.Module):
    def __init__(self, din, dout, r, alpha):
        super().__init__()
        self.A = nn.Parameter(torch.randn(r, din) / r)
        self.B = nn.Parameter(torch.zeros(dout, r))
        self.scale = alpha / r

    def forward(self, x):
        return ((x.float() @ self.A.t()) @ self.B.t() * self.scale).to(x.dtype)


class RCABlock(nn.Module):

    def __init__(self, lm, c, yes_no=None):
        super().__init__()
        self.__dict__["lm"] = lm
        self.l0, self.read = int(_cfg(c, "rca_l0", 8)), int(_cfg(c, "rca_read_layer", 20))
        assert 0 <= self.l0 < self.read <= len(lm.layers), (self.l0, self.read, len(lm.layers))
        self.ckpt = bool(_cfg(c, "rca_ckpt", True))
        self.shared_grad = bool(_cfg(c, "rca_shared_grad", False))
        cfg = lm.config
        d, hq, hkv, hd = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.hq, self.hkv, self.hd = hq, hkv, hd
        r = int(_cfg(c, "rca_lora", 16))
        shapes = {"q": (d, hq * hd), "k": (d, hkv * hd), "v": (d, hkv * hd), "o": (hq * hd, d),
                  "gate": (d, cfg.intermediate_size), "up": (d, cfg.intermediate_size), "down": (cfg.intermediate_size, d)}
        self.lora = nn.ModuleList([nn.ModuleDict({k: _LoRA(a, b, r, 2 * r) for k, (a, b) in shapes.items()}) if r > 0 else nn.ModuleDict()
                                   for _ in range(self.l0, self.read)])
        self.norm = nn.RMSNorm(d, eps=cfg.rms_norm_eps)
        with torch.no_grad():
            self.norm.weight.copy_(lm.norm.weight.float())
        self.head = nn.Linear(d, 2)
        with torch.no_grad():
            self.head.bias.zero_()
            if str(_cfg(c, "rca_head_init", "yesno")) == "yesno" and yes_no is None:
                from transformers import AutoTokenizer
                yes_no = yes_no_vector(lm, AutoTokenizer.from_pretrained(lm.config._name_or_path))
            if str(_cfg(c, "rca_head_init", "yesno")) == "yesno":
                self.head.weight.copy_(yes_no.float()[None].expand(2, -1))
            else:
                self.head.weight.zero_()
        self.none_logit = nn.Parameter(torch.zeros(2))

    def _lin(self, base, x, lo):
        if hasattr(base, "base") and hasattr(base, "A") and not self.shared_grad:
            y = base.base(x) + ((x.float() @ base.A.detach().t()) @ base.B.detach().t() * base.scale).to(x.dtype)
        else:
            y = base(x)
        return y + lo(x) if lo is not None else y

    def _layer(self, li, x, pos_emb, req_k, req_v, mask):
        L = self.lm.layers[li]; at = L.self_attn; lo = self.lora[li - self.l0]
        g = (lambda k: lo[k]) if len(lo) else (lambda k: None)
        B, S, _ = x.shape
        h = L.input_layernorm(x)
        q = at.q_norm(self._lin(at.q_proj, h, g("q")).view(B, S, self.hq, self.hd)).transpose(1, 2)
        k = at.k_norm(self._lin(at.k_proj, h, g("k")).view(B, S, self.hkv, self.hd)).transpose(1, 2)
        v = self._lin(at.v_proj, h, g("v")).view(B, S, self.hkv, self.hd).transpose(1, 2)
        cos, sin = pos_emb
        cos, sin = cos[:, None], sin[:, None]
        rot = lambda t: torch.cat([-t[..., t.shape[-1] // 2:], t[..., :t.shape[-1] // 2]], -1)
        q, k = q * cos + rot(q) * sin, k * cos + rot(k) * sin
        K = torch.cat([req_k.to(k.dtype), k], 2); V = torch.cat([req_v.to(v.dtype), v], 2)
        rep = self.hq // self.hkv
        if rep > 1:
            K = K.repeat_interleave(rep, 1); V = V.repeat_interleave(rep, 1)
        o = F.scaled_dot_product_attention(q, K, V, attn_mask=mask, scale=at.scaling)
        x = x + self._lin(at.o_proj, o.transpose(1, 2).reshape(B, S, self.hq * self.hd), g("o"))
        h = L.post_attention_layernorm(x); m = L.mlp
        a = m.act_fn(self._lin(m.gate_proj, h, g("gate"))) * self._lin(m.up_proj, h, g("up"))
        return x + self._lin(m.down_proj, a, g("down"))

    def forward(self, req_kv, req_mask, unit_states, unit_mask, pos0):
        assert len(req_kv) == self.read - self.l0, (len(req_kv), self.l0, self.read)
        B, M, T, d = unit_states.shape
        S = M * T; dev = unit_states.device
        x = unit_states.reshape(B, S, d)
        um = unit_mask.reshape(B, S)
        blk = torch.arange(S, device=dev) // T; inb = torch.arange(S, device=dev) % T
        own = (blk[:, None] == blk[None, :]) & (inb[:, None] >= inb[None, :])
        own = own[None] & um[:, None, :]
        own = own | torch.eye(S, dtype=torch.bool, device=dev)[None]
        mask = torch.cat([req_mask[:, None, :].expand(B, S, -1), own], 2)[:, None]
        pos = pos0[:, None] + inb[None, :]
        pos_emb = self.lm.rotary_emb(x, pos)
        for li in range(self.l0, self.read):
            k_, v_ = req_kv[li - self.l0]
            if self.ckpt and self.training and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(self._layer, li, x, pos_emb, k_, v_, mask, use_reentrant=False)
            else:
                x = self._layer(li, x, pos_emb, k_, v_, mask)
        last = (unit_mask.sum(-1).clamp(min=1) - 1)
        h = x.reshape(B, M, T, d).gather(2, last[..., None, None].expand(-1, -1, 1, d))[:, :, 0]
        return self.head(self.norm(h.float()))


def rca_candidates(key, active, M, gold=None, p_swap=0.5, gen=None):
    B, n = key.shape; M = min(M, n)
    k = key.detach().float().masked_fill(~active, float("-inf"))
    idx = k.topk(M, dim=1).indices
    valid = active.gather(1, idx)
    if gold is None or p_swap <= 0:
        return idx, valid
    idx, valid = idx.clone(), valid.clone()
    g_in = torch.zeros_like(active); g_in.scatter_(1, idx, valid)
    miss = gold & active & ~g_in
    if not bool(miss.any()):
        return idx, valid
    draw = torch.rand(miss.shape, generator=gen, device="cpu").to(miss.device) < p_swap
    take = miss & draw
    is_gold = gold.gather(1, idx) & valid
    for b in take.any(1).nonzero().flatten().tolist():
        new = take[b].nonzero().flatten()
        new = new[k[b, new].argsort(descending=True)]
        slots = [j for j in range(M - 1, -1, -1) if not bool(is_gold[b, j])]
        for u, j in zip(new.tolist(), slots):
            idx[b, j] = u; valid[b, j] = True
    return idx, valid


def rca_loss(scores, valid, req_gold, help_gold, teacher, none_logit, kd_w=1.0, kd_tau=0.5):
    B, M, _ = scores.shape
    lg = scores.float().masked_fill(~valid[..., None], float("-inf"))
    out, st = 0.0, {}
    if none_logit is None:
        none_logit = scores.new_zeros(2)
    for ch, pos in ((0, req_gold & valid), (1, (req_gold | help_gold) & valid)):
        z = torch.cat([lg[..., ch], none_logit[ch].float().expand(B, 1)], 1)
        tgt = torch.cat([pos.float(), (~pos.any(1, keepdim=True)).float()], 1)
        tgt = tgt / tgt.sum(1, keepdim=True)
        lp = F.log_softmax(z, 1)
        ce = -(tgt * lp.masked_fill(tgt == 0, 0.0)).sum(1).mean()
        out = out + ce; st[f"rca_ce{ch}"] = ce.item()
    if teacher is not None and kd_w > 0:
        g = teacher.float()
        ok = valid & ~torch.isnan(g) & (g >= 0)
        use = (ok.sum(1) >= 2) & (g.masked_fill(~ok, 0).amax(1) > 0)
        if bool(use.any()):
            t = F.softmax((g / kd_tau).masked_fill(~ok, float("-inf")), 1)
            lp = F.log_softmax(lg[..., 0].masked_fill(~ok, float("-inf")), 1)
            kd = -(t * lp.masked_fill(~ok, 0.0)).sum(1)[use].mean()
            out = out + kd_w * kd; st["rca_kd"] = kd.item()
    st["rca_pos_rate"] = float((req_gold & valid).any(1).float().mean())
    return out, st


def yes_no_vector(lm, tok, lm_head_weight=None):
    W = lm_head_weight if lm_head_weight is not None else lm.embed_tokens.weight
    y, n = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
    return (W[y] - W[n]).detach().float()


def n_params(block):
    return sum(p.numel() for p in block.parameters() if p.requires_grad)


__all__ = ["encode_units", "RCABlock", "rca_candidates", "rca_loss", "yes_no_vector", "PREFIX", "SUFFIX", "n_params"]
