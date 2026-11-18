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
"""Frozen unit-key encoder (fine-tuned Qwen3-Embedding-0.6B tower): last-token pooling, L2 normalised."""
import torch
import torch.nn.functional as F

QINSTR = ("Given a user's request to their personal assistant (with the conversation so far), retrieve the records from the user's "
          "personal knowledge base that the assistant needs to handle the request")


class KeyEncoder:
    def __init__(self, path, device="cuda", dtype=torch.bfloat16, max_len=512, q_trunc="left"):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path, padding_side="left")
        self.m = AutoModel.from_pretrained(path, dtype=dtype, attn_implementation="sdpa").to(device).eval()
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.device, self.max_len = device, max_len
        assert q_trunc in ("left", "right"), q_trunc
        self.q_trunc = q_trunc

    def _tail(self, t, n):
        ids = self.tok(t, add_special_tokens=False)["input_ids"]
        return t if len(ids) <= n else self.tok.decode(ids[-n:])

    def _head(self, t, n):
        ids = self.tok(t, add_special_tokens=False)["input_ids"]
        return t if len(ids) <= n else self.tok.decode(ids[:n])

    @torch.no_grad()
    def _run(self, texts, bs):
        out = []
        for i in range(0, len(texts), bs):
            b = self.tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=self.max_len).to(self.device)
            out.append(F.normalize(self.m(**b).last_hidden_state[:, -1].float(), dim=-1))
        return torch.cat(out) if out else torch.zeros(0, self.m.config.hidden_size, device=self.device)

    def encode(self, texts, bs=64):
        self.tok.truncation_side = "right"
        return self._run(list(texts), bs).cpu()

    def encode_query(self, texts, bs=64):
        self.tok.truncation_side = "left"
        cut = self._head if self.q_trunc == "right" else self._tail
        return self._run([f"Instruct: {QINSTR}\nQuery:" + cut(t, self.max_len - 60) for t in texts], bs)
