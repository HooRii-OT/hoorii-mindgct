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
"""Load GCT-0.6B, compile a graph view and retrieve the must-mention units of a request.

    from gct import GCT
    gct = GCT.from_pretrained("path/to/this/repo")            # or a Hugging Face repo id
    graph = gct.compile_graph(world)                          # world: one graph (gct_world_v1), see examples/
    out = gct.retrieve(graph, "Book a table for Lin and me on Friday.", now="2026-03-14 18:30")

    gct.save_graph(graph, "graphs/example")                   # safetensors + JSON, no pickle
    server = GCT.from_pretrained("path/to/this/repo", encoders=False)   # retrieval only: without the two encoders
    graph = server.load_graph("graphs/example")               # same scores as the graph compiled above

A compiled graph holds tensors computed from the unit texts by the frozen encoders. Together with the public weights they
carry the content of the graph: treat a compiled graph with the same care as the graph itself.
"""
import json
import os
from dataclasses import fields
from pathlib import Path

import torch

from . import compile as _compile
from . import data as _data
from . import embed as _embed
from . import lex as _lex
from . import model as _model
from .keyenc import KeyEncoder

ACTIVE_FLOOR = -29.9          # units below this logit were never activated by the graph layers and are never returned


def _resolve(path_or_repo, revision=None, what="model"):
    p = Path(str(path_or_repo)).expanduser()
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download
    try:
        return Path(snapshot_download(str(path_or_repo), revision=revision))
    except Exception as e:
        hint = (" Download it yourself (any way you like) and pass the local directory with base_model=DIR"
                " (quickstart: --base-model DIR); every file must be identical to that revision.") if what == "base" else ""
        raise RuntimeError(f"could not download {path_or_repo}" + (f" at revision {revision}" if revision else "")
                           + f": {type(e).__name__}: {e}.{hint}") from e


class CompiledGraph:
    """One caller's view of a graph, compiled for this model: unit ids and texts plus the request-independent tensors."""

    def __init__(self, world, view, tensors):
        self.world, self.view, self.tensors = world, view, tensors

    @property
    def unit_ids(self):
        return list(self.view.unit_ids)

    def __len__(self):
        return self.view.n


GRAPH_FORMAT = "gct-compiled-graph/1"
_RUNTIME_ONLY = ("RCAP", "prev")       # re-created when a graph is loaded / set per request; keys starting with "_" are caches


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


class GCT:
    def __init__(self, model_dir, base_model_dir, device=None, close_read=None, encoders=True):
        model_dir = Path(model_dir)
        cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        self.config = cfg
        self.model_dir, self._digests = model_dir, {}
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        r = cfg["rendering"]
        # rendering of units and requests (module settings shared by the compiler, the lexical channel and the encoders)
        _compile.CTX_RECENT_FIRST = r["context_recent_first"]
        _compile.CTX_RANDOM_WINDOWS = None
        _compile.CTX_MAX_MSGS = None if r["context_max_messages"] is None or r["context_max_messages"] < 0 else r["context_max_messages"]
        _compile.TIME_IV = r["unit_time_intervals"]
        _compile.VAL_AT = False
        _compile.GRAPH_EDGES, _compile.REWIRE_SEED = "all", 0
        _lex.REQ_CLEAN, _lex.LEX_W, _lex.DAYS, _lex.DATES = r["lex_request_clean"], r["lex_words"], r["lex_day_tokens"], r["lex_date_tokens"]
        _lex.LEX_V = 1 << r["lex_bits"]
        _embed.EMB_MAXLEN = r["unit_embedding_tokens"]
        self.subject_name = r["subject_name"]
        base = str(base_model_dir)
        _embed.QWEN = _model.QWEN = base
        tok_dir = str(model_dir / "tokenizer") if (model_dir / "tokenizer" / "tokenizer.json").is_file() else base
        _embed.TOKENIZER = _model.TOKENIZER = tok_dir

        # the two encoders are used only to compile graphs; encoders=False leaves them out (the instance loads compiled graphs)
        self.encoder = _embed.TextEncoder(device) if encoders else None    # unit encoder: first layers of the frozen backbone
        self.key_encoder = (KeyEncoder(str(model_dir / cfg["key_encoder"]["path"]), device, q_trunc=cfg["key_encoder"]["request_truncation"])
                            if encoders else None)
        self.unit_memory_tokens = r["unit_memory_tokens"]

        arch = dict(cfg["architecture"])
        for k in ("graph_after", "hop_kmax"):
            if arch.get(k) is not None:
                arch[k] = tuple(arch[k])
        if arch.get("hop_train_k") is not None:
            arch["hop_train_k"] = tuple(tuple(x) for x in arch["hop_train_k"])
        if close_read is not None:
            arch["rca_m"] = int(close_read)
        known = {f.name for f in fields(_model.Cfg)}
        unknown = set(arch) - known
        if unknown:
            raise ValueError(f"config.json has unknown architecture fields: {sorted(unknown)}")
        c = _model.Cfg(**arch)
        from safetensors.torch import load_file
        sd = load_file(str(model_dir / cfg["weights"]), device="cpu")
        if "sit_E" in sd:
            c.n_sit = sd["sit_E"].shape[0]
        self.model = _model.GraphFusedModel(c, device)
        res = self.model.load_state_dict(sd, strict=False)
        trainable = {n for n, p in self.model.named_parameters() if p.requires_grad}
        missing = sorted(set(res.missing_keys) & trainable)
        if missing or res.unexpected_keys:
            raise RuntimeError(f"weights do not match the model: missing {missing[:5]}, unexpected {res.unexpected_keys[:5]}")
        self.model.__dict__["_kenc"] = self.key_encoder
        self.cfg = c
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        # module keys of the schema layer: the same texts for every graph, encoded once, alone (saved with every compiled graph)
        self.module_E = _embed.EmbCache(None, self.encoder).get("modules", _data.module_texts()).to(device) if encoders else None
        self.request_token_limit = cfg["inference"]["request_token_limit"]

    @classmethod
    def from_pretrained(cls, path_or_repo=".", revision=None, base_model=None, device=None, close_read=None, encoders=True):
        """path_or_repo: a local copy of this repository or its Hugging Face id. base_model: a local copy of the frozen
        Qwen3-0.6B (every file identical to the pinned upstream revision); default: download that revision.
        encoders=False: do not load the unit and key encoders, which only compile_graph uses; such an instance retrieves
        over graphs written by save_graph (load_graph) and needs less memory."""
        model_dir = _resolve(path_or_repo, revision)
        cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        b = cfg["base_model"]
        if base_model and not Path(str(base_model)).expanduser().is_dir():
            raise FileNotFoundError(f"base_model={base_model!r} is not a directory")
        base_dir = _resolve(base_model, None) if base_model else _resolve(b["repo_id"], b["revision"], what="base")
        return cls(model_dir, base_dir, device=device, close_read=close_read, encoders=encoders)

    def _identity(self):
        """what a compiled graph depends on: the release, its weights, the frozen backbone revision and the unit rendering"""
        cfg = self.config
        for name, rel in (("weights_sha256", cfg["weights"]), ("key_encoder_sha256", Path(cfg["key_encoder"]["path"]) / "model.safetensors")):
            if name not in self._digests:
                self._digests[name] = _sha256(self.model_dir / rel)
        return {"release": cfg["release"], **self._digests, "base_model_revision": cfg["base_model"]["revision"],
                "rendering": cfg["rendering"]}

    # ------------------------------------------------------------------------------------------------ graph
    @torch.inference_mode()
    def compile_graph(self, world, caller="o"):
        """world: one graph in the gct_world_v1 format (entities, facts, edges, meta.owner, meta.now; `vis` lists the callers
        that may see an item). Only what `caller` may see is compiled.

        Every graph is encoded with its own caches, so its tensors and scores do not depend on which graphs were compiled
        before in the same process."""
        if self.encoder is None:
            raise RuntimeError("this instance was loaded with encoders=False: it loads compiled graphs (load_graph) and cannot compile them")
        view = _compile.compile_view(world, caller, include_subject_name=self.subject_name)
        unit_cache = _embed.EmbCache(None, self.encoder)              # in memory, per graph (no files written)
        rel_cache = _embed.EmbCache(None, self.encoder)
        key_cache = _embed.EmbCache(None, self.key_encoder)
        tokens = _embed.TokenProvider(self.encoder, T_u=self.unit_memory_tokens)
        c = self.cfg
        tokens.rcaprov = _embed.RCAStateProvider(self.model.lm, self.model.tok, c.rca_l0, c.rca_t, c.rca_suffix, seg=c.rca_seg)
        tokens.rcaprov.lazy = True                                    # close-read states of a unit are encoded on first use
        unit_cache.get(view.world_id, view.unit_text)
        rel_cache.get("relations", view.rel_text)
        rel_cache.get("modules_all", view.mod_text)
        key_cache.get(view.world_id, view.unit_text)
        tokens.get(view)
        _data.KEY_CACHE = key_cache
        try:
            G = _data.graph_tensors(view, unit_cache, rel_cache, self.device, tokens)
        finally:
            _data.KEY_CACHE = None
        G.pop("_memo", None)
        G.pop("_cg", None)
        G["module_E"] = self.module_E
        return CompiledGraph(world, view, G)

    def _close_read_provider(self, view):
        c = self.cfg
        prov = _embed.RCAStateProvider(self.model.lm, self.model.tok, c.rca_l0, c.rca_t, c.rca_suffix, seg=c.rca_seg)
        prov.lazy = True                                              # as after compile_graph: states encoded on first use
        return prov, view

    def save_graph(self, graph, path):
        """Write a compiled graph to the new directory `path`: graph.json (format, the weights it was compiled for, the
        caller's view and the request-header fields of the world) and tensors.safetensors. No pickle is used.
        The files contain the graph's content (what the caller may see): store them as you store the graph itself."""
        from dataclasses import asdict
        from safetensors.torch import save_file
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        tensors, values = {}, {}
        for k, v in graph.tensors.items():
            if k in _RUNTIME_ONLY or k.startswith("_"):
                continue
            if torch.is_tensor(v):
                tensors[k] = v.detach().to("cpu").contiguous().clone()
            elif v is None or isinstance(v, (bool, int, float, str, list)):
                values[k] = v
            else:
                raise TypeError(f"graph entry {k!r} of type {type(v).__name__} cannot be saved")
        view = asdict(graph.view)
        payload_tensors = {}                                          # compile-time tensors kept in the view (lexical tables)
        for k, v in list(view["payload"].items()):
            if torch.is_tensor(v) or (isinstance(v, (tuple, list)) and v and all(torch.is_tensor(x) for x in v)):
                parts = [v] if torch.is_tensor(v) else list(v)
                for i, x in enumerate(parts):
                    tensors[f"payload/{k}/{i}"] = x.detach().to("cpu").contiguous().clone()
                payload_tensors[k] = None if torch.is_tensor(v) else len(parts)
                del view["payload"][k]
        w = graph.world
        world = {k: w[k] for k in ("world_id", "lang") if k in w}
        world["meta"] = dict(w.get("meta", {}))
        if (w.get("seed") or {}).get("now"):
            world["seed"] = {"now": w["seed"]["now"]}
        meta = {"format": GRAPH_FORMAT, "model": self._identity(), "world": world, "view": view, "payload_tensors": payload_tensors,
                "values": values, "tensors": sorted(tensors)}
        (path / "graph.json").write_text(json.dumps(meta, ensure_ascii=False) + "\n", encoding="utf-8")
        save_file(tensors, str(path / "tensors.safetensors"), metadata={"format": GRAPH_FORMAT})
        return path

    def load_graph(self, path):
        """Load a graph written by save_graph for the same release and weights. Scores are identical to those of the
        graph returned by compile_graph in the compiling process, given the same requests in the same order."""
        from safetensors.torch import load_file
        path = Path(path)
        meta = json.loads((path / "graph.json").read_text(encoding="utf-8"))
        if meta.get("format") != GRAPH_FORMAT:
            raise ValueError(f"{path} is not a compiled graph of format {GRAPH_FORMAT}")
        mine, theirs = self._identity(), meta["model"]
        diff = sorted(k for k in set(mine) | set(theirs) if mine.get(k) != theirs.get(k))
        if diff:
            raise ValueError(f"{path} was compiled for other weights or settings (differs in {diff}): compile it again")
        tensors = load_file(str(path / "tensors.safetensors"), device=str(self.device))
        if sorted(tensors) != meta["tensors"]:
            raise ValueError(f"{path}: tensors.safetensors does not match graph.json")
        for k, n in meta["payload_tensors"].items():
            parts = [tensors.pop(f"payload/{k}/{i}").cpu() for i in range(1 if n is None else n)]
            meta["view"]["payload"][k] = parts[0] if n is None else tuple(parts)
        view = _compile.CompiledGraph(**meta["view"])
        G = {**tensors, **meta["values"]}
        G["RCAP"] = self._close_read_provider(view)
        return CompiledGraph(meta["world"], view, G)

    # ------------------------------------------------------------------------------------------------ request
    def request_text(self, graph, text, context=(), now=None, caller=None):
        """the model input for one request: header with the reference time, the current turn, earlier turns newest first"""
        world = graph.world
        if now is not None:
            world = {**world, "meta": {**world.get("meta", {}), "now": now}}
        ctx = [["u" if m["role"] == "user" else "a", m["content"]] for m in (context or [])]
        q = {"caller": caller or graph.view.caller, "id": "request", "text": text, "ctx": ctx}
        return _compile.query_text(world, q)

    @torch.inference_mode()
    def score(self, graph, text, context=(), now=None, previous_unit_ids=()):
        """-> (model input text, must-mention logits (n,) over the view's units, raw model outputs)"""
        q = self.request_text(graph, text, context, now)
        G, view = graph.tensors, graph.view
        G["prev"] = torch.zeros(1, view.n, device=self.device)
        prev = [view.index[u] for u in (previous_unit_ids or []) if u in view.index]
        if prev:
            G["prev"][0, torch.tensor(prev, device=self.device)] = 1.0
        out = self.model([q], G, None)
        lg = out["unit"][0, :, 0].float()
        ro = view.payload.get("route_only")
        if ro:
            lg[ro] = -1e4
        return q, lg, out

    def retrieve(self, graph, text, context=(), now=None, previous_unit_ids=(), top_k=5, positive_only=True):
        """must-mention units of one request, best first.

        text: the current turn. context: earlier turns [{"role": "user"|"assistant", "content": str}], oldest first.
        now: reference time of the request (recommended "YYYY-MM-DD HH:MM" in the graph's local time); default meta.now.
        previous_unit_ids: units already handed to the assistant earlier in this conversation and still in force.
        A positive score means "must mention". positive_only (default): return only those, at most top_k, best first;
        positive_only=False: the top_k units by score whatever their sign. Scores are logits, not calibrated probabilities."""
        q, lg, out = self.score(graph, text, context, now, previous_unit_ids)
        view = graph.view
        valid = lg > ACTIVE_FLOOR
        if positive_only:
            valid &= lg > 0
        k = min(int(top_k), int(valid.sum()))
        idx = lg.masked_fill(~valid, float("-inf")).topk(k).indices.tolist() if k > 0 else []
        n_tok = len(self.model.tok(q, add_special_tokens=True)["input_ids"])
        results = [{"id": view.unit_ids[i], "type": "fact" if view.unit_type[i] else "entity", "text": view.unit_text[i],
                    "score": round(lg[i].item(), 4)} for i in idx]
        return {"results": results, "input_tokens": n_tok, "request_truncated": n_tok > self.request_token_limit}
