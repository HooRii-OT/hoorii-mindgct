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
"""x86 CPU path of GCT-0.6B for must-mention retrieval: INT8 (W8A8) linear layers, fused CPU kernels, graphs compiled on
the CPU with fp32 encoders.

    from gct import GCT
    from gct.edge import x86
    gct = GCT.from_pretrained(".", device="cpu")
    x86.install(gct)                    # once, before compiling graphs
    graph = gct.compile_graph(world)    # compiled on the CPU
    out = gct.retrieve(graph, "Can you book a restaurant for Lin's birthday dinner?", now="2026-03-14 18:30")

The INT8 layers make the scores differ slightly from the GPU release; the model card reports the measured recall.
The first install compiles a small C++ extension with the PyTorch toolchain (a C++ compiler with OpenMP is required).
"""
import types

import torch

from .. import graph as _graph, linear1, writeback
from . import closeread, encoders, request, unit_states
from .common import STATE


class _Host:
    """the attributes the write-back and graph installers read"""

    def __init__(self, model):
        self.model, self.info, self.loaded = model, {}, None

    def graph(self, case):
        raise RuntimeError("graphs are compiled with GCT.compile_graph")


def _wrap_provider(prov):
    for name in ("get_units", "build_table", "get"):
        fn = getattr(prov, name, None)
        if fn is None:
            continue

        def wrapped(*a, _fn=fn, **k):
            saved = STATE["on"]
            STATE["on"] = False
            try:
                return _fn(*a, **k)
            finally:
                STATE["on"] = saved
        setattr(prov, name, wrapped)


def install(gct, kind="w8a8", close_read=32, threads=None, request_prefix=True):
    """Install the x86 path on a GCT instance loaded on the CPU.

    kind: "w8a8" (INT8 weights and activations, the measured configuration) or "fp32" (same fused kernels, no INT8).
    close_read: close-read candidates per request (32 = the measured configuration).
    threads: torch intra-op threads (default: leave the current setting)."""
    if str(gct.device) != "cpu":
        raise ValueError("the x86 path runs on the CPU: load GCT with device='cpu'")
    if getattr(gct, "_edge_x86", None) is not None:
        raise RuntimeError("the x86 path is already installed on this instance")
    if kind not in ("w8a8", "fp32"):
        raise ValueError("kind must be 'w8a8' or 'fp32'")
    if threads:
        torch.set_num_threads(int(threads))
    model = gct.model
    info = {"kind": kind, "close_read": int(close_read)}
    info["encoders"] = encoders.install(types.SimpleNamespace(enc=gct.encoder, kenc=gct.key_encoder), fp32=True, sort=True)
    model.float()
    fused = request.install(model, kind, request_prefix)
    info.update(closeread.install_rca(model.rca, kind, last_row=True))
    model.c.rca_m = int(close_read)
    original = model.forward

    def forward(self, *a, _orig=original, **k):
        STATE["on"] = True
        try:
            return _orig(*a, **k)
        finally:
            STATE["on"] = False
    model.forward = types.MethodType(forward, model)
    host = _Host(model)
    tables = writeback.install(host, eager_units=8000)
    gstate = _graph.install(host, tables, enabled=True, native_library=None)
    m1 = linear1.install(model)
    unit_states.install()
    layers = list(model.layers)
    compile_graph = gct.compile_graph

    def compile_graph_x86(world, caller="o", _orig=compile_graph):
        g = _orig(world, caller)
        rcap = g.tensors.get("RCAP")
        if rcap is not None:
            _wrap_provider(rcap[0])
        tables.prepare(g.tensors, layers)
        return g
    gct.compile_graph = compile_graph_x86
    info.update(graph_fused={"P": gstate.get("fused_P"), "H": gstate.get("fused_H")}, single_row_linear=len(m1["modules"]))
    gct._edge_x86 = {"info": info, "request": fused, "tables": tables, "graph": gstate}
    return info
