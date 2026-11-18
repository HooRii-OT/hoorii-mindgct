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
"""Graph write-back token projections from per-graph tables (exact row-wise maps of the compiled graph)."""
import inspect
import textwrap
import types

OLD_K = 'keys = self.Wtk(tokk).reshape(B, K * Tu, -1)'
OLD_V = 'vals = self.Wtv(tokk).reshape(B, K * Tu, -1)'
NEW_K = 'keys = _edge_wb_table(G, self, "Wtk", ui, tokk).reshape(B, K * Tu, -1)'
NEW_V = 'vals = _edge_wb_table(G, self, "Wtv", ui, tokk).reshape(B, K * Tu, -1)'


def row_invariance(rows=4096, out=1024, subset=512, seed=0):
    import torch
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, 1024, generator=g)
    w = torch.randn(out, 1024, generator=g) * .03
    b = torch.randn(out, generator=g) * .1
    idx = torch.randperm(rows, generator=g)[:subset]
    full = torch.nn.functional.linear(x, w, b)
    part = torch.nn.functional.linear(x[idx], w, b)
    return bool(torch.equal(full[idx], part))


class Tables:
    def __init__(self, eager_units=800):
        self.eager_units = eager_units
        self.enabled = True
        self.stats = {'eager_builds': 0, 'lazy_rows': 0, 'build_ms': 0.}

    def table(self, G, layer, name, ui, tokk):
        import time
        import torch
        if not self.enabled and tokk is not None:
            return getattr(layer, name)(tokk)
        memo = G.setdefault('_edge_wb', {})
        key = (id(layer), name)
        lin = getattr(layer, name)
        TOK = G['TOK']
        if key not in memo:
            n = TOK.shape[0]
            started = time.perf_counter()
            if n <= self.eager_units:
                memo[key] = (lin(TOK.float()), None)
                self.stats['eager_builds'] += 1
            else:
                memo[key] = (torch.empty(n, TOK.shape[1], lin.out_features), torch.zeros(n, dtype=torch.bool))
            self.stats['build_ms'] += (time.perf_counter() - started) * 1000
        table, ready = memo[key]
        if ready is not None:
            flat = ui.reshape(-1)
            missing = flat[~ready[flat]].unique()
            if len(missing):
                started = time.perf_counter()
                table[missing] = lin(TOK[missing].float())
                ready[missing] = True
                self.stats['lazy_rows'] += int(len(missing))
                self.stats['build_ms'] += (time.perf_counter() - started) * 1000
        return table[ui]

    def prepare(self, G, layers):
        import torch
        if not self.enabled or G['TOK'].shape[0] > self.eager_units:
            return
        with torch.inference_mode():
            for layer in layers:
                for name in ('Wtk', 'Wtv'):
                    self.table(G, layer, name, torch.zeros(1, 1, dtype=torch.long), None)


def install(runtime, eager_units=800):
    import torch
    model = runtime.model
    layers = list(model.layers)
    cls = type(layers[0])
    source = textwrap.dedent(inspect.getsource(cls.forward))
    assert source.count(OLD_K) == 1 and source.count(OLD_V) == 1, 'write-back source differs from the audited release'
    source = source.replace(OLD_K, NEW_K).replace(OLD_V, NEW_V)
    module = inspect.getmodule(cls)
    tables = Tables(eager_units)
    namespace = dict(vars(module))
    namespace['_edge_wb_table'] = tables.table
    code = compile(source, inspect.getsourcefile(cls) + ':edge_wb', 'exec')
    exec(code, namespace)
    patched = namespace['forward']
    for layer in layers:
        assert type(layer) is cls
        layer.forward = types.MethodType(patched, layer)
    original_graph = runtime.graph

    def graph(case, _orig=original_graph):
        before = runtime.loaded
        saved = _orig(case)
        if saved is not before:
            tables.prepare(saved['G'], layers)
        return saved
    runtime.graph = graph
    runtime.info.update(edge_wb_tables=True, edge_wb_eager_units=eager_units)
    return tables
