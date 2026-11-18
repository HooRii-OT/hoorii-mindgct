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
"""Quickstart on an x86 CPU server: the same example graphs and requests as examples/quickstart.py, served by the x86 path
(INT8 linear layers, 32 close-read candidates, graphs compiled on the CPU).

    python cpu-x86/quickstart_x86.py                       # from the repository root
    python cpu-x86/quickstart_x86.py --threads 16          # pin the process to one socket for stable latency, e.g.
    taskset -c 0-15 python cpu-x86/quickstart_x86.py --threads 16

    python cpu-x86/quickstart_x86.py --save-graphs graphs  # also write each compiled graph with gct.save_graph
    python cpu-x86/quickstart_x86.py --from-saved graphs   # serve from the saved graphs, without the two encoders
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPO_ID = "HooRiiTechnology/hoorii-mindgct"
DEFAULT_MODEL = str(ROOT) if (ROOT / "config.json").is_file() else REPO_ID   # a code-only clone downloads the weights

import torch  # noqa: E402

from gct import GCT  # noqa: E402
from gct.edge import x86  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"the model repository, local path or Hugging Face id (default: this directory if it holds config.json, else {REPO_ID})")
    ap.add_argument("--base-model", default=None, help="local copy of the pinned Qwen3-0.6B revision (default: download it)")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--requests", default=str(ROOT / "examples" / "example_requests.json"))
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--save-graphs", default=None, help="directory: save each compiled graph there (gct.save_graph)")
    ap.add_argument("--from-saved", default=None, help="directory written by --save-graphs: load the graphs instead of compiling")
    o = ap.parse_args()
    t0 = time.perf_counter()
    gct = GCT.from_pretrained(o.model, base_model=o.base_model, device="cpu", encoders=o.from_saved is None)
    x86.install(gct, threads=o.threads)
    print(f"loaded GCT-0.6B with the x86 path in {time.perf_counter() - t0:.1f} s ({torch.get_num_threads()} threads)")
    graphs = {}
    for r in json.loads(Path(o.requests).read_text(encoding="utf-8")):
        name = r["graph"]
        if name not in graphs:
            t1 = time.perf_counter()
            if o.from_saved:
                graphs[name] = gct.load_graph(Path(o.from_saved) / Path(name).stem)
                print(f"\nloaded {name}: {len(graphs[name])} units in {time.perf_counter() - t1:.1f} s")
            else:
                world = json.loads((Path(o.requests).parent / name).read_text(encoding="utf-8"))
                graphs[name] = gct.compile_graph(world, caller="o")
                print(f"\ncompiled {name} on the CPU: {len(graphs[name])} units in {time.perf_counter() - t1:.1f} s")
                if o.save_graphs:
                    print(f"saved to {gct.save_graph(graphs[name], Path(o.save_graphs) / Path(name).stem)}")
        t1 = time.perf_counter()
        out = gct.retrieve(graphs[name], r["text"], context=r.get("context", []), now=r.get("now"), top_k=o.top_k,
                           positive_only=False)
        ms = 1000 * (time.perf_counter() - t1)
        print(f"\nrequest ({ms:.0f} ms): {r['text']}")
        for m in r.get("context", []):
            print(f"  earlier ({m['role']}): {m['content']}")
        for x in out["results"]:
            mark = "must mention" if x["score"] > 0 else "            "
            print(f"  {x['score']:+8.3f}  {mark}  {x['id']:>4}  {x['text']}")


if __name__ == "__main__":
    main()
