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
"""Quickstart: compile the example graphs and print the top units of a few requests with their must-mention scores
(score > 0: must mention; the scores are logits, not calibrated probabilities).

    pip install -r requirements.txt
    python examples/quickstart.py                       # from the repository root; downloads Qwen/Qwen3-0.6B once
    python examples/quickstart.py --base-model DIR      # use a local copy of the pinned Qwen3-0.6B revision
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

from gct import GCT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"the model repository, local path or Hugging Face id (default: this directory if it holds config.json, else {REPO_ID})")
    ap.add_argument("--base-model", default=None, help="local copy of the frozen Qwen3-0.6B; default: download the pinned revision")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda when available)")
    ap.add_argument("--requests", default=str(ROOT / "examples" / "example_requests.json"))
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--positive-only", action="store_true", help="list only units with a positive score (must mention)")
    o = ap.parse_args()

    t0 = time.perf_counter()
    gct = GCT.from_pretrained(o.model, base_model=o.base_model, device=o.device)
    print(f"loaded GCT-0.6B on {gct.device} in {time.perf_counter() - t0:.1f} s")
    graphs = {}
    for r in json.loads(Path(o.requests).read_text(encoding="utf-8")):
        name = r["graph"]
        if name not in graphs:
            world = json.loads((Path(o.requests).parent / name).read_text(encoding="utf-8"))
            graphs[name] = gct.compile_graph(world, caller="o")
            print(f"\ncompiled {name}: {len(graphs[name])} units visible to the caller")
        out = gct.retrieve(graphs[name], r["text"], context=r.get("context", []), now=r.get("now"), top_k=o.top_k,
                           positive_only=o.positive_only)
        print(f"\nrequest: {r['text']}")
        for m in r.get("context", []):
            print(f"  earlier ({m['role']}): {m['content']}")
        if not out["results"]:
            print("  (no unit)")
        for x in out["results"]:
            mark = "must mention" if x["score"] > 0 else "            "
            print(f"  {x['score']:+8.3f}  {mark}  {x['id']:>4}  {x['text']}")


if __name__ == "__main__":
    main()
