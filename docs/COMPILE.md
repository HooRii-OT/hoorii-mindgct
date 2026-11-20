# Compiling a graph for GCT

GCT reads a user's graph in compiled form. Compiling turns one view of a graph into the tensors the model reads, with frozen encoders that come with the weights. It does not train anything, and it does not depend on the request: compile a graph once per version and per caller, then send it as many requests as you like.

This page covers what compilation does, when to compile again, how to compile on a GPU and on a CPU, how to save and load compiled graphs, how to compile for a caller with restricted access, what it costs on our hardware, and how to store compiled graphs safely. Everything here uses the code in `gct/` and the released weights. Building a graph from raw material is covered in [Building a graph](BUILD_GRAPH.md).

## What compilation does

`gct.compile_graph(world, caller="o")` takes a graph in the HooRii Mind format (a JSON object with `entities`, `facts`, `edges` and `meta`) and the code of a caller, and returns a `CompiledGraph`. It runs in three steps.

**1. The caller's view.** Facts and edges whose `vis` list does not include the caller are left out, and so are edges in the state `old`. Entities have no `vis` field: every entity enters the view with its name, aliases and description. Facts in the state `superseded` stay in the view and are marked as an old value. To compile for a caller with restricted access, remove what that caller may not see first; see [Views for other callers](#views-for-other-callers).

**2. Units and links.** Every entity and every fact in the view becomes one unit, a short line of text in the language of the graph. With the example graph in `examples/example_graph.json`:

```text
Lin Park (Person)
Lin Park | Health · Allergies: Lin is allergic to peanuts and carries an EpiPen. (confirmed)
Lin's birthday dinner | Notes · General: Lin's birthday dinner is planned for Saturday March 21 at 19:00; the restaurant is not booked yet. (pending confirmation)
```

A fact unit names its subject, the label of its module and its state. Edges between entities become typed links in both directions, and every fact is linked to its subject and to the entities it mentions.

**3. Request-independent tensors.** The frozen encoders encode every unit once:

- a vector per unit from the first 14 layers of the frozen Qwen3-0.6B backbone (mean over the first 96 tokens of the unit);
- a key per unit from the unit-key encoder in `key_encoder/` (fine-tuned from Qwen3-Embedding-0.6B);
- the states of the first 16 tokens of each unit, which the graph layers read;
- vectors for the labels of the relations and modules that occur in the view;
- lexical tables (the words and dates of each unit, hashed and weighted by how rare they are in the view), the time table (when each fact applies, from the dates in its text, `observed_at`, `valid_from` and `valid_to`, with relative dates anchored at `meta.now`), and the link structure.

One part is left for request time. For the close reading of the best candidates, GCT needs the states of each candidate's tokens in the frozen backbone. They are computed from the unit text the first time a unit becomes a candidate and kept in memory for later requests. They are not saved with the graph, and a process that serves many requests on a large graph keeps more of them in memory.

Every graph is encoded with its own encoder caches, and its time table is cached by the content of the graph, so its tensors do not depend on graphs compiled before in the same process.

## When to compile again

Compile the graph again when:

- anything in the graph changes: an entity, a fact, an edge, a `vis` list or a state. This release always compiles a whole view; it has no incremental compilation;
- you serve a different caller (one compiled graph per caller);
- `meta.now` of the graph changes, because the time table anchors relative dates in fact values at it;
- you move to a different release of the weights. `load_graph` refuses a graph compiled for other weights (see [Common problems](#common-problems)).

You do not need to compile again for new requests. The request text, the earlier turns, the request time (`now`) and `previous_unit_ids` are all read at request time.

A changed date or `meta.now` shows in the time table of the compiled view. To look at it, for checking only (`view.payload` is an internal field and can change between releases):

```python
import datetime, json
from gct import GCT

gct = GCT.from_pretrained("HooRiiTechnology/hoorii-mindgct")
world = json.load(open("examples/example_graph.json", encoding="utf-8"))
graph = gct.compile_graph(world, caller="o")
rows = dict(zip(graph.view.unit_ids, graph.view.payload["t_iv"]))   # one row per unit, in the order of view.unit_ids
print(rows["f11"], datetime.date.fromordinal(rows["f11"][0]))
```

```text
[739696, 739696, 739685, 14] 2026-03-21
```

Each row holds four numbers: the first and the last day the unit refers to, as day numbers (`datetime.date.fromordinal`), or `-1` and `-2` when it has no date; the day the fact was observed (`observed_at`), or `-1`; and the index of the group of units it is compared with in time (for a fact, mostly the facts with the same subject and module). Fact `f11` of the example graph, Lin's birthday dinner on Saturday March 21, was observed on March 10.

## Compile on a GPU or a CPU

The same call compiles on any device; the device of the `GCT` instance decides where the encoders run.

```python
import json
from gct import GCT

gct = GCT.from_pretrained("HooRiiTechnology/hoorii-mindgct")                  # CUDA when available
world = json.load(open("examples/example_graph.json", encoding="utf-8"))
graph = gct.compile_graph(world, caller="o")
print(len(graph), "units")

cpu = GCT.from_pretrained("HooRiiTechnology/hoorii-mindgct", device="cpu")    # the reference implementation on a CPU
graph = cpu.compile_graph(world, caller="o")
```

On an x86 CPU server, install the accelerated path before compiling. Graphs are then compiled on the CPU with fp32 encoders, and requests use INT8 linear layers. `x86.install` changes how the whole process computes close-read states, so run it in a process of its own and do not use instances without the x86 path in that process:

```python
import json
from gct import GCT
from gct.edge import x86

cpu = GCT.from_pretrained("HooRiiTechnology/hoorii-mindgct", device="cpu")
x86.install(cpu, threads=16)          # once per process, before compiling or loading graphs
world = json.load(open("examples/example_graph.json", encoding="utf-8"))
graph = cpu.compile_graph(world, caller="o")
print(len(graph), "units")
```

On a CPU the scores depend on the number of threads, and units with nearly equal scores can swap ranks between thread counts and between a GPU and a CPU. On a GPU, two runs of the same requests can also differ slightly, because some GPU kernels do not add numbers in a fixed order: over the 27 graphs of the example cases, with our test requests (24 per case, 648 in all), two runs on the same A100 returned the same five best units for every request and in the same order for 644 of them, while single scores moved by up to 0.56. Between different devices a few requests can differ by more than a near-tie: a unit that one device leaves inactive can be activated on another and come among the five best. With our test requests on the 27 graphs of the example cases, this happened in 8 of 648 requests between an A100 and an RTX 5090, and in 3 of the 72 requests on the three graphs built from raw material between an A100 and the reference implementation on a Xeon CPU. Compare runs on the same hardware with the same number of threads, and compare the units returned and their order rather than the last digits of the scores.

## Save and load

A compiled graph can be written to disk and loaded by another process, which then needs neither the two encoders nor the original graph:

```python
graph = gct.compile_graph(world, caller="o")
gct.save_graph(graph, "graphs/example")      # a new directory: graph.json + tensors.safetensors, no pickle

server = GCT.from_pretrained("HooRiiTechnology/hoorii-mindgct", encoders=False)
graph = server.load_graph("graphs/example")  # same scores as the graph compiled above
out = server.retrieve(graph, "Can you book a restaurant for Lin's birthday dinner?", now="2026-03-14 18:30")
print(out["results"][0]["id"], out["results"][0]["display_text"])
```

- `graph.json` holds the format version, the identity of the weights the graph was compiled for, and the caller's view: unit ids, unit texts, links and the per-fact fields. `tensors.safetensors` holds the tensors.
- `save_graph` writes into a directory that must not exist yet and stops with `FileExistsError` otherwise.
- `load_graph` checks that the graph was compiled for the same release, the same weights and key encoder, the same backbone revision and the same unit rendering, and refuses it otherwise.
- An instance loaded with `encoders=False` cannot compile (`compile_graph` raises an error) but serves loaded graphs with less memory.
- A loaded graph gives the same results as the compiled one on the same machine, given the same requests in the same order, up to the run-to-run variation described above. Of our 72 test requests on the three case graphs built from raw material, all 72 gave identical results with the reference implementation on a CPU, 71 with the x86 path and 69 on the A100.
- `save_graph` creates the directory and both files readable by their owner only, because they hold the content of the view (see [Protecting compiled graphs](#protecting-compiled-graphs)). Give other accounts access deliberately, for example the account of a serving process.

## Views for other callers

A request carries the caller of the compiled graph in its header (`o` the user, `f` a family assistant, `w` a work assistant, `d` a driver assistant), and the view decides what GCT can find. Because entities are not filtered by `vis`, prepare the graph for a restricted caller before compiling it: drop the entities that caller may not see, the facts about or mentioning them, the edges that touch them, and, if the view should hold no history, the superseded facts.

```python
import copy

def restrict(world, caller, hidden_entities=(), keep_history=True):
    """the part of a graph that a restricted caller may see, ready for compile_graph"""
    hidden = set(hidden_entities)
    w = copy.deepcopy(world)
    w["entities"] = [e for e in w["entities"] if e["id"] not in hidden]
    w["edges"] = [e for e in w["edges"] if caller in e["vis"] and not {e["src"], e["dst"]} & hidden]
    w["facts"] = [f for f in w["facts"] if caller in f["vis"] and f["subject"] not in hidden
                  and not hidden & set(f.get("mentions") or ()) and (keep_history or f["state"] != "superseded")]
    return w
```

`mentions` only lists the entities the graph builder linked; a fact can still name a person in its text. Review the facts of a restricted view, or build the graph so that every named entity is in `mentions`.

An example with the example graph: share Sam's job and his team meeting with a work assistant, and keep his sister, his dog and his friend's birthday dinner out of that view.

```python
world = json.load(open("examples/example_graph.json", encoding="utf-8"))
for f in world["facts"]:
    if f["id"] in ("f8", "f9"):                  # Sam's role and his team meeting
        f["vis"] = ["o", "w"]
for e in world["edges"]:
    if e["rel"] == "works_at":
        e["vis"] = ["o", "w"]

print(gct.compile_graph(world, caller="w").unit_ids)
print(gct.compile_graph(restrict(world, "w", hidden_entities=["e2", "e4", "e5"]), caller="w").unit_ids)
```

```text
['e0', 'e1', 'e2', 'e3', 'e4', 'e5', 'f8', 'f9']
['e0', 'e1', 'e3', 'f8', 'f9']
```

Filtered by `vis` alone, the work assistant's view still holds Maya Carter (`e2`), Biscuit (`e4`) and Lin's birthday dinner (`e5`) as entity units. Prepared with `restrict`, it holds Sam, Lin, Northwind Labs and the two shared facts.

## Measured cost

We compiled the two example graphs and the three graphs of the example cases in `examples/cases/` built from the cases' raw material with GPT-6.1 Sol (`gpt-6.1-sol`) for the owner's view, `caller="o"`, once each:

| Graph | Entities | Facts | Edges | Units | GPU | CPU, reference implementation | CPU, x86 path | Saved size |
|---|---|---|---|---|---|---|---|---|
| `examples/example_graph.json` | 6 | 13 | 7 | 18 | 0.23 s | 4.1 s | 1.3 s | 2.3 MiB |
| `examples/example_graph_zh.json` | 4 | 7 | 3 | 11 | 0.13 s | 1.8 s | 0.8 s | 2.0 MiB |
| `assistant_en` case | 263 | 496 | 391 | 759 | 1.5 s | 158 s | 39 s | 33.5 MiB |
| `chat_en` case | 308 | 773 | 525 | 1,081 | 2.0 s | 239 s | 52 s | 46.9 MiB |
| `family_zh` case | 257 | 537 | 425 | 794 | 1.6 s | 158 s | 38 s | 35.2 MiB |

- **GPU**: one NVIDIA A100-SXM4-80GB in a server whose other GPUs ran other jobs; PyTorch 2.11 with CUDA 12.9, transformers 5.13.1. Compiling a case graph took 4.0 to 4.4 GiB of GPU memory at its peak, model included.
- **CPU**: 16 cores of one socket of a 2 × Xeon Platinum 8380 server, one thread per core, with other jobs running on the machine; PyTorch 2.11, transformers 5.13.1. The x86 path compiles with fp32 encoders and its own fused kernels, and is about four times faster than the reference implementation on the case graphs.
- **Saved size** is the size of `graph.json` plus `tensors.safetensors`, the same on every device. It grows with the number of units, about 45 KiB per unit on the case graphs; `graph.json`, the view as text, is 0.3 to 0.4 MB of it. Loading a saved case graph took 0.1 to 0.2 s.
- **The first requests** on a newly compiled or loaded graph also compute the close-read states of their candidates. On the case graphs the first pass over our 24 test requests of a case averaged 115 to 128 ms per request on the GPU against a median of 94 to 103 ms in a second pass; 404 to 465 ms against 240 to 288 ms with the x86 path; and 3.9 to 4.3 s against 2.3 to 2.6 s with the reference implementation on the CPU.

On a GPU, compiling is cheap enough to do whenever the graph changes. On a CPU, use the x86 path and compile in the background, while the process keeps serving the previous version of the graph.

A graph saved on one device loads on another, as long as the release and weights are the same, but its scores then differ more from those of a graph compiled where it is served. We loaded the three case graphs compiled on the A100 into CPU processes and sent our 72 test requests: with the reference implementation, the five best units came out in the same order as with the graphs compiled on that CPU for 46 of the 72 requests; with the x86 path, which compiles with fp32 encoders, for 11 of them. To get the scores the x86 path was measured with, compile with the x86 path.

## Protecting compiled graphs

A compiled graph carries the content of the view it was compiled from. `graph.json` contains the unit texts as plain text, and the tensors are computed from those texts by encoders that anyone can download with the weights; with them, the texts can be recovered from the tensors. Compiling does not anonymise a graph and cannot be treated as one-way.

- Store and transfer compiled graphs with the same protection as the graph itself, and delete them when the graph is deleted.
- Compile one view per caller and give each process only the views it serves. A view compiled for the user's own assistant holds everything that assistant may see.
- Anyone who holds the files of a view can read that view. To keep a graph on hardware its owner controls, compile and serve it there; GCT-0.6B runs on a CPU for that purpose.

## Common problems

- **`ValueError: ... was compiled for other weights or settings (differs in [...]): compile it again`.** The graph was compiled with another release, other weights, another backbone revision or another unit rendering. Compile the original graph again with the current weights.
- **`FileExistsError` from `save_graph`.** The target directory exists. Choose a new directory, or remove the old one.
- **`RuntimeError: this instance was loaded with encoders=False ...`.** Load GCT without `encoders=False` to compile; keep `encoders=False` for processes that only serve saved graphs.
- **A view with more than 2,048 units.** The model card lists views of up to 2,048 units as a limit of the released model. Keep each view within that size.
- **An edited graph compiled again in the same process.** Earlier versions of `gct/timex.py` cached the time table of a graph under its `world_id` and its numbers of entities and facts, so a graph edited without changing those numbers (a corrected date, a moved `meta.now`) and compiled again in the same process kept the time table of the earlier version. This release caches the time table by the content of the graph, so the new compile is up to date.
- **Different scores on another machine.** See the note on threads and hardware under [Compile on a GPU or a CPU](#compile-on-a-gpu-or-a-cpu). Compare the units returned and their order, not the last digits of the scores.
