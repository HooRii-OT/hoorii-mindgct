# GCT-0.6B on an x86 CPU server

Must-mention retrieval with the released weights on the CPU: INT8 weights and activations in the linear layers
(per-output-channel int8 weights, per-row dynamic int8 activations, int32 accumulation), fused CPU kernels, 32
close-read candidates, and graphs compiled on the CPU with fp32 encoders. Nothing is quantised on disk: the weights are
converted when the path is installed.

## Requirements

- Linux on x86-64. The measured configuration is a Xeon with AVX-512 VNNI; other x86 CPUs run the same code more slowly.
- The packages of `requirements.txt`, and a C++ compiler with OpenMP: the first `install` compiles a small extension with
  the PyTorch toolchain (about a minute) and reuses it afterwards.
- Memory, as peak resident size of the process: 14 GiB (graphs of a few hundred units) to 17 GiB (24,988 units) for a
  process that compiles graphs and serves requests; 4.8 GiB (1,962 units) to 6.8 GiB (24,988 units) for a process
  that only loads saved graphs (`encoders=False`, below). A serving process grows as it caches the close-read states
  of the units it has read: 8.2 GiB after 400 different requests on the 24,988-unit graph (the cache is capped at
  12 GiB).

## Use

```python
from gct import GCT
from gct.edge import x86

gct = GCT.from_pretrained("HooRiiTechnology/hoorii-mindgct", device="cpu")
x86.install(gct, threads=16)          # once per process, before compiling or loading graphs
graph = gct.compile_graph(world)      # compiled on the CPU
out = gct.retrieve(graph, "Can you book a restaurant for Lin's birthday dinner?", now="2026-03-14 18:30")
```

```bash
taskset -c 0-15 python cpu-x86/quickstart_x86.py --threads 16
```

A first run downloads what is not yet in the Hugging Face cache: the frozen Qwen3-0.6B backbone (about 1.5 GB) and,
from a code-only copy, the Hugging Face repository of GCT (about 2.5 GB). Meanwhile only a "Fetching … files" progress
bar is shown.

## Compile once, serve from saved graphs

A process that only serves requests does not need the two encoders that compilation uses:

```python
gct.save_graph(graph, "graphs/lin")          # graph.json + tensors.safetensors, no pickle

server = GCT.from_pretrained("HooRiiTechnology/hoorii-mindgct", device="cpu", encoders=False)
x86.install(server, threads=16)
graph = server.load_graph("graphs/lin")      # same scores as the graph compiled above
```

```bash
python cpu-x86/quickstart_x86.py --save-graphs graphs      # compile and save the example graphs
python cpu-x86/quickstart_x86.py --from-saved graphs       # serve them without the encoders
```

`gct.save_graph(graph, path)` writes into `path`, which must not exist yet, and stops with `FileExistsError`
otherwise. `--save-graphs DIR` saves each example graph as `DIR/<graph name>` (`DIR/example_graph` and
`DIR/example_graph_zh`), creating `DIR` if needed, and stops in the same way when one of them already exists: choose
another `DIR`, or remove the old graph directories, before running it again.

A saved graph can only be loaded with the same release and weights. Its files contain the graph's content (what the
caller may see): store and transfer them as you store the graph itself.

A graph saved on one device loads on another, but its scores then differ more from those of a graph compiled where it
is served: on our test requests for the case graphs, graphs compiled on an A100 and served by the x86 path returned
the five best units in the same order as graphs compiled with the x86 path for 11 of 72 requests. Compile with the
path you serve with. `save_graph` creates the directory and both files readable by their owner only.

## Notes

`x86.install` replaces the close-read state function for the whole process, so keep instances without the x86 path out
of that process; mixing them raises a `RuntimeError` about mismatched dtypes (float and BFloat16) on the first
request.
Pin the process to the cores of one socket and give it one thread per physical core. Each graph is compiled with its
own caches, so a graph's scores do not depend on the graphs compiled before it. The scores depend on the number of
threads: units with nearly equal scores can swap ranks between thread counts, so compare runs at the same thread count.
The INT8 layers make the scores differ slightly from the GPU release. Measured with its own caches per graph, the
recall@5 of this path matches the GPU release on our main development set, LongMemEval_S and PerLTQA and is 0.019
lower on LoCoMo. The model card's x86 figures (Table 5 and its x86 section) were measured with the deployment
runtime, the latency and memory figures in a serving process that loads compiled graphs.

Latency. On the server of the model card's Table 5 (2 × Xeon Platinum 8358P, one socket of a shared machine), we
timed this path, serving saved graphs, and the deployment runtime behind the card's x86 numbers alternately, three
rounds each, with 8, 16 and 32 cores on graphs of 1,962, 7,976 and 24,988 units. With the environment above (PyTorch
2.15, CPU build), a request on this path took 1.04 times as long as on the deployment runtime at the median of the 27
rounds, and between 0.95 and 1.07 times in the middle half; single rounds varied by up to 30% on that machine. With
PyTorch 2.11 the median was 1.11.
