# GCT-0.6B on an x86 CPU server

Must-mention retrieval with the released weights on the CPU: INT8 weights and activations in the linear layers
(per-output-channel int8 weights, per-row dynamic int8 activations, int32 accumulation), fused CPU kernels, 32
close-read candidates, and graphs compiled on the CPU with fp32 encoders. Nothing is quantised on disk: the weights are
converted when the path is installed.

## Requirements

- Linux on x86-64. The measured configuration is a Xeon with AVX-512 VNNI; other x86 CPUs run the same code more slowly.
- The packages of `requirements.txt`, and a C++ compiler with OpenMP: the first `install` compiles a small extension with
  the PyTorch toolchain (about a minute) and reuses it afterwards.
- About 9 GB of memory for the quickstart. Compiling and serving large graphs in one process needs more.

## Use

```python
from gct import GCT
from gct.edge import x86

gct = GCT.from_pretrained("HooRiiTechnology/hoorii-mindgct", device="cpu")
x86.install(gct, threads=16)          # once per process, before compiling graphs
graph = gct.compile_graph(world)      # compiled on the CPU
out = gct.retrieve(graph, "Can you book a restaurant for Lin's birthday dinner?", now="2026-03-14 18:30")
```

```bash
taskset -c 0-15 python cpu-x86/quickstart_x86.py --threads 16
```

Pin the process to the cores of one socket and give it one thread per physical core. Each graph is compiled with its
own caches, so a graph's scores do not depend on the graphs compiled before it. The INT8 layers make the scores differ
slightly from the GPU release; the model card reports the measured recall and latency.
