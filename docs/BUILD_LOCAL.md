# Building a graph with a local open model

The build module (`gct/build/`) talks to any server that offers the OpenAI Chat Completions API, so the same code that builds a graph with GPT-6.1 Sol can build it with an open model on your own GPU: start an OpenAI-compatible server, point `OPENAI_BASE_URL` at it and name the model with `--model`. No source text leaves the machine. This page gives the two models we measured, the commands we ran and how their graphs differ from those built by GPT-6.1 Sol. How the build works is described in [BUILD_GRAPH.md](BUILD_GRAPH.md), and the input files in [BUILD_INPUTS.md](BUILD_INPUTS.md).

[Recommendation](#recommendation) · [Setup we measured](#setup-we-measured) · [Steps](#steps) · [Results](#results) · [Known issues](#known-issues)

## Recommendation

| | Qwen3.6-35B-A3B (recommended) | Qwen3.5-9B |
|---|---|---|
| Hugging Face repository | [`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | [`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B) |
| Revision we measured | `995ad96eacd98c81ed38be0c5b274b04031597b0` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| Licence | Apache-2.0 | Apache-2.0 |
| Weights as served | The official BF16 weights, quantised to FP8 by vLLM when it loads them (`--quantization fp8`) | The official BF16 weights |
| GPU memory | 47,491 MiB at most while building (nvidia-smi, sampled every 5 s), with the server capped at 0.58 of an A100-80GB | 23,339 MiB at most while building (nvidia-smi, sampled every 5 s), with the server capped at 0.28 of an A100-80GB |

Use Qwen3.6-35B-A3B when a GPU with about 48 GB is available (a 48 GB card has 49,152 MiB): on the three example cases it dropped far fewer facts for an undefined subject (7 against 83), split fewer people into extra entities (8 against 11), left fewer time codes it could not resolve (52 against 85), returned fewer invalid answers (1 of 71 requests against 6 of 79) and built faster; Qwen3.5-9B's graphs hold slightly more facts (1,179 against 1,091) and their time codes disagree less often with dates written in the facts (52 against 73) ([Results](#results)). Qwen3.5-9B needs about 24 GB (a 24 GB card has 24,576 MiB); see also [Known issues](#known-issues). We measured both on an A100-80GB with the memory caps above, not on 24 GB or 48 GB cards. For the best graphs, use GPT-6.1 Sol ([BUILD_GRAPH.md](BUILD_GRAPH.md)).

## Setup we measured

- One NVIDIA A100-80GB per model, with the memory vLLM may use capped by `--gpu-memory-utilization` to the share given above, so that the measured footprint is what a smaller card would have to hold. We did not run these models on 24 GB or 48 GB cards.
- vLLM 0.25.0 (with PyTorch 2.11.0, CUDA 12.9 build), started as shown below, context length 32,768 tokens, at most 16 sequences.
- The build module of this repository, two requests in flight, the model's thinking mode switched off (`--extra-body '{"chat_template_kwargs": {"enable_thinking": false}}'`), all other sampling settings left to the model's defaults.
- The same three example cases as in [BUILD_GRAPH.md](BUILD_GRAPH.md), each built once from an empty cache.

## Steps

**1. Get the weights.** Download the measured revision (the `hf` command comes with huggingface_hub):

```bash
hf download Qwen/Qwen3.6-35B-A3B --revision 995ad96eacd98c81ed38be0c5b274b04031597b0 --local-dir weights/qwen3.6-35b-a3b
hf download Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --local-dir weights/qwen3.5-9b
```

We served copies of the weights that we checked, file by file, against these revisions on Hugging Face: every weight, configuration, tokenizer and chat-template file is identical (SHA-256 for the weights, the Git object hash for the other files).

**2. Start the server.** We ran vLLM 0.25.0 (with PyTorch 2.11.0 built for CUDA 12.9) in its own Python environment; install it as the vLLM documentation describes for your CUDA version. Then start one server per model:

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve weights/qwen3.6-35b-a3b --served-model-name Qwen/Qwen3.6-35B-A3B \
    --quantization fp8 --language-model-only --max-model-len 32768 --max-num-seqs 16 --gpu-memory-utilization 0.58 \
    --host 127.0.0.1 --port 8000

VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve weights/qwen3.5-9b --served-model-name Qwen/Qwen3.5-9B \
    --language-model-only --max-model-len 32768 --max-num-seqs 16 --gpu-memory-utilization 0.28 \
    --host 127.0.0.1 --port 8001
```

What the options do: `--quantization fp8` stores the weights of the 35B model in FP8 so that it fits in about 48 GB; `--language-model-only` skips the vision encoder of these models, which the build does not use; `--gpu-memory-utilization` is the share of the GPU's memory vLLM may take (0.58 and 0.28 of 80 GB here; on a card with 48 GB or 24 GB, raise it so that the same amount fits); `--max-num-seqs 16` keeps the per-sequence state of these models within the memory cap; `VLLM_USE_FLASHINFER_SAMPLER=0` uses vLLM's built-in sampler, which needs no compiler at start-up. `--host 127.0.0.1` keeps the server on the local machine. The server is ready when its log says `Application startup complete`; on our machine that took about 1 minute for Qwen3.5-9B and 3 minutes for Qwen3.6-35B-A3B, with vLLM's compile cache from an earlier start.

**3. Build.** From the repository root of this project, in the environment described in the [README](../README.md):

```bash
OPENAI_BASE_URL=http://127.0.0.1:8000/v1 OPENAI_API_KEY=EMPTY python -m gct.build --case examples/cases/chat_en \
    --model Qwen/Qwen3.6-35B-A3B --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}' \
    --cache cache/qwen3.6-35b-a3b/chat_en --out graphs/chat_en.qwen3.6-35b-a3b.json
python -m gct.build.validate graphs/chat_en.qwen3.6-35b-a3b.json
```

This writes to `graphs/` and leaves the example graphs in `examples/cases/` untouched. (We built the published `examples/cases/<case>/graph.raw.<model>.json` with the same command and `--out` pointing there, which overwrites them.) For Qwen3.5-9B use port 8001 and `--model Qwen/Qwen3.5-9B`. `OPENAI_API_KEY` can be any value for a local server. `--model` must be the name given to `--served-model-name`. Everything else in [BUILD_GRAPH.md](BUILD_GRAPH.md) applies unchanged, including `--sources` for the AI-memory variants.

## Results

The three example cases, built from their raw material with each model:

| Case | Builder | Entities / facts / edges | Facts against GPT-6.1 Sol | Requests | Build time |
|---|---|---|---|---|---|
| `chat_en` | GPT-6.1 Sol | 308 / 773 / 525 | | 20 | 17 min |
| | Qwen3.6-35B-A3B | 247 / 436 / 333 | 56% | 18 | 5.6 min |
| | Qwen3.5-9B | 283 / 456 / 403 | 59% | 19 | 9.0 min |
| `assistant_en` | GPT-6.1 Sol | 263 / 496 / 391 | | 22 | 14 min |
| | Qwen3.6-35B-A3B | 181 / 332 / 227 | 67% | 23 | 4.3 min |
| | Qwen3.5-9B | 195 / 347 / 295 | 70% | 24 | 6.7 min |
| `family_zh` | GPT-6.1 Sol | 257 / 537 / 425 | | 31 | 16 min |
| | Qwen3.6-35B-A3B | 182 / 323 / 295 | 60% | 30 | 5.9 min |
| | Qwen3.5-9B | 222 / 376 / 362 | 70% | 36 | 10.3 min |

Requests and build times are those of the first build from an empty cache; requests include retries. Build times are the wall time of one build with two requests in flight, for GPT-6.1 Sol and for the local models with the setup above (the Qwen3.6-35B-A3B build of `family_zh` shared its server with other builds). The graphs were later rebuilt from the same caches after two rounds of fixes to the build's rules, which re-sent 33 occasion-grouping requests to Qwen3.5-9B, 20 to Qwen3.6-35B-A3B and 6 to GPT-6.1 Sol (no extraction request); the counts of entities, facts and edges are those of the rebuilt graphs.

Beyond the number of facts, these counts show where the local graphs fall short. Each comes from the build report or from reading the graph.

| Case | Builder | Facts dropped (subject undefined) | Time code disagrees with a date written in the fact | Superseded facts that are real replacements | Extra entities for one person |
|---|---|---|---|---|---|
| `chat_en` | GPT-6.1 Sol | 0 | 2 of 60 | 4 of 4 | 0 |
| | Qwen3.6-35B-A3B | 4 | 23 of 46 | 1 of 7 | 5 |
| | Qwen3.5-9B | 52 | 14 of 33 | 0 of 2 | 5 |
| `assistant_en` | GPT-6.1 Sol | 0 | 2 of 103 | 2 of 2 | 0 |
| | Qwen3.6-35B-A3B | 0 | 26 of 74 | 1 of 3 | 3 |
| | Qwen3.5-9B | 11 | 23 of 73 | 0 of 5 | 6 |
| `family_zh` | GPT-6.1 Sol | 0 | 0 of 74 | 4 of 4 | 0 |
| | Qwen3.6-35B-A3B | 3 | 24 of 54 | 0 of 1 | 0 |
| | Qwen3.5-9B | 20 | 15 of 42 | (none superseded) | 0 |

*Facts dropped*: facts whose subject the model named with an id it never defined; the build leaves them out (`facts_dropped_unknown_subject` in the report). *Time code disagrees*: of the facts that write a date in their text, those whose time code points elsewhere; the build keeps the code in most of these cases ([BUILD_GRAPH.md](BUILD_GRAPH.md#time)), so many of the local dates are wrong. *Real replacements*: our reading of every `superseded` fact against the fact that replaced it. *Extra entities for one person*: person entities that repeat a name already in the graph. The local graphs hold 56% to 70% as many facts as GPT-6.1 Sol's, and on every count in this table both local models are behind GPT-6.1 Sol.

We also built the three AI-memory variants of each case ([BUILD_INPUTS.md](BUILD_INPUTS.md)) with Qwen3.6-35B-A3B; the graphs are in the case directories as `graph.memory_*.qwen3.6-35b-a3b.json`. They hold fewer facts than GPT-6.1 Sol's graphs of the same summaries (57 to 217 against 274 to 426).

## Known issues

- **Language of Chinese material.** Without an explicit instruction, both models wrote many facts of Chinese material in English: in a first build of `family_zh`, Qwen3.6-35B-A3B wrote 275 of 382 facts in English and Qwen3.5-9B 110 of 375. The build now adds a line to its prompts for Chinese cases (`"lang": "zh"`) asking for Chinese names and texts; with it, no fact of the published `family_zh` graphs is in English apart from the dates in brackets (0 of 280 for Qwen3.6-35B-A3B, 0 of 314 for Qwen3.5-9B), and the results above were measured. GPT-6.1 Sol wrote Chinese without the line.
- **Fewer facts.** The local models extract fewer facts per session than GPT-6.1 Sol, so details mentioned in passing are more often missing.
- **Facts with an unknown subject.** The local models sometimes name a fact's subject by an id they did not define. The build leaves such facts out and counts them in the report (`facts_dropped_unknown_subject`): Qwen3.5-9B lost 11, 52 and 20 facts in `assistant_en`, `chat_en` and `family_zh` (of 328, 421 and 334 extracted; most name a subject with an id such as "M1" that the answer never defines), Qwen3.6-35B-A3B 0, 4 and 3 (of 306, 383 and 283), GPT-6.1 Sol none. A subject given by the exact name of a known entity ("Leo's dad") is resolved.
- **Dates.** The time codes of the local models disagree more often with a date written in the same fact: in 23, 14 and 15 facts of `assistant_en`, `chat_en` and `family_zh` for Qwen3.5-9B (of 73, 33 and 42 facts with a written date), 26, 23 and 24 for Qwen3.6-35B-A3B (of 74, 46 and 54), against 2, 2 and 0 for GPT-6.1 Sol (of 103, 60 and 74). The build replaces a code by the written date only in the five cases of [BUILD_GRAPH.md](BUILD_GRAPH.md#time), because a written date is often not the date of the fact; so it keeps the code in most of these (19, 11 and 13 for Qwen3.5-9B, 18, 21 and 16 for Qwen3.6-35B-A3B), and many of those kept codes are wrong. Time codes that the build could not turn into a date (`when_unresolved` in the report) numbered 23 to 35 per case for Qwen3.5-9B and 6 to 24 for Qwen3.6-35B-A3B, against 1 to 3 for GPT-6.1 Sol. Check the dates of local graphs before you rely on them.
- **Corrections.** The local models give the same slot name to facts that are not values of one thing (being lactose intolerant and wearing contacts both as "diet"; a date and becoming maid of honour as "relationship status"; two medicines as "medication schedule"), and the build then marks the older fact as replaced. Of the 29 `superseded` facts in the 15 local graphs, we judged 3 to be real replacements (a new address, a new job, a new lease); in the 12 GPT-6.1 Sol graphs all 11 are. Read `superseded` facts in local graphs with care and check them as in [BUILD_GRAPH.md](BUILD_GRAPH.md#check-the-graph).
- **Wrong aliases.** The smaller model sometimes lists another person's name among an entity's aliases, and the build keeps it when it merges the entity into a known one: in Qwen3.5-9B's graph of `family_zh`, 周子航 carries "小满" (his sister's name) as an alias. Aliases are only used to match later mentions; check them in local graphs before you add more sources.
- **One person as several entities.** The local models name the same person with changing roles (Leo's partner Sam as a friend, a partner, a child), and the build keeps two people apart when their roles are incompatible. Counting person entities that repeat a name already in the graph, Qwen3.5-9B's graphs have 6, 5 and 0 such extra entities in `assistant_en`, `chat_en` and `family_zh`, Qwen3.6-35B-A3B's 3, 5 and 0 (before the build learnt to look at every entity with a name and to treat a friend and a partner as compatible: 16, 7 and 1, and 5, 16 and 1; 13 entities were named Sam in one graph), GPT-6.1 Sol's none.
- **Invalid answers.** In the first builds, Qwen3.5-9B returned an answer that was not valid JSON, or skipped a session of the batch, in 1 of 19 requests of `chat_en` and 5 of 36 of `family_zh`; Qwen3.6-35B-A3B in 1 of 30 requests of `family_zh`. The build retried them and every retry succeeded; raise `--retries` if failures persist.
- **Thinking mode.** We measured both models with thinking switched off; we did not measure them with it on.
- **torchcodec without FFmpeg.** In our vLLM 0.25.0 environment the `torchcodec` package was installed but the machine had no FFmpeg libraries, and importing vLLM's API server failed with "Could not load libtorchcodec". We started the servers with a stand-in `torchcodec` module on `PYTHONPATH` that only raises `ImportError`, so that vLLM treats it as absent; with it the server imported and ran. Installing FFmpeg's libraries, or uninstalling `torchcodec` from the server's environment (the build sends only text), should also work; we did not try those.
- **Server options.** Without `--max-num-seqs 16`, vLLM refused to start Qwen3.5-9B under the 0.28 memory cap ("max_num_seqs (256) exceeds available Mamba cache blocks"). Without `VLLM_USE_FLASHINFER_SAMPLER=0`, vLLM tried to compile FlashInfer's sampler at start-up and failed on our machine, which had no `ninja` on the path.
