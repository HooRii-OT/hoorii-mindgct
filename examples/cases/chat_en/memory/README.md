# AI-memory variants of this case

The files in this directory hold the same case as `../sources.jsonl` (44 text-message conversations), written as memories that another AI assistant might keep about its user instead of the raw material. They let you compare a graph built from raw material with one built from summaries ([docs/BUILD_GRAPH.md](../../../../docs/BUILD_GRAPH.md#ai-summarised-memories)).

| File | Form ([docs/BUILD_INPUTS.md](../../../../docs/BUILD_INPUTS.md)) | Content |
|---|---|---|
| `memory_list.jsonl` | `memory_list` | 221 memory items, one fact each, dated with the source they come from |
| `memory_summaries.jsonl` | `memory_summaries` | 44 summaries, one per source, dated with it |
| `memory_profile.jsonl` | `memory_profile` | a Markdown profile (91 entries after splitting by heading), dated with the last source |

## How they were made

Each file was generated once from `../sources.jsonl` with GPT-6.1 Sol (`gpt-6.1-sol`, reasoning effort `medium`) by `gct/build/summarize.py`, which holds the prompts. The prompts see only the sources, in date order, and the owner's name, and ask for what an assistant should remember for later conversations; they see nothing else of the case. We did not edit the output.

From the repository root, with `OPENAI_API_KEY` set. These are the commands we ran: they write into this directory and overwrite the published files. To make variants of your own, give `--out` another path, for example `--out my_memory/chat_en.memory_list.jsonl`.

```bash
python -m gct.build.summarize --case examples/cases/chat_en --form list --id memory_list --out examples/cases/chat_en/memory/memory_list.jsonl --cache cache/summarize/chat_en
python -m gct.build.summarize --case examples/cases/chat_en --form sessions --id memory_summaries --out examples/cases/chat_en/memory/memory_summaries.jsonl --cache cache/summarize/chat_en
python -m gct.build.summarize --case examples/cases/chat_en --form profile_md --id memory_profile --out examples/cases/chat_en/memory/memory_profile.jsonl --cache cache/summarize/chat_en
```

A new run gives different wording, because the model writes the summaries anew.

## Building a graph from them

```bash
python -m gct.build --case examples/cases/chat_en --sources memory/memory_list.jsonl --out graphs/chat_en.memory_list.json --cache cache/chat_en
```

and the same with `memory_summaries.jsonl` and `memory_profile.jsonl`. This writes to `graphs/`; the published graphs, built with `--out` pointing next to the case (which overwrites them), are `graph.memory_list.gpt-6.1-sol.json`, `graph.memory_summaries.gpt-6.1-sol.json` and `graph.memory_profile.gpt-6.1-sol.json`. Their facts are marked as second-hand (`hearsay`), and plans as `pending`, because the sources are another AI's summaries and not the user's words.
