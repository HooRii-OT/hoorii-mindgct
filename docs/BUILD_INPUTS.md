# Inputs for building a graph

`gct.build` builds a HooRii Mind graph from a **case directory**:

```
my_case/
  case.json        # who the graph belongs to and when it is built
  sources.jsonl    # the material, one source per line, in any of the six forms below
```

How the graph is built from these files is described in [BUILD_GRAPH.md](BUILD_GRAPH.md); this page describes the files.

## case.json

```json
{"world_id": "maya_en", "lang": "en", "owner": "Maya Chen", "now": "2026-03-14 18:30",
 "timezone": "Europe/London", "description": "One sentence about the case (not used by the build)."}
```

| Field | Meaning |
|---|---|
| `world_id` | Name of the graph: 1 to 64 characters from `A-Z a-z 0-9 _ . -` |
| `lang` | `en` or `zh`; write the material in this language |
| `owner` | The owner's name, spelled exactly as the owner's `speaker` name in the chats |
| `now` | When the graph is built, `YYYY-MM-DD` or `YYYY-MM-DD HH:MM` in the owner's local time, not earlier than the last source |
| `timezone` | The owner's IANA time zone, copied into the graph |
| `description` | Free text, ignored by the build |

Every source has an `id` (1 to 64 characters from `A-Z a-z 0-9 _ . : -`, unique in the case). Dates are `YYYY-MM-DD` and times `HH:MM`. Relative times in the text ("yesterday", "next Friday", "上周二") stay as they were said: the build turns them into dates from the date of their source.

## The six forms

| # | Form | `type` | How the build treats it | Example case |
|---|---|---|---|---|
| 1 | Chats between people (two or more) | `chat` | One source is one conversation. Its numbered turns are read by the extraction step, and every fact keeps the turns it came from | [`examples/cases/chat_en`](../examples/cases/chat_en), [`examples/cases/family_zh`](../examples/cases/family_zh) |
| 2 | Conversations between the user and an AI assistant | `chat` (an `assistant` participant, or `messages`) | As form 1. What the assistant says is context only: a suggestion counts when the user accepts it | [`examples/cases/assistant_en`](../examples/cases/assistant_en) |
| 3 | Notes and documents (memos, itineraries, order confirmations, emails, notices) | `note` | Read as a conversation written by its author, one turn per paragraph, with the title in front | [`examples/cases/assistant_en`](../examples/cases/assistant_en), [`examples/cases/family_zh`](../examples/cases/family_zh) |
| 4 | AI-summarised memories: a flat list of memory items | `memory_list` | The memory-import step splits items into facts, in batches of up to 40 items; the facts are second-hand (`hearsay`) unless the source is marked as confirmed by the user; plans and intentions are `pending` | `memory/memory_list.jsonl` of each example case |
| 5 | AI-summarised memories: one summary per conversation | `memory_summaries` | Each summary is a dated memory item, imported as in form 4 | `memory/memory_summaries.jsonl` of each example case |
| 6 | AI-summarised memories: a profile document (Markdown sections) or a categorised JSON object | `memory_profile` | Split into "Section > entry" items by headings or JSON keys, imported as in form 4 | `memory/memory_profile.jsonl` of each example case (JSON in `assistant_en`, Markdown in the others) |

Forms 1 to 3 are raw material: what people said or wrote. Forms 4 to 6 are memories that another AI has already summarised, not the user's words, and the graph keeps them marked as second-hand: their facts have the state `hearsay` ("hearsay, unconfirmed" in the text GCT reads) unless the source carries `"confirmed": true`, and their value starts with the date of the memory in brackets. A plan or intention in a memory ("Maya plans to visit Porto in May") gets `pending`, as a plan in a chat does: it is still something not yet done, and `pending` already tells the reader it is unconfirmed. One case may mix all six forms.

### 1. Chats between people

```json
{"id": "chat-2026-03-02", "type": "chat", "date": "2026-03-02", "time": "19:40", "channel": "text messages",
 "participants": ["Maya Chen", "Lin Park"],
 "turns": [{"speaker": "Lin Park", "text": "Are we still on for Saturday?"},
           {"speaker": "Maya Chen", "text": "Yes! I booked Sichuan Garden for 7. Remember I can't do peanuts."}]}
```

List every speaker of the conversation, the owner included, in `participants`; every `speaker` must be one of them. `time` and `channel` are optional. Write one record per conversation, each with its own `date`.

### 2. Conversations with an AI assistant

Mark the assistant in `participants`:

```json
{"id": "asst-0305", "type": "chat", "date": "2026-03-05",
 "participants": ["Maya Chen", {"name": "Assistant", "kind": "assistant"}],
 "turns": [{"speaker": "Maya Chen", "text": "Plan a 3-day Lisbon trip for me and my sister in April."},
           {"speaker": "Assistant", "text": "Sure. Day 1: Alfama ..."}]}
```

or write the messages with roles, where `user` is the owner:

```json
{"id": "asst-0305", "type": "chat", "date": "2026-03-05", "assistant": "Assistant",
 "messages": [{"role": "user", "content": "Plan a 3-day Lisbon trip for me and my sister in April."},
              {"role": "assistant", "content": "Sure. Day 1: Alfama ..."}]}
```

Only text messages with the roles `user` and `assistant` are read.

### 3. Notes and documents

```json
{"id": "note-flight", "type": "note", "date": "2026-03-08", "title": "Flight confirmation", "author": "Maya Chen",
 "text": "Atlantica Air AT1363, London Heathrow -> Lisbon, 17 April 2026 08:15.\n\nBooking ref K7Q2LM. Seats 14A and 14B (Maya and Sofia)."}
```

Paragraphs are separated by blank lines. `author` is optional and defaults to the owner; for an email, give the sender. A long document is cut at paragraph boundaries into parts of about 9,000 characters.

### 4. AI memories: a flat list

```json
{"id": "mem-list", "type": "memory_list", "date": "2026-03-10", "assistant": "AI assistant",
 "items": ["Maya is allergic to peanuts.",
           {"text": "Maya's sister Sofia is visiting London in April.", "date": "2026-02-27"},
           "Prefers short answers."]}
```

`date` is the day the list was saved or exported; an item may carry its own `date`. `assistant` names the AI that kept the memories and is written into each fact's `source` as `import:<name>`. `confirmed` (default `false`) marks memories the user has confirmed.

### 5. AI memories: one summary per conversation

```json
{"id": "mem-sessions", "type": "memory_summaries", "assistant": "AI assistant",
 "summaries": [{"date": "2026-03-02", "text": "Maya confirmed Saturday dinner with Lin at Sichuan Garden at 7pm and reminded Lin about her peanut allergy."},
               {"date": "2026-03-05", "text": "Maya asked for a 3-day Lisbon itinerary for herself and her sister Sofia in April."}]}
```

Each summary has the date of the conversation it summarises; relative times in it are read from that date. `title` is optional.

### 6. AI memories: a profile

As Markdown:

```json
{"id": "mem-profile-md", "type": "memory_profile", "date": "2026-03-10", "format": "markdown",
 "text": "# About Maya\n## Health\n- Allergic to peanuts\n## Family\n- Sister: Sofia (lives in Porto)\n## Work\n- Product designer at Northwind"}
```

As JSON:

```json
{"id": "mem-profile-json", "type": "memory_profile", "date": "2026-03-10", "format": "json",
 "data": {"health": {"allergies": ["peanuts"]}, "family": {"sister": {"name": "Sofia", "city": "Porto"}},
          "work": {"role": "product designer", "employer": "Northwind"}}}
```

In Markdown, headings give the section path and every list item or other non-empty line is one item ("About Maya > Health > Allergic to peanuts"). In JSON, every leaf value is one item with its key path ("family > sister: name: Sofia; city: Porto"; an object whose values are all plain values stays one item). `date` is the day the profile was saved.

## Converting common files

`python -m gct.build.convert` turns two common shapes into a chat record and prints it as one JSON line, to append to `sources.jsonl`:

```bash
# a plain-text transcript with one "Name: text" turn per line
python -m gct.build.convert transcript chat.txt --id chat-0302 --date 2026-03-02 --owner "Maya Chen"
# a JSON array of {"role", "content"} messages (user = the owner)
python -m gct.build.convert messages messages.json --id asst-0305 --date 2026-03-05
```

Add `--assistant NAME` to mark an AI assistant speaker in a transcript and `--time HH:MM` for the start time. A time stamp in front of a line ("[19:40] Maya Chen: ...", "2026-03-02 19:40 Maya Chen: ...") is removed. Without `--speakers`, every "Name:" at the start of a line starts a turn, so a line such as "Note: bring the tickets" becomes a speaker called Note; the names found are printed so you can check them. `--speakers "Maya Chen,Sofia Chen"` lets only those names (and the owner and the assistant) start a turn, and keeps other lines as text of the turn before. `--id`, `--date` and `--time` are checked as in `sources.jsonl`. These converters read the generic shapes above. They do not claim to read the export format of any particular product.

## Making AI-memory variants of a case

`python -m gct.build.summarize` asks a model to keep, from a case's raw chats and notes, what an assistant's memory feature would keep, and writes one memory record (forms 4 to 6). We used it to derive the AI-memory variants of the example cases; the prompts are in `gct/build/summarize.py` and see only the sources and the owner's name.

```bash
python -m gct.build.summarize --case examples/cases/chat_en --form sessions --id memory_summaries --out my_memory/chat_en.memory_summaries.jsonl --cache cache/summarize/chat_en
```

`--form` is `list`, `sessions`, `profile_md` or `profile_json`. The record goes to `my_memory/`; to build from it, pass it to the build with `--sources my_memory/chat_en.memory_summaries.jsonl` (a path relative to the repository root also works). The commands we used to make the example files are in `examples/cases/<case>/memory/README.md`; they write into the case's `memory/` directory and overwrite the published files.
