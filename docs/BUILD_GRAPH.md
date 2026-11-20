# Building a graph with GPT-6.1 Sol

GCT retrieves from a personal graph: the people, places, plans and things in a user's life, the facts about them and the links between them. This page describes how we build such a graph from chats, notes and AI-summarised memories with GPT-6.1 Sol (`gpt-6.1-sol`) through the OpenAI API, using the build module in this repository, `gct/build/`. [BUILD_INPUTS.md](BUILD_INPUTS.md) describes the input files, [BUILD_LOCAL.md](BUILD_LOCAL.md) the same build with an open model on your own GPU, and [COMPILE.md](COMPILE.md) what to do with the graph afterwards.

[What you need](#what-you-need) · [Build a graph](#build-a-graph) · [What the build does](#what-the-build-does) · [What the graph looks like](#what-the-graph-looks-like-and-why) · [Time](#time) · [Fact states](#fact-states) · [Corrections](#corrections-cancellations-and-old-information) · [AI-summarised memories](#ai-summarised-memories) · [Check the graph](#check-the-graph) · [Cost](#cost) · [Common problems](#common-problems) · [Options](#options)

## What you need

- This repository, installed as described in the [README](../README.md). The build itself only sends HTTP requests and needs no GPU, but `python -m gct.build` imports the `gct` package, so the packages of `requirements.txt` must be installed.
- An OpenAI API key with access to `gpt-6.1-sol`, in the environment variable `OPENAI_API_KEY`. The build calls `https://api.openai.com/v1` unless `OPENAI_BASE_URL` names another OpenAI-compatible address.
- A case directory: `case.json` and `sources.jsonl` ([BUILD_INPUTS.md](BUILD_INPUTS.md)). The three example cases in `examples/cases/` can be used as they are.

## Build a graph

From the repository root:

```bash
export OPENAI_API_KEY=...            # your key
python -m gct.build --case examples/cases/chat_en --out graphs/chat_en.json --cache cache/chat_en --report graphs/chat_en.report.json
python -m gct.build.validate graphs/chat_en.json
```

The build prints a line for every finished extraction or memory-import request and one summary line for the organising step (its occasion-grouping requests included), and writes the graph, a report (counts, requests, tokens, time) and a cache. If it stops, run the same command again: every answer is in the cache, and only the missing requests are sent. When sources are added to the case later, the sessions already extracted are taken from the cache as well. The graph, the cache and the raw answers hold the content of your sources: the build creates them readable by your user only (file mode 0600, directories 0700); keep them that way when you copy them.

To build from an AI-memory variant of a case instead of its raw material, name the sources file:

```bash
python -m gct.build --case examples/cases/chat_en --sources memory/memory_summaries.jsonl --out graphs/chat_en.summaries.json --cache cache/chat_en
```

The graph is a JSON file in the HooRii Mind format, the same format as `examples/example_graph.json`, and can be compiled and queried as in the README:

```python
import json
from gct import GCT

MODEL = "HooRiiTechnology/hoorii-mindgct"     # or the path of a local copy of that repository
gct = GCT.from_pretrained(MODEL)
graph = gct.compile_graph(json.load(open("graphs/chat_en.json")))
out = gct.retrieve(graph, "I want to mail Maddie a card. What's her address?", now="2026-03-05 20:15", top_k=3, positive_only=False)
for unit in out["results"]:
    print(round(unit["score"], 3), unit["display_text"])
```

This prints the three best units of the request with their scores; with `positive_only=True` (the default) only units with a positive, must-mention score are returned.

The graphs we built for the example cases are in `examples/cases/<case>/graph.<input form>.<model>.json`; `graph.raw.gpt-6.1-sol.json` was built from the raw material with the command above, with `--out` pointing into the case directory. The one of `family_zh` was built before the build added its instruction on the language of Chinese material ([BUILD_LOCAL.md](BUILD_LOCAL.md#known-issues)); GPT-6.1 Sol wrote every fact in Chinese without it.

## What the build does

The build follows the pipeline we use for HooRii Mind, without the parts that need our service (privacy levels, several callers, versions). It has six steps. Steps 2, 4 and 5 call the model; everything else is a program, so the same answers always give the same graph.

**1. Load.** `case.json` and `sources.jsonl` are read and checked. Each chat or note becomes one *session*: a chat as it is, a note as a session written by its author with one turn per paragraph. Sessions are put in date order. Memory sources become dated memory items (step 5).

**2. Extract.** Sessions are sent in batches of at most 9,000 characters and three sessions, each turn numbered (`T1 Maya Chen: ...`). For every session the model returns the entities the facts are about (with aliases and how each relates to a participant), the facts, one piece of information each, and the typed relations between them. Every fact names the turns it comes from, a module of the vocabulary in `gct/vocab.json` (`health.allergies`, `work.role`, ...), a state (`confirmed`, `pending` for a plan or intention, `hearsay` for something uncertain or second-hand) and a time code, and, when it gives the current value of something that can change (home city, employer), a slot name. The prompt is in `gct/build/prompts.py`. Two choices matter:

- **The model never sees a calendar date.** The start of each session is called S, and the model answers "S-1d", "S+1w", "S-Tue" (the Tuesday before), "Y-1:05-10" (May 10 of the previous year) and so on. The program turns these codes into dates with calendar arithmetic, so the date arithmetic is done by the program and is repeatable, and a cached extraction can be reused when a session is re-dated.
- **Facts keep the words as said.** "Maddie ran 11 miles yesterday" stays in the fact, with the resolved date written after it, so the reader of the fact sees both.

The answer of each session is cached on its own, keyed by the prompt, the model and its request settings (wire format, reasoning effort, temperature, extra request fields), the participants and the session text.

**3. Assemble.** Sessions are applied in date order, the way new conversations arrive. Each entity is matched against what the graph already holds: by its proper name and aliases, by a unique role ("Leo's father"), or by a descriptive name ("Maya's car"). Words that are not names ("Dad", "my mom", "外婆", pronouns) are never used to match. A first name alone does not join two people whose roles are incompatible (a colleague and a plumber both called Marcus, a brother and a friend both called Alex); every known entity with the name is considered, and the one with a compatible role is taken, so a later mention of the plumber finds the plumber. A friend and a partner count as compatible (the friend who is later called a partner stays one person). A role never joins two people with different proper names (a business partner and a girlfriend). A title with a surname joins the full name ("Dr. Huang" and "Dr. Ana Huang"; 王老师 and 王丽老师), a Chinese given name joins the full name it ends ("子航" and "周子航"), and a participant's first name joins the participant when no one else has it ("Rachel" for the owner Rachel Moreno); a surname with a title alone (王老师) is otherwise matched together with whose teacher it is. A fact's subject may be an id, a participant, or the exact name of an entity of the session or the graph ("Leo's dad"); a fact whose subject is none of these is left out and counted in the report (`facts_dropped_unknown_subject`); the owner never receives it by default. Facts get stable ids and keep their source turns. A newer value of a single-valued slot marks the older fact as superseded (see below). Relations become edges.

**4. Organise.** Local rules give the graph an event-centred shape, repeated until nothing changes:

- entities of compatible kind whose names or aliases coincide are merged;
- a fact on a person that mentions a more specific thing (an event or trip, a project, a possession, a pet, a place or organisation) moves to that thing, and the person becomes a mention, so that "Leo started at Ashby High" sits on Ashby High and is reached from both;
- for each person, one model call groups the facts that describe the same occasion (a trip, a party, an appointment, a project) into an event, trip, appointment or project entity, and names existing occasions that are the same one under two names;
- an occasion with more than 15 facts is split into one occasion per date;
- occasions get attendance edges, and "Maya's pottery class" gets the aliases "pottery class" and "the class".

Then every fact value gets its date in brackets at the end, newer values and retractions supersede older facts, and every occasion gets a dated schedule fact ("Lin's birthday dinner: 21 March 2026"), except an occasion that was called off and not planned again. When none of an occasion's facts gives a date, the schedule fact takes the date of its first source, the day it was talked about.

**5. Memories.** Memory items another AI has summarised (forms 4 to 6 in [BUILD_INPUTS.md](BUILD_INPUTS.md)) are sent in batches of at most 40 items through their own prompt, which asks for facts as close to the memory as possible. Their entities are matched against the graph by exact name or alias, by a first name, a title with a surname or a Chinese given name when exactly one known person of a compatible role has it (never two full names with different surnames, never a title such as "Dr." taken for a first name), or by a unique role (mother, father, spouse); an alias that several entities share goes to the entity whose own name it is; words that are not names ("哥哥", "Mom") are never used to match, and two different proper names are never joined. A fact may name its subject by id or by name; a subject the batch does not define is left out and counted. Plans and intentions are `pending`; every other memory fact is second-hand (`hearsay`) unless the source says the user confirmed it (then `confirmed`). The memory's date is in brackets at the start of the value.

**6. Finalise.** Every entity is linked to the owner (a chat partner with no stated relation gets `custom.talks_with`, any other unlinked entity `custom.mentions`), and the graph is written with `vis: ["o"]` on every fact and edge: the owner's view. Each fact keeps `prov`, the source turns it came from (`"chat-0302#4"`), `observed_at`, the date of its source, and `valid_from` / `valid_to` when the time it refers to is known. `meta.build` records the input form and the model.

## What the graph looks like, and why

A graph built this way may look different from a hand-made profile. These are results of the pipeline; this is the form our own pipeline produces:

- **Organised around occasions.** About half of the facts sit on events, trips, appointments and projects (52%, 52% and 46% in `assistant_en`, `chat_en` and `family_zh`), each with a dated schedule fact. A question about a trip finds the trip and everything said about it.
- **A fact about a person often hangs on the thing it is about.** When a fact about someone names a more specific thing or occasion, it moves there and keeps the person as a mention: a fact about a child getting carsick can sit under the travel-sickness medicine it was mentioned with, a landing time under the flight. Few facts stay directly on the owner (1.6%, 3.5% and 2.8% of the facts), because most of what the material says about the owner involves something more specific; another 12% to 23% of the facts are about the owner and sit on such a thing, with the owner kept as a mention of the fact.
- **About half of the facts have the general module.** `notes.general` covers 44%, 57% and 51% of the facts: a fact keeps the module the extraction gave it only when that module fits the kind of the entity it ends up on.
- **Many small entities.** Things mentioned once (a receipt, a shop, a single visit) become entities of their own; 20% to 26% of the entities have exactly one fact.
- **Split occasions carry the day they were talked about.** An occasion with more than 15 facts is split into one occasion per date, and each part gets a schedule fact with the date of its own facts, which is often the day the occasion was discussed rather than the day it takes place. In `chat_en`, Grace and Daniel's wedding (16 May 2026) is split into 9 parts named after their dates, with schedule facts such as "Grace and Daniel's wedding: 27 September 2025" and "Grace and Daniel's wedding: 15 March 2026"; the fact that gives the wedding date ("Grace and Daniel have set their wedding date for May 16 (16 May 2026)") sits on the part of 16 May 2026. The facts themselves keep their own dates. This is the form our pipeline produces; read a split occasion's schedule fact as "talked about on", and look for the fact that states the date.

## Time

Relative times ("yesterday", "next Friday", "上周二") only mean something together with the day they were said, so the build anchors them explicitly.

- `observed_at` is the date of the source: when it was said or written.
- `valid_from` is the time the fact refers to. It comes from the model's time code ("S-1w", "S+Fri", "04-18"), turned into a date by the program. A day is written as `valid_from`; a month or a year as `valid_from` and `valid_to` (the first and last day). "Next month" said on 1 October is November, and "last year" said in 2026 is 2025.
- A date written in the fact itself ("25号", "April 24", "Monday the 30th") replaces the code only in five cases: the code says no more than "the day it was said" (or there is none); the fact says something was moved ("rescheduled to the 20th", "从24号改签到25号": the date after the move); the code is a weekday and a written date falls on that weekday ("Wednesday the 17th" with the code "next Wednesday"); the code is a month or a year and the written date lies in it ("上个月25号" with the code "last month"); or the code is more than a year away while the written month-day is not (a code six years ahead beside "5月9日"). In every other case the code is kept, because a written date is often not the date of the fact: "the week before December 21", "from Monday to Sunday the 30th", "苏晓雯4月9日说上周已完成体检", "Ava, whose birthday is March 14, is sleeping over on Friday". A written date that replaces a contradicting code is counted in the report (`date_code_overridden_by_text`).
- A month and day without a year ("June 8") is read as the nearest such day, the later one when two are equally near. For a plan it is the coming one, unless that day passed at most 60 days ago ("promised on February 19 to RSVP", said in March, stays in this year); a weekday written in the same sentence ("3月26日周四") picks the year in which the day falls on it.
- A day without a month ("the 30th", "18号") is read in the month of the source when that day is not later than the source, otherwise as the nearest such day (the later one on a tie); a weekday written with it ("Monday the 6th") picks among the previous, this and the next month. For a plan, and for a yearly date such as a birthday ("Leo's birthday is on the 30th"), it is the coming such day.
- Only full month words, or abbreviations that are followed by a date-like context, count as months ("3 novels", "Mayan" and "Jan 5 times" hold no date); house numbers after 路 / 街 / 道 / 巷 / 弄 / 村 / 区 or 门牌 ("文三路 12号", "门牌18号") are not days.
- The date in brackets at the end of the value is `valid_from` when known, otherwise the date of the source. It is written in English ("7 May 2023", "May 2023", "2023"), also in Chinese graphs: the default, as in our own pipeline. `--date-lang zh` writes Chinese dates instead ("2026年4月28日"); we keep English as the default, as in our own pipeline.
- What still goes wrong. The rules are about the words, not the meaning, so some dates stay off: a wrong code is kept when the fact also names the right date in a way the five cases above do not cover (a model code "04-27" next to "4月26日出发"); a fact that spans a period gets only its first day ("from the 17th to the 19th"); a day without a month that was meant for next month but is said after the same day of this month is read as this month's ("the 3rd", said on the 10th about the coming 3rd, unless the fact is a plan); and a fact about a recurring date ("autopay on the 1st") gets one occurrence. Check dates as described in [Check the graph](#check-the-graph).
- What the build does not do: it does not turn a plan into a past event when its date has passed. A plan stays `pending` until a later source says it happened, was cancelled or moved. Compare `valid_from` with the date of the request when you present such facts. Times of day stay in the text.
- Sources need a full date. A source without a date cannot be anchored; give it the best date you have.

## Fact states

Every fact has one of four states, which GCT reads as part of the fact ("confirmed", "pending confirmation", "hearsay, unconfirmed", "superseded old value"):

| State | When the build gives it | Example from the example graphs |
|---|---|---|
| `confirmed` | The source states it as true. This is also the state when the model gives none, and the state of AI-memory facts from a source marked `"confirmed": true` | "Priya's new address is 2290 SE Linnet St, Apt 4B, Portland, OR 97202 (9 March 2026)" (`assistant_en`) |
| `pending` | A plan, an intention or something not yet done. A schedule fact of an occasion takes the state of the occasion's facts | "Maddie Cho was invited to an Aspen Lane on-site interview in Denver on Thursday, September 4 (4 September 2025)" (`chat_en`) |
| `hearsay` | Uncertain or second-hand: what someone else reports, guesses or expects. Every fact from AI-summarised memories that is not a plan, unless the source is marked as confirmed | "据子航说，启明英语的Amy老师下个月调走。 (April 2026)" (`family_zh`: "According to Zihang, Amy, the teacher at Qiming English, moves away next month") |
| `superseded` | A later fact replaced it ([next section](#corrections-cancellations-and-old-information)); `superseded_by` names that fact | "四（3）班家长会定于4月23日（周四）晚上7点召开。 (23 April 2026)", replaced by "家长会因学校教研活动改至4月28日（周二）晚上7点。" (`family_zh`: the parent meeting moved from 23 to 28 April) |

About three in ten facts of the example graphs are `pending`: 144 of 496 in `assistant_en`, 190 of 773 in `chat_en` and 172 of 537 in `family_zh`. These cases are about getting things done, so much of what they hold is plans and appointments. A plan stays `pending` after its date: the build does not guess that it happened.

## Corrections, cancellations and old information

People change their minds, move house and cancel plans. Nothing is deleted from the graph: a fact that no longer holds gets the state `superseded` (GCT reads it as "superseded old value"), and `superseded_by` names the fact that replaced it. Four rules do this:

- **Single-valued slots.** When a fact gives the current value of something that can change (home city, employer, address, school), the newer value supersedes the older one, unless it restates it. Names of people are left out before comparing, so "Priya lives in Chicago" does replace "Priya lives in Boston". When both facts carry numbers and the newer one has a number the older one lacks, it is a new value ("6 years old" → "7 years old", "7 pm" → "8 pm", "3岁" → "4岁"). Otherwise a restatement has mostly the same words, the same numbers (a time, a room, an address), only words or names the older one already has ("works at Ashby" after "works at Ashby High"), or, in Chinese, content the older one contains ("住在南京" after "家住南京市建邺区…"); a vaguer restatement never replaces the exact value.
- **Retractions.** A confirmed fact that removes, cancels or stops something ("no longer", "gave up", "取消", "不再") supersedes earlier facts about the same thing; a plan or a rumour ("is thinking about cancelling") retracts nothing. Time words are not part of what is retracted, and "stopped by", "dropped (someone) off" and "took the day off" are not retractions. One that says something was finished supersedes only the plans and the ongoing work about it ("is doing a branding project"), not facts that stay true ("the project is for a bakery"). One that moves something to another date supersedes only facts that state a date. A fact with the same date or the same numbers as an earlier one restates it and does not retract it.
- **Cancelled occasions.** When an occasion is called off (a confirmed cancellation that names the occasion: "Grace called off the bridal shower", "厦门旅行取消了"), its earlier plans and its confirmed arrangements still ahead are superseded, and it gets no schedule fact unless it is planned again later. A new plan said in the same conversation as the cancellation ("cancelled Saturday, the party is the Saturday after instead") counts as later.
- **Moved occasions.** When an occasion is moved ("rescheduled to March 23", "改到4月28日"; a move ahead is usually `pending`, a possible move such as "might be moved" does not count), its earlier facts that state a different time are superseded, whatever their module, and its schedule fact takes the new date. An earlier statement that still holds ("a trip in October" after "moved to October 24") stays.

These rules work on words, not meaning, and they lean towards leaving an old fact current. A retraction is recognised only when its words name what it retracts: "Priya quit her job at Acme" does not supersede "Priya works at Acme as a designer", "小满不再挑食了" does not supersede "小满挑食，不吃胡萝卜", and when a train is cancelled, the plan to pick someone up from that train stays `pending`. What is cancelled is the head of the phrase, so a cancellation must name the occasion itself: "cancelled the hotel breakfast add-on for the Vancouver trip" and "Lena's Thursday flight for her Lisbon trip was cancelled" leave the trip alone (the second still supersedes the fact about that flight), and a cancellation that names a date ("the flight on the 24th was cancelled") leaves facts about other dates alone. They can still go wrong the other way: a retraction supersedes an earlier fact on the same thing that shares most of its words, even when that fact still holds, and two different facts that the model gave the same slot name ("diet") count as old and new values. Read superseded facts together with the fact that replaced them when you present them.

A change inside one conversation is left to extraction, which sees the whole exchange: in our builds it wrote the outcome, for example "Leo Brennan chose a vintage film camera rather than a record player for Sam's birthday (1 February 2026)" (`chat_en`). To remove a source completely (a chat the user wants forgotten), delete its line from `sources.jsonl` and build again with the same cache. No fact of that source remains; the other sessions come from the cache, and an occasion-grouping request is sent again only when the facts or occasions it lists have changed.

By default superseded facts stay in the graph, so that "Where did Maddie live before?" can be answered; this is also the default of our own pipeline. `--drop-superseded` leaves them out.

## AI-summarised memories

Building from raw chats keeps more than building from another AI's summary of them. A summary has already chosen what to keep: it keeps the latest state of things and leaves out how they changed, it can drop exact times, amounts and side details, and what it says is second-hand. The graphs of our example cases show this. All were built with GPT-6.1 Sol; the summaries were made from the same raw material by GPT-6.1 Sol, as described in `examples/cases/<case>/memory/README.md`:

| Case | Input | Entities / facts / edges | Facts with a time of day | Superseded facts |
|---|---|---|---|---|
| `chat_en` | raw chats | 308 / 773 / 525 | 5 | 4 |
| | flat memory list | 162 / 399 / 258 | 5 | 1 |
| | per-conversation summaries | 94 / 304 / 134 | 5 | 0 |
| | Markdown profile | 105 / 287 / 163 | 5 | 0 |
| `assistant_en` | raw chats and documents | 263 / 496 / 391 | 42 | 2 |
| | flat memory list | 178 / 426 / 281 | 34 | 0 |
| | per-conversation summaries | 126 / 329 / 201 | 33 | 0 |
| | JSON profile | 116 / 351 / 180 | 40 | 0 |
| `family_zh` | raw group chats and notes | 257 / 537 / 425 | 25 | 4 |
| | flat memory list | 116 / 285 / 172 | 16 | 0 |
| | per-conversation summaries | 106 / 288 / 156 | 21 | 0 |
| | Markdown profile | 115 / 274 / 160 | 20 | 0 |

The graphs built from summaries hold 37% to 86% as many facts as those built from the raw material, and fewer entities: things mentioned in passing are often gone. Times of day survive in some forms and not in others (the same 5 facts in `chat_en`, 33 to 40 against 42 in `assistant_en`, 16 to 21 against 25 in `family_zh`). Old values are mostly gone, because a summary keeps only the latest state: the raw-material graphs hold 2 to 4 superseded facts each, the summary graphs 0 or 1. And every fact from a summary is second-hand (`hearsay`) or a plan (`pending`), none `confirmed`, because the summary is another AI's account, not the user's words; in the raw-material graphs, 64% to 74% of the facts are `confirmed`.

Use the raw material when you have it. Use summaries when they are all you have (an export from another assistant, a profile the user wrote with an AI): import them as they are, keep them marked as second-hand, and mark a source `"confirmed": true` only when the user has confirmed its content. When both exist, build from the raw material, and add a summary to the same `sources.jsonl` only for periods or topics the raw material does not cover.

## Check the graph

**Validator.** `python -m gct.build.validate graph.json` checks the structure, ids and references, the vocabulary (entity kinds, fact modules, relations), states, `vis` lists, dates and the size of the owner's view (at most 2,048 units), and compiles the view with the reference compiler. It exits with code 1 on any error and lists warnings (duplicates, occasions whose names differ only in whose they are, unusual modules, facts without a source turn, unlinked entities).

**Read a sample.** We suggest reading a sample of every graph before you use it, in this order:

1. People with the same first name or the same title (two Alexes, two 王老师): one entity each, with their own facts.
2. Every `superseded` fact: is the replacing fact really a newer value? `superseded_by` points to it.
3. Twenty facts with relative times ("yesterday", "next Friday", "上周二"): compare the bracketed date with the date of the source.
4. Plans (`pending`): do they carry the date of the plan, not of the conversation?
5. Facts with no `prov` (the model named no source turn), facts on the owner that clearly belong to another entity, and the report's `facts_dropped_unknown_subject` (facts left out because their subject was unknown).
6. The module and kind counts: a graph with almost everything in `notes.general`, or with no events, usually means a source format problem.

**Retrieve.** Compile the graph ([COMPILE.md](COMPILE.md)) and ask it what an assistant would need on a typical day; look at the five best units of each request and at the facts you thought it would find.

## Cost

Measured with GPT-6.1 Sol, reasoning effort `medium`, two requests in flight. Requests and tokens are those whose answers make up the published graph of each example case; the time is that of the first build from an empty cache (the graphs were later rebuilt from the cache after rule fixes, which changes their counts slightly but not the extraction).

| Case | Material | Requests (extraction + occasion grouping) | Input tokens | Output tokens | Time of the first build | Graph: entities / facts / edges |
|---|---|---|---|---|---|---|
| `chat_en` | 44 chats, 543 turns | 20 (15 + 5) | 44,210 | 93,607 | 17 min | 308 / 773 / 525 |
| `assistant_en` | 43 assistant chats (191 turns), 17 documents | 22 (20 + 2) | 42,191 | 78,759 | 14 min | 263 / 496 / 391 |
| `family_zh` | 46 group chats (250 turns), 16 notes | 31 (21 + 10) | 55,997 | 89,809 | 16 min | 257 / 537 / 425 |

Output tokens dominate: 1.6 to 2.1 output tokens for every input token. An extraction request took between 50 seconds and a little over 3 minutes (median 74 to 123 seconds per case), an occasion-grouping request between 2 and 33 seconds. A rebuild from the cache sends only what changed: when the organisation rules changed, rebuilding sent only the occasion-grouping requests whose facts had changed (4, 0 and 6 requests for the three cases); two later rounds of fixes to the rules sent 2 more for `chat_en`, 4 more for `family_zh` and none for `assistant_en`.

## Common problems

| Problem | Cause and what to do |
|---|---|
| `stopped: HTTP 401` or `HTTP 403` | The key is missing or wrong. The build stops at once and sends nothing more. |
| `stopped: HTTP 404 ...` (model not found) | The model name is wrong, or the key has no access to it: OpenAI answers 404 for a model the key cannot use. Check `--model` and the key's project. The build stops at once. |
| `HTTP 400` / `422` in the request log, then `sources or batches failed` | The server rejected a field of the request, for example a reasoning effort it does not know. These are not retried; after three in a row the build stops. For a server without reasoning effort, pass `--effort ""`. |
| `stopped: ... quota` / `billing` | The account is out of credit. Run the same command again later; the cache keeps what was done. |
| HTTP 429 or 5xx errors, then success | Rate limits or a busy service. The build waits and retries each request up to `--retries` times (3 by default); lower `--concurrency` if it happens often. |
| `WARNING: ... sources or batches failed` | A request failed for good (for example the answer was not valid JSON after all retries). The graph is written without those sessions, the exit code is 1, and running the command again retries only them. |
| `error: source ...: turn N speaker ... is not a participant` | A turn's `speaker` is missing from `participants`, or the owner's name is spelled differently from `case.json`. Other problems with `case.json` or `sources.jsonl` are reported the same way, naming the source. |
| `warning: model ... is not an OpenAI model name but requests go to https://api.openai.com/v1` | You meant a local server: set `OPENAI_BASE_URL` (or `--base-url`) to its address ([BUILD_LOCAL.md](BUILD_LOCAL.md)). |
| Two people merged, or one person split | Check that names are spelled consistently in the sources. Merges come from shared names and aliases; splits from two different names for one person ("Ashby" and "Ashby High" stay two entities). |
| A short name sits on the wrong one of two people | When the model writes a person's short name as an alias of the full name, the alias stays on that person even if someone else has the same short name: in the `chat_en` graph built from the Markdown profile, Leo's friend Alex Moreau carries the alias "Alex" while Leo's brother is called Alex. A source added later that says only "Alex", with no role, can then be attached to either of them. Give such sources the full name, or check the person entities after adding them. |
| One occasion as two entities | When two sources name the same occasion after different people ("Dana's quarterly review" in Dana's message, "Ines's quarterly review" in Ines's own note), the build keeps two entities: different possessors usually mean different occasions ("Leo's birthday", "Sam's birthday"). The one whose facts give no date then gets the date of the note as its schedule date. `python -m gct.build.validate` warns about events, trips, appointments and projects whose names differ only in whose they are; merge such pairs in the sources (use one name) if they are the same. |
| A plan shows the conversation date | The time was too vague for a code ("one of these days"). The text keeps the words; add the date to the source if it matters. |
| `caller o sees N units ... at most 2048` | The case is too large for one view. Split it into several graphs (for example by year) or drop sources that do not matter. |
| A cancelled or moved plan still looks current | The cancellation named the plan differently from the plan itself. Look for the pair in the graph and correct the source if needed. |

## Options

`python -m gct.build --help` lists them all. The ones that matter most:

| Option | Default | Meaning |
|---|---|---|
| `--case DIR` | (required) | Case directory with `case.json` and `sources.jsonl` |
| `--sources FILE` | `<case>/sources.jsonl` | Another sources file (a path, or relative to the case directory) |
| `--out FILE` | (required) | Graph file to write |
| `--model NAME` | `gpt-6.1-sol` | Model name |
| `--api responses\|chat` | `responses` for `gpt-*` models, else `chat` | Wire format: Responses API or Chat Completions |
| `--effort LEVEL` | `medium` for `gpt-*` models, else none | Reasoning effort; `--effort ""` sends none |
| `--extra-body JSON` | none | Extra request fields, e.g. `'{"chat_template_kwargs": {"enable_thinking": false}}'` for a local model |
| `--cache DIR` | none | Cached answers and the request log (`calls.jsonl`); strongly recommended |
| `--save-raw DIR` | none | Keep every raw answer, including answers that were not valid JSON |
| `--concurrency N` | 2 | Requests in flight |
| `--retries N` | 3 | Extra attempts per request |
| `--no-retry-unknown` | off | Do not re-send a request whose connection broke after it was sent (it may have been billed) |
| `--max-calls N` | none | Stop sending after N requests |
| `--drop-superseded` | off | Leave out superseded facts |
| `--report FILE` | none | Write the build report |

## Next

Compile the graph and serve it: [COMPILE.md](COMPILE.md).
