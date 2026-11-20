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
"""The two prompts of the build: extraction (one call per batch of sessions) and occasion grouping (one call per person)."""

PROMPT_VERSION = "extract-1"

# entity kinds the extractor may use, the fact modules allowed for each kind and the typed relations (all in gct/vocab.json,
# except custom.mentions / custom.talks_with, which the graph declares in novel_relations)
KINDS = "person pet organization group place event asset project vehicle document service trip".split()
KIND_MODS = {
    "person": """identity.basic identity.demographics identity.residency identity.roles relationship.roles relationship.important_dates
relationship.obligations communication.style preferences.food preferences.entertainment preferences.travel preferences.shopping
preferences.gifts preferences.dressing preferences.spending health.conditions health.allergies health.mental_wellbeing health.medications
health.providers schedule.routines schedule.availability schedule.recurring work.role work.organization work.responsibilities
work.preferences work.collaborators education.enrollment education.activities finance.budgets notes.general""",
    "pet": "pet.identity pet.species pet.health pet.care_routine pet.diet notes.general",
    "organization": "organization.type organization.capabilities organization.people organization.locations organization.policies notes.general",
    "group": "group.members group.purpose group.schedule group.history notes.general",
    "place": "location.address location.amenities location.environment location.hours notes.general",
    "event": "event.schedule event.attendees event.location event.purpose event.outcome event.followup event.status notes.general",
    "asset": "asset.kind asset.state asset.value asset.ownership asset.lifecycle notes.general",
    "project": "project.objective project.status project.milestones project.history project.next_actions project.people project.artifacts notes.general",
    "vehicle": "vehicle.state vehicle.maintenance notes.general",
    "document": "document.type document.status notes.general",
    "service": "service.provider service.history service.capabilities notes.general",
    "trip": "trip.itinerary trip.travelers trip.history trip.preferences trip.status notes.general"}
KIND_MODS = {k: v.split() for k, v in KIND_MODS.items()}
REL_LIST = """parent_of child_of spouse_of partner_of sibling_of grandparent_of grandchild_of relative_of friend_of neighbor_of colleague_of
pet_owner_of member_of employed_by works_at student_at teacher_of client_of lives_at frequently_visits attends organizes owns uses
leads enrolled_in treated_by located_at part_of subscribes_to custom.mentions""".split()
NOVEL = {"custom.mentions": "The person talked about this entity in conversation; no more specific relation is known (person -> entity).",
         "custom.talks_with": "The owner has talked with this person; no more specific relation is known (person -> person)."}
STATES = ("confirmed", "pending", "hearsay")

EXTRACT = """You are the ingestion step of a personal memory system. You read chat sessions and write down, as a small knowledge graph, what a personal assistant should remember about the people in them.

PARTICIPANTS (speaker label: what it is):
{participants}

Each session is shown with its turns numbered T1, T2, ...; the session start time is called S (the calendar date is not shown). A session marked "note" is a note or document written by the speaker; each paragraph is one turn.

For EACH session write:
1. "entities": the specific things the facts are about or mention, other than the participants: people (family, friends, colleagues, named or only described), pets, organisations, groups, places, events and occasions, trips, possessions, creations, documents, long-running projects or activities, services or products the person uses. Fields:
   {{"id": "E1", "name": proper name if given, otherwise "<participant>'s <thing>" (e.g. "Melanie's daughter", "User's car"), "kind": one of {kinds}, "aliases": other ways the session refers to it (e.g. "my mom", "Mom", a nickname), "desc": <= 12 words, "of": the participant or entity id it belongs to or is related to, "role": one word for what it is to "of" (e.g. mother, daughter, friend, dog, employer, car, hometown, wedding), "rel": relation such that "<of> <rel> <entity>" is true, one of {rels}}}
   Do not create entities for abstract topics (a hobby in general, feelings, themes) or for general-knowledge things the person only asks about without a personal tie.
2. "facts": the personal information revealed, one piece of information per fact. Fields:
   {{"about": participant label or entity id: the person when the statement says what the person did, does, has, feels, likes, plans or is; a specific other entity when the statement describes that entity itself (its name, attributes, state, what happened to it),
    "module": profile module that fits the kind of "about": {kind_mods},
    "text": short third-person statement (<= 25 words) in the language of the session that names its subject and keeps the specific details (names, numbers, places, titles, brands, amounts; time words as said, e.g. "yesterday", "last year"),
    "when": the time the statement refers to, relative to S: "S" (during or around the session); "S-1d", "S-3d", "S-2w", "S-1m", "S-1y" (N days / weeks / months / years before S, any N); "S+1d", "S+1w", "S+1m" (after S, for plans); "S-Tue" (the nearest Tuesday before S, e.g. "last Tuesday", "on Tuesday" in the past), "S+Fri" (the nearest Friday after S, e.g. "this Friday", "on Friday" for a plan), "S-1w:Tue" / "S+1w:Fri" (Tuesday of the previous week / Friday of the next week, e.g. "上周二", "下周五"); a date stated in full ("2019", "2022-06", "2023-05-04"); a month-day without a year as "05-10" (read as the nearest such day), or with a year relative to S as "Y:06-08" (June 8 of this year), "Y-1:05-10" (e.g. "last year on May 10") or "Y:09" (September of this year); or "" when timeless or too vague,
    "state": "confirmed" (stated as true), "pending" (a plan, intention or something not yet done), or "hearsay" (uncertain or second-hand),
    "slot": a short attribute name when the fact gives the CURRENT value of a single-valued attribute that may change later (e.g. "home city", "job", "employer", "relationship status", "number of pets", "car model", "school"), else "",
    "turns": [numbers of the turns the statement comes from],
    "mentions": [entity ids, other than "about", that the statement names]}}
   Cover everything personal, including details mentioned in passing inside a request (e.g. "I need meal ideas for my three kids" -> the person has three kids). Include what the person did, experienced, bought, made, visited, learned, owns, likes or dislikes, plans, their health, work, family and friends, and what happened with them. A suggestion of the assistant only counts when the person accepts it or says they will use it.
   Do NOT record general knowledge, explanations by an assistant, task content without a personal tie (code, essay text, translations), greetings, thanks or small talk.
3. "relations": typed links between participants and entities that are stated or clearly implied: {{"src": ..., "rel": one of the relations above, "dst": ..., "turns": [...]}}.

{sessions}

Return JSON only: {{"sessions": [{{"id": "<session id>", "entities": [...], "facts": [...], "relations": [...]}}, ...]}} with one entry per session id above, in order. A session with nothing personal gets empty lists."""

GROUP = """You organise a personal memory graph. Below are short facts recorded about {who} from their chats and notes (each with a label and the date it refers to), and the events, trips and projects that already exist in the graph.

Group the facts that describe the SAME specific real-life occasion: something that happened, happens or is planned at a particular time (a race, a trip, a class, a party, a visit, a concert, a move, an interview, a doctor's appointment), or a long-running activity done on several occasions (a hobby project, a course, a volunteering role). For each group give
  {{"name": short name of the occasion, <= 8 words, naming whose it is (e.g. "Priya's trip to the pottery fair"), in the language of the facts,
    "kind": "event" | "trip" | "appointment" | "project",
    "date": when it happened / happens ("YYYY-MM-DD", "YYYY-MM" or ""),
    "existing": label of an existing event / trip / project below that is the same occasion, or "",
    "facts": {{fact label: its role in the occasion}}}}
with role one of "when" (date / time / schedule), "who" (who took part), "where" (place / route), "why" (purpose / reason), "what" (what happened / was done / was made), "plan" (planned / intended / status), "feeling" (how it felt / opinion of it), "followup" (what happened afterwards because of it).
A fact is in at most one group. Lasting traits, general preferences, general feelings, family, work role, identity and health facts are NOT occasions: leave them out of every group. A group may have a single fact when that fact clearly describes one occasion.
Also list the existing events / trips / projects below that are the same real-life occasion as each other (the same party or trip named in two ways): "same": [["X1", "X3"], ...], or [] when there are none.

FACTS
{facts}

EXISTING EVENTS / TRIPS / PROJECTS
{existing}

Return JSON only: {{"groups": [...], "same": [...]}}"""

MEMORY = """You are the import step of a personal memory system. Another AI assistant has kept the memories below about its user, {owner}. They are that assistant's own summaries, not the user's words. Turn them into a small knowledge graph.

Memories (M<number>, [date the memory was saved or the conversation it summarises took place]):
{items}

Return JSON only: {{"entities": [...], "facts": [...], "relations": [...]}}
1. "entities": specific people, pets, organisations, places, events, possessions, projects the memories name, other than {owner}:
   {{"id": "E1", "name": proper name if given, otherwise "{owner}'s <thing>", "kind": one of {kinds}, "aliases": [...], "desc": <= 12 words,
     "of": "OWNER" or an entity id it belongs to, "role": one word (sister, dog, employer, ...), "rel": relation such that "<of> <rel> <entity>" is true, one of {rels}}}
2. "facts": one piece of information per fact, as close to the memory as possible (do not add details the memory does not state):
   {{"about": "OWNER" or an entity id, "module": a module for that kind: {kind_mods},
     "text": short third-person statement (<= 25 words) in the language of the memories, naming its subject,
     "when": time the statement refers to, relative to the memory's date D: "D", "D-3d", "D-1w", "D-1m", "D-1y" (before it, any number), "D+1d", "D+1w" (after it, for plans), "D-Tue" / "D+Fri" (the nearest Tuesday before / Friday after D), a date stated in full ("2019", "2022-06", "2023-05-04"), a month-day without a year as "05-10", or "" when timeless or unknown,
     "state": "confirmed" (stated as a fact), "pending" (a plan or intention),
     "slot": short name of a single-valued attribute whose CURRENT value this gives, using these names when they fit: home city, employer, job, relationship status, diet, school, car model, number of pets; else "",
     "items": [memory numbers it comes from], "mentions": [other entity ids it names]}}
   Skip memories about how the assistant should answer (tone, format, language) unless they state a personal fact; record those as a fact about OWNER with module communication.style.
3. "relations": {{"src": ..., "rel": ..., "dst": ..., "items": [...]}} between OWNER and entities, or between entities."""

# appended to the extraction, memory and grouping prompts of a Chinese case (nothing is added for English)
LANG_NOTE = {"zh": "LANGUAGE: the material is in Chinese. Write every entity name, alias, description, fact text and occasion name in "
                   "Chinese, keeping names that the material writes in another script as they are."}
