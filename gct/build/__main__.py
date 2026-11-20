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
"""python -m gct.build --case DIR --out graph.json [options]   (docs/BUILD_GRAPH.md, docs/BUILD_LOCAL.md)"""
import argparse
import json
import os
import sys
from pathlib import Path

from .case import CaseError
from .llm import LLM, StopBuild, address_warning
from .pipeline import build_graph
from .validate import check


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m gct.build", description="Build a HooRii Mind graph from a case directory "
                                 "(case.json + sources.jsonl) with an LLM behind an OpenAI-compatible API. The API key and "
                                 "address come from OPENAI_API_KEY and OPENAI_BASE_URL.")
    ap.add_argument("--case", required=True, help="case directory holding case.json and sources.jsonl")
    ap.add_argument("--sources", default=None, help="read this sources file instead of <case>/sources.jsonl (e.g. an "
                    "AI-memory variant; a path, or relative to the case directory)")
    ap.add_argument("--out", required=True, help="graph file to write")
    ap.add_argument("--model", default="gpt-6.1-sol", help="model name (default gpt-6.1-sol)")
    ap.add_argument("--api", choices=("responses", "chat"), default=None,
                    help="wire format: the Responses API or Chat Completions (default: responses for gpt-* models, else chat)")
    ap.add_argument("--base-url", default=None, help="API address (default: OPENAI_BASE_URL, else https://api.openai.com/v1)")
    ap.add_argument("--effort", default=None, help='reasoning effort (default: medium for gpt-* models, else not sent; '
                    '--effort "" sends none, for servers that reject the field)')
    ap.add_argument("--temperature", type=float, default=None, help="sampling temperature (default: not sent)")
    ap.add_argument("--max-output-tokens", type=int, default=None, help="output limit per call (default: not sent)")
    ap.add_argument("--json-mode", action="store_true", help="ask the server for a JSON object answer (response_format)")
    ap.add_argument("--extra-body", default=None, help='extra request fields as JSON, e.g. \'{"chat_template_kwargs": '
                    '{"enable_thinking": false}}\'')
    ap.add_argument("--cache", default=None, help="directory for cached answers and the call log; re-running resumes from it")
    ap.add_argument("--save-raw", default=None, help="directory to keep every raw answer")
    ap.add_argument("--concurrency", type=int, default=2, help="calls in flight (default 2)")
    ap.add_argument("--retries", type=int, default=3, help="extra attempts per call after an error or an invalid answer (default 3)")
    ap.add_argument("--no-retry-unknown", action="store_true", help="do not re-send a call whose connection broke after sending")
    ap.add_argument("--max-calls", type=int, default=None, help="stop sending after this many requests")
    ap.add_argument("--timeout", type=float, default=900, help="seconds to wait for the server to start answering or to send "
                    "more of a streamed answer (default 900); a long answer may take more time in total")
    ap.add_argument("--drop-superseded", action="store_true", help="leave out facts replaced by a newer value")
    ap.add_argument("--no-occasions", action="store_true", help="skip the occasion-grouping calls")
    ap.add_argument("--date-lang", choices=("en", "zh"), default="en", help="language of the dates written after fact values")
    ap.add_argument("--report", default=None, help="write the build report (counts, calls, tokens) to this JSON file")
    a = ap.parse_args(argv)
    log = lambda m: print(m, file=sys.stderr, flush=True)  # noqa: E731
    os.umask(os.umask(0o077) | 0o077)                     # graph, cache and raw answers: readable by you only
    for p in (a.out, a.report):                           # fail now, not after an hour of calls
        try:
            if p:
                Path(p).parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            ap.error(f"cannot create the directory of {p}: {e}")
    effort = a.effort if a.effort is not None else ("medium" if a.model.startswith("gpt-") else None)
    try:
        extra = json.loads(a.extra_body) if a.extra_body else None
    except ValueError as e:
        ap.error(f"--extra-body is not JSON: {e}")
    llm = LLM(model=a.model, api=a.api, base_url=a.base_url, effort=effort or None, temperature=a.temperature,
              max_output_tokens=a.max_output_tokens, json_mode=a.json_mode, cache=a.cache, save_raw=a.save_raw,
              max_calls=a.max_calls, retries=a.retries, retry_unknown=not a.no_retry_unknown, timeout=a.timeout,
              extra_body=extra, concurrency=a.concurrency)
    if address_warning(llm):
        log(address_warning(llm))
    try:
        world, report = build_graph(a.case, llm, sources=a.sources, keep_history=not a.drop_superseded, occasions=not a.no_occasions,
                                    date_lang=a.date_lang, workers=a.concurrency, log=log)
    except StopBuild as e:
        log(f"stopped: {e}")
        return 2
    except (CaseError, FileNotFoundError) as e:
        log(f"error: {e}")
        return 2
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(world, ensure_ascii=False, indent=1), encoding="utf-8")
    err, warn, summary = check(world)
    report["validate"] = {"errors": err, "warnings": len(warn)}
    if a.report:
        Path(a.report).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"wrote {a.out}: {report['counts']['entities']} entities, {report['counts']['facts']} facts, "
        f"{report['counts']['edges']} edges; {report['llm']['calls']} requests, {report['llm']['cached']} cached, "
        f"{report['llm']['input_tokens']} input / {report['llm']['output_tokens']} output tokens; "
        f"{len(err)} validation errors, {len(warn)} warnings")
    if report["failed"]:
        log(f"WARNING: {len(report['failed'])} sources or batches failed and are missing from the graph: {report['failed']}")
    return 1 if (err or report["failed"]) else 0


if __name__ == "__main__":
    sys.exit(main())
