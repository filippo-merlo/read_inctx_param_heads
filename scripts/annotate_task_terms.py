"""Task-descriptive words for every NQ-Swap question, from GPT, for ``lexical_task_heads.py``.

Yang et al. (2026) call a head a lexical task head when its top-10 tokens contain a word
that describes the task (their Section 2.2.2); their Appendix O has such word lists written
by LLMs, GPT-5.4 Mini among them. They do not publish that prompt, so the one below is ours,
with some of their human-written lists as examples. NQ-Swap has no fixed tasks, so each
distinct question of the correctly answered counterfactual items gets its own list.
Questions already in the output are skipped, so an interrupted run resumes; questions
that fail are reported and left for the next run.

The model is an Azure OpenAI deployment, reached through its chat-completions endpoint.

    .venv/bin/python scripts/annotate_task_terms.py --limit 5
    # needs AZURE_OPENAI_API_KEY, and AZURE_OPENAI_ENDPOINT or --endpoint
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECODED = ROOT / "data" / "decode" / "targets_all_heads_top10.npz"
OUT_DIR = ROOT / "data" / "task_terms"
API_VERSION = "2024-12-01-preview"

# Lists from the study's task_relation_dict.json (commit f81ef39), shown as examples.
INSTRUCTIONS = """\
Some attention heads in a language model write words that describe the task a prompt asks
for, not its answer. Given a question, name the task in a few words, then list single words
that describe that task or the kind of answer it asks for. Do not give the answer, and do
not use names of specific people, places, works or organisations. Lists from the study, for
reference:
country-capital: city, cities, capital, capitals
product-company: company, companies, firm, firms, manufacturers, corporations, corporation, organisations, organisation, institution, institutions
person-occupation: occupation, jobs, job, career
person-university: university, college, school, academy
university-year: year"""

SCHEMA = {
    "type": "object",
    "properties": {"task": {"type": "string"}, "terms": {"type": "array", "items": {"type": "string"}}},
    "required": ["task", "terms"],
    "additionalProperties": False,
}


def questions() -> list[str]:
    """Distinct questions of the correct counterfactual items, in order of first appearance."""
    import numpy as np

    sys.path.insert(0, str(ROOT))
    from heads_reading import load_run

    records = load_run("counterfactual")
    items = np.load(DECODED)["cf_items"]
    return list(dict.fromkeys(records[i]["question"] for i in items))


def annotate(client, url: str, question: str, attempts: int = 8) -> dict:
    body = {
        "messages": [
            {"role": "system", "content": INSTRUCTIONS},
            {"role": "user", "content": f"Question: {question}"},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "task_terms", "strict": True, "schema": SCHEMA},
        },
        "max_completion_tokens": 16384,
    }
    for attempt in range(attempts):
        wait = 2 ** attempt
        try:
            response = client.post(url, json=body)
        except Exception as error:  # network errors: retry
            if attempt == attempts - 1:
                raise
            print(f"retry ({type(error).__name__})", file=sys.stderr)
        else:
            if response.status_code == 200:
                data = response.json()
                answer = json.loads(data["choices"][0]["message"]["content"])
                return {"question": question, **answer, "model": data["model"], "response_id": data["id"]}
            try:
                message = response.json()["error"]["message"]
            except (ValueError, KeyError, TypeError):
                message = response.text[:500]
            if response.status_code != 429 and response.status_code < 500 or attempt == attempts - 1:
                raise RuntimeError(f"HTTP {response.status_code}: {message}")
            wait = max(wait, float(response.headers.get("retry-after", 0)))
            print(f"retry (HTTP {response.status_code})", file=sys.stderr)
        time.sleep(wait)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--deployment", default="gpt-5.5")
    parser.add_argument("--endpoint", default=os.environ.get("AZURE_OPENAI_ENDPOINT"))
    parser.add_argument("--limit", type=int, default=None, help="annotate at most this many new questions")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    key = os.environ.get("AZURE_OPENAI_API_KEY")
    if not key:
        sys.exit("AZURE_OPENAI_API_KEY is not set")
    if not args.endpoint:
        sys.exit("no endpoint: set AZURE_OPENAI_ENDPOINT or pass --endpoint")
    url = (f"{args.endpoint.rstrip('/')}/openai/deployments/{args.deployment}"
           f"/chat/completions?api-version={API_VERSION}")
    import httpx

    out = OUT_DIR / f"{args.deployment}.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(line)["question"] for line in out.read_text().splitlines()}
    todo = [q for q in questions() if q not in done][: args.limit]
    print(f"{len(done)} done, {len(todo)} to annotate -> {out}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    failed = []
    with httpx.Client(headers={"api-key": key}, timeout=300) as client, \
            out.open("a") as f, ThreadPoolExecutor(args.workers) as pool:

        def run(question: str) -> None:
            try:
                line = json.dumps(annotate(client, url, question), ensure_ascii=False)
            except Exception as error:
                failed.append(question)
                print(f"failed: {question!r}: {error}", file=sys.stderr)
                return
            with lock:
                f.write(line + "\n")
                f.flush()

        for n, _ in enumerate(pool.map(run, todo), 1):
            if n % 100 == 0:
                print(f"{n}/{len(todo)}")
    print(f"{len(todo) - len(failed)} annotated, {len(failed)} failed")


if __name__ == "__main__":
    main()
