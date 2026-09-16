"""Step 3 (optimizer): revise the compression prompt from the counterexamples.

    python -m pair.propose --input B/propose/proposer_input.json --out B/propose/proposals.json \
        [--candidates 5] [--analysis-workers 8] [--reuse-analyses B/propose/proposals.analyses.json] [--fill-invalid]

Stage A diagnoses one counterexample per call: what the two continuations did differently, and which
statements of the summary would change a continuation's actions, each judged against the task, the
agent's operating rules (for a prefix-conditioned incumbent) and the raw segment. The diagnoses are
written next to the output as `<out>.analyses.json`.

Stage B draws the candidates. Each candidate first groups the diagnoses by failure mechanism (turn 1),
then revises the two templates under the instructions with the skeleton locked (turn 2): section headers
verbatim and in order, Jinja variables kept, only the guidance text inside each section changed, every
rule citing a counterexample. A revision that fails validation is redrawn up to three times; refused
revisions are kept in `<out>.rejected.json`. For a prefix-conditioned incumbent the
<downstream-agent-contract> block is fixed: the optimizer omits it and pair.materialize re-attaches it.
Candidates are drawn independently at temperature 1.
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from pair import llm

ANALYSIS = (
    "Below are the incumbent summarization templates of a history-compression module and ONE verified "
    "counterexample: a real compression event with the raw history segment the compressor consumed, the summary "
    "it produced, the previous summary if any, and the settled outcomes of fresh continuations run from the "
    "identical environment state under (a) the raw representation and (b) that summary. The field "
    "agent_prompt_seen_by_compressor, when present, is the exact prefix the agent and the compressor both saw: "
    "it contains the numbered operating rules of the environment and the task statement.\n\n"
    "TASK (analysis only, no proposals). Judge the summary against three things: the task statement, the numbered "
    "operating rules, and the raw segment.\n"
    "1. State what the two sides did differently: final submissions, actions used, settled scores.\n"
    "2. Find the deviations in summary_produced that would change what a continuation does: a task predicate, "
    "filter or scope replaced or dropped; a target set listed incompletely; a requirement, blocker, answer or next "
    "step the history never established; completed work reopened; a fact altered. For each, quote the summary "
    "phrase, quote the raw evidence, and name what it violates: 'rule N' of the operating rules, the task "
    "statement, or the raw observation. Do not list wording differences that would not change the continuation's "
    "actions.\n"
    "3. Rank the deviations by how likely each is to account for the settled divergence, most likely first. When "
    "an operating rule or the raw evidence settles the question, state the conclusion plainly. Say 'cannot be "
    "determined' only when the settled outcomes leave several deviations equally possible.\n"
    "4. Channel: ERROR (wrong or missing submission) or BURDEN (extra actions).\n"
    "Use only this counterexample.\n\n")

SYNTHESIS = (
    "Below are the incumbent summarization templates of a history-compression module and verified "
    "counterexamples. Each counterexample was analysed separately against the task statement, the environment's "
    "operating rules and the raw history; you are given its channel, its local effect (settled gap between raw "
    "and summary continuations), the summary the compressor produced, and that analysis. The raw histories are "
    "not repeated here.\n\n"
    "TASK (analysis only, no proposals yet): group the counterexamples by failure mechanism: what the summary did "
    "(replaced or dropped a predicate, listed targets incompletely, invented a requirement or an answer, reopened "
    "completed work, altered a fact) and how that changed the continuation. For each group cite the counterexample "
    "ids, the operating rule or task clause involved, and the decisive summary phrases. Keep ERROR and BURDEN "
    "mechanisms separate. The templates you will write next are read by the compressor at run time, which knows "
    "nothing about counterexamples, evidence or causation: guidance about analysing evidence or attributing outcomes "
    "belongs here, never in the templates.\n\n")

BLOCK = "downstream-agent-contract"
DIAGNOSIS_TOKENS = 4000
SYNTHESIS_TOKENS = 9000
REVISION_TOKENS = 16000


def headers(template: str) -> list[str]:
    return [l.strip() for l in template.split("\n") if l.strip().startswith("## ")]


def revision_prompt(instructions: str, heads: list[str], has_block: bool) -> str:
    return ("Now, based ONLY on your own analysis above, revise the two templates. " + instructions +
            "\nADDITIONAL TERM: STRUCTURE IS LOCKED. Both templates must keep EXACTLY these section headers, "
            "verbatim and in this order, with no sections added, removed, renamed, or reordered:\n   " +
            "\n   ".join(heads) + "\n" +
            (f"   The <{BLOCK}>...</{BLOCK}> block at the top of each "
             "template is FIXED and will be re-attached mechanically: OMIT it from your output and start each "
             f"template at the first line after </{BLOCK}>; do not reproduce, paraphrase or "
             "reference it.\n" if has_block else "") +
            "   Change only the guideline text (bracketed criteria and instruction "
            "paragraphs). Preserve the jinja variables: first_summary uses {{ history }}; update_summary uses "
            "{{ history }} and {{ prev_summary }}.\nReply with ONE JSON object and NOTHING else (no prose before "
            "or after; escape newlines inside strings): {\"rules\": [{\"clause\": str, \"channel\": \"error|burden\", "
            "\"counterexample_ids\": [...]}], \"first_summary\": str, \"update_summary\": str}.")


def validate(c: dict, heads: list[str]) -> str | None:
    if not all(k in c for k in ("rules", "first_summary", "update_summary")):
        return "keys"
    if BLOCK in c["first_summary"] or BLOCK in c["update_summary"]:
        return "block_present"
    if headers(c["first_summary"]) != heads or headers(c["update_summary"]) != heads:
        return "headers"
    if "{{ history }}" not in c["first_summary"] or "{{ history }}" not in c["update_summary"] \
            or "{{ prev_summary }}" not in c["update_summary"]:
        return "vars"
    if not all(r.get("counterexample_ids") for r in c["rules"]):
        return "uncited"
    return None


def sidecar(out: str, suffix: str) -> str:
    return (out[:-5] if out.endswith(".json") else out) + suffix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--candidates", type=int, default=5)
    ap.add_argument("--analysis-workers", type=int, default=8, help="parallel Stage A calls")
    ap.add_argument("--reuse-analyses", default=None, metavar="ANALYSES_JSON",
                    help="skip Stage A and take the diagnoses from an earlier run")
    ap.add_argument("--fill-invalid", action="store_true",
                    help="redraw only the candidates marked invalid in an existing --out; keep the others")
    a = ap.parse_args()
    inp = json.load(open(a.input))
    packs = inp["counterexamples"]
    heads = headers(inp["incumbent_templates"]["first_summary"])
    assert heads == headers(inp["incumbent_templates"]["update_summary"]), "incumbent templates disagree on headers"
    has_block = BLOCK in inp["incumbent_templates"]["first_summary"]
    t2 = revision_prompt(inp["instructions"], heads, has_block)
    client, model = llm.client(timeout=1200), llm.model_name()

    def call(msgs, tokens):
        return client.chat.completions.create(model=model, messages=msgs, temperature=1.0,
                                              **llm.completion_kwargs(tokens)).choices[0].message.content

    # Stage A: one diagnosis per counterexample.
    shared = {"incumbent_templates": inp["incumbent_templates"]}
    if "agent_prompt_at_runtime" in inp:
        shared["agent_prompt_at_runtime"] = inp["agent_prompt_at_runtime"]
    if a.reuse_analyses:
        analyses = json.load(open(a.reuse_analyses))
        assert set(analyses) == {p["id"] for p in packs}, "reused diagnoses do not match the counterexamples"
    else:
        def diagnose(p):
            prompt = ANALYSIS + json.dumps({**shared, "counterexample": p}, ensure_ascii=False)
            return p["id"], call([{"role": "user", "content": prompt}], DIAGNOSIS_TOKENS)
        analyses = {}
        with ThreadPoolExecutor(a.analysis_workers) as ex:
            for fut in as_completed([ex.submit(diagnose, p) for p in packs]):
                pid, text = fut.result()
                analyses[pid] = text
                print(f"diagnosed {pid}", flush=True)
        json.dump(analyses, open(sidecar(a.out, ".analyses.json"), "w"), indent=1, ensure_ascii=False)

    # Stage B: candidates from the diagnoses and the summaries; the raw segments are not repeated.
    synthesis_input = {"instructions": inp["instructions"], "incumbent_templates": inp["incumbent_templates"],
                       **({"agent_prompt_at_runtime": inp["agent_prompt_at_runtime"]} if "agent_prompt_at_runtime" in inp else {}),
                       "counterexamples": [{"id": p["id"], "channel": p["channel"], "boundary_id": p["boundary_id"],
                                            "local_effect": p["local_effect"], "summary_produced": p["summary_produced"],
                                            "analysis": analyses[p["id"]]} for p in packs]}
    turn1 = SYNTHESIS + json.dumps(synthesis_input, ensure_ascii=False)
    rejected = []

    def draw(i):
        msgs = [{"role": "user", "content": turn1}]
        analysis = call(msgs, SYNTHESIS_TOKENS)
        for attempt in range(3):
            txt = call(msgs + [{"role": "assistant", "content": analysis}, {"role": "user", "content": t2}], REVISION_TOKENS)
            try:
                cand = json.loads(txt[txt.find("{"):txt.rfind("}") + 1])
            except ValueError:
                print(f"candidate {i + 1} attempt {attempt + 1}: unparsable JSON", flush=True)
                rejected.append({"candidate": i + 1, "attempt": attempt + 1, "reason": "json", "text": txt})
                continue
            why = validate(cand, heads)
            print(f"candidate {i + 1} attempt {attempt + 1}: {'ok' if why is None else 'rejected (' + why + ')'}", flush=True)
            if why is None:
                cand["analysis_turn"] = analysis
                return i, cand
            rejected.append({"candidate": i + 1, "attempt": attempt + 1, "reason": why, "text": txt})
        return i, {"invalid": True, "analysis_turn": analysis}

    outs, todo = [None] * a.candidates, list(range(a.candidates))
    if a.fill_invalid and os.path.exists(a.out):
        outs = json.load(open(a.out))
        assert len(outs) == a.candidates, "--fill-invalid: the existing output has a different number of candidates"
        todo = [i for i, c in enumerate(outs) if not c or c.get("invalid")]
    with ThreadPoolExecutor(max(1, len(todo))) as ex:
        for fut in as_completed([ex.submit(draw, i) for i in todo]):
            i, c = fut.result()
            outs[i] = c
    if rejected:
        json.dump(rejected, open(sidecar(a.out, ".rejected.json"), "w"), indent=1, ensure_ascii=False)
    json.dump(outs, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"{sum(1 for c in outs if not c.get('invalid'))}/{a.candidates} valid candidates -> {a.out}")


if __name__ == "__main__":
    main()
