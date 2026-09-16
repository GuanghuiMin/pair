"""Step 3 (input): turn retained boundaries into counterexamples for the optimizer.

    python -m pair.evidence --benchmark appworld|officebench|tau2 --boundaries B/boundaries.jsonl \
        --rollouts B/rollouts.jsonl --effects B/effects.jsonl --incumbent prompts/p0 --out B/propose \
        [--agent-prefix-from RUN_DIR] [--obs-chars N] [--keep BOUNDARY_ID ...]

One counterexample per retained boundary: the raw segment the compressor consumed, the previous
summary, the summary it produced, its local effect, and the settled facts of every PRE and POST
continuation (steps used, how it ended, what it submitted, how much re-work it did). Writes
packs.jsonl and proposer_input.json (instructions + incumbent templates + counterexamples).

The optimizer never sees the agent's own prompt unless the incumbent is prefix-conditioned, in
which case --agent-prefix-from attaches to every counterexample the prefix its compressor saw for
that task (`agent_prompt_seen_by_compressor`), so the summary can be judged against the task and
the agent's operating rules while the rules written against {{ agent_prompt }} must hold for every task.
"""
import argparse
import collections
import importlib
import json
from pathlib import Path

INSTRUCTIONS = """You revise the summarization prompt of a history-compression module used
inside a long-horizon tool-using agent. You are given the incumbent prompt
templates and a set of verified counterexamples. Each counterexample shows,
for one real compression event: the raw history segment the compressor
consumed, the summary it produced, and the SETTLED outcomes of fresh
continuations run from the identical environment state under (a) the raw
representation and (b) that summary — including what each side finally
submitted and how many actions each needed.

Rules:
1. Propose revisions to the summarization templates only. Every revised or
   added clause MUST cite at least one counterexample id it addresses.
2. Rules must be task-general: never copy task-instance entities (person or
   account names, emails, amounts, dates, answers) from the counterexamples
   into the templates.
3. Do not claim any single compression event caused a final outcome; the
   evidence shows local, settled divergences, not global attributions.
4. Do not invent environment capabilities, APIs, or behaviors not present in
   the evidence.
5. ERROR-channel counterexamples (wrong submissions) and BURDEN-channel
   counterexamples (extra actions) are different failure classes: address
   them with separate clauses; never trade one against the other in a
   single rule.
6. Output the full revised first_summary and update_summary templates, plus
   a rule->counterexample-id mapping. No scoring, no self-evaluation."""


def cut(text, n):
    text = text or ""
    return text if n is None or len(text) <= n else text[:n] + f" …[{len(text) - n} more chars]"


def continuation_facts(rollouts, bench):
    return [{"steps_used": len(r["steps"]), "terminal": bench.terminal(r), **bench.facts(r)}
            for r in sorted(rollouts, key=lambda r: r["draw"])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=("appworld", "officebench", "tau2"), required=True)
    ap.add_argument("--boundaries", required=True)
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--effects", required=True, help="output of pair.verifier")
    ap.add_argument("--incumbent", required=True, help="prompt directory being revised")
    ap.add_argument("--out", required=True)
    ap.add_argument("--agent-prefix-from", default=None, metavar="RUN_DIR",
                    help="prefix-conditioned incumbent: run whose trajectories supply the agent prefix")
    ap.add_argument("--obs-chars", type=int, default=None, help="truncate observations in the evidence to N chars")
    ap.add_argument("--keep", nargs="*", default=None, help="restrict to these boundary ids")
    a = ap.parse_args()
    bench = importlib.import_module(f"pair.benchmarks.{a.benchmark}.evidence")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    effects = [json.loads(l) for l in open(a.effects) if l.strip()]
    bounds = {b["boundary_id"]: b for b in map(json.loads, (l for l in open(a.boundaries) if l.strip()))}
    rolls = collections.defaultdict(lambda: {"PRE": [], "POST": []})
    for r in map(json.loads, (l for l in open(a.rollouts) if l.strip())):
        if not r.get("error"):
            rolls[r["boundary_id"]][r["arm"]].append(r)

    retained = [e for e in effects if e["retained"] and (a.keep is None or e["boundary_id"] in a.keep)]
    retained.sort(key=lambda e: (e["channel"], -(e["burden"] if e["channel"] == "burden" else e["hazard"])))
    cache, packs, n = {}, [], collections.Counter()
    for e in retained:
        b = bounds[e["boundary_id"]]
        prev, produced = bench.summaries(b, cache)
        n[e["channel"]] += 1
        packs.append({
            "id": f"{e['channel']}-{n[e['channel']]:02d}", "channel": e["channel"], "boundary_id": e["boundary_id"],
            "local_effect": {"pass_rate_gap_raw_minus_summary": e["hazard"], "extra_actions_under_summary": e["burden"]},
            "previous_summary": prev,
            **({"agent_prompt_seen_by_compressor": bench.agent_prefix(Path(a.agent_prefix_from), b["task_id"])}
               if a.agent_prefix_from else {}),
            "raw_segment_consumed": bench.segment(b, cache, a.obs_chars),
            "summary_produced": produced,
            "continuations": {"raw_representation": continuation_facts(rolls[e["boundary_id"]]["PRE"], bench),
                              "this_summary": continuation_facts(rolls[e["boundary_id"]]["POST"], bench)},
        })
    with open(out / "packs.jsonl", "w") as f:
        for p in packs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    inc = Path(a.incumbent)
    prompt = {"instructions": INSTRUCTIONS,
              "incumbent_templates": {k: (inc / f"{k}.jinja").read_text() for k in ("first_summary", "update_summary")},
              "counterexamples": packs}
    if a.agent_prefix_from:
        prompt["agent_prompt_at_runtime"] = {"note": bench.PREFIX_NOTE}
    blob = json.dumps(prompt, ensure_ascii=False)
    if not a.agent_prefix_from and packs:
        probe = bench.leak_probe(bounds[packs[0]["boundary_id"]], cache)
        assert probe not in blob, "the agent's own prompt leaked into the optimizer input"
    (out / "proposer_input.json").write_text(json.dumps(prompt, indent=1, ensure_ascii=False))
    print(f"{len(packs)} counterexamples ({n['error']} error, {n['burden']} burden); "
          f"proposer_input.json ~{len(blob) // 4 // 1000}K tokens -> {out}")
    for p in packs:
        le = p["local_effect"]
        print(f"  {p['id']:<10} {p['boundary_id']:<40} hazard={le['pass_rate_gap_raw_minus_summary']:<7} "
              f"burden={le['extra_actions_under_summary']:<6} segment={len(p['raw_segment_consumed'])} "
              f"PRE={[c['steps_used'] for c in p['continuations']['raw_representation']]} "
              f"POST={[c['steps_used'] for c in p['continuations']['this_summary']]}")


if __name__ == "__main__":
    main()
