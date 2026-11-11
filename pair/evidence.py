import argparse
import collections
import importlib
import json
from pathlib import Path

from pair import halving
from pair.propose import INSTRUCTIONS

CONTINUATION_OBS_CHARS = 20000


def cut(text, n):
    text = text or ""
    return text if n is None or len(text) <= n else text[:n] + f"...[{len(text) - n} more characters]"


def pick(channel, pre, post):
    if channel == "error":
        raw = next(r for r in pre if r["success"])
        summ = next(r for r in post if not r["success"])
    elif channel == "burden":
        raw = min(pre, key=lambda r: (len(r["steps"]), r["draw"]))
        summ = max(post, key=lambda r: (len(r["steps"]), -r["draw"]))
    else:
        raise ValueError(channel)
    return raw, summ


def continuation(r, bench, obs_chars):
    return {"steps": [{"step": i + 1, **{k: cut(v, obs_chars) if k == "observation" else v for k, v in bench.step(s).items()}}
                      for i, s in enumerate(r["steps"])],
            "passed": bool(r["success"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=("appworld", "officebench", "tau2"), required=True)
    ap.add_argument("--boundaries", required=True)
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--effects", required=True)
    ap.add_argument("--incumbent", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=halving.ROUNDS)
    ap.add_argument("--agent-prefix-from", default=None)
    ap.add_argument("--obs-chars", type=int, default=None)
    ap.add_argument("--continuation-obs-chars", type=int, default=CONTINUATION_OBS_CHARS)
    ap.add_argument("--keep", nargs="*", default=None)
    a = ap.parse_args()
    bench = importlib.import_module(f"pair.benchmarks.{a.benchmark}.evidence")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    effects = [json.loads(l) for l in open(a.effects) if l.strip()]
    bounds = {b["boundary_id"]: b for b in map(json.loads, (l for l in open(a.boundaries) if l.strip()))}
    rolls = collections.defaultdict(lambda: {"PRE": [], "POST": []})
    for r in map(json.loads, (l for l in open(a.rollouts) if l.strip())):
        if not r.get("error") and r["draw"] < a.rounds:
            rolls[r["boundary_id"]][r["arm"]].append(r)

    retained = [e for e in effects if e["retained"] and (a.keep is None or e["boundary_id"] in a.keep)]
    retained.sort(key=lambda e: (e["channel"], -(e["burden"] if e["channel"] == "burden" else e["hazard"])))
    cache, packs, picks = {}, [], []
    for k, e in enumerate(retained, 1):
        b = bounds[e["boundary_id"]]
        prev, produced = bench.summaries(b, cache)
        pre = sorted(rolls[e["boundary_id"]]["PRE"], key=lambda r: r["draw"])
        post = sorted(rolls[e["boundary_id"]]["POST"], key=lambda r: r["draw"])
        raw, summ = pick(e["channel"], pre, post)
        event_type = "pass" if e["channel"] == "error" else "burden"
        packs.append({
            "id": f"event-{k:02d}", "boundary_id": e["boundary_id"], "event_type": event_type,
            "previous_summary": prev,
            **({"agent_prompt_seen_by_compressor": bench.agent_prefix(Path(a.agent_prefix_from), b["task_id"])}
               if a.agent_prefix_from else {}),
            "raw_segment_consumed": bench.segment(b, cache, a.obs_chars),
            "summary_produced": produced,
            "raw_continuations": [continuation(raw, bench, a.continuation_obs_chars)],
            "summary_continuations": [continuation(summ, bench, a.continuation_obs_chars)],
        })
        picks.append({"id": f"event-{k:02d}", "boundary_id": e["boundary_id"], "event_type": event_type,
                      "channel": e["channel"], "raw_draw": raw["draw"], "raw_passed": bool(raw["success"]),
                      "raw_steps": len(raw["steps"]), "summary_draw": summ["draw"],
                      "summary_passed": bool(summ["success"]), "summary_steps": len(summ["steps"])})
    with open(out / "packs.jsonl", "w") as f:
        for p in packs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(out / "selected_continuations.jsonl", "w") as f:
        for p in picks:
            f.write(json.dumps(p) + "\n")

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
    n = collections.Counter(p["event_type"] for p in packs)
    print(f"{len(packs)} events ({n['pass']} pass, {n['burden']} burden); "
          f"proposer_input.json ~{len(blob) // 4 // 1000}K tokens -> {out}")
    for p in picks:
        print(f"  {p['id']} {p['boundary_id']:<40} {p['event_type']:<6} raw draw {p['raw_draw']} "
              f"({p['raw_steps']} steps, passed={p['raw_passed']})  summary draw {p['summary_draw']} "
              f"({p['summary_steps']} steps, passed={p['summary_passed']})")


if __name__ == "__main__":
    main()
