import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from pair import llm

INSTRUCTIONS = """You revise the summarization prompt of a history-compression module used
inside a long-horizon tool-using agent. You are given the incumbent prompt
templates and analyses of real compression events. Each analysis was written
from the steps the agent actually took after that compression.

Rules:
1. Revise the summarization templates only. Every revised or added clause
   must cite the ids of the events whose analysis supports it.
2. Rules must be task-general: never copy task-instance entities (person or
   account names, emails, amounts, dates, answers) from the events into the
   templates.
3. Do not add guidance that no event analysis supports.
4. Do not invent environment capabilities, APIs, or behaviors not present in
   the evidence.
5. At run time the compressor sees only the conversation up to the compression
   point (and the previous summary, if any); it never sees what happens after
   that point. Every clause must be something the compressor can apply from
   that history alone. What the continuations did after the compression point
   can show what the summary should have contained, but must not be written
   as if the history already showed it.
6. Output the full revised first_summary and update_summary templates, plus
   a clause->event-id mapping. No scoring, no self-evaluation."""

EVENT_TYPES = {
    "pass": ("event_type \"pass\": from this state, continuations given the raw history passed more often than continuations "
             "given the summary. Shown: one continuation given the raw history that passed (raw_continuations) and one "
             "given the summary that failed (summary_continuations)."),
    "burden": ("event_type \"burden\": from this state, continuations given the summary needed more steps than continuations "
               "given the raw history. Shown: the continuation given the raw history with the fewest steps "
               "(raw_continuations) and the continuation given the summary with the most steps (summary_continuations)."),
}

ANALYSIS = (
    "Below are the incumbent summarization templates of a history-compression module and ONE compression event from a "
    "real agent run: the raw history segment the compressor consumed, the previous summary if any, the summary it "
    "produced, and two continuations run from the identical environment state: one given the raw history instead of "
    "the summary (raw_continuations) and one given the summary (summary_continuations). Each continuation is every step "
    "the agent took until the episode ended (reasoning, code, observation) and whether the episode passed. The field "
    "event_type says why this event was selected and how the two continuations were picked:\n"
    "- " + EVENT_TYPES["pass"] + "\n- " + EVENT_TYPES["burden"] + "\n\n"
    "TASK (analysis only, no proposals). Read both continuations step by step. Explain whether, and how, what the agent "
    "did differed between the two sides, and whether those differences depended on summary_produced. For each point "
    "where the two sides acted differently, quote the sentence of summary_produced the summary-side agent acted on (or "
    "the information it lacked), and state whether the raw segment established it, quoting the raw step; if not, say "
    "whether the summary added it, stated it more firmly than the raw segment (e.g. a tentative intention written as a "
    "decision or a required step), or dropped it. Only the raw segment and the previous summary are what the compressor "
    "saw; observations inside the continuations happened after the compression point and must not be cited as the raw "
    "segment. The passed field is the outcome; do not re-judge which side served the task better. Cite the side (raw or "
    "summary) and step numbers for every point you make. If the summary made no difference to what the agent did, say "
    "so. Use only this event.\n\n")

SYNTHESIS = (
    "Below are the incumbent summarization templates of a history-compression module and analyses of real compression "
    "events.\n\n"
    "How each analysis was produced: for every event, a separate call received the incumbent templates and that "
    "event's data (the raw history segment the compressor consumed, the previous summary if any, the summary it "
    "produced, the event type, and two continuations run from the identical environment state: one given the raw "
    "history instead of the summary, one given the summary, each with every step and whether it passed), together with "
    "exactly this instruction:\n<<<\n" + ANALYSIS.rstrip("\n") + "\n>>>\n"
    "So each analysis compares what the agent did on the two sides, traces each difference to a sentence of "
    "summary_produced (or to information it lacked), and says whether the raw segment established that sentence, or "
    "whether the summary added it, stated it more firmly, or dropped it. Here you are given, for every event, the event "
    "type, the summary the compressor produced, whether each continuation passed, and that analysis; the raw segments "
    "and the continuation steps are not repeated.\n\n"
    "TASK (analysis only, no proposals yet): from these analyses, work out what, if anything, should change in how the "
    "templates make the compressor write summaries. Cite the event ids and the steps quoted in the analyses for each "
    "point. The templates you will write next are read by the compressor at run time, which knows nothing about events, "
    "evidence or causation: guidance about analysing evidence belongs here, never in the templates. At run time the "
    "compressor sees only the conversation up to the compression point (and the previous summary, if any); it never "
    "sees what happens after that point, including anything the continuations did.\n\n")

BLOCK = "downstream-agent-contract"
SYNTHESIS_TOKENS = 9000
REVISION_TOKENS = 16000
STAGE_B_KEYS = ("id", "boundary_id", "event_type", "summary_produced", "raw_continuations", "summary_continuations")


def headers(template):
    return [l.strip() for l in template.split("\n") if l.strip().startswith("## ")]


def shared_context(inp):
    s = {"incumbent_templates": inp["incumbent_templates"]}
    if "agent_prompt_at_runtime" in inp:
        s["agent_prompt_at_runtime"] = inp["agent_prompt_at_runtime"]
    return s


def analysis_prompt(inp, event):
    return ANALYSIS + json.dumps({**shared_context(inp), "counterexample": event})


def stage_b_view(key, value):
    if key in ("raw_continuations", "summary_continuations"):
        return [{"passed": c["passed"]} for c in value]
    return value


def synthesis_prompt(inp, analyses):
    return SYNTHESIS + json.dumps({
        "instructions": INSTRUCTIONS, "incumbent_templates": inp["incumbent_templates"], "protocol": "r1",
        **({"agent_prompt_at_runtime": inp["agent_prompt_at_runtime"]} if "agent_prompt_at_runtime" in inp else {}),
        "counterexamples": [{**{k: stage_b_view(k, p[k]) for k in STAGE_B_KEYS if k in p},
                             "analysis": analyses.get(p["id"], "<stage A output>")} for p in inp["counterexamples"]]})


def revision_prompt(heads, has_block):
    return ("Now, based ONLY on your own analysis above, revise the two templates. " + INSTRUCTIONS +
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
            "or after; escape newlines inside strings): {\"rules\": [{\"clause\": str, "
            "\"event_ids\": [...]}], \"first_summary\": str, \"update_summary\": str}.")


def validate(c, heads):
    if not all(k in c for k in ("rules", "first_summary", "update_summary")):
        return "keys"
    if BLOCK in c["first_summary"] or BLOCK in c["update_summary"]:
        return "block_present"
    if headers(c["first_summary"]) != heads or headers(c["update_summary"]) != heads:
        return "headers"
    if "{{ history }}" not in c["first_summary"] or "{{ history }}" not in c["update_summary"] \
            or "{{ prev_summary }}" not in c["update_summary"]:
        return "vars"
    if not all(isinstance(r, dict) and r.get("event_ids") for r in c["rules"]):
        return "uncited"
    return None


def sidecar(out, suffix):
    return (out[:-5] if out.endswith(".json") else out) + suffix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--candidates", type=int, default=5)
    ap.add_argument("--analysis-workers", type=int, default=8)
    ap.add_argument("--analysis-tries", type=int, default=5)
    ap.add_argument("--reuse-analyses", default=None)
    ap.add_argument("--fill-invalid", action="store_true")
    a = ap.parse_args()
    inp = json.load(open(a.input))
    packs = inp["counterexamples"]
    heads = headers(inp["incumbent_templates"]["first_summary"])
    assert heads == headers(inp["incumbent_templates"]["update_summary"]), "incumbent templates disagree on headers"
    t2 = revision_prompt(heads, BLOCK in inp["incumbent_templates"]["first_summary"])
    client, model = llm.client(timeout=1200), llm.model_name()

    def request(msgs, tokens):
        r = client.chat.completions.create(model=model, messages=msgs, temperature=1.0,
                                           **(llm.completion_kwargs(tokens) if tokens else {}))
        return r.choices[0].message.content or "", r.choices[0].finish_reason

    def checked(msgs, tokens, label):
        for attempt in range(1, a.analysis_tries + 1):
            try:
                txt, fin = request(msgs, tokens)
                if txt.strip() and fin == "stop":
                    return txt
                print(f"{label} attempt {attempt}: {'empty' if not txt.strip() else 'finish=' + str(fin)}", flush=True)
            except Exception as e:
                print(f"{label} attempt {attempt}: {type(e).__name__}", flush=True)
        return None

    analyses_path = sidecar(a.out, ".analyses.json")
    if a.reuse_analyses or a.fill_invalid:
        analyses = json.load(open(a.reuse_analyses or analyses_path))
        assert set(analyses) == {p["id"] for p in packs}, "reused analyses do not match the events"
    else:
        analyses = {}
        with ThreadPoolExecutor(a.analysis_workers) as ex:
            futs = {ex.submit(checked, [{"role": "user", "content": analysis_prompt(inp, p)}], None, f"stage A {p['id']}"): p["id"]
                    for p in packs}
            for fut in as_completed(futs):
                analyses[futs[fut]] = fut.result()
                print(f"stage A {futs[fut]}: {'ok' if analyses[futs[fut]] else 'FAILED'}", flush=True)
        failed = [k for k, v in analyses.items() if not v]
        assert not failed, f"stage A failed: {failed}"
        json.dump(analyses, open(analyses_path, "w"), indent=1)

    turn1 = synthesis_prompt(inp, analyses)
    rejected = []

    def draw(i):
        msgs = [{"role": "user", "content": turn1}]
        analysis = checked(msgs, SYNTHESIS_TOKENS, f"candidate {i + 1} synthesis")
        if analysis is None:
            return i, {"invalid": True, "reason": "synthesis turn failed"}
        for attempt in range(3):
            txt, _ = request(msgs + [{"role": "assistant", "content": analysis}, {"role": "user", "content": t2}],
                             REVISION_TOKENS)
            try:
                cand = json.loads(txt[txt.find("{"):txt.rfind("}") + 1])
            except ValueError:
                print(f"candidate {i + 1} attempt {attempt + 1}: unparsable JSON", flush=True)
                rejected.append({"candidate": i + 1, "attempt": attempt + 1, "reason": "json", "text": txt})
                continue
            why = validate(cand, heads)
            print(f"candidate {i + 1} attempt {attempt + 1}: {'ok' if why is None else 'rejected (' + why + ')'}", flush=True)
            if why is None:
                for r in cand["rules"]:
                    r["counterexample_ids"] = r["event_ids"]
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
            json.dump(outs, open(a.out, "w"), indent=1)
    if rejected:
        rp = sidecar(a.out, ".rejected.json")
        old = json.load(open(rp)) if a.fill_invalid and os.path.exists(rp) else []
        json.dump(old + rejected, open(rp, "w"), indent=1)
    json.dump(outs, open(a.out, "w"), indent=1)
    print(f"{sum(1 for c in outs if c and not c.get('invalid'))}/{a.candidates} valid candidates -> {a.out}")


if __name__ == "__main__":
    main()
