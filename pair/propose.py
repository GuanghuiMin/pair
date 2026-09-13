"""Step 3 (optimizer): revise the compression prompt from the counterexamples.

    python -m pair.propose --input B/propose/proposer_input.json --out B/propose/proposals.json [--candidates 5]

Two turns per candidate. Turn 1 analyses every counterexample: what the two continuations did
differently and what in the summary accounts for it. Turn 2 revises the two templates under the
instructions with the skeleton locked: section headers verbatim and in order, Jinja variables kept,
only the guidance text inside each section changed, every rule citing a counterexample. A candidate
that fails validation is redrawn up to three times. For a prefix-conditioned incumbent the
<downstream-agent-contract> block is fixed: the optimizer omits it and pair.materialize re-attaches it.
Candidates are drawn independently at temperature 1.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from pair import llm

T1 = ("Below are the incumbent summarization templates of a history-compression module, and "
      "verified counterexamples. Each counterexample shows one real compression event: the raw "
      "history the compressor consumed, the summary it produced, and settled outcomes of fresh "
      "continuations from the identical state under (a) the raw representation and (b) that summary.\n\n"
      "TASK (analysis only, no proposals yet): for EACH counterexample, observe carefully and explain: "
      "what did the two sides end up doing differently, and what in the summary (present, absent, or "
      "phrased) accounts for the divergence? Quote the specific evidence. Be concrete and exhaustive.\n\n")

BLOCK = "downstream-agent-contract"
ANALYSIS_TOKENS = 9000
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--candidates", type=int, default=5)
    a = ap.parse_args()
    inp = json.load(open(a.input))
    heads = headers(inp["incumbent_templates"]["first_summary"])
    assert heads == headers(inp["incumbent_templates"]["update_summary"]), "incumbent templates disagree on headers"
    has_block = BLOCK in inp["incumbent_templates"]["first_summary"]
    t2 = revision_prompt(inp["instructions"], heads, has_block)
    client, model = llm.client(timeout=1200), llm.model_name()

    def draw(i):
        msgs = [{"role": "user", "content": T1 + json.dumps(inp, ensure_ascii=False)}]
        analysis = client.chat.completions.create(model=model, messages=msgs, temperature=1.0,
                                                  **llm.completion_kwargs(ANALYSIS_TOKENS)).choices[0].message.content
        for attempt in range(3):
            txt = client.chat.completions.create(
                model=model, temperature=1.0, **llm.completion_kwargs(REVISION_TOKENS),
                messages=msgs + [{"role": "assistant", "content": analysis}, {"role": "user", "content": t2}]).choices[0].message.content
            try:
                cand = json.loads(txt[txt.find("{"):txt.rfind("}") + 1])
            except ValueError:
                print(f"candidate {i + 1} attempt {attempt + 1}: unparsable JSON", flush=True)
                continue
            why = validate(cand, heads)
            print(f"candidate {i + 1} attempt {attempt + 1}: {'ok' if why is None else 'rejected (' + why + ')'}", flush=True)
            if why is None:
                cand["analysis_turn"] = analysis
                return i, cand
        return i, {"invalid": True, "analysis_turn": analysis}

    outs = [None] * a.candidates
    with ThreadPoolExecutor(a.candidates) as ex:
        for fut in as_completed([ex.submit(draw, i) for i in range(a.candidates)]):
            i, c = fut.result()
            outs[i] = c
    json.dump(outs, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"{sum(1 for c in outs if not c.get('invalid'))}/{a.candidates} valid candidates -> {a.out}")


if __name__ == "__main__":
    main()
