import argparse
import json
import re
import shutil
from pathlib import Path

IDS = r"(?:event|boundary|error|burden)-\d+"
LABELS = r"(?:Supported by|Supports?|Evidence|counterexamples?|events?|event ids)"
BASE_TAGS = [
    r"\s*[\[(]\s*Addresses[^\])]*[\])]\.?",
    r"\s*\[(ERROR|BURDEN)s?:[^\]]*\]\.?",
    r"\s*[\[(]\s*(error|burden)-\d+[^\])]*[\])]\.?",
    r"\s*Cite\s+(error|burden)-\d+(?:\s*,\s*(?:error|burden)-\d+)*\.?",
    r"\s*(ERROR|BURDEN) rule:\s*",
    r"\b(ERROR|BURDEN)( CHECK)?:\s*",
    r"[ \t]*\[(ERROR|BURDEN)\s*[—-][^\]]*\][ \t]*",
    r"[ \t]*(\[R\d+\])+",
    r"^[ \t]*-[^\n]*\bERROR\b[^\n]*\bBURDEN\b[^\n]*\n",
    r"\s*Addresses:[ \t]*(?:(?:error|burden)-\d+[ \t]*,?[ \t]*)+\.?",
    r"[ \t]*\((?:ERROR|BURDEN)\s*/\s*(?:(?:error|burden)-\d+[ \t]*,?[ \t]*)+\)\.?",
    r"[ \t]*(?:(?:error|burden)-\d+[ \t]*[,;]?[ \t]*)+(?=\]|\n|$)",
]
BASE_FLAGS = [re.I, re.I, re.I, re.I, re.I, 0, 0, 0, re.M, re.I, 0, 0]
CITATION_TAGS = [
    re.compile(r"\s*\[" + LABELS + r"\s*:?\s*(?:" + IDS + r"\s*[,;]?\s*)+\]\.?", re.I),
    re.compile(r"\s*\(" + LABELS + r"\s*:?\s*(?:" + IDS + r"\s*[,;]?\s*)+\)\.?", re.I),
    re.compile(r"\s*\[(?:" + IDS + r"\s*[,;]?\s*)+\]\.?"),
    re.compile(r"\s*(?:Supported by|Supports|Evidence)\s*:\s*(?:" + IDS + r"\s*[,;]?\s*)+(?=\]|\n|$)", re.I),
] + [re.compile(p.replace("(error|burden)", "(error|burden|boundary|event)")
                .replace("(?:error|burden)", "(?:error|burden|boundary|event)"), f) for p, f in zip(BASE_TAGS, BASE_FLAGS)]
LEFTOVER = re.compile(r"\b(ERROR|BURDEN)( CHECK)?:|\[R\d+\]|\b(error|burden|boundary|event)-\d+|Supported by|Supports:|Evidence:"
                      r"|counterexample|\[[A-Za-z ]{0,30}:\s*\]")
EMAIL = re.compile(r"[\w.]+@[\w.]+")
BLOCK = re.compile(r"<downstream-agent-contract>.*?</downstream-agent-contract>\s*", re.S)


def strip_citations(t):
    prev = None
    while prev != t:
        prev = t
        for rx in CITATION_TAGS:
            t = rx.sub("", t)
    t = re.sub(r"^\s*(?:-|\d+\.|- \[[ x]\])\s*\]\s*\n", "", t, flags=re.M)
    return re.sub(r"[ \t]+\n", "\n", t)


def headers(t):
    return [l.strip() for l in t.split("\n") if l.strip().startswith("## ")]


def event_ids(proposals):
    base = Path(proposals).parent
    if (base / "packs.jsonl").exists():
        return {json.loads(l)["id"] for l in open(base / "packs.jsonl") if l.strip()}
    if (base / "proposer_input.json").exists():
        return {p["id"] for p in json.load(open(base / "proposer_input.json"))["counterexamples"]}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposals", required=True)
    ap.add_argument("--incumbent", required=True)
    ap.add_argument("--prefix", required=True)
    a = ap.parse_args()
    inc = Path(a.incumbent)
    inc_first = (inc / "first_summary.jinja").read_text()
    heads = headers(inc_first)
    m = BLOCK.search(inc_first)
    block = m.group(0) if m else ""
    known = event_ids(a.proposals)

    for i, c in enumerate(json.load(open(a.proposals)), 1):
        d = Path(f"{a.prefix}{i}")
        if not c or c.get("invalid"):
            print(f"{d.name}: invalid candidate, skipped")
            continue
        if d.exists():
            shutil.rmtree(d)
        (d / "raw").mkdir(parents=True)
        (d / "raw" / "first_summary.jinja").write_text(c["first_summary"])
        (d / "raw" / "update_summary.jinja").write_text(c["update_summary"])
        (d / "RULES.json").write_text(json.dumps({"rules": c["rules"], "source": Path(a.proposals).name}, indent=1))
        first, update = strip_citations(c["first_summary"]), strip_citations(c["update_summary"])
        assert "downstream-agent-contract" not in first + update, f"{d.name}: candidate reproduced the fixed block"
        first, update = block + first.lstrip("\n"), block + update.lstrip("\n")
        shutil.copy(inc / "system_prompt.jinja", d / "system_prompt.jinja")
        (d / "first_summary.jinja").write_text(first)
        (d / "update_summary.jinja").write_text(update)

        ok_headers = headers(first) == heads and headers(update) == heads and first.startswith(block) and update.startswith(block)
        ok_vars = "{{ history }}" in first and "{{ history }}" in update and "{{ prev_summary }}" in update
        cited = {x for r in c["rules"] for x in (r.get("event_ids") or r.get("counterexample_ids") or [])}
        unknown = len(cited - known) if known is not None else "?"
        leftover = len(LEFTOVER.findall(first + update))
        print(f"{d.name}: rules={len(c['rules'])} unknown_citations={unknown} leftover_tags={leftover} headers_locked={ok_headers} "
              f"vars={ok_vars} emails={len(EMAIL.findall(first + update))} chars={len(first)}/{len(update)}")


if __name__ == "__main__":
    main()
