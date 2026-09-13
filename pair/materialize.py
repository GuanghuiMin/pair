"""Write the optimizer's candidates as compression prompt directories.

    python -m pair.materialize --proposals B/propose/proposals.json --incumbent prompts/p0 --prefix prompts/p0_r1c

Candidate i becomes <prefix><i>/ with system_prompt.jinja copied from the incumbent and the two revised
templates. raw/ keeps the optimizer's output verbatim and RULES.json its rule -> counterexample map; the
templates the compressor loads have the citation tags and channel labels removed (they are bookkeeping,
not instructions). For a prefix-conditioned incumbent the fixed <downstream-agent-contract> block is
re-attached. One audit line per candidate reports headers, variables, leftover tags and leaked entities.
"""
import argparse
import json
import re
import shutil
from pathlib import Path

CITATION_TAGS = [
    re.compile(r"\s*[\[(]\s*Addresses[^\])]*[\])]\.?", re.I),
    re.compile(r"\s*\[(ERROR|BURDEN)s?:[^\]]*\]\.?", re.I),
    re.compile(r"\s*[\[(]\s*(error|burden)-\d+[^\])]*[\])]\.?", re.I),
    re.compile(r"\s*Cite\s+(error|burden)-\d+(?:\s*,\s*(?:error|burden)-\d+)*\.?", re.I),
    re.compile(r"\s*(ERROR|BURDEN) rule:\s*", re.I),
    re.compile(r"\b(ERROR|BURDEN)( CHECK)?:\s*"),
    re.compile(r"[ \t]*\[(ERROR|BURDEN)\s*[—-][^\]]*\][ \t]*"),
    re.compile(r"[ \t]*(\[R\d+\])+"),
    re.compile(r"^[ \t]*-[^\n]*\bERROR\b[^\n]*\bBURDEN\b[^\n]*\n", re.M),
    re.compile(r"\s*Addresses:[ \t]*(?:(?:error|burden)-\d+[ \t]*,?[ \t]*)+\.?", re.I),
    re.compile(r"[ \t]*\((?:ERROR|BURDEN)\s*/\s*(?:(?:error|burden)-\d+[ \t]*,?[ \t]*)+\)\.?"),
    re.compile(r"[ \t]*(?:(?:error|burden)-\d+[ \t]*[,;]?[ \t]*)+(?=\]|\n|$)"),
]
EMAIL = re.compile(r"[\w.]+@[\w.]+")
BLOCK = re.compile(r"<downstream-agent-contract>.*?</downstream-agent-contract>\s*", re.S)


def strip_citations(t: str) -> str:
    prev = None
    while prev != t:                      # tags nest; repeat until stable
        prev = t
        for rx in CITATION_TAGS:
            t = rx.sub("", t)
    t = re.sub(r"^\s*(?:-|\d+\.|- \[[ x]\])\s*\]\s*\n", "", t, flags=re.M)   # list items that held only a tag
    return re.sub(r"[ \t]+\n", "\n", t)


def headers(t: str) -> list[str]:
    return [l.strip() for l in t.split("\n") if l.strip().startswith("## ")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposals", required=True)
    ap.add_argument("--incumbent", required=True)
    ap.add_argument("--prefix", required=True, help="output directories <prefix>1, <prefix>2, ...")
    a = ap.parse_args()
    inc = Path(a.incumbent)
    inc_first = (inc / "first_summary.jinja").read_text()
    heads = headers(inc_first)
    m = BLOCK.search(inc_first)
    block = m.group(0) if m else ""
    packs = Path(a.proposals).parent / "packs.jsonl"
    pack_ids = {json.loads(l)["id"] for l in open(packs) if l.strip()} if packs.exists() else None

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
        (d / "RULES.json").write_text(json.dumps({"rules": c["rules"], "source": str(a.proposals)}, indent=1, ensure_ascii=False))
        first, update = strip_citations(c["first_summary"]), strip_citations(c["update_summary"])
        if block:
            assert "downstream-agent-contract" not in first + update, f"{d.name}: candidate reproduced the fixed block"
            first, update = block + first.lstrip("\n"), block + update.lstrip("\n")
        shutil.copy(inc / "system_prompt.jinja", d / "system_prompt.jinja")
        (d / "first_summary.jinja").write_text(first)
        (d / "update_summary.jinja").write_text(update)

        ok_headers = headers(first) == heads and headers(update) == heads and (not block or (first.startswith(block) and update.startswith(block)))
        ok_vars = "{{ history }}" in first and "{{ history }}" in update and "{{ prev_summary }}" in update
        cited = {x for r in c["rules"] for x in r.get("counterexample_ids", [])}
        unknown = len(cited - pack_ids) if pack_ids is not None else "?"
        leftover = len(re.findall(r"\b(ERROR|BURDEN)( CHECK)?:|\[R\d+\]|\b(error|burden)-\d+", first + update))
        removed = len(c["first_summary"]) + len(c["update_summary"]) - len(first) - len(update)
        print(f"{d.name}: rules={len(c['rules'])} unknown_citations={unknown} leftover_tags={leftover} headers_locked={ok_headers} "
              f"vars={ok_vars} emails={len(EMAIL.findall(first + update))} chars={len(first)}/{len(update)} stripped={removed}")


if __name__ == "__main__":
    main()
