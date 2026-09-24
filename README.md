# PAIR: Prompt Adaptation from Verified Compaction Boundaries

This repo contains the code for PAIR, a method that adapts the summarization prompt of a history-compression module inside a long-horizon tool-using agent. Instead of scoring whole trajectories, PAIR locates the individual compaction events that hurt the agent, verifies each one with counterfactual continuations, and lets an optimizer LLM revise the compression prompt against that evidence.

## 🔗 Quick Links
- [PAIR: Prompt Adaptation from Verified Compaction Boundaries](#pair-prompt-adaptation-from-verified-compaction-boundaries)
  - [🔗 Quick Links](#-quick-links)
  - [Install Requirements](#install-requirements)
  - [Benchmark Setup](#benchmark-setup)
  - [Repository Layout](#repository-layout)
  - [Prompt Adaptation Pipeline](#prompt-adaptation-pipeline)
    - [Step 1: Collecting compressed trajectories](#step-1-collecting-compressed-trajectories)
    - [Step 2: Verifying adverse compaction boundaries](#step-2-verifying-adverse-compaction-boundaries)
    - [Step 3: Adapting the compression prompt](#step-3-adapting-the-compression-prompt)
    - [Step 4: Selecting the adapted prompt](#step-4-selecting-the-adapted-prompt)
  - [Evaluation](#evaluation)
  - [Other Benchmarks](#other-benchmarks)
  - [Bugs or Questions?](#bugs-or-questions)

## Install Requirements

**Step 1**: Install the package and its dependencies (Python 3.10 or newer):

```bash
cd pair
pip install -r requirements.txt
pip install -e .
```

**Step 2**: Copy the environment template and edit it:

```bash
cp env.example.sh env.sh
```

Put your OpenAI API key in `env.sh` as `OPENAI_API_KEY`; the agent, the compressor and the optimizer all read it from there (`ACON_VLLM_API_KEY` and `PAIR_COMPRESSOR_API_KEY` default to it). `env.sh` is git-ignored, so the key never enters the repository. In the same file set the model names (`PAIR_AGENT_MODEL` for the agent, `PAIR_COMPRESSOR_MODEL` for the compressor and the optimizer), `PAIR_RUNS` (where runs are written) and the paths of the benchmark harnesses. Then load it in every shell you work from:

```bash
source env.sh
```

## Benchmark Setup

PAIR does not ship an agent; the agents, environments and evaluators come from the benchmarks' own harnesses:

- **AppWorld** and **OfficeBench** run through a patched checkout of [ACON](https://github.com/microsoft/acon). Apply `third_party/acon.patch`, install AppWorld, and export OfficeBench's task tree to a read-only directory.
- **OfficeBench** runs each task in a container built from `docker/officebench/Dockerfile`.
- **tau2-bench** (retail) is used unmodified in its own Python environment.

The exact commands are in [third_party/README.md](third_party/README.md).

## Repository Layout

```
pair/compressor.py        summary compressor driven by a prompt directory (system_prompt / first_summary / update_summary)
pair/halving.py           Step 2: successive-halving allocation of PRE/POST continuations
pair/verifier.py          Step 2: outcome hazard and interaction burden per boundary, retention rule
pair/evidence.py          Step 3: retained boundaries -> counterexamples for the optimizer
pair/propose.py           Step 3: optimizer LLM, one diagnosis per counterexample, five candidates, skeleton locked
pair/materialize.py       Step 3: candidates -> prompt directories
pair/fit_tasks.py         Step 4: the 12 longest training trajectories
pair/selection.py         Step 4: Pass^2, ties by fewer steps
pair/report.py            pass rate, Pass^k, Pass@k over evaluation runs
pair/benchmarks/<name>/   run.py (agent runner), boundaries / rollouts (Steps 1-2), evidence.py (benchmark-specific rendering)
prompts/p0                the original structured compression prompt P0 (history-only start)
prompts/p0_prefix         P0 conditioned on the agent's own prefix ({{ agent_prompt }}) (prefix-conditioned start)
prompts/<benchmark>/      the adapted prompts selected by the pipeline, one per start: history/ and prefix/
output/<benchmark>/       held-out evaluation runs of those prompts, <start>/s{1,2,3}/ (run_records.jsonl + trajectories/)
data/tasklists            AppWorld and OfficeBench train / test task lists (tau2 uses its own splits)
scripts/pipeline_*.sh     the four steps in order, one script per benchmark
```

`prompts/appworld`, `prompts/officebench` and `prompts/tau2_retail` hold the prompts that came out of the pipeline for
each benchmark (`history/` adapted from `prompts/p0`, `prefix/` from `prompts/p0_prefix`); they are drop-in replacements
for the start prompts. `output/` holds the three-seed held-out runs behind the reported numbers, in the runner's own
layout, so any row can be recomputed with `pair.report`. Machine-specific paths inside the OfficeBench trajectories are
replaced by `<OB_ROOT>` and `<OB_CANONICAL_TASKS>`.

Every runner writes the same layout: `<run>/run_records.jsonl` with one line per task (`task_id`, `success`, `num_steps`, `n_compressions`, ...) and `<run>/trajectories/<task_id>.json` with the steps and the compaction events. Every command is resumable and skips work that is already recorded.

## Prompt Adaptation Pipeline

The commands below adapt `prompts/p0` on AppWorld at a context budget of 4096 tokens. `scripts/pipeline_appworld.sh` runs them in sequence; `scripts/pipeline_officebench.sh` and `scripts/pipeline_tau2.sh` are the same pipeline on the other two benchmarks. To adapt the prefix-conditioned start instead, set `START=prompts/p0_prefix`.

```bash
RUNS=$PAIR_RUNS/appworld
B=$RUNS/boundaries/p0_w4096
```

### Step 1: Collecting compressed trajectories

Run every training task once with the compressor `C_{P_0}` under the target context budget, then list every compaction boundary of every trajectory that compacted at least once. Trajectories are kept regardless of their final outcome: a successful trajectory may still contain outcome-degrading compactions, and interaction burden occurs in successful and failed executions alike.

```bash
python -m pair.benchmarks.appworld.run --tasks data/tasklists/appworld_train.jsonl --out $RUNS/p0_w4096_s1 \
    --method prompts/p0 --window 4096 --seed 1 --workers 20
python -m pair.benchmarks.appworld.boundaries --run $RUNS/p0_w4096_s1 --out $B
```

`$B/boundaries.jsonl` has one line per compaction: the trajectory, the compaction index `t`, the step it fired before, and the range of raw steps it consumed.

### Step 2: Verifying adverse compaction boundaries

For every boundary, run continuations from the pre-compaction context (PRE) and from the post-compaction context (POST). Both arms restore the environment by replaying the recorded actions, then let the frozen agent continue with no further compression until it finishes or exhausts the remaining step budget; the end state is graded by the benchmark's evaluator. Continuations are allocated by three-round successive halving (`pair/halving.py`): every boundary first receives one PRE/POST pair; after each round, the half of that round's boundaries with the strongest current evidence of harm, `max(mean(pass PRE - pass POST) / 0.5, mean(steps POST - steps PRE) / 5)`, receives the next pair, up to three pairs. On average a boundary costs 3.5 continuations instead of 6.

```bash
python -m pair.benchmarks.appworld.rollouts --boundaries $B/boundaries.jsonl --out $B --rounds 3 --workers 20
python -m pair.verifier --rollouts $B/rollouts.jsonl --out $B/effects.jsonl --hazard-min 0.5 --burden-min 5
```

The verifier computes the empirical outcome hazard `H_t` (PRE pass rate minus POST pass rate) and the interaction burden `B_t` (mean POST steps minus mean PRE steps) and retains a boundary that completed all three rounds if `H_t >= 0.5` or `B_t >= 5`. Retained boundaries are labelled `error` or `burden`; the label selects the evidence channel shown to the optimizer.

### Step 3: Adapting the compression prompt

Each retained boundary becomes one counterexample: the raw segment the compressor consumed, the previous summary, the summary it produced, and the settled facts of the PRE and POST continuations. The optimizer works in two stages. It first diagnoses each counterexample in a separate call, judging the summary against the task, the agent's operating rules and the raw segment, and ranking the deviations that would change a continuation's actions. It then groups the diagnoses by failure mechanism and revises the two summary templates with the section structure, required fields, compression scope and output format fixed. Only the guidance inside each section changes, and every new clause cites the counterexamples it addresses. Diagnoses are saved next to the proposals and can be reused with `--reuse-analyses`; `--fill-invalid` redraws only candidates that failed validation.

```bash
python -m pair.evidence --benchmark appworld --boundaries $B/boundaries.jsonl --rollouts $B/rollouts.jsonl \
    --effects $B/effects.jsonl --incumbent prompts/p0 --out $B/propose_r1
python -m pair.propose --input $B/propose_r1/proposer_input.json --out $B/propose_r1/proposals.json --candidates 5
python -m pair.materialize --proposals $B/propose_r1/proposals.json --incumbent prompts/p0 --prefix prompts/p0_w4096_r1_c
```

This writes `prompts/p0_w4096_r1_c1` ... `prompts/p0_w4096_r1_c5`, each a drop-in replacement for the original prompt directory. For a prefix-conditioned incumbent add `--agent-prefix-from $RUNS/p0_prefix_w4096_s1` to `pair.evidence`: every counterexample then carries the prefix its compressor saw for that task, so the diagnosis can check the summary against the task statement and the agent's operating rules.

### Step 4: Selecting the adapted prompt

Boundary evidence was collected under the original compressor, and a revised prompt changes both the summaries and the boundaries visited, so candidates are validated end to end. Take the 12 training tasks with the longest compressed trajectories, run each candidate twice on them, and select by Pass^2 with ties broken by the lower mean number of interaction steps.

```bash
python -m pair.fit_tasks --run $RUNS/p0_w4096_s1 --n 12 --out $B/fit_tasks.jsonl
for i in 1 2 3 4 5; do for s in 1 2; do
    python -m pair.benchmarks.appworld.run --tasks $B/fit_tasks.jsonl --out $RUNS/fit_r1_c${i}_s$s \
        --method prompts/p0_w4096_r1_c$i --window 4096 --seed $s --workers 12
done; done
python -m pair.selection --candidate c1=$RUNS/fit_r1_c1_s1,$RUNS/fit_r1_c1_s2 --candidate c2=$RUNS/fit_r1_c2_s1,$RUNS/fit_r1_c2_s2 \
    --candidate c3=$RUNS/fit_r1_c3_s1,$RUNS/fit_r1_c3_s2 --candidate c4=$RUNS/fit_r1_c4_s1,$RUNS/fit_r1_c4_s2 \
    --candidate c5=$RUNS/fit_r1_c5_s1,$RUNS/fit_r1_c5_s2
```

The selected prompt `P*` is used without further adaptation for held-out evaluation.

## Evaluation

Run the selected prompt and the reference arms on the held-out tasks with several seeds, then summarise:

```bash
for s in 1 2 3; do
    python -m pair.benchmarks.appworld.run --tasks data/tasklists/appworld_test_normal.jsonl --out $RUNS/test_full_s$s --seed $s --workers 20
    python -m pair.benchmarks.appworld.run --tasks data/tasklists/appworld_test_normal.jsonl --out $RUNS/test_p0_s$s \
        --method prompts/p0 --window 4096 --seed $s --workers 20
    python -m pair.benchmarks.appworld.run --tasks data/tasklists/appworld_test_normal.jsonl --out $RUNS/test_pair_s$s \
        --method prompts/p0_w4096_r1_c3 --window 4096 --seed $s --workers 20
done
python -m pair.report --arm full=$RUNS/test_full_s1,$RUNS/test_full_s2,$RUNS/test_full_s3 \
    --arm p0=$RUNS/test_p0_s1,$RUNS/test_p0_s2,$RUNS/test_p0_s3 \
    --arm pair=$RUNS/test_pair_s1,$RUNS/test_pair_s2,$RUNS/test_pair_s3 --out $RUNS/summary.md
```

`pair.report` prints the pass rate over all runs, Pass^k (a task passes in every run), Pass@k (in at least one), and the mean numbers of steps and compactions.

## Other Benchmarks

The pipeline is identical on OfficeBench and tau2-bench; only the runner and the boundary tools change:

| | AppWorld | OfficeBench | tau2-bench (retail) |
|---|---|---|---|
| run tasks | `pair.benchmarks.appworld.run` | `pair.benchmarks.officebench.run --tag TAG` | `pair.benchmarks.tau2.run --domain retail --split train\|test` |
| boundaries | `pair.benchmarks.appworld.boundaries` | `pair.benchmarks.officebench.rollouts mine` | `pair.benchmarks.tau2.rollouts mine` (tau2 env) |
| continuations | `pair.benchmarks.appworld.rollouts` | `pair.benchmarks.officebench.rollouts run` | `pair.benchmarks.tau2.rollouts run` (tau2 env) |
| `pair.evidence --benchmark` | `appworld` | `officebench` | `tau2` |
| context budget used | 4096 | 2048 | 2048 |

OfficeBench runs and continuations execute inside containers; `pair.benchmarks.officebench.rollouts audit` replays every recorded trajectory without an LLM call and reports whether it reproduces. tau2-bench has no separate task text, so the compressor summarises the dialogue itself and the prefix-conditioned prompt receives the agent's system prompt.
