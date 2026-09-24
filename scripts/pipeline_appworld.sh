#!/usr/bin/env bash
# The four PAIR steps on AppWorld, in order. Each command is resumable; rerun the script to continue.
set -euo pipefail
cd "$(dirname "$0")/.." && source env.sh

START=${START:-prompts/p0}          # incumbent compression prompt (prompts/p0 or prompts/p0_prefix)
WINDOW=${WINDOW:-4096}              # context budget in tokens
ROUND=${ROUND:-r1}
RUNS=$PAIR_RUNS/appworld
NAME=$(basename "$START")_w${WINDOW}
B=$RUNS/boundaries/$NAME
PREFIX_ARG=$( [ "$(basename "$START")" = p0_prefix ] && echo "--agent-prefix-from $RUNS/${NAME}_s1" || true )

# Step 1: collect compressed trajectories on the training tasks, then list every compaction boundary.
python -m pair.benchmarks.appworld.run --tasks data/tasklists/appworld_train.jsonl --out $RUNS/${NAME}_s1 \
    --method $START --window $WINDOW --seed 1 --workers 20
python -m pair.benchmarks.appworld.boundaries --run $RUNS/${NAME}_s1 --out $B

# Step 2: PRE/POST continuations at every boundary, then the hazard / burden verifier.
python -m pair.benchmarks.appworld.rollouts --boundaries $B/boundaries.jsonl --out $B --rounds 3 --workers 20
python -m pair.verifier --rollouts $B/rollouts.jsonl --out $B/effects.jsonl --hazard-min 0.5 --burden-min 5

# Step 3: counterexamples -> optimizer -> five candidate prompts.
python -m pair.evidence --benchmark appworld --boundaries $B/boundaries.jsonl --rollouts $B/rollouts.jsonl \
    --effects $B/effects.jsonl --incumbent $START --out $B/propose_$ROUND $PREFIX_ARG
python -m pair.propose --input $B/propose_$ROUND/proposer_input.json --out $B/propose_$ROUND/proposals.json --candidates 5
python -m pair.materialize --proposals $B/propose_$ROUND/proposals.json --incumbent $START --prefix prompts/${NAME}_${ROUND}_c

# Step 4: two runs of each candidate on the 12 longest training trajectories, then Pass^2 selection.
python -m pair.fit_tasks --run $RUNS/${NAME}_s1 --n 12 --out $B/fit_tasks.jsonl
SELECT_ARGS=()
for i in 1 2 3 4 5; do
    C=prompts/${NAME}_${ROUND}_c$i
    [ -d $C ] || continue
    for s in 1 2; do
        python -m pair.benchmarks.appworld.run --tasks $B/fit_tasks.jsonl --out $RUNS/fit_${NAME}_${ROUND}_c${i}_s$s \
            --method $C --window $WINDOW --seed $s --workers 12
    done
    SELECT_ARGS+=(--candidate c$i=$RUNS/fit_${NAME}_${ROUND}_c${i}_s1,$RUNS/fit_${NAME}_${ROUND}_c${i}_s2)
done
python -m pair.selection "${SELECT_ARGS[@]}"
