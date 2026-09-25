#!/usr/bin/env bash
# The four PAIR steps on tau2-bench retail, in order. Each command is resumable; rerun the script to continue.
# Boundary mining and continuations run in the tau2 interpreter ($TAU2_PY); everything else in this one.
set -euo pipefail
cd "$(dirname "$0")/.." && source env.sh

START=${START:-prompts/p0}
WINDOW=${WINDOW:-2048}
ROUND=${ROUND:-r1}
RUNS=$PAIR_RUNS/tau2
NAME=$(basename "$START")_w${WINDOW}
B=$RUNS/boundaries/$NAME
PREFIX_ARG=$( [ "$(basename "$START")" = p0_prefix ] && echo "--agent-prefix-from $RUNS/${NAME}_s1" || true )

# Step 1
python -m pair.benchmarks.tau2.run --domain retail --split train --out $RUNS/$NAME --method $START --window $WINDOW \
    --seeds 1 --concurrency 12
(cd $TAU2_HOME && $TAU2_PY -m pair.benchmarks.tau2.rollouts mine --run $RUNS/${NAME}_s1 --domain retail --out $B)

# Step 2
(cd $TAU2_HOME && $TAU2_PY -m pair.benchmarks.tau2.rollouts run --boundaries $B/boundaries.jsonl --out $B --rounds 3 --workers 24)
python -m pair.verifier --rollouts $B/rollouts.jsonl --out $B/effects.jsonl --hazard-min 0.5 --burden-min 5

# Step 3
python -m pair.evidence --benchmark tau2 --boundaries $B/boundaries.jsonl --rollouts $B/rollouts.jsonl \
    --effects $B/effects.jsonl --incumbent $START --out $B/propose_$ROUND $PREFIX_ARG
python -m pair.propose --input $B/propose_$ROUND/proposer_input.json --out $B/propose_$ROUND/proposals.json --candidates 5
python -m pair.materialize --proposals $B/propose_$ROUND/proposals.json --incumbent $START --prefix prompts/tau2_${NAME}_${ROUND}_c

# Step 4
python -m pair.fit_tasks --run $RUNS/${NAME}_s1 --n 12 --out $B/fit_tasks.jsonl
SELECT_ARGS=()
for i in 1 2 3 4 5; do
    C=prompts/tau2_${NAME}_${ROUND}_c$i
    [ -d $C ] || continue
    python -m pair.benchmarks.tau2.run --domain retail --split train --out $RUNS/fit_${NAME}_${ROUND}_c$i \
        --method $C --window $WINDOW --seeds 1 --concurrency 12 --tasks $B/fit_tasks.jsonl
    SELECT_ARGS+=(--candidate c$i=$RUNS/fit_${NAME}_${ROUND}_c${i}_s1)
done
python -m pair.selection "${SELECT_ARGS[@]}"
