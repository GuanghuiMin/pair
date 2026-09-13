# Copy to env.sh, fill in the values, then `source env.sh` before running anything.
export PAIR_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$PAIR_HOME${PYTHONPATH:+:$PYTHONPATH}"

# Where runs, boundaries and continuations are written.
export PAIR_RUNS="$PAIR_HOME/runs"

# Your OpenAI API key. It is used by the agent (through the patched ACON harness and tau2-bench),
# by the compressor and by the optimizer; the two *_API_KEY variables below default to it.
export OPENAI_API_KEY=""

# Agent LLM (read by the patched ACON harness on AppWorld and OfficeBench).
export PAIR_AGENT_MODEL="gpt-5.6-luna"
export ACON_VLLM_BASE_URL="https://api.openai.com/v1"
export ACON_VLLM_API_KEY="$OPENAI_API_KEY"
export ACON_VLLM_API_STYLE="responses"          # Responses API with native tool calls
export ACON_VLLM_REASONING_EFFORT="medium"
export ACON_VLLM_NO_THINKING_KWARG=1

# Compressor and optimizer LLM (pair.compressor, pair.propose).
export PAIR_COMPRESSOR_MODEL="gpt-5.6-luna"
export PAIR_COMPRESSOR_BASE_URL="$ACON_VLLM_BASE_URL"
export PAIR_COMPRESSOR_API_KEY="$OPENAI_API_KEY"
export PAIR_COMPRESSOR_REASONING="medium"

# Benchmark harnesses.
export ACON_ROOT="$PAIR_HOME/../acon"                     # patched ACON checkout (third_party/README.md)
export OB_CANONICAL_TASKS="$PAIR_HOME/../officebench_tasks" # read-only export of OfficeBench's tasks/ tree
export PAIR_OB_IMAGE="pair-officebench"                    # docker/officebench/Dockerfile
export PAIR_OB_SCRATCH="/tmp/pair_officebench"
export TAU2_HOME="$PAIR_HOME/../tau2-bench"                # tau2-bench checkout
export TAU2_PY="$(command -v python)"                       # interpreter with tau2 installed
export PAIR_TAU2_AGENT_LLM="openai/responses/$PAIR_AGENT_MODEL"
