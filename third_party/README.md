# Third-party harnesses

PAIR does not ship an agent of its own. The AppWorld and OfficeBench agents come from
[ACON](https://github.com/microsoft/acon); the tau2-bench agent comes from
[tau2-bench](https://github.com/sierra-research/tau2-bench). Both are used as installed packages.

## ACON (AppWorld and OfficeBench)

`acon.patch` applies to ACON commit `d63f9ae` and touches six files under `src/productive_agents/`:
native tool calling for the AppWorld agent (`agents/appworld/agent.py`), tool-role messages and their
rendering in the memory manager (`agents/memory.py`, `agents/unified_agent.py`), the configurable
endpoint, function tools and the OpenAI Responses API in the LLM client (`llm.py`, `agents/utils.py`),
and correct shell quoting of OfficeBench actions (`env/officebench/env.py`). The runners select the
behaviour through the `ACON_VLLM_*` and `ACON_APPWORLD_*` variables in `env.example.sh`.

```bash
git clone https://github.com/microsoft/acon ../acon
cd ../acon && git checkout d63f9ae && git apply ../pair/third_party/acon.patch
pip install -e .
```

Then install AppWorld and its data as described in ACON's `experiments/appworld/README.md`, and point
`ACON_ROOT` at the checkout.

## OfficeBench tasks

OfficeBench's `tasks/` tree (from the [OfficeBench](https://github.com/zlwang-cs/OfficeBench) repository,
as used by ACON) is exported once to a read-only directory and referenced through `OB_CANONICAL_TASKS`.
Every run rebuilds its working directory from that export, so a recorded trajectory can be replayed
exactly for continuations.

```bash
git clone https://github.com/zlwang-cs/OfficeBench /tmp/OfficeBench
cp -r /tmp/OfficeBench/tasks ../officebench_tasks && chmod -R a-w ../officebench_tasks
docker build -t pair-officebench docker/officebench
```

## tau2-bench (retail)

tau2-bench is used unmodified, including its `train` / `test` task splits for the retail domain.

```bash
git clone https://github.com/sierra-research/tau2-bench ../tau2-bench
cd ../tau2-bench && pip install -e .
```

Set `TAU2_HOME` to the checkout and `TAU2_PY` to the interpreter that has tau2 installed; the tau2
runner and its boundary continuations execute in that interpreter.
