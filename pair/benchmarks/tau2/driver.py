"""tau2 driver executed inside the tau2 environment by pair.benchmarks.tau2.run: one domain, one split.

    python -m pair.benchmarks.tau2.driver --domain retail --split test --agent llm_agent|pair_agent --seed 300 --save-to NAME

The agent LLM runs at temperature 1 with medium reasoning; the user simulator is tau2's default at
temperature 0. Evaluation is tau2's official evaluator with the per-component breakdown stored.
"""
import argparse
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="retail")
    ap.add_argument("--split", default="test")
    ap.add_argument("--agent", default="llm_agent", choices=("llm_agent", "pair_agent"))
    ap.add_argument("--llm", default=os.environ.get("PAIR_TAU2_AGENT_LLM", "openai/responses/gpt-5.6-luna"))
    ap.add_argument("--user-llm", default="gpt-4.1-2025-04-14")
    ap.add_argument("--user-temperature", type=float, default=0.0)
    ap.add_argument("--reasoning-effort", default="medium")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--task-ids", nargs="*", default=None)
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=300)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--save-to", required=True)
    a = ap.parse_args()
    import litellm
    litellm.suppress_debug_info = True
    if a.agent == "pair_agent":
        from pair.benchmarks.tau2 import agent
        agent.register()
    from tau2.runner.helpers import load_tasks
    from tau2.run import run_tasks
    from tau2.evaluator.evaluator import EvaluationType
    tasks = load_tasks(a.domain, task_split_name=a.split)
    if a.task_ids:
        tasks = [t for t in tasks if t.id in set(a.task_ids)]
    print(f"{a.domain}/{a.split}: {len(tasks)} tasks x {a.trials} trials, agent={a.agent} llm={a.llm} "
          f"user={a.user_llm}@T{a.user_temperature} method={os.environ.get('PAIR_TAU2_METHOD')} "
          f"window={os.environ.get('PAIR_TAU2_WINDOW')}", flush=True)
    run_tasks(domain=a.domain, tasks=tasks, agent=a.agent, user="user_simulator",
              llm_agent=a.llm, llm_args_agent={"temperature": 1.0, "reasoning_effort": a.reasoning_effort},
              llm_user=a.user_llm, llm_args_user={"temperature": a.user_temperature},
              num_trials=a.trials, max_steps=a.max_steps, save_to=a.save_to, max_concurrency=a.concurrency,
              seed=a.seed, console_display=False, log_level="WARNING", auto_resume=True,
              evaluation_type=EvaluationType.ALL)


if __name__ == "__main__":
    main()
