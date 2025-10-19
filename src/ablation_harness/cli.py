"""
CLI Orchestrator. Calls and organizes commands.

Usage:
    python -m ablation_harness.cli run \
    --config experiments/study_test.yaml \
    --metric val/acc --goal max \
    --out_dir runs/del_test



TODO:
Dir deletetion with --clean_run currently WIP (not reliable on windows).
"""

import argparse
import sys

from . import executor, planner


def main(argv=None):
    """CLI Orchestrator."""
    p = argparse.ArgumentParser("ablation_harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", required=True)
    common.add_argument("--trainer", default="ablation_harness.train")
    common.add_argument("--out_dir", default="runs/ablation")
    common.add_argument("--seed", type=int, default=None)
    common.add_argument("--metric", default="val/acc")
    common.add_argument("--goal", choices=("max", "min"), default="max")

    _ = sub.add_parser("plan", parents=[common])
    sr = sub.add_parser("run", parents=[common])
    sr.add_argument("--concurrency", type=int, default=1)
    sr.add_argument("--resume_failed", action="store_true")
    sr.add_argument("--dry_run", action="store_true")
    sr.add_argument("--clean_run", action="store_true")  # currently wip, not reliable. Delete dirs manually before starting on windows.
    sr.add_argument("--resume", action="store_true")
    sr.add_argument("--max_fail", type=int, default=3)

    args = p.parse_args(argv)

    spec = planner.load_yaml(args.config)
    runs = planner.plan(spec, cli_seed=args.seed)
    if args.cmd == "plan":
        planner.print_preview(runs, metric=args.metric)  # docstring is wrong
        return 0

    # run
    result = executor.run_many(
        runs=runs,
        trainer_mod=args.trainer,
        out_dir=args.out_dir,
        metric=args.metric,
        goal=args.goal,
        concurrency=args.concurrency,
        dry_run=args.dry_run,
        resume_failed=args.resume_failed,
        clean_run=args.clean_run,
        resume=args.resume,
        max_fail=args.max_fail,
    )

    print(f"[cli.py] wrote {result.jsonl_path}; ok={result.n_ok}, err={result.n_err}")
    return 1 if result.n_err else 0


if __name__ == "__main__":
    sys.exit(main())
