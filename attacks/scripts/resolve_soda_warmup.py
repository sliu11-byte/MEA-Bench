from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attacks.core.base import AttackRunConfig
from attacks.core.hf_query_pool import hf_query_pool_for_budget, resolve_query_pool_and_ordering
from attacks.methods.soda_warmup import resolve_soda_warmup


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve the compatible completed SeqKD warmup for SODA.")
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--query-pool", default="auto")
    parser.add_argument("--query-ordering", default="auto")
    parser.add_argument("--stage1-config", default="attacks/configs/formal_stage1_budget.yaml")
    parser.add_argument("--transcript-dir")
    parser.add_argument("--output-env", help="Optional shell file receiving WARMUP_MODEL and SHARED_TRANSCRIPT_DIR.")
    args = parser.parse_args()

    pool_spec = hf_query_pool_for_budget(args.budget) if args.query_pool == "auto" else args.query_pool
    pool_path, ordering_path = resolve_query_pool_and_ordering(pool_spec, args.query_ordering)
    config = AttackRunConfig(
        attack="soda",
        budget=args.budget,
        query_pool_path=pool_path,
        query_ordering_path=ordering_path,
        output_dir=Path(args.output_dir).expanduser().resolve(),
        stage1_config_path=Path(args.stage1_config).expanduser().resolve(),
        student_model=args.student_model,
    )
    transcript_dir = None if not args.transcript_dir else Path(args.transcript_dir).expanduser().resolve()
    resolution = resolve_soda_warmup(config, transcript_dir=transcript_dir)
    if args.output_env:
        if not resolution.transcript_dir:
            raise SystemExit("Resolved SeqKD warmup does not record a teacher transcript directory")
        output = Path(args.output_env).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            f"export WARMUP_MODEL={shlex.quote(resolution.checkpoint)}\n"
            f"export SHARED_TRANSCRIPT_DIR={shlex.quote(resolution.transcript_dir)}\n",
            encoding="utf-8",
        )
    else:
        print(resolution.checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
