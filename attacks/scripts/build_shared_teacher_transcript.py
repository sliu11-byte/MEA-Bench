from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attacks.core.hf_query_pool import hf_query_pool_for_budget, resolve_query_pool_and_ordering
from attacks.methods.stage1_budget_impl.build_teacher_transcript import build_teacher_transcript

DEFAULT_STAGE1_CONFIG = REPO_ROOT / "attacks" / "configs" / "formal_stage1_budget.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "shared_teacher_transcripts"


def _set_env(name: str, value: str | None) -> None:
    if value is not None:
        os.environ[name] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or validate a shared Stage-1 teacher transcript bundle.")
    parser.add_argument("--budget", required=True, type=int)
    parser.add_argument("--query-pool", default="auto")
    parser.add_argument("--query-ordering", default="auto")
    parser.add_argument("--stage1-config", default=str(DEFAULT_STAGE1_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--latest-path", help="Optional file that receives the resolved bundle directory.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--teacher-backend")
    parser.add_argument("--teacher-model")
    parser.add_argument("--teacher-endpoint-url")
    parser.add_argument("--teacher-request-model")
    parser.add_argument("--teacher-api-key")
    parser.add_argument("--teacher-max-new-tokens", type=int)
    parser.add_argument("--student-model")
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    query_pool_spec = hf_query_pool_for_budget(args.budget) if args.query_pool in (None, "auto") else args.query_pool
    query_pool_path, query_ordering_path = resolve_query_pool_and_ordering(query_pool_spec, args.query_ordering)

    _set_env("STAGE1_QUERY_POOL_PATH", str(query_pool_path))
    if query_ordering_path is not None:
        _set_env("STAGE1_QUERY_ORDERING_PATH", str(query_ordering_path))
    _set_env("STAGE1_TEACHER_BACKEND", args.teacher_backend)
    _set_env("STAGE1_TEACHER_MODEL_PATH", args.teacher_model)
    _set_env("STAGE1_TEACHER_MODEL_NAME", args.teacher_request_model or args.teacher_model)
    _set_env("STAGE1_TEACHER_BASE_URL", args.teacher_endpoint_url)
    _set_env("STAGE1_TEACHER_API_KEY", args.teacher_api_key)
    _set_env("STAGE1_STUDENT_MODEL_PATH", args.student_model)
    if args.teacher_max_new_tokens is not None:
        _set_env("STAGE1_TEACHER_MAX_NEW_TOKENS", str(args.teacher_max_new_tokens))
    if args.seed is not None:
        _set_env("STAGE1_SEED", str(args.seed))
    _set_env("STAGE1_OUTPUT_ROOT", str(Path(args.output_dir).expanduser().resolve()))

    result: dict[str, Any] = build_teacher_transcript(
        config_path=Path(args.stage1_config).expanduser().resolve(),
        budget=args.budget,
        backend_override=args.teacher_backend,
        output_dir_override=Path(args.output_dir).expanduser().resolve(),
        dry_run=args.dry_run,
        validate_only=args.validate_only,
    )
    if args.latest_path and result.get("bundle_dir"):
        latest_path = Path(args.latest_path).expanduser().resolve()
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        latest_path.write_text(str(result["bundle_dir"]) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
