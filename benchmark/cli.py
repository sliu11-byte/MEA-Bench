from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attacks.core.hf_query_pool import (
    file_sha256,
    hf_query_pool_for_budget,
    load_query_pool_records,
    resolve_query_pool_and_ordering,
)
from benchmark.manifests import (
    SCHEMA_VERSION,
    find_resumable_manifest,
    git_source,
    make_run_id,
    portable_path,
    software_versions,
    stable_hash,
    write_experiment_manifest,
)
from benchmark.profiles import ProfileError, load_profile
from benchmark.registry import ATTACKS, RegistryError, validate_attack_budget


class CLIError(RuntimeError):
    pass


def _repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    return (REPO_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _attack_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("attack", help="Run or plan one model extraction attack.")
    parser.add_argument("--attack", required=True, choices=sorted(ATTACKS))
    parser.add_argument("--budget", required=True, type=int)
    parser.add_argument("--profile", default="paper")
    parser.add_argument("--query-pool")
    parser.add_argument("--query-ordering")
    parser.add_argument("--teacher-endpoint")
    parser.add_argument("--student-endpoint")
    parser.add_argument("--teacher-serving", choices=("external", "local"), default="local")
    parser.add_argument("--student-serving", choices=("external", "local"), default="local")
    parser.add_argument("--transcript-dir")
    parser.add_argument("--warmup-model")
    parser.add_argument("--output-root")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-write-manifest", action="store_true", help=argparse.SUPPRESS)
    parser.set_defaults(handler=_run_attack)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mea-benchmark", description="Portable MEA benchmark runner.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _attack_parser(subparsers)
    return parser


def _attack_plan(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    profile = load_profile(args.profile)
    spec = validate_attack_budget(args.attack, args.budget, profile.attack_budgets)
    attack_profile = profile.section("attack")
    data_profile = profile.section("data")

    output_root = _repo_path(args.output_root or str(attack_profile["output_root"]))
    stage1_config = _repo_path(str(attack_profile["stage1_config"]))
    if not stage1_config.is_file():
        raise CLIError(f"stage-1 config does not exist: {stage1_config}")

    query_pool_spec = args.query_pool or str(data_profile["query_pool"])
    if query_pool_spec == "auto":
        query_pool_spec = hf_query_pool_for_budget(args.budget)
    query_ordering_spec = args.query_ordering or str(data_profile["query_ordering"])
    query_pool_path, query_ordering_path = resolve_query_pool_and_ordering(
        query_pool_spec,
        query_ordering_spec,
    )
    records, pool_meta = load_query_pool_records(query_pool_path)
    if len(records) < args.budget:
        raise CLIError(
            f"query pool contains {len(records)} records but budget {args.budget} was requested"
        )

    teacher = dict(attack_profile["teacher"])
    student = dict(attack_profile["student"])
    if args.teacher_endpoint:
        teacher["endpoint"] = args.teacher_endpoint
    if args.student_endpoint:
        student["endpoint"] = args.student_endpoint
    seed = args.seed if args.seed is not None else int(attack_profile["seed"])
    transcript_dir = _repo_path(args.transcript_dir) if args.transcript_dir else None
    if transcript_dir is not None and not transcript_dir.is_dir():
        raise CLIError(f"transcript directory does not exist: {transcript_dir}")

    portable_query_pool = (
        query_pool_spec
        if query_pool_spec.startswith("hf://")
        else portable_path(query_pool_path, repo_root=REPO_ROOT, output_root=output_root)
    )
    config = {
        "attack": args.attack,
        "budget": args.budget,
        "profile": profile.name,
        "seed": seed,
        "query_pool": portable_query_pool,
        "query_ordering": query_ordering_spec,
        "stage1_config": portable_path(stage1_config, repo_root=REPO_ROOT, output_root=output_root),
        "teacher": teacher,
        "student": student,
        "serving": {"teacher": args.teacher_serving, "student": args.student_serving},
        "training": dict(attack_profile["training"]),
        "method": dict(attack_profile["methods"].get(args.attack, {})),
        "transcript_dir": None
        if transcript_dir is None
        else portable_path(transcript_dir, repo_root=REPO_ROOT, output_root=output_root),
        "warmup_model": args.warmup_model,
    }
    inputs = {
        "query_pool": {
            "uri": portable_query_pool,
            "path": portable_path(query_pool_path, repo_root=REPO_ROOT, output_root=output_root),
            "sha256": file_sha256(query_pool_path),
            "records": len(records),
            "tier": pool_meta.get("tier"),
        },
        "query_ordering": None
        if query_ordering_path is None
        else {
            "path": portable_path(query_ordering_path, repo_root=REPO_ROOT, output_root=output_root),
            "sha256": file_sha256(query_ordering_path),
        },
        "transcript": None
        if transcript_dir is None
        else {"path": portable_path(transcript_dir, repo_root=REPO_ROOT, output_root=output_root)},
    }
    config_hash = stable_hash({"config": config, "inputs": inputs})
    run_root = output_root / "attacks" / args.attack / f"b{args.budget}"
    resumed = find_resumable_manifest(run_root, config_hash) if args.resume else None
    if resumed is not None:
        manifest_path, previous = resumed
        run_id = str(previous["run_id"])
        started_at = previous.get("started_at")
    else:
        run_id = make_run_id(args.attack, args.budget)
        manifest_path = run_root / run_id / "attack_manifest.json"
        started_at = None

    steps = ["resolve_query_pool", "validate_profile_and_method"]
    if spec.uses_shared_transcript:
        steps.append("reuse_shared_transcript" if transcript_dir else "build_shared_teacher_transcript")
    else:
        steps.append("run_method_specific_teacher_acquisition")
    if "seqkd_warmup" in spec.prerequisites:
        steps.append("resolve_seqkd_warmup")
    steps.extend((f"run_{args.attack}", "write_attack_manifest"))

    now = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_type": "attack",
        "run_id": run_id,
        "status": "planned",
        "method": args.attack,
        "attack": args.attack,
        "budget": args.budget,
        "seed": seed,
        "profile": profile.name,
        "config": config,
        "config_hash": config_hash,
        "inputs": inputs,
        "upstream_manifests": [],
        "artifacts": {
            "run_dir": portable_path(manifest_path.parent, repo_root=REPO_ROOT, output_root=output_root),
            "attack_manifest": portable_path(manifest_path, repo_root=REPO_ROOT, output_root=output_root),
        },
        "metrics": {},
        "software_versions": software_versions(),
        "source_revision": git_source(REPO_ROOT),
        "started_at": started_at or now,
        "completed_at": None,
        "resume": {"requested": bool(args.resume), "matched_existing_run": resumed is not None},
        "execution_plan": {
            "dry_run": True,
            "required_services": list(spec.required_services),
            "steps": steps,
        },
    }
    return manifest, manifest_path


def _run_attack(args: argparse.Namespace) -> int:
    if not args.dry_run:
        raise CLIError(
            "real attack execution is not connected to the consolidated runner yet; "
            "rerun with --dry-run while this interface is being implemented"
        )
    manifest, manifest_path = _attack_plan(args)
    if not args.no_write_manifest:
        write_experiment_manifest(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (CLIError, ProfileError, RegistryError, OSError, ValueError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
