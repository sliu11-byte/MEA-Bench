"""End-to-end defense -> countermeasure -> attack pipeline.

This script keeps the benchmark attack code unchanged except for manifest labels:
it runs a transcript-generating defense, applies a response-only WaterPark
countermeasure, then trains the selected attack from the cleaned teacher
transcript via attacks/scripts/run_attack.py --teacher-transcript-path.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attacks.core.hf_query_pool import hf_query_pool_for_budget, resolve_query_pool_and_ordering
from countermeasures.waterpark_response import CountermeasureConfig, METHODS, rewrite_transcript
from defenses.core.io_utils import ensure_dir

TRANSCRIPT_NAME = "defended_teacher.jsonl"

DEFENSE_MODULES = {
    "adfp": "defenses.adfp.run",
    "ads": "defenses.ads.run",
    "doge": "defenses.doge.run",
    "ginsew": "defenses.ginsew.run",
    "radioactivity": "defenses.radioactivity.run",
    "trace_rewriting": "defenses.trace_rewriting.run",
}

DEFENSE_CONFIG_FLAGS = {
    "adfp": "--adfp_config",
    "ads": "--ads_config",
    "ginsew": "--watermark_config",
    "radioactivity": "--watermark_config",
    "trace_rewriting": "--trace_rewriting_config",
}


@dataclass(frozen=True)
class PipelinePaths:
    pipeline_dir: Path
    defense_dir: Path | None
    defended_transcript: Path
    countermeasure_dir: Path
    cleaned_transcript: Path
    countermeasure_manifest: Path
    attack_dir: Path
    pipeline_manifest: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def command_text(command: Sequence[str]) -> str:
    return shlex.join([str(part) for part in command])


def split_extra_args(raw: str | None) -> list[str]:
    if not raw:
        return []
    return shlex.split(raw)


def run_command(command: Sequence[str], *, log_path: Path, cwd: Path = REPO_ROOT) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + command_text(command) + "\n\n")
        log.flush()
        proc = subprocess.run(
            [str(part) for part in command],
            cwd=str(cwd),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with exit code {proc.returncode}; see {log_path}")


def latest_attack_manifest(attack_dir: Path, attack: str) -> Path | None:
    root = attack_dir / attack
    if not root.exists():
        return None
    manifests = sorted(root.glob("*/attack_manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return manifests[0] if manifests else None


def load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_paths(args: argparse.Namespace, run_id: str) -> PipelinePaths:
    output_root = Path(args.output_dir).expanduser().resolve() if args.output_dir else REPO_ROOT / "outputs" / "countermeasures" / "pipelines"
    pipeline_dir = output_root if output_root.name == run_id else output_root / run_id
    countermeasure_name = f"waterpark_{args.countermeasure}"
    countermeasure_dir = pipeline_dir / "countermeasure" / countermeasure_name
    cleaned_transcript = countermeasure_dir / "transcripts" / "cleaned_teacher.jsonl"
    countermeasure_manifest = countermeasure_dir / "countermeasure_manifest.json"
    attack_dir = pipeline_dir / "attack"

    if args.defended_transcript:
        defended_transcript = Path(args.defended_transcript).expanduser().resolve()
        defense_dir = None
    else:
        if not args.defense:
            raise SystemExit("Pass --defense, or pass --defended-transcript to reuse an existing defense output.")
        defense_dir = pipeline_dir / "defense" / args.defense
        defended_transcript = defense_dir / run_id / "transcripts" / TRANSCRIPT_NAME

    return PipelinePaths(
        pipeline_dir=pipeline_dir,
        defense_dir=defense_dir,
        defended_transcript=defended_transcript,
        countermeasure_dir=countermeasure_dir,
        cleaned_transcript=cleaned_transcript,
        countermeasure_manifest=countermeasure_manifest,
        attack_dir=attack_dir,
        pipeline_manifest=pipeline_dir / "countermeasure_pipeline_manifest.json",
    )


def build_defense_command(
    args: argparse.Namespace,
    *,
    run_id: str,
    defense_dir: Path,
    query_pool_path: Path,
) -> list[str]:
    defense = args.defense
    if defense not in DEFENSE_MODULES:
        available = ", ".join(sorted(DEFENSE_MODULES))
        raise SystemExit(f"Unsupported defense {defense!r}; available: {available}")
    if defense == "query_traffic":
        raise SystemExit("query_traffic is detector-only and has no native text countermeasure stage.")

    command = [sys.executable, "-m", DEFENSE_MODULES[defense]]
    if defense == "doge":
        command.append("run")

    command.extend([
        "--query_pool_path",
        str(query_pool_path),
        "--output_dir",
        str(defense_dir),
        "--run_id",
        run_id,
    ])

    if args.teacher_model:
        command.extend(["--teacher_model", args.teacher_model])
    elif defense != "trace_rewriting":
        raise SystemExit("--teacher-model is required when running the defense stage.")

    if args.max_queries is not None:
        command.extend(["--max_queries", str(args.max_queries)])
    if args.device:
        command.extend(["--device", args.device])
    if args.defense_config and defense in DEFENSE_CONFIG_FLAGS:
        command.extend([DEFENSE_CONFIG_FLAGS[defense], args.defense_config])

    if defense in {"adfp", "doge"}:
        if not args.proxy_model:
            raise SystemExit(f"--proxy-model is required for {defense}.")
        command.extend(["--proxy_model", args.proxy_model])
    if defense == "ads" and args.proxy_model:
        command.extend(["--proxy_student", args.proxy_model])
    if defense == "trace_rewriting":
        if not args.rewriter_model:
            raise SystemExit("--rewriter-model is required for trace_rewriting.")
        command.extend(["--rewriter_model", args.rewriter_model])
        command.extend(["--rewriter_backend", args.rewriter_backend])
        if args.teacher_backend:
            command.extend(["--teacher_backend", args.teacher_backend])
        if args.teacher_base_url:
            command.extend(["--teacher_base_url", args.teacher_base_url])
        if args.teacher_request_model:
            command.extend(["--teacher_request_model", args.teacher_request_model])
        if args.teacher_api_key:
            command.extend(["--teacher_api_key", args.teacher_api_key])
        if args.rewriter_base_url:
            command.extend(["--rewriter_base_url", args.rewriter_base_url])
        if args.rewriter_request_model:
            command.extend(["--rewriter_request_model", args.rewriter_request_model])
        if args.rewriter_api_key:
            command.extend(["--rewriter_api_key", args.rewriter_api_key])
    if defense == "doge":
        if not args.doge_train_file:
            raise SystemExit("--doge-train-file is required for doge.")
        command.extend(["--train_file", args.doge_train_file])

    command.extend(split_extra_args(args.defense_extra_args))
    return command


def build_attack_command(
    args: argparse.Namespace,
    *,
    query_pool_path: Path,
    query_ordering_path: Path | None,
    cleaned_transcript: Path,
    countermeasure_manifest: Path,
    attack_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        "attacks/scripts/run_attack.py",
        "--attack",
        args.attack,
        "--budget",
        str(args.budget),
        "--query-pool",
        str(query_pool_path),
        "--query-ordering",
        str(query_ordering_path) if query_ordering_path is not None else args.query_ordering,
        "--teacher-transcript-path",
        str(cleaned_transcript),
        "--countermeasure",
        f"waterpark_{args.countermeasure}",
        "--countermeasure-manifest",
        str(countermeasure_manifest),
        "--output-dir",
        str(attack_dir),
    ]
    if args.teacher_model:
        command.extend(["--teacher-model", args.teacher_model])
    if args.student_model:
        command.extend(["--student-model", args.student_model])
    if args.stage1_config:
        command.extend(["--stage1-config", args.stage1_config])
    if args.seed is not None:
        command.extend(["--seed", str(args.seed)])
    if args.dry_run:
        command.append("--dry-run")
    command.extend(split_extra_args(args.attack_extra_args))
    return command


def resolve_benchmark_query_inputs(args: argparse.Namespace) -> tuple[Path, Path | None]:
    query_pool_spec = hf_query_pool_for_budget(args.budget) if args.query_pool in (None, "auto") else args.query_pool
    return resolve_query_pool_and_ordering(query_pool_spec, args.query_ordering)


def write_pipeline_manifest(
    *,
    args: argparse.Namespace,
    run_id: str,
    paths: PipelinePaths,
    query_pool_path: Path,
    query_ordering_path: Path | None,
    defense_command: Sequence[str] | None,
    attack_command: Sequence[str],
    attack_manifest_path: Path | None,
) -> dict[str, Any]:
    payload = {
        "schema_version": "countermeasure_pipeline_manifest_v1",
        "created_at": utc_now(),
        "run_id": run_id,
        "defense": args.defense,
        "attack": args.attack,
        "budget": args.budget,
        "query_pool": str(query_pool_path),
        "query_ordering": None if query_ordering_path is None else str(query_ordering_path),
        "countermeasure_enabled": True,
        "countermeasure": f"waterpark_{args.countermeasure}",
        "countermeasure_family": "waterpark_response_only",
        "paths": {key: None if value is None else str(value) for key, value in asdict(paths).items()},
        "commands": {
            "defense": None if defense_command is None else list(defense_command),
            "countermeasure": [
                sys.executable,
                "-m",
                "countermeasures.waterpark_response",
                "--method",
                args.countermeasure,
                "--input",
                str(paths.defended_transcript),
                "--output",
                str(paths.cleaned_transcript),
                "--manifest",
                str(paths.countermeasure_manifest),
            ],
            "attack": list(attack_command),
        },
        "manifests": {
            "countermeasure": str(paths.countermeasure_manifest),
            "attack": None if attack_manifest_path is None else str(attack_manifest_path),
        },
        "notes": {
            "attack_input": "The attack is trained from the countermeasure-cleaned teacher transcript.",
            "teacher_queries_added_by_countermeasure": 0,
            "requires_clean_teacher_outputs": False,
        },
    }
    paths.pipeline_manifest.parent.mkdir(parents=True, exist_ok=True)
    paths.pipeline_manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run defense -> WaterPark countermeasure -> attack.")
    parser.add_argument("--defense", choices=sorted(DEFENSE_MODULES), default=None)
    parser.add_argument("--defended-transcript", default=None, help="Reuse an existing defended_teacher.jsonl and skip the defense stage.")
    parser.add_argument("--countermeasure", choices=METHODS, required=True)
    parser.add_argument("--attack", default="seqkd", choices=["seqkd", "lord", "soda", "qedks", "model_leeching", "gad"])
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--query-pool", default="auto")
    parser.add_argument("--query-ordering", default="auto")
    parser.add_argument("--teacher-model", default=None)
    parser.add_argument("--student-model", default=None)
    parser.add_argument("--proxy-model", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--defense-config", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--stage1-config", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--lex", type=int, default=40, help="DIPPER lexical diversity. MEA default: 40.")
    parser.add_argument("--order", type=int, default=0, help="DIPPER order diversity. MEA default: 0.")
    parser.add_argument("--sent-interval", type=int, default=3)
    parser.add_argument("--with-context", action="store_true")
    parser.add_argument("--countermeasure-model-name", default=None)
    parser.add_argument("--countermeasure-tokenizer-name", default=None)
    parser.add_argument("--countermeasure-device", default=None)
    parser.add_argument("--countermeasure-limit", type=int, default=None)

    parser.add_argument("--teacher-backend", choices=["local_hf", "openai_compatible"], default=None)
    parser.add_argument("--teacher-base-url", default=None)
    parser.add_argument("--teacher-request-model", default=None)
    parser.add_argument("--teacher-api-key", default="EMPTY")
    parser.add_argument("--rewriter-model", default=None)
    parser.add_argument("--rewriter-backend", choices=["local_hf", "openai_compatible"], default="local_hf")
    parser.add_argument("--rewriter-base-url", default=None)
    parser.add_argument("--rewriter-request-model", default=None)
    parser.add_argument("--rewriter-api-key", default="EMPTY")
    parser.add_argument("--doge-train-file", default=None)

    parser.add_argument("--defense-extra-args", default="", help="Shell-like string appended to the defense command.")
    parser.add_argument("--attack-extra-args", default="", help="Shell-like string appended to attacks/scripts/run_attack.py.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S") + f"_{args.defense or 'external'}_{args.attack}_waterpark_{args.countermeasure}"
    paths = build_paths(args, run_id)
    ensure_dir(paths.pipeline_dir)

    query_pool_path, query_ordering_path = resolve_benchmark_query_inputs(args)

    defense_command = None
    if paths.defense_dir is not None:
        defense_command = build_defense_command(args, run_id=run_id, defense_dir=paths.defense_dir, query_pool_path=query_pool_path)
        run_command(defense_command, log_path=paths.pipeline_dir / "logs" / "defense.log")
        if not paths.defended_transcript.exists():
            raise RuntimeError(f"defense completed but transcript was not found: {paths.defended_transcript}")
    elif not paths.defended_transcript.exists():
        raise SystemExit(f"defended transcript does not exist: {paths.defended_transcript}")

    cm_config = CountermeasureConfig(
        method=args.countermeasure,
        input_path=str(paths.defended_transcript),
        output_path=str(paths.cleaned_transcript),
        manifest_path=str(paths.countermeasure_manifest),
        lex=args.lex,
        order=args.order,
        sent_interval=args.sent_interval,
        no_ctx=not args.with_context,
        model_name=args.countermeasure_model_name,
        tokenizer_name=args.countermeasure_tokenizer_name,
        device=args.countermeasure_device,
        seed=args.seed,
        limit=args.countermeasure_limit,
    )
    countermeasure_payload = rewrite_transcript(config=cm_config)

    attack_command = build_attack_command(
        args,
        query_pool_path=query_pool_path,
        query_ordering_path=query_ordering_path,
        cleaned_transcript=paths.cleaned_transcript,
        countermeasure_manifest=paths.countermeasure_manifest,
        attack_dir=paths.attack_dir,
    )
    run_command(attack_command, log_path=paths.pipeline_dir / "logs" / "attack.log")
    attack_manifest_path = latest_attack_manifest(paths.attack_dir, args.attack)

    payload = write_pipeline_manifest(
        args=args,
        run_id=run_id,
        paths=paths,
        query_pool_path=query_pool_path,
        query_ordering_path=query_ordering_path,
        defense_command=defense_command,
        attack_command=attack_command,
        attack_manifest_path=attack_manifest_path,
    )
    payload["countermeasure_manifest_payload"] = countermeasure_payload
    payload["attack_manifest_payload"] = load_json(attack_manifest_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
