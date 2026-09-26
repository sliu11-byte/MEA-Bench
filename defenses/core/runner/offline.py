from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional

from attacks.core.hf_query_pool import hf_query_pool_for_budget, load_query_pool_records, resolve_query_pool_and_ordering
from defenses.core.detector.commands import run_detector_command
from defenses.core.generator.attack import run_attack_process
from defenses.core.generator.result import GeneratorResult
from defenses.core.io_utils import ensure_dir, write_jsonl
from defenses.core.generator.oracle import wait_for_oracle
from defenses.core.process_logging import run_logged_command, stage_log_path, start_logged_process
from defenses.core.runner.online import (
    DETECTOR_FACTORIES,
    DetectorFactory,
    default_output_dir,
    find_free_port,
    make_run_id,
    parse_common_args,
    repo_root_from_here,
    run_online_defense,
)
from defenses.core.runner.result import DefenseRunResult


OFFLINE_TRANSCRIPT_ATTACKS = {"seqkd", "lord", "gad", "soda"}
OFFLINE_GENERATOR_DEFENSES = {"clean", "ginsew", "radioactivity", "adfp", "ads", "trace_rewriting", "doge"}


def _start_soda_student(
    *, repo_root: Path, checkpoint: Path, student_model: str, output_dir: Path, timeout: float
) -> tuple[subprocess.Popen, str, str]:
    served_model = "soda-seqkd-initialization"
    model = checkpoint
    lora_args: list[str] = []
    adapter_config = checkpoint / "adapter_config.json"
    if adapter_config.is_file():
        payload = json.loads(adapter_config.read_text(encoding="utf-8"))
        base_model = payload.get("base_model_name_or_path") or student_model
        model = Path(str(base_model))
        lora_args = ["--enable-lora", "--lora-modules", f"{served_model}={checkpoint}"]

    host = "127.0.0.1"
    port = find_free_port(host)
    base_url = f"http://{host}:{port}/v1"
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model),
        "--served-model-name",
        served_model,
        "--host",
        host,
        "--port",
        str(port),
        "--tensor-parallel-size",
        os.environ.get("SODA_STUDENT_TENSOR_PARALLEL_SIZE", "1"),
        "--gpu-memory-utilization",
        os.environ.get("SODA_STUDENT_GPU_MEMORY_UTILIZATION", "0.85"),
        "--max-model-len",
        os.environ.get("SODA_STUDENT_MAX_MODEL_LEN", "4096"),
        "--dtype",
        os.environ.get("SODA_STUDENT_DTYPE", "bfloat16"),
        *lora_args,
    ]
    process = start_logged_process(
        command,
        cwd=repo_root,
        log_path=stage_log_path(output_dir / "soda_student_server.log"),
        env={"CUDA_VISIBLE_DEVICES": os.environ.get("SODA_STUDENT_CUDA_VISIBLE_DEVICES", "0")},
    )
    try:
        wait_for_oracle(base_url, process, timeout=timeout)
    except BaseException:
        process.terminate()
        process.wait(timeout=20)
        raise
    return process, base_url, served_model


def _stop_process(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=20)
    thread = getattr(process, "log_thread", None)
    if thread is not None:
        thread.join(timeout=20)


def _run_soda_with_prerequisites(
    *, repo_root: Path, args: argparse.Namespace, attack_dir: Path, transcript_path: Path,
    attack_extra_args: Optional[Iterable[str]], defense: str,
) -> dict:
    extra = list(attack_extra_args or ())
    seqkd = run_attack_process(
        repo_root=repo_root,
        attack="seqkd",
        budget=args.budget,
        teacher_base_url=None,
        served_model_name=f"offline-defended-{defense}",
        teacher_model=args.teacher_model,
        student_model=args.student_model,
        output_dir=attack_dir,
        query_pool=args.query_pool,
        query_ordering=args.query_ordering,
        teacher_mode=args.teacher_mode,
        teacher_temperature=args.teacher_temperature,
        teacher_top_p=args.teacher_top_p,
        teacher_max_tokens=args.teacher_max_tokens,
        stage1_config=args.stage1_config,
        extra_args=extra,
        teacher_transcript_path=transcript_path,
    )
    checkpoint_value = seqkd.get("checkpoint_path")
    if not checkpoint_value:
        raise RuntimeError("automatic SODA prerequisite did not produce a SeqKD checkpoint")
    checkpoint = Path(checkpoint_value).resolve()

    process, student_url, request_model = _start_soda_student(
        repo_root=repo_root,
        checkpoint=checkpoint,
        student_model=str(args.student_model),
        output_dir=attack_dir,
        timeout=float(args.startup_timeout),
    )
    try:
        run_attack_process(
            repo_root=repo_root,
            attack="soda",
            budget=args.budget,
            teacher_base_url=None,
            served_model_name=f"offline-defended-{defense}",
            teacher_model=args.teacher_model,
            student_model=args.student_model,
            output_dir=attack_dir,
            query_pool=args.query_pool,
            query_ordering=args.query_ordering,
            teacher_mode=args.teacher_mode,
            teacher_temperature=args.teacher_temperature,
            teacher_top_p=args.teacher_top_p,
            teacher_max_tokens=args.teacher_max_tokens,
            stage1_config=args.stage1_config,
            extra_args=[
                *extra,
                "--warmup-model", str(checkpoint),
                "--student-endpoint-url", student_url,
                "--student-request-model", request_model,
                "--execution-stage", "prepare",
            ],
            teacher_transcript_path=transcript_path,
        )
    finally:
        _stop_process(process)

    manifests = sorted(
        (attack_dir / "soda").glob("*/prepare_manifest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not manifests:
        raise RuntimeError("automatic SODA preparation did not write prepare_manifest.json")
    prepared_run = manifests[0].parent
    result = run_attack_process(
        repo_root=repo_root,
        attack="soda",
        budget=args.budget,
        teacher_base_url=None,
        served_model_name=f"offline-defended-{defense}",
        teacher_model=args.teacher_model,
        student_model=args.student_model,
        output_dir=attack_dir,
        query_pool=args.query_pool,
        query_ordering=args.query_ordering,
        teacher_mode=args.teacher_mode,
        teacher_temperature=args.teacher_temperature,
        teacher_top_p=args.teacher_top_p,
        teacher_max_tokens=args.teacher_max_tokens,
        stage1_config=args.stage1_config,
        extra_args=[
            *extra,
            "--warmup-model", str(checkpoint),
            "--execution-stage", "train",
            "--prepared-run-dir", str(prepared_run),
        ],
        teacher_transcript_path=transcript_path,
    )
    result["prerequisites"] = {"seqkd": seqkd, "prepared_run": str(prepared_run)}
    return result


def parse_runner_args(defense: str, argv: Optional[list[str]] = None) -> tuple[argparse.Namespace, list[str]]:
    args, attack_extra = parse_common_args(defense, argv)
    if not hasattr(args, "execution_mode"):
        args.execution_mode = "auto"
    return args, attack_extra


def _ordered_query_file(*, args: argparse.Namespace, output_dir: Path) -> Path:
    query_pool_spec = hf_query_pool_for_budget(args.budget) if args.query_pool in (None, "auto") else args.query_pool
    query_pool_path, ordering_path = resolve_query_pool_and_ordering(query_pool_spec, args.query_ordering)
    records, _ = load_query_pool_records(query_pool_path)

    if ordering_path is not None:
        ordering = json.loads(ordering_path.read_text(encoding="utf-8"))
        indices = [int(i) for i in ordering.get("ordered_indices", [])]
    else:
        indices = list(range(len(records)))
    if len(indices) < args.budget:
        raise ValueError(f"query ordering has {len(indices)} rows, smaller than budget={args.budget}")

    selected = []
    for order_index, pool_index in enumerate(indices[: args.budget]):
        raw = records[pool_index]
        query_id = str(raw.get("id") or raw.get("prompt_id") or raw.get("query_id") or f"row_{pool_index:08d}")
        query = str(raw.get("prompt") or raw.get("prompt_text") or raw.get("query") or "")
        if not query:
            raise ValueError(f"query pool record {pool_index} has no prompt/query text")
        selected.append(
            {
                "id": query_id,
                "query_id": query_id,
                "prompt": query,
                "query": query,
                "order_index": order_index,
                "source_pool_index": pool_index,
            }
        )

    path = output_dir / "attack_queries.jsonl"
    write_jsonl(selected, path)
    return path


def _write_teacher_query_log(transcript_path: Path, output_dir: Path) -> Path:
    rows = []
    with transcript_path.open("r", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            query = record.get("query") or record.get("prompt") or record.get("prompt_text")
            rows.append(
                {
                    "query_id": record.get("query_id") or record.get("id") or f"offline_{i:06d}",
                    "query": "" if query is None else str(query),
                    "source": "offline_batch_defended_transcript",
                }
            )
    path = output_dir / "teacher_received_queries.jsonl"
    write_jsonl(rows, path)
    return path


def _run_command(command: list[str], *, cwd: Path, log_path: Path) -> None:
    log_path = stage_log_path(log_path)
    proc = run_logged_command(command, cwd=cwd, log_path=log_path)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with exit code {proc.returncode}; see {log_path}")


def _generation_command(defense: str, args: argparse.Namespace, query_file: Path, oracle_dir: Path) -> list[str]:
    config = args.defense_config or "{}"
    common = ["--query_pool_path", str(query_file), "--output_dir", str(oracle_dir), "--max_queries", str(args.budget)]
    if defense == "clean":
        return [sys.executable, "-m", "defenses.oracle.generate_clean_teacher",
                "--teacher_model", args.teacher_model, *common,
                "--defense_config", config, "--device", args.device or "cuda",
                "--temperature", str(args.teacher_temperature), "--top_p", str(args.teacher_top_p),
                "--max_new_tokens", str(args.teacher_max_tokens), "--mode", args.teacher_mode]
    if defense == "ginsew":
        return [
            sys.executable,
            "-m",
            "defenses.ginsew.run",
            "--teacher_model",
            args.teacher_model,
            *common,
            "--watermark_config",
            config,
            "--device",
            args.device or "cuda",
            "--temperature",
            str(args.teacher_temperature),
            "--max_new_tokens",
            str(args.teacher_max_tokens),
        ]
    if defense == "radioactivity":
        return [
            sys.executable,
            "-m",
            "defenses.radioactivity.run",
            "--teacher_model",
            args.teacher_model,
            *common,
            "--watermark_config",
            config,
            "--device",
            args.device or "cuda",
            "--temperature",
            str(args.teacher_temperature),
            "--max_new_tokens",
            str(args.teacher_max_tokens),
        ]
    if defense == "adfp":
        return [
            sys.executable,
            "-m",
            "defenses.adfp.run",
            "--teacher_model",
            args.teacher_model,
            "--proxy_model",
            args.proxy_model or args.student_model or args.teacher_model,
            *common,
            "--adfp_config",
            config,
            "--batch_size",
            str(int(getattr(args, "adfp_batch_size", 8) or 8)),
            "--temperature",
            str(args.teacher_temperature),
            "--top_p",
            str(args.teacher_top_p),
            "--max_new_tokens",
            str(args.teacher_max_tokens),
        ]
    if defense == "ads":
        command = [
            sys.executable,
            "-m",
            "defenses.ads.run",
            "--teacher_model",
            args.teacher_model,
            *common,
            "--ads_config",
            config,
            "--proxy_student",
            args.proxy_model or args.student_model or args.teacher_model,
            "--device",
            args.device or "cuda",
            "--batch_size",
            str(int(getattr(args, "ads_batch_size", 8) or 8)),
            "--temperature",
            str(args.teacher_temperature),
            "--top_p",
            str(args.teacher_top_p),
            "--max_new_tokens",
            str(args.teacher_max_tokens),
        ]
        if args.grad_path:
            command.extend(["--grad_path", args.grad_path])
        return command
    if defense == "trace_rewriting":
        command = [
            sys.executable,
            "-m",
            "defenses.trace_rewriting.run",
            "--teacher_model",
            args.teacher_model,
            "--rewriter_model",
            args.rewriter_model or args.teacher_model,
            *common,
            "--trace_rewriting_config",
            config,
            "--device",
            args.device or "cuda",
        ]
        if args.teacher_base_url:
            command.extend(
                [
                    "--teacher_backend",
                    "openai_compatible",
                    "--teacher_base_url",
                    args.teacher_base_url,
                    "--teacher_request_model",
                    args.teacher_request_model or args.teacher_model,
                    "--teacher_api_key",
                    args.teacher_api_key,
                ]
            )
        if args.rewriter_backend:
            command.extend(["--rewriter_backend", args.rewriter_backend])
        if args.rewriter_base_url:
            command.extend(["--rewriter_base_url", args.rewriter_base_url])
        if args.rewriter_request_model:
            command.extend(["--rewriter_request_model", args.rewriter_request_model])
        if args.rewriter_api_key:
            command.extend(["--rewriter_api_key", args.rewriter_api_key])
        return command
    if defense == "doge":
        if not args.doge_checkpoint:
            raise ValueError("offline DOGe generation requires --doge-checkpoint")
        transcript_path = oracle_dir / "transcripts" / "defended_teacher.jsonl"
        manifest_path = oracle_dir / "defense_manifest.json"
        return [
            sys.executable,
            "-m",
            "defenses.doge.run",
            "generate",
            "--checkpoint",
            args.doge_checkpoint,
            "--input",
            str(query_file),
            "--output",
            str(transcript_path),
            "--manifest",
            str(manifest_path),
            "--max-new-tokens",
            str(args.teacher_max_tokens),
            "--temperature",
            str(args.teacher_temperature),
        ]
    raise ValueError(f"unsupported offline defense: {defense}")


def can_run_offline(defense: str, args: argparse.Namespace) -> bool:
    if defense not in OFFLINE_GENERATOR_DEFENSES:
        return False
    if args.attack not in OFFLINE_TRANSCRIPT_ATTACKS:
        return False
    if args.countermeasure != "none":
        return False
    return True


def run_offline_batch_defense(
    *,
    defense: str,
    args: argparse.Namespace,
    attack_extra_args: Optional[Iterable[str]] = None,
    detector_factory: Optional[DetectorFactory] = None,
) -> DefenseRunResult:
    repo_root = repo_root_from_here()
    if not can_run_offline(defense, args):
        raise ValueError(f"{defense}/{args.attack} does not support offline_batch execution")

    run_id = args.run_id or make_run_id(defense, args.attack, args.budget)
    output_dir = ensure_dir(Path(args.output_dir).expanduser().resolve() if args.output_dir else default_output_dir(repo_root, defense, run_id))
    oracle_dir = ensure_dir(output_dir / "oracle")
    attack_dir = ensure_dir(output_dir / "attack")

    replayed = bool(getattr(args, "reuse_defense_transcript", None))
    if replayed:
        if args.attack != "soda":
            raise ValueError("--reuse-defense-transcript is currently restricted to SODA warmup replay")
        transcript_path = Path(args.reuse_defense_transcript).expanduser().resolve()
        if not transcript_path.is_file():
            raise FileNotFoundError(f"reused defended transcript does not exist: {transcript_path}")
        artifacts_dir = (
            Path(args.reuse_defense_artifacts).expanduser().resolve()
            if getattr(args, "reuse_defense_artifacts", None) else transcript_path.parent.parent / "artifacts"
        )
        if not args.no_detector and DETECTOR_FACTORIES.get(defense) is not None and not artifacts_dir.is_dir():
            raise FileNotFoundError(f"reused defense artifacts do not exist: {artifacts_dir}")
        if getattr(args, "reuse_teacher_query_log", None):
            teacher_query_log = Path(args.reuse_teacher_query_log).expanduser().resolve()
            if not teacher_query_log.is_file():
                raise FileNotFoundError(f"reused teacher query log does not exist: {teacher_query_log}")
        else:
            teacher_query_log = _write_teacher_query_log(transcript_path, oracle_dir)
        oracle_manifest = (
            Path(args.reuse_oracle_manifest).expanduser().resolve()
            if getattr(args, "reuse_oracle_manifest", None) else transcript_path.parent.parent / "defense_manifest.json"
        )
        command = None
        query_file = None
    else:
        query_file = _ordered_query_file(args=args, output_dir=oracle_dir)
        command = _generation_command(defense, args, query_file, oracle_dir)
        _run_command(command, cwd=repo_root, log_path=oracle_dir / f"{defense}_offline_generation.log")

        transcript_path = oracle_dir / "transcripts" / "defended_teacher.jsonl"
        if not transcript_path.exists():
            raise FileNotFoundError(f"offline generation did not create transcript: {transcript_path}")
        teacher_query_log = _write_teacher_query_log(transcript_path, oracle_dir)
        oracle_manifest = oracle_dir / "defense_manifest.json"
        artifacts_dir = oracle_dir / "artifacts"

    if args.attack == "soda" and not args.dry_run and not replayed:
        attack_payload = _run_soda_with_prerequisites(
            repo_root=repo_root,
            args=args,
            attack_dir=attack_dir,
            transcript_path=transcript_path,
            attack_extra_args=attack_extra_args,
            defense=defense,
        )
    else:
        attack_payload = run_attack_process(
            repo_root=repo_root,
            attack=args.attack,
            budget=args.budget,
            teacher_base_url=None,
            served_model_name=f"offline-defended-{defense}",
            teacher_model=args.teacher_model,
            student_model=args.student_model,
            output_dir=attack_dir,
            query_pool=args.query_pool,
            query_ordering=args.query_ordering,
            teacher_mode=args.teacher_mode,
            teacher_temperature=args.teacher_temperature,
            teacher_top_p=args.teacher_top_p,
            teacher_max_tokens=args.teacher_max_tokens,
            stage1_config=args.stage1_config,
            dry_run=args.dry_run,
            extra_args=attack_extra_args,
            teacher_transcript_path=transcript_path,
        )

    attack_manifest = Path(attack_payload["manifest_path"]) if attack_payload.get("manifest_path") else None
    checkpoint = Path(attack_payload["checkpoint_path"]) if attack_payload.get("checkpoint_path") else None
    generator_result = GeneratorResult(
        defense=defense,
        run_id=run_id,
        output_dir=output_dir,
        oracle_dir=oracle_dir,
        oracle_base_url=f"offline_batch://{defense}",
        oracle_manifest_path=oracle_manifest,
        teacher_query_log_path=teacher_query_log,
        defended_transcript_path=transcript_path,
        defense_artifacts_dir=artifacts_dir,
        attack_output_dir=attack_dir,
        attack_manifest_path=attack_manifest,
        student_checkpoint_path=checkpoint,
        status="ok",
        metadata={
            "execution_mode": "offline_replay" if replayed else "offline_batch",
            "attack": attack_payload,
            "offline_generation_command": command,
            "offline_generation_log": None if replayed else str(stage_log_path(oracle_dir / f"{defense}_offline_generation.log").resolve()),
            "query_file": None if query_file is None else str(query_file.resolve()),
            "reused_defended_transcript": str(transcript_path) if replayed else None,
            "reused_defense_artifacts": str(artifacts_dir) if replayed else None,
        },
    )

    detector_result = None
    factory = detector_factory if detector_factory is not None else DETECTOR_FACTORIES.get(defense)
    if not args.no_detector and factory is not None and not args.dry_run:
        detector_command = factory(args, generator_result)
        if detector_command:
            detector_result = run_detector_command(
                repo_root=repo_root,
                detector=defense,
                command=detector_command,
                output_dir=generator_result.output_dir / "detector" / defense,
            )

    result = DefenseRunResult(
        defense=defense,
        run_id=run_id,
        output_dir=output_dir,
        generator_result=generator_result,
        detector_result=detector_result,
        metadata={"execution_mode": "offline_replay" if replayed else "offline_batch"},
    )
    manifest_path = output_dir / "defense_run_manifest.json"
    manifest_path.write_text(json.dumps(result.to_manifest(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({**result.to_manifest(), "manifest_path": str(manifest_path)}, indent=2, ensure_ascii=False))
    return result


def run_defense(
    *,
    defense: str,
    args: argparse.Namespace,
    attack_extra_args: Optional[Iterable[str]] = None,
    detector_factory: Optional[DetectorFactory] = None,
) -> DefenseRunResult:
    mode = args.execution_mode
    if mode == "offline_batch" or (mode == "auto" and can_run_offline(defense, args)):
        return run_offline_batch_defense(
            defense=defense,
            args=args,
            attack_extra_args=attack_extra_args,
            detector_factory=detector_factory,
        )
    return run_online_defense(
        defense=defense,
        args=args,
        attack_extra_args=attack_extra_args,
        detector_factory=detector_factory,
    )


def main_for_defense(defense: str, argv: Optional[list[str]] = None) -> None:
    args, attack_extra = parse_runner_args(defense, argv)
    run_defense(defense=defense, args=args, attack_extra_args=attack_extra)
