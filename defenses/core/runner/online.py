from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
from dataclasses import replace
import json
import socket
import subprocess
import sys
import time
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

from defenses.core.detector.commands import adfp_detector_command, run_detector_command, watermark_detector_command
from defenses.core.detector.result import DetectorResult
from defenses.core.generator.attack import run_attack_process
from defenses.core.generator.oracle import start_oracle
from defenses.core.generator.result import GeneratorResult
from defenses.core.io_utils import ensure_dir
from defenses.core.process_logging import run_logged_command, stage_log_path, start_logged_process
from defenses.core.runner.result import DefenseRunResult

DetectorFactory = Callable[[argparse.Namespace, GeneratorResult], Optional[list[str]]]


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[3]


def make_run_id(defense: str, attack: str, budget: int) -> str:
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{defense}_{attack}_b{budget}"


def find_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def split_passthrough(values: list[str]) -> tuple[list[str], list[str]]:
    if "--" not in values:
        return values, []
    idx = values.index("--")
    return values[:idx], values[idx + 1:]


def add_common_online_args(parser: argparse.ArgumentParser, *, defense: str) -> None:
    parser.add_argument("--defense", default=defense, help=argparse.SUPPRESS)
    parser.add_argument("--attack", default="seqkd")
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--query-pool", default="auto")
    parser.add_argument("--query-ordering", default="auto")
    parser.add_argument("--teacher-model", default=None)
    parser.add_argument("--student-model", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0, help="0 chooses a free local port")
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--defense-config", default=None, help="Inline JSON or path to JSON config passed to the oracle")
    parser.add_argument("--device", default=None)
    parser.add_argument("--teacher-base-url", default=None)
    parser.add_argument("--teacher-request-model", default=None)
    parser.add_argument("--teacher-api-key", default="EMPTY")
    parser.add_argument("--teacher-mode", default="chat", choices=["chat", "completion"])
    parser.add_argument("--teacher-temperature", type=float, default=0.0)
    parser.add_argument("--teacher-top-p", type=float, default=1.0)
    parser.add_argument("--teacher-max-tokens", type=int, default=1536)
    parser.add_argument("--stage1-config", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--startup-timeout", type=float, default=600.0)
    parser.add_argument("--no-detector", action="store_true")
    parser.add_argument("--probe-queries", default=None, help="Defaults to oracle teacher_received_queries.jsonl for watermark detectors")
    parser.add_argument("--reuse-defense-transcript", default=None, help="Existing defended transcript to replay instead of querying the teacher")
    parser.add_argument("--reuse-defense-artifacts", default=None, help="Defense artifacts paired with --reuse-defense-transcript")
    parser.add_argument("--reuse-oracle-manifest", default=None, help="Oracle/defense manifest paired with a replayed transcript")
    parser.add_argument("--reuse-teacher-query-log", default=None, help="Existing received-query log paired with a replayed transcript")
    parser.add_argument("--detector-max-queries", type=int, default=None)
    parser.add_argument("--detector-max-new-tokens", type=int, default=64)
    parser.add_argument("--detector-temperature", type=float, default=0.7)
    parser.add_argument("--detector-seed", type=int, default=42)
    parser.add_argument("--detector-extra-args", default="", help="Extra detector args as a JSON list or shell-like string is not parsed; prefer -- passthrough per script if needed")
    parser.add_argument("--proxy-model", default=None)
    parser.add_argument("--grad-path", default=None)
    parser.add_argument("--rewriter-model", default=None)
    parser.add_argument("--rewriter-backend", choices=["local_hf", "openai_compatible"], default="local_hf")
    parser.add_argument("--rewriter-base-url", default=None)
    parser.add_argument("--rewriter-request-model", default=None)
    parser.add_argument("--rewriter-api-key", default="EMPTY")
    parser.add_argument("--doge-checkpoint", default=None)
    parser.add_argument("--adfp-batch-size", type=int, default=8)
    parser.add_argument("--ads-batch-size", type=int, default=8)
    parser.add_argument("--countermeasure", choices=["none", "dipper", "translation"], default="none")
    parser.add_argument("--counter-baselines", action="store_true", help="Train and detect clean and defense-only students in this counter job")
    parser.add_argument("--prepare-counter-baselines", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--require-counter-baselines", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--countermeasure-lex", type=int, default=40, help="DIPPER lexical diversity. MEA default: 40.")
    parser.add_argument("--countermeasure-order", type=int, default=0, help="DIPPER order diversity. MEA default: 0.")
    parser.add_argument("--countermeasure-sent-interval", type=int, default=3)
    parser.add_argument("--countermeasure-with-context", action="store_true")
    parser.add_argument("--countermeasure-model-name", default=None)
    parser.add_argument("--countermeasure-tokenizer-name", default=None)
    parser.add_argument("--countermeasure-device", default=None)
    parser.add_argument(
        "--countermeasure-timeout",
        type=float,
        default=3600.0,
        help="Seconds the countermeasure proxy waits for one defended-teacher response.",
    )
    parser.add_argument(
        "--execution-mode",
        choices=["auto", "offline_batch", "online"],
        default="auto",
        help="auto uses offline_batch when this defense/attack pair supports a precomputed transcript.",
    )


def parse_common_args(defense: str, argv: Optional[list[str]] = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=f"Run {defense} as an online defended-teacher benchmark")
    add_common_online_args(parser, defense=defense)
    front, attack_extra = split_passthrough(list(sys.argv[1:] if argv is None else argv))
    args = parser.parse_args(front)
    return args, attack_extra


def default_output_dir(repo_root: Path, defense: str, run_id: str) -> Path:
    return repo_root / "outputs" / "defenses" / defense / run_id


def detector_for_watermark(module: str, detector_name: str) -> DetectorFactory:
    def build(args: argparse.Namespace, result: GeneratorResult) -> Optional[list[str]]:
        if result.attack_manifest_path is None:
            return None
        probe_queries = Path(args.probe_queries).expanduser().resolve() if args.probe_queries else result.teacher_query_log_path
        return watermark_detector_command(
            module=module,
            artifacts_dir=result.defense_artifacts_dir,
            probe_queries=probe_queries,
            attack_manifest=result.attack_manifest_path,
            output_dir=result.output_dir / "detector" / detector_name,
            max_queries=args.detector_max_queries,
            max_new_tokens=args.detector_max_new_tokens,
            temperature=args.detector_temperature,
            seed=getattr(args, "detector_seed", 42),
        )
    return build


def detector_for_adfp(args: argparse.Namespace, result: GeneratorResult) -> Optional[list[str]]:
    if result.attack_manifest_path is None:
        return None
    return adfp_detector_command(
        artifacts_dir=result.defense_artifacts_dir,
        transcript_path=result.defended_transcript_path,
        attack_manifest=result.attack_manifest_path,
        output_dir=result.output_dir / "detector" / "adfp",
        max_contexts=args.detector_max_queries,
    )


def start_countermeasure_proxy(
    *,
    repo_root: Path,
    args: argparse.Namespace,
    target_base_url: str,
    output_dir: Path,
    port: int,
) -> tuple[subprocess.Popen, str, Path, Path, Path]:
    proxy_dir = ensure_dir(output_dir / "countermeasure" / f"waterpark_{args.countermeasure}")
    log_path = stage_log_path(proxy_dir / "countermeasure_proxy.log")
    manifest_path = proxy_dir / "countermeasure_manifest.json"
    transcript_path = proxy_dir / "countermeasure_teacher_transcript.jsonl"
    command = [
        sys.executable,
        "-m",
        "countermeasures.openai_proxy",
        "--target-base-url",
        target_base_url,
        "--host",
        args.host,
        "--port",
        str(port),
        "--output-dir",
        str(proxy_dir),
        "--countermeasure",
        args.countermeasure,
        "--api-key",
        args.teacher_api_key,
        "--timeout",
        str(int(args.countermeasure_timeout)),
        "--lex",
        str(args.countermeasure_lex),
        "--order",
        str(args.countermeasure_order),
        "--sent-interval",
        str(args.countermeasure_sent_interval),
    ]
    if args.countermeasure_with_context:
        command.append("--with-context")
    if args.countermeasure_model_name:
        command.extend(["--model-name", args.countermeasure_model_name])
    if args.countermeasure_tokenizer_name:
        command.extend(["--tokenizer-name", args.countermeasure_tokenizer_name])
    if args.countermeasure_device:
        command.extend(["--device", args.countermeasure_device])

    process = start_logged_process(command, cwd=repo_root, log_path=log_path)
    base_url = f"http://{args.host}:{port}/v1"
    try:
        from defenses.core.generator.oracle import wait_for_oracle

        wait_for_oracle(base_url, process, timeout=args.startup_timeout)
    except Exception:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        raise
    return process, base_url, manifest_path, transcript_path, log_path


def terminate_process(process: Optional[subprocess.Popen], timeout: float = 20.0) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout)


DETECTOR_FACTORIES: Dict[str, DetectorFactory] = {
    "ginsew": detector_for_watermark("defenses.ginsew.detector_run", "ginsew"),
    "radioactivity": detector_for_watermark("defenses.radioactivity.detector_run", "radioactivity"),
    "adfp": detector_for_adfp,
}


def run_online_defense(
    *,
    defense: str,
    args: argparse.Namespace,
    attack_extra_args: Optional[Iterable[str]] = None,
    detector_factory: Optional[DetectorFactory] = None,
) -> DefenseRunResult:
    repo_root = repo_root_from_here()
    run_id = args.run_id or make_run_id(defense, args.attack, args.budget)
    output_dir = ensure_dir(Path(args.output_dir).expanduser().resolve() if args.output_dir else default_output_dir(repo_root, defense, run_id))
    if getattr(args, "prepare_counter_baselines", False):
        placeholder = GeneratorResult(
            defense=defense, run_id="baseline_preparation", output_dir=output_dir,
            oracle_dir=output_dir / "oracle", oracle_base_url="baseline_preparation",
            oracle_manifest_path=output_dir / "oracle_manifest.json",
            teacher_query_log_path=output_dir / "queries.jsonl",
            defended_transcript_path=output_dir / "transcript.jsonl",
            defense_artifacts_dir=output_dir / "artifacts", attack_output_dir=output_dir / "attack",
            attack_manifest_path=None, student_checkpoint_path=None, status="pending",
        )
        baselines = ensure_shared_baselines(defense=defense, args=args,
                                            attack_extra_args=attack_extra_args, reference=placeholder)
        print(json.dumps({"status": "completed", "baseline_manifests": {
            key: str(value.output_dir / "baseline_manifest.json") for key, value in baselines.items()
        }}, indent=2))
        return DefenseRunResult(defense=defense, run_id="baseline_preparation", output_dir=output_dir,
                                generator_result=baselines.get("defense_only", baselines["clean"]),
                                metadata={"baseline_preparation": True})
    if (args.countermeasure != "none" and getattr(args, "counter_baselines", False)
            and args.attack in {"seqkd", "lord", "soda", "gad"} and not args.dry_run):
        return run_shared_transcript_counter(defense=defense, args=args, output_dir=output_dir,
                                             run_id=run_id, attack_extra_args=attack_extra_args,
                                             detector_factory=detector_factory)
    oracle_dir = ensure_dir(output_dir / "oracle")
    attack_dir = ensure_dir(output_dir / "attack")
    port = args.port or find_free_port(args.host)
    served_model_name = args.served_model_name or f"defended-{defense}"

    oracle = start_oracle(
        repo_root=repo_root,
        defense=defense,
        output_dir=oracle_dir,
        host=args.host,
        port=port,
        served_model_name=served_model_name,
        teacher_model=args.teacher_model,
        defense_config=args.defense_config,
        device=args.device,
        teacher_base_url=args.teacher_base_url,
        teacher_request_model=args.teacher_request_model,
        teacher_api_key=args.teacher_api_key,
        proxy_model=args.proxy_model,
        grad_path=args.grad_path,
        rewriter_model=args.rewriter_model,
        rewriter_backend=args.rewriter_backend,
        rewriter_base_url=args.rewriter_base_url,
        rewriter_request_model=args.rewriter_request_model,
        rewriter_api_key=args.rewriter_api_key,
        doge_checkpoint=args.doge_checkpoint,
        run_id=run_id,
        startup_timeout=args.startup_timeout,
    )
    attack_payload: Dict[str, Any]
    countermeasure_enabled = args.countermeasure != "none"
    countermeasure_process: Optional[subprocess.Popen] = None
    countermeasure_base_url: Optional[str] = None
    countermeasure_manifest_path: Optional[Path] = None
    countermeasure_transcript_path: Optional[Path] = None
    countermeasure_log_path: Optional[Path] = None
    release_monitor: Optional[threading.Thread] = None
    release_monitor_error: list[BaseException] = []
    try:
        attack_teacher_base_url = oracle.base_url
        if countermeasure_enabled:
            proxy_port = find_free_port(args.host)
            (
                countermeasure_process,
                countermeasure_base_url,
                countermeasure_manifest_path,
                countermeasure_transcript_path,
                countermeasure_log_path,
            ) = start_countermeasure_proxy(
                repo_root=repo_root,
                args=args,
                target_base_url=oracle.base_url,
                output_dir=output_dir,
                port=proxy_port,
            )
            attack_teacher_base_url = countermeasure_base_url

        attack_env = None
        if args.attack == "qedks" and not args.dry_run:
            release_request = attack_dir / ".release_teacher.request"
            release_ack = attack_dir / ".release_teacher.ack"
            release_request.unlink(missing_ok=True)
            release_ack.unlink(missing_ok=True)
            attack_env = {
                "QEDKS_RELEASE_TEACHER_REQUEST": str(release_request),
                "QEDKS_RELEASE_TEACHER_ACK": str(release_ack),
            }

            def release_teacher_before_training() -> None:
                try:
                    while not release_request.exists():
                        if oracle.process.poll() is not None:
                            return
                        time.sleep(0.2)
                    terminate_process(countermeasure_process)
                    oracle.terminate()
                    release_ack.write_text("teacher resources released\n", encoding="utf-8")
                except BaseException as exc:
                    release_monitor_error.append(exc)

            release_monitor = threading.Thread(
                target=release_teacher_before_training,
                name="qedks-teacher-release",
                daemon=True,
            )
            release_monitor.start()

        attack_payload = run_attack_process(
            repo_root=repo_root,
            attack=args.attack,
            budget=args.budget,
            teacher_base_url=attack_teacher_base_url,
            served_model_name=served_model_name,
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
            countermeasure=f"waterpark_{args.countermeasure}" if countermeasure_enabled else None,
            countermeasure_manifest=countermeasure_manifest_path,
            extra_env=attack_env,
        )
        if release_monitor is not None:
            release_monitor.join(timeout=10)
        if release_monitor_error:
            raise RuntimeError("failed to release QEDKS teacher before training") from release_monitor_error[0]
    finally:
        terminate_process(countermeasure_process)
        oracle.terminate()

    attack_manifest = Path(attack_payload["manifest_path"]) if attack_payload.get("manifest_path") else None
    checkpoint = Path(attack_payload["checkpoint_path"]) if attack_payload.get("checkpoint_path") else None
    generator_result = GeneratorResult(
        defense=defense,
        run_id=run_id,
        output_dir=output_dir,
        oracle_dir=oracle.output_dir,
        oracle_base_url=oracle.base_url,
        oracle_manifest_path=oracle.manifest_path,
        teacher_query_log_path=oracle.query_log_path,
        defended_transcript_path=oracle.transcript_path,
        defense_artifacts_dir=oracle.artifacts_dir,
        attack_output_dir=attack_dir,
        attack_manifest_path=attack_manifest,
        student_checkpoint_path=checkpoint,
        status="ok",
        metadata={
            "attack": attack_payload,
            "oracle_log_path": str(oracle.log_path),
            "countermeasure_enabled": countermeasure_enabled,
            "countermeasure": f"waterpark_{args.countermeasure}" if countermeasure_enabled else None,
            "countermeasure_base_url": countermeasure_base_url,
            "countermeasure_manifest_path": None if countermeasure_manifest_path is None else str(countermeasure_manifest_path),
            "countermeasure_transcript_path": None if countermeasure_transcript_path is None else str(countermeasure_transcript_path),
            "countermeasure_log_path": None if countermeasure_log_path is None else str(countermeasure_log_path),
        },
    )

    detector_result: Optional[DetectorResult] = None
    factory = detector_factory if detector_factory is not None else DETECTOR_FACTORIES.get(defense)
    shared_baselines = countermeasure_enabled and getattr(args, "counter_baselines", False)
    if not args.no_detector and factory is not None and not args.dry_run and not shared_baselines:
        command = factory(args, generator_result)
        if command:
            detector_result = run_detector_command(
                repo_root=repo_root,
                detector=defense,
                command=command,
                output_dir=generator_result.output_dir / "detector" / defense,
            )

    result = DefenseRunResult(
        defense=defense,
        run_id=run_id,
        output_dir=output_dir,
        generator_result=generator_result,
        detector_result=detector_result,
        metadata={
            "online_oracle": True,
            "countermeasure_enabled": countermeasure_enabled,
            "countermeasure": f"waterpark_{args.countermeasure}" if countermeasure_enabled else None,
        },
    )
    if countermeasure_enabled and getattr(args, "counter_baselines", False) and not args.dry_run:
        comparison, detector_result = run_counter_baselines(
            defense=defense, args=args, attack_extra_args=attack_extra_args,
            reference=generator_result, counter_detector=detector_result, factory=factory,
        )
        result = replace(result, detector_result=detector_result,
                         metadata={**result.metadata, "comparison": str(comparison)})
    manifest_path = output_dir / "defense_run_manifest.json"
    manifest_path.write_text(json.dumps(result.to_manifest(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({**result.to_manifest(), "manifest_path": str(manifest_path)}, indent=2, ensure_ascii=False))
    return result


def run_counter_baselines(
    *, defense: str, args: argparse.Namespace,
    attack_extra_args: Optional[Iterable[str]], reference: GeneratorResult,
    counter_detector: Optional[DetectorResult], factory: Optional[DetectorFactory],
    baselines: Optional[Dict[str, GeneratorResult]] = None,
) -> tuple[Path, Optional[DetectorResult]]:
    repo_root = repo_root_from_here()
    if baselines is None:
        baselines = ensure_shared_baselines(defense=defense, args=args,
                                            attack_extra_args=attack_extra_args, reference=reference)

    # Shared defense-only artifacts and probes define all three detection runs.
    detection_reference = baselines["defense_only"]
    reports: Dict[str, Any] = {}
    report_paths: Dict[str, str] = {}
    for group, student in {**baselines, "counter": reference}.items():
        if factory and not args.no_detector:
            label = "negative" if group == "clean" else "positive"
            if group == "counter":
                target_dir = reference.output_dir
                target = replace(detection_reference, output_dir=target_dir,
                                 attack_manifest_path=student.attack_manifest_path,
                                 student_checkpoint_path=student.student_checkpoint_path)
                command = factory(args, target)
                detected = None
                if command:
                    command[command.index("--label") + 1] = label
                    detected = run_detector_command(repo_root=repo_root, detector=defense, command=command,
                                                    output_dir=target_dir / "detector" / defense)
                counter_detector = detected or counter_detector
            else:
                detected = run_shared_baseline_detection(
                    defense=defense, args=args, detection_reference=detection_reference,
                    student=student, group=group, label=label, factory=factory,
                )
            if detected and detected.report_path:
                reports[group] = json.loads(detected.report_path.read_text(encoding="utf-8"))
                report_paths[group] = str(detected.report_path.resolve())
    path = reference.output_dir / "comparison_report.json"
    path.write_text(json.dumps({"defense": defense, "attack": args.attack,
                               "budget": args.budget, "reports": reports,
                               "report_paths": report_paths,
                               "baseline_manifests": {k: str(v.output_dir / "baseline_manifest.json")
                                                      for k, v in baselines.items()}}, indent=2) + "\n", encoding="utf-8")
    return path, counter_detector


def run_shared_baseline_detection(
    *, defense: str, args: argparse.Namespace, detection_reference: GeneratorResult,
    student: GeneratorResult, group: str, label: str, factory: DetectorFactory,
) -> Optional[DetectorResult]:
    """Run each Clean/Protected detector once and reuse it across counter branches."""
    if group not in {"clean", "defense_only"}:
        raise ValueError(f"Unsupported shared detector baseline group: {group}")
    identity = {
        "schema_version": 1,
        "defense": defense,
        "attack": args.attack,
        "budget": args.budget,
        "group": group,
        "label": label,
        "artifacts": str(detection_reference.defense_artifacts_dir.resolve()),
        "probes": str(detection_reference.teacher_query_log_path.resolve()),
        "attack_manifest": str(student.attack_manifest_path.resolve()),
        "student_checkpoint": str(student.student_checkpoint_path.resolve()),
        "detector_max_queries": getattr(args, "detector_max_queries", None),
        "detector_max_new_tokens": getattr(args, "detector_max_new_tokens", None),
        "detector_temperature": getattr(args, "detector_temperature", None),
        "detector_seed": getattr(args, "detector_seed", 42),
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    cache_dir = ensure_dir(detection_reference.output_dir / "shared_detector_baselines" / group / digest)
    detector_dir = cache_dir / "detector" / defense
    protocol_path = cache_dir / "protocol.json"
    report_path = detector_dir / "detector_report.json"
    manifest_path = detector_dir / "detector_manifest.json"
    log_path = detector_dir / f"{defense}.log"
    with (cache_dir / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if protocol_path.is_file() and report_path.is_file():
            if json.loads(protocol_path.read_text(encoding="utf-8")) != identity:
                raise RuntimeError(f"Shared detector baseline protocol mismatch: {cache_dir}")
            print(f"[counter] Reusing shared {defense} detector baseline {group}: {report_path}", flush=True)
            return DetectorResult(defense, "ok", detector_dir, report_path,
                                  manifest_path if manifest_path.is_file() else None,
                                  log_path if log_path.is_file() else None)
        target = replace(detection_reference, output_dir=cache_dir,
                         attack_manifest_path=student.attack_manifest_path,
                         student_checkpoint_path=student.student_checkpoint_path)
        command = factory(args, target)
        if not command:
            return None
        command[command.index("--label") + 1] = label
        print(f"[counter] Running shared {defense} detector baseline {group}: {cache_dir}", flush=True)
        detected = run_detector_command(repo_root=repo_root_from_here(), detector=defense, command=command,
                                        output_dir=detector_dir)
        if not detected.report_path:
            raise RuntimeError(f"Shared detector baseline produced no report: {cache_dir}")
        temporary = protocol_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(protocol_path)
        return detected


def ensure_shared_baselines(
    *, defense: str, args: argparse.Namespace,
    attack_extra_args: Optional[Iterable[str]], reference: GeneratorResult,
) -> Dict[str, GeneratorResult]:
    repo_root = repo_root_from_here()
    extra_args = list(attack_extra_args or [])
    cfg = load_baseline_config(args.defense_config)
    identity = {key: getattr(args, key, None) for key in (
        "attack", "budget", "teacher_model", "student_model", "query_pool", "query_ordering",
        "teacher_mode", "teacher_temperature", "teacher_top_p", "teacher_max_tokens",
        "stage1_config", "device",
    )}
    identity["attack_extra_args"] = extra_args
    if args.attack in {"seqkd", "lord"}:
        identity["stage1_runtime_flags_version"] = 1
    identity["system_prompt"] = cfg.get("system_prompt")
    identity["training_environment"] = {
        key: value for key, value in os.environ.items()
        if key.startswith(("LORD_", "STAGE1_SEQKD_")) or key == "STAGE1_SEED"
    }
    for key in ("stage1_config", "query_pool", "query_ordering"):
        value = identity.get(key)
        if value and Path(str(value)).is_file():
            identity[key + "_sha256"] = hashlib.sha256(Path(str(value)).read_bytes()).hexdigest()
    storage = Path(os.environ.get("STORAGE_ROOT", str(repo_root)))
    shared_root = Path(os.environ.get("COUNTER_BASELINE_ROOT", str(storage / "outputs" / "counter_baselines")))
    baselines: Dict[str, GeneratorResult] = {}
    groups = ("clean",) if os.environ.get("COUNTER_CLEAN_ONLY") == "1" else ("clean", "defense_only")
    for group in groups:
        group_identity = {**identity, "schema_version": 2}
        if group == "defense_only":
            group_identity.update(defense=defense, defense_config=cfg, proxy_model=args.proxy_model)
        digest = hashlib.sha256(json.dumps(group_identity, sort_keys=True).encode()).hexdigest()[:20]
        group_dir = ensure_dir(shared_root / f"{args.attack}_b{args.budget}" / group /
                               (defense if group == "defense_only" else "shared") / digest)
        baseline_args = argparse.Namespace(**vars(args))
        baseline_args.countermeasure = "none"
        baseline_args.counter_baselines = False
        baseline_args.prepare_counter_baselines = False
        baseline_args.no_detector = True
        baseline_args.output_dir = str(group_dir)
        baseline_args.run_id = f"{reference.run_id}_{group}"
        if group == "clean":
            baseline_args.teacher_base_url = None
            baseline_args.defense_config = json.dumps({"system_prompt": cfg.get("system_prompt")})
        # Hold the lock through training; publish only successful, usable results.
        with (group_dir / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            manifest = group_dir / "baseline_manifest.json"
            baseline = load_completed_baseline(manifest)
            if baseline is None:
                if getattr(args, "require_counter_baselines", False):
                    raise RuntimeError(f"required shared {group} baseline is missing or incomplete: {manifest}")
                print(f"[counter] Training shared {group}: {group_dir}", flush=True)
                baseline = train_shared_baseline(
                    defense="clean" if group == "clean" else defense,
                    args=baseline_args, attack_extra_args=extra_args,
                ).generator_result
                if baseline.attack_manifest_path is None or baseline.student_checkpoint_path is None:
                    raise RuntimeError(f"{group} completed without an attack manifest or checkpoint")
                temporary = manifest.with_suffix(".tmp")
                temporary.write_text(json.dumps(baseline.to_manifest(), indent=2) + "\n", encoding="utf-8")
                temporary.replace(manifest)
            else:
                print(f"[counter] Reusing shared {group}: {manifest}", flush=True)
            baselines[group] = baseline

    return baselines


def train_shared_baseline(
    *, defense: str, args: argparse.Namespace, attack_extra_args: Optional[Iterable[str]],
) -> DefenseRunResult:
    if defense in {"clean", "ginsew", "radioactivity", "adfp"} and args.attack in {"seqkd", "lord", "soda", "gad"}:
        from defenses.core.runner.offline import run_offline_batch_defense

        return run_offline_batch_defense(defense=defense, args=args, attack_extra_args=attack_extra_args)
    return run_online_defense(defense=defense, args=args, attack_extra_args=attack_extra_args)


def run_shared_transcript_counter(
    *, defense: str, args: argparse.Namespace, output_dir: Path, run_id: str,
    attack_extra_args: Optional[Iterable[str]], detector_factory: Optional[DetectorFactory],
) -> DefenseRunResult:
    repo_root = repo_root_from_here()
    extra_args = list(attack_extra_args or [])
    placeholder = GeneratorResult(
        defense=defense, run_id=run_id, output_dir=output_dir,
        oracle_dir=output_dir / "oracle", oracle_base_url="shared_transcript",
        oracle_manifest_path=output_dir / "oracle_manifest.json",
        teacher_query_log_path=output_dir / "queries.jsonl",
        defended_transcript_path=output_dir / "transcript.jsonl",
        defense_artifacts_dir=output_dir / "artifacts", attack_output_dir=output_dir / "attack",
        attack_manifest_path=None, student_checkpoint_path=None, status="pending",
    )
    baselines = ensure_shared_baselines(defense=defense, args=args,
                                       attack_extra_args=extra_args, reference=placeholder)
    defended = baselines["defense_only"]
    source = defended.defended_transcript_path
    if not source.is_file():
        raise FileNotFoundError(f"shared defense-only transcript is missing: {source}")
    counter_dir = ensure_dir(output_dir / "countermeasure" / f"waterpark_{args.countermeasure}")
    transcript = counter_dir / "cleaned_teacher.jsonl"
    manifest = counter_dir / "countermeasure_manifest.json"
    log_path = stage_log_path(counter_dir / "countermeasure_rewrite.log")
    command = [sys.executable, "-m", "countermeasures.waterpark_response",
               "--method", args.countermeasure, "--input", str(source),
               "--output", str(transcript), "--manifest", str(manifest),
               "--lex", str(args.countermeasure_lex), "--order", str(args.countermeasure_order),
               "--sent-interval", str(args.countermeasure_sent_interval)]
    if args.countermeasure_with_context:
        command.append("--with-context")
    for flag, value in (("--model-name", args.countermeasure_model_name),
                        ("--tokenizer-name", args.countermeasure_tokenizer_name),
                        ("--device", args.countermeasure_device)):
        if value:
            command.extend([flag, value])
    print(f"[counter] Rewriting shared defense-only transcript: {source}", flush=True)
    process = run_logged_command(command, cwd=repo_root, log_path=log_path)
    if process.returncode:
        raise RuntimeError(f"countermeasure rewrite failed with exit code {process.returncode}; see {log_path}")
    print(f"[counter] Rewrite completed; starting {args.attack} training", flush=True)
    payload = run_attack_process(
        repo_root=repo_root, attack=args.attack, budget=args.budget,
        teacher_base_url=None, served_model_name=f"defended-{defense}",
        teacher_model=args.teacher_model, student_model=args.student_model,
        output_dir=output_dir / "attack", query_pool=args.query_pool, query_ordering=args.query_ordering,
        teacher_mode=args.teacher_mode, teacher_temperature=args.teacher_temperature,
        teacher_top_p=args.teacher_top_p, teacher_max_tokens=args.teacher_max_tokens,
        stage1_config=args.stage1_config, extra_args=extra_args,
        teacher_transcript_path=transcript, countermeasure=f"waterpark_{args.countermeasure}",
        countermeasure_manifest=manifest,
    )
    generator = replace(
        defended, run_id=run_id, output_dir=output_dir, attack_output_dir=output_dir / "attack",
        attack_manifest_path=Path(payload["manifest_path"]) if payload.get("manifest_path") else None,
        student_checkpoint_path=Path(payload["checkpoint_path"]) if payload.get("checkpoint_path") else None,
        metadata={"execution_mode": "shared_transcript_counter", "attack": payload,
                  "countermeasure_enabled": True, "countermeasure": f"waterpark_{args.countermeasure}",
                  "source_defended_transcript": str(source),
                  "countermeasure_transcript_path": str(transcript),
                  "countermeasure_manifest_path": str(manifest), "countermeasure_log_path": str(log_path)},
    )
    factory = detector_factory if detector_factory is not None else DETECTOR_FACTORIES.get(defense)
    comparison, detector = run_counter_baselines(
        defense=defense, args=args, attack_extra_args=extra_args, reference=generator,
        counter_detector=None, factory=factory, baselines=baselines,
    )
    result = DefenseRunResult(
        defense=defense, run_id=run_id, output_dir=output_dir,
        generator_result=generator, detector_result=detector,
        metadata={"execution_mode": "shared_transcript_counter", "comparison": str(comparison)},
    )
    path = output_dir / "defense_run_manifest.json"
    path.write_text(json.dumps(result.to_manifest(), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**result.to_manifest(), "manifest_path": str(path)}, indent=2))
    return result


def load_completed_baseline(path: Path) -> Optional[GeneratorResult]:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("attack_manifest_path", "student_checkpoint_path"):
        if not payload.get(key) or not Path(payload[key]).exists():
            return None
    if payload.get("status") != "ok":
        return None
    for key in ("defended_transcript_path", "teacher_query_log_path", "defense_artifacts_dir"):
        if not payload.get(key) or not Path(payload[key]).exists():
            return None
    for key in ("output_dir", "oracle_dir", "oracle_manifest_path", "teacher_query_log_path",
                "defended_transcript_path", "defense_artifacts_dir", "attack_output_dir",
                "attack_manifest_path", "student_checkpoint_path"):
        payload[key] = Path(payload[key]) if payload.get(key) else None
    return GeneratorResult(**payload)


def load_baseline_config(raw: Optional[str]) -> Dict[str, Any]:
    if not raw:
        return {}
    path = Path(raw)
    return json.loads(path.read_text(encoding="utf-8") if path.is_file() else raw)


def main_for_defense(defense: str, argv: Optional[list[str]] = None) -> None:
    args, attack_extra = parse_common_args(defense, argv)
    if args.prepare_counter_baselines:
        output = ensure_dir(Path(args.output_dir).expanduser().resolve())
        placeholder = GeneratorResult(
            defense=defense, run_id="baseline_preparation", output_dir=output,
            oracle_dir=output / "oracle", oracle_base_url="baseline_preparation",
            oracle_manifest_path=output / "oracle_manifest.json",
            teacher_query_log_path=output / "queries.jsonl",
            defended_transcript_path=output / "transcript.jsonl",
            defense_artifacts_dir=output / "artifacts", attack_output_dir=output / "attack",
            attack_manifest_path=None, student_checkpoint_path=None, status="pending",
        )
        baselines = ensure_shared_baselines(defense=defense, args=args,
                                            attack_extra_args=attack_extra, reference=placeholder)
        print(json.dumps({"status": "completed", "baseline_manifests": {
            key: str(value.output_dir / "baseline_manifest.json") for key, value in baselines.items()
        }}, indent=2))
        return
    run_online_defense(defense=defense, args=args, attack_extra_args=attack_extra)
