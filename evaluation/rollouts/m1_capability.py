from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

from evaluation.tasks.adapters import AdaptedExample
from evaluation.core.clients import OpenAICompatibleClient, RequestResult, sanitize_base_url
from evaluation.core.code_execution import execute_humaneval
from evaluation.core.config import ConfigError, M1Config, apply_cli_overrides, load_config
from evaluation.core.metadata import finalize_manifest, initial_manifest, new_run_id, utc_now
from evaluation.metrics.m1_capability import report_markdown, write_summary_artifacts
from evaluation.core.models import collect_model_metadata
from evaluation.tasks.registry import DatasetDiscovery, DatasetRegistry, build_sample_manifest
from evaluation.core.validation import SchemaValidationError, validate_prediction_record
from evaluation.core.writers import PredictionWriter, read_jsonl, write_csv, write_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M1 fixed-model capability-profile rollout benchmark")
    parser.add_argument("--config", default="evaluation/configs/m1_rollout.yaml")
    parser.add_argument("--mode", choices=["dry-run", "smoke", "pilot", "full"], required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-split-fallback", action="store_true")
    parser.add_argument("--allow-code-execution", action="store_true")
    parser.add_argument("--preflight-run-id")
    parser.add_argument("--confirm-full-run", action="store_true")
    parser.add_argument("--output-root")
    parser.add_argument("--endpoint", action="append", default=[], metavar="MODEL_KEY=URL")
    parser.add_argument("--request-model", action="append", default=[], metavar="MODEL_KEY=NAME")
    parser.add_argument(
        "--single-split-per-dataset",
        action="store_true",
        help="Evaluate only the first available requested split for each dataset.",
    )

    parser.add_argument("--seed", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", dest="top_p", type=float)
    parser.add_argument("--top-k", dest="top_k", type=int)
    parser.add_argument("--repetition-penalty", dest="repetition_penalty", type=float)
    parser.add_argument("--max-tokens-multiple-choice", dest="max_tokens_multiple_choice", type=int)
    parser.add_argument("--max-tokens-gsm8k", dest="max_tokens_gsm8k", type=int)
    parser.add_argument("--max-tokens-domain-qa", dest="max_tokens_domain_qa", type=int)
    parser.add_argument("--max-tokens-humaneval", dest="max_tokens_humaneval", type=int)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--stop-sequence", action="append", default=None)

    parser.add_argument("--retry-max-attempts", type=int)
    parser.add_argument("--retry-initial-backoff-seconds", type=float)
    parser.add_argument("--retry-max-backoff-seconds", type=float)
    parser.add_argument("--retry-multiplier", type=float)
    parser.add_argument("--retry-request-timeout-seconds", type=float)
    parser.add_argument("--manifest-pool-size", type=int)
    parser.add_argument("--smoke-examples-per-split", type=int)
    parser.add_argument("--pilot-examples-per-split", type=int)
    parser.add_argument("--humaneval-pilot-examples", type=int)
    return parser.parse_args(argv)


def _check_mode_preconditions(config: M1Config, args: argparse.Namespace) -> None:
    if args.mode == "full" and not args.confirm_full_run:
        raise ConfigError("Full M1 is disabled until explicitly requested; pass --confirm-full-run")
    if args.mode == "pilot":
        if not args.preflight_run_id:
            raise ConfigError("Pilot requires --preflight-run-id from a passed M1 smoke run")
        status_path = config.output_root / args.preflight_run_id / "summaries" / "preflight_status.json"
        if not status_path.exists():
            raise ConfigError(f"Preflight status not found: {status_path}")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("passed") is not True:
            raise ConfigError(f"Preflight run {args.preflight_run_id} did not pass")


def _prepare_run(config: M1Config, args: argparse.Namespace) -> tuple[str, Path, dict[str, Any]]:
    run_id = args.run_id or new_run_id(args.mode)
    run_dir = config.output_root / run_id
    if run_dir.exists() and not args.resume:
        raise ConfigError(f"Run directory already exists; use --resume only for the same run: {run_dir}")
    if args.resume:
        manifest_path = run_dir / "manifest.json"
        resolved_path = run_dir / "resolved_config.yaml"
        if not manifest_path.exists() or not resolved_path.exists():
            raise ConfigError(f"Cannot resume incomplete run directory: {run_dir}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("run_id") != run_id:
            raise ConfigError("Resume run_id does not match manifest")
        previous_config = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
        if previous_config != config.sanitized_dict():
            raise ConfigError("Resolved config differs from the existing run; start a new run_id")
        manifest["end_time_utc"] = None
        manifest["status"] = "resumed"
        write_json(manifest_path, manifest)
        return run_id, run_dir, manifest

    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = initial_manifest(config, run_id, run_dir, args.mode, sys.argv)
    return run_id, run_dir, manifest


def _split_rows(discoveries: list[DatasetDiscovery]) -> list[dict[str, Any]]:
    rows = []
    for discovery in discoveries:
        selections = {item.requested_split: item for item in discovery.selections}
        if discovery.dataset_key == "humaneval":
            official = discovery.selections[0]
            rows.append({
                "dataset": discovery.dataset_key,
                "available_splits": "|".join(discovery.available_splits),
                "requested_train_split": "not_requested",
                "requested_validation_split": "not_requested",
                "train_status": "not_requested",
                "validation_status": "not_requested",
                "fallback_used": official.fallback_used,
                "notes": f"official HumanEval split requested={official.requested_split}; {official.note}",
            })
            continue
        train = selections.get("train")
        validation = selections.get("validation")
        rows.append({
            "dataset": discovery.dataset_key,
            "available_splits": "|".join(discovery.available_splits),
            "requested_train_split": "train",
            "requested_validation_split": "validation",
            "train_status": train.status if train else "not_requested",
            "validation_status": validation.status if validation else "not_requested",
            "fallback_used": any(item.fallback_used for item in discovery.selections),
            "notes": "; ".join(item.note for item in discovery.selections),
        })
    return rows


def _mode_count(config: M1Config, mode: str, dataset_key: str, available_count: int) -> int:
    if mode == "smoke":
        return min(available_count, config.sampling.smoke_examples_per_split)
    if mode == "pilot":
        configured = (
            config.sampling.humaneval_pilot_examples
            if dataset_key == "humaneval"
            else config.sampling.pilot_examples_per_split
        )
        return min(available_count, configured)
    if mode == "dry-run":
        return min(available_count, config.sampling.smoke_examples_per_split)
    return available_count


def _selected_split_items(discovery: DatasetDiscovery, single_split_per_dataset: bool) -> list[Any]:
    available = [selection for selection in discovery.selections if selection.actual_split is not None]
    if not single_split_per_dataset or not available:
        return available
    for preferred in ("validation", "train", "test"):
        for selection in available:
            if selection.actual_split == preferred or selection.requested_split == preferred:
                return [selection]
    return [available[0]]


def _prediction_path(run_dir: Path, model_key: str, dataset: str, split: str, mode: str) -> Path:
    suffix = {
        "smoke": "smoke",
        "pilot": "pilot",
        "dry-run": "dry_run",
        "full": "full",
    }[mode]
    return run_dir / "predictions" / model_key / f"{dataset}_{split}_{suffix}.jsonl"


def _generation_dict(config: M1Config, max_tokens: int) -> dict[str, Any]:
    generation = config.generation
    return {
        "temperature": generation.temperature,
        "top_p": generation.top_p,
        "top_k": generation.top_k,
        "max_tokens": max_tokens,
        "seed": generation.seed,
        "repetition_penalty": generation.repetition_penalty,
        "stop_sequences": generation.stop_sequences,
    }


def _base_record(
    run_id: str,
    config: M1Config,
    model_key: str,
    model_metadata: dict[str, Any],
    discovery: DatasetDiscovery,
    requested_split: str,
    actual_split: str,
    fallback_used: bool,
    example: AdaptedExample,
) -> dict[str, Any]:
    model = config.models[model_key]
    adapter = discovery.adapter
    split_dataset = adapter.get_split(actual_split)
    max_tokens = config.generation.max_tokens_for(adapter.spec.task_type)
    record = {
        "run_id": run_id,
        "timestamp_utc": utc_now(),
        "model_key": model_key,
        "model_id": model.model_id,
        "model_role": model.role,
        "model_family": model.family,
        "checkpoint_type": model.checkpoint_type,
        "model_revision": model_metadata.get("Hugging_Face_revision_or_commit_hash_if_available"),
        "tokenizer_name": model_metadata.get("tokenizer_name"),
        "tokenizer_revision": model_metadata.get("tokenizer_revision_or_commit_hash_if_available"),
        "backend": model.backend,
        "endpoint_base_url": sanitize_base_url(model.base_url),
        "dataset": adapter.spec.key,
        "dataset_source": adapter.spec.identifier,
        "dataset_config": adapter.spec.config_name,
        "dataset_revision": adapter.spec.revision or getattr(split_dataset, "_fingerprint", None),
        "split": actual_split,
        "requested_split": requested_split,
        "split_fallback_used": fallback_used,
        "example_id": example.example_id,
        "example_index": example.example_index,
        "prompt_template_id": adapter.prompt_template_id,
        "prompt_template_version": adapter.prompt_template_version,
        "prompt_rendering_mode": (
            "completions" if adapter.spec.key == "humaneval" else model.prompt_rendering_mode
        ),
        "logical_prompt": example.logical_prompt,
        "rendered_prompt": example.rendered_prompt,
        "generation_config": _generation_dict(config, max_tokens),
        "raw_response": "",
        "parsed_prediction": None,
        "normalized_prediction": None,
        "gold_answer": example.gold_answer,
        "normalized_gold_answer": example.normalized_gold_answer,
        "parse_status": "not_attempted",
        "parser_name": getattr(example.parser, "name", type(example.parser).__name__),
        "correct": None if adapter.spec.key == "humaneval" else False,
        "request_status": "failed",
        "retry_count": 0,
        "error_type": None,
        "error_message": None,
        "latency_seconds": 0.0,
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
    }
    if adapter.spec.key == "humaneval":
        record.update({
            "task_id": example.extra["task_id"],
            "canonical_prompt": example.extra["canonical_prompt"],
            "generated_completion": "",
            "execution_enabled": False,
            "execution_status": "not_enabled",
            "test_result": None,
            "error_trace_if_any": None,
        })
    return record


def _evaluate_one(
    record: dict[str, Any],
    example: AdaptedExample,
    client: OpenAICompatibleClient | None,
    config: M1Config,
    mode: str,
    allow_code_execution: bool,
) -> dict[str, Any]:
    if mode == "dry-run":
        record["request_status"] = "dry_run"
        record["parse_status"] = "not_attempted"
        validate_prediction_record(record)
        return record

    if client is None:
        record.update({
            "request_status": "failed",
            "parse_status": "not_attempted",
            "error_type": "EndpointConfigurationError",
            "error_message": "Model base_url is empty; configure the model endpoint",
        })
        validate_prediction_record(record)
        return record

    max_tokens = record["generation_config"]["max_tokens"]
    result: RequestResult = client.generate(
        example.rendered_prompt,
        config.generation,
        max_tokens,
        force_completions=record["dataset"] == "humaneval",
    )
    record.update({
        "request_status": result.status,
        "retry_count": result.retry_count,
        "error_type": result.error_type,
        "error_message": result.error_message,
        "latency_seconds": result.latency_seconds,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "total_tokens": result.total_tokens,
        "raw_response": result.text,
    })
    if result.status != "completed":
        record["parse_status"] = "not_attempted"
        validate_prediction_record(record)
        return record

    parsed = example.parser.parse(result.text)
    record.update({
        "parsed_prediction": parsed.parsed_prediction,
        "normalized_prediction": parsed.normalized_prediction,
        "parse_status": parsed.status,
        "parser_name": parsed.parser_name,
    })
    if record["dataset"] == "humaneval":
        completion = parsed.normalized_prediction or ""
        record["generated_completion"] = completion
        record["execution_enabled"] = allow_code_execution
        if allow_code_execution and parsed.status == "success":
            execution = execute_humaneval(
                example.extra["canonical_prompt"], completion, example.extra["test"], example.extra["entry_point"]
            )
            record["execution_status"] = execution.status
            record["test_result"] = execution.test_result
            record["error_trace_if_any"] = execution.error_trace
            record["correct"] = execution.status == "passed"
        else:
            record["execution_status"] = "not_enabled" if not allow_code_execution else "not_executed_parse_failed"
            record["correct"] = None
    elif parsed.status == "success":
        record["correct"] = parsed.normalized_prediction == record["normalized_gold_answer"]
    validate_prediction_record(record)
    return record


def _write_dataset_metadata(
    run_dir: Path,
    discoveries: list[DatasetDiscovery],
    selected_counts: dict[str, dict[str, int]],
) -> None:
    for discovery in discoveries:
        adapter = discovery.adapter
        if discovery.load_error:
            metadata = {
                "dataset_name": adapter.spec.display_name,
                "dataset_source": adapter.spec.source,
                "local_path_or_Hugging_Face_identifier": adapter.spec.local_path or adapter.spec.identifier,
                "dataset_revision_or_version_if_available": adapter.spec.revision,
                "config_name": adapter.spec.config_name,
                "split": [],
                "total_available_examples": {},
                "selected_example_count": {},
                "task_type": adapter.spec.task_type,
                "label_space": adapter.label_space,
                "label_mapping": adapter.label_mapping,
                "prompt_template_id": adapter.prompt_template_id,
                "prompt_template_version_or_hash": adapter.prompt_template_hash,
                "few_shot_setting": adapter.few_shot_setting,
                "preprocessing_or_filtering_rule": adapter.preprocessing_rule,
                "load_error": discovery.load_error,
                "provenance_note": adapter.spec.provenance_note,
            }
        else:
            metadata = adapter.dataset_metadata(discovery.selections)
            metadata["selected_example_count"] = selected_counts.get(discovery.dataset_key, {})
        write_json(run_dir / "datasets" / f"{discovery.dataset_key}.json", metadata)


def run(config: M1Config, args: argparse.Namespace) -> tuple[str, Path, bool]:
    _check_mode_preconditions(config, args)
    run_id, run_dir, manifest = _prepare_run(config, args)
    status = "failed"
    try:
        model_metadata = {}
        for key, model in config.models.items():
            metadata = collect_model_metadata(model, config.retry)
            model_metadata[key] = metadata
            write_json(run_dir / "models" / f"{key}.json", metadata)

        discoveries = DatasetRegistry(config).discover(args.allow_split_fallback)
        split_rows = _split_rows(discoveries)
        write_csv(
            run_dir / "summaries" / "split_availability.csv",
            split_rows,
            [
                "dataset", "available_splits", "requested_train_split", "requested_validation_split",
                "train_status", "validation_status", "fallback_used", "notes",
            ],
        )

        work: list[tuple[DatasetDiscovery, Any, str, list[AdaptedExample]]] = []
        selected_counts: dict[str, dict[str, int]] = {}
        for discovery in discoveries:
            if discovery.load_error:
                continue
            for selection in _selected_split_items(
                discovery, getattr(args, "single_split_per_dataset", False)
            ):
                adapter = discovery.adapter
                actual_split = selection.actual_split
                split_dataset = adapter.get_split(actual_split)
                pool_requested = (
                    len(split_dataset) if args.mode == "full" else config.sampling.manifest_pool_size
                )
                sample_manifest, pool_indices = build_sample_manifest(
                    adapter, actual_split, config.sampling.selection_seed, pool_requested
                )
                output_split = (
                    adapter.spec.official_split_output_name
                    if adapter.spec.key == "humaneval" and adapter.spec.official_split_output_name
                    else actual_split
                )
                write_json(
                    run_dir / "sample_manifests" / adapter.spec.key / f"{output_split}.json",
                    sample_manifest,
                )
                count = _mode_count(config, args.mode, adapter.spec.key, len(pool_indices))
                chosen = pool_indices[:count]
                examples = adapter.adapt_many(
                    ((index, dict(split_dataset[index])) for index in chosen), actual_split
                )
                selected_counts.setdefault(adapter.spec.key, {})[actual_split] = len(examples)
                work.append((discovery, selection, output_split, examples))

        _write_dataset_metadata(run_dir, discoveries, selected_counts)

        for model_key, model in config.models.items():
            client = OpenAICompatibleClient(model, config.retry) if model.base_url and args.mode != "dry-run" else None
            for discovery, selection, output_split, examples in work:
                path = _prediction_path(
                    run_dir, model_key, discovery.dataset_key, output_split, args.mode
                )
                writer = PredictionWriter(path)
                completed = writer.completed_keys()
                pending: list[tuple[AdaptedExample, dict[str, Any]]] = []
                for example in examples:
                    key = (run_id, model_key, discovery.dataset_key, selection.actual_split, example.example_id)
                    if key in completed:
                        continue
                    record = _base_record(
                        run_id, config, model_key, model_metadata[model_key], discovery,
                        selection.requested_split, selection.actual_split, selection.fallback_used, example,
                    )
                    pending.append((example, record))

                if config.generation.concurrency == 1:
                    for example, record in pending:
                        writer.append(_evaluate_one(
                            record, example, client, config, args.mode, args.allow_code_execution
                        ))
                else:
                    with ThreadPoolExecutor(max_workers=config.generation.concurrency) as executor:
                        futures = {
                            executor.submit(
                                _evaluate_one, record, example, client, config, args.mode,
                                args.allow_code_execution,
                            ): example.example_index
                            for example, record in pending
                        }
                        completed_records = []
                        for future in as_completed(futures):
                            completed_records.append(future.result())
                        for record in sorted(completed_records, key=lambda item: item["example_index"]):
                            writer.append(record)

        prediction_paths = list((run_dir / "predictions").glob("*/*.jsonl"))
        records = read_jsonl(prediction_paths)
        write_summary_artifacts(run_dir / "summaries", records, list(config.models), args.mode)

        coverage = {
            (record.get("model_key"), record.get("dataset"))
            for record in records
            if record.get("request_status") == "completed"
        }
        expected_coverage = {
            (model_key, dataset_key)
            for model_key in config.models
            for dataset_key in config.datasets
        }
        request_failures = sum(record.get("request_status") == "failed" for record in records)
        parse_failures = sum(record.get("parse_status") == "failed" for record in records)
        dataset_load_failures = [item.dataset_key for item in discoveries if item.load_error]
        preflight_passed = (
            args.mode == "smoke"
            and coverage == expected_coverage
            and request_failures == 0
            and parse_failures == 0
            and not dataset_load_failures
        )
        if args.mode == "smoke":
            write_json(run_dir / "summaries" / "preflight_status.json", {
                "run_id": run_id,
                "passed": preflight_passed,
                "coverage_pairs": len(coverage),
                "expected_coverage_pairs": len(expected_coverage),
                "request_failures": request_failures,
                "parse_failures": parse_failures,
                "dataset_load_failures": dataset_load_failures,
            })

        if args.mode == "smoke":
            status = "passed" if preflight_passed else "needs_review"
        elif args.mode == "dry-run":
            status = "completed_with_blockers" if dataset_load_failures else "completed"
        else:
            status = (
                "needs_review"
                if dataset_load_failures or request_failures or parse_failures or coverage != expected_coverage
                else "completed"
            )
        report_name = "pilot_report.md" if args.mode == "pilot" else "smoke_report.md"
        (run_dir / report_name).write_text(
            report_markdown(run_id, args.mode, records, split_rows, status), encoding="utf-8"
        )
        finalize_manifest(run_dir, manifest, status)
        return run_id, run_dir, preflight_passed
    except Exception:
        finalize_manifest(run_dir, manifest, status)
        raise


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config)
        if args.output_root:
            config.output_root = Path(args.output_root)
            config.raw_resolved["output_root"] = args.output_root
        apply_cli_overrides(config, args)
        if args.stop_sequence is not None:
            config.generation.stop_sequences = args.stop_sequence
            config.raw_resolved.setdefault("generation", {})["stop_sequences"] = args.stop_sequence
        run_id, run_dir, preflight_passed = run(config, args)
    except (ConfigError, SchemaValidationError, ValueError) as exc:
        print(f"M1 configuration/pipeline error: {exc}", file=sys.stderr)
        return 2
    print(f"run_id={run_id}")
    print(f"artifacts={run_dir.resolve()}")
    if args.mode == "smoke":
        print(f"preflight_passed={str(preflight_passed).lower()}")
        return 0 if preflight_passed else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

