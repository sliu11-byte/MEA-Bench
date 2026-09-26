"""Rebuild deterministic shared detector baselines for completed counter runs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

from defenses.core.runner.online import (
    DETECTOR_FACTORIES,
    load_completed_baseline,
    run_shared_baseline_detection,
)
from evaluation.attack_eval.evaluate_attack_outputs import write_json
from evaluation.defense_eval.evaluate_counter_seqkd import detector_stats, read_json, write_table


DEFENSES = ("adfp", "ginsew", "radioactivity")
COUNTERS = ("dipper", "translation")


def comparison_path(storage: Path, attack: str, budget: int, counter: str, defense: str) -> Path:
    return storage / "outputs" / "countermeasures" / f"{attack}_b{budget}" / counter / defense / "comparison_report.json"


def shared_baselines_ready(
    storage: Path,
    attack: str,
    budget: int,
    seed: int,
    temperature: float,
    max_queries: int,
    max_new_tokens: int,
) -> bool:
    expected = {
        "shared_across_countermeasures": True,
        "seed": seed,
        "temperature": temperature,
        "max_queries": max_queries,
        "max_new_tokens": max_new_tokens,
    }
    for defense in DEFENSES:
        reports = []
        for counter in COUNTERS:
            path = comparison_path(storage, attack, budget, counter, defense)
            if not path.is_file():
                return False
            report = read_json(path)
            if report.get("baseline_detection_protocol") != expected:
                return False
            reports.append(report)
        for group in ("clean", "defense_only"):
            if reports[0].get("reports", {}).get(group) != reports[1].get("reports", {}).get(group):
                return False
    return True


def load_shared_baselines(paths: dict[str, Path]) -> dict[str, object]:
    manifests: dict[str, Path] = {}
    for group in ("clean", "defense_only"):
        values = {Path(read_json(path)["baseline_manifests"][group]).resolve() for path in paths.values()}
        if len(values) != 1:
            raise ValueError(f"{group} baseline manifest differs between counter branches: {sorted(map(str, values))}")
        manifests[group] = values.pop()
    baselines = {group: load_completed_baseline(path) for group, path in manifests.items()}
    missing = [group for group, result in baselines.items() if result is None]
    if missing:
        raise ValueError(f"Incomplete or missing baseline manifests: {', '.join(missing)}")
    return baselines


def atomic_write(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def refresh_completed_evaluation(
    output: Path,
    reports: dict[tuple[str, str], dict],
    attack: str,
    budget: int,
) -> bool:
    complete = output / "complete.json"
    models_path = output / "models_summary.json"
    if not complete.is_file() or not models_path.is_file():
        return False
    model_rows = read_json(models_path)
    scores = {row["model"]: {"acc": row["acc"], "bertscore": row["bertscore"]} for row in model_rows}
    required = {"clean", *(f"{defense}_defense_only" for defense in DEFENSES),
                *(f"{defense}_{counter}" for defense in DEFENSES for counter in COUNTERS)}
    if not required.issubset(scores):
        raise ValueError(f"Completed evaluation is missing models: {sorted(required - scores.keys())}")
    smoke = bool(read_json(complete).get("smoke"))
    rows = []
    for (counter, defense), report in reports.items():
        for group, model in (("clean", "clean"), ("defense_only", f"{defense}_defense_only"),
                             ("counter", f"{defense}_{counter}")):
            rows.append({"attack": attack, "budget": budget, "defense": defense,
                         "countermeasure": counter, "condition": group, "model": model,
                         **scores[model], **detector_stats(report["reports"][group]), "smoke": smoke})
    write_json(output / "input_detection_reports.json",
               {f"{counter}/{defense}": report for (counter, defense), report in reports.items()})
    write_json(output / "summary.json", rows)
    write_table(output / "summary.csv", rows)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-root", required=True)
    parser.add_argument("--attack", default="seqkd")
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--detector-seed", type=int, default=42)
    parser.add_argument("--detector-max-queries", type=int, default=1000)
    parser.add_argument("--detector-max-new-tokens", type=int, default=124)
    parser.add_argument("--detector-temperature", type=float, default=0.7)
    parser.add_argument("--evaluation-root")
    parser.add_argument("--skip-if-shared", action="store_true")
    args = parser.parse_args()
    storage = Path(args.storage_root).resolve()
    if args.skip_if_shared and shared_baselines_ready(
        storage,
        args.attack,
        args.budget,
        args.detector_seed,
        args.detector_temperature,
        args.detector_max_queries,
        args.detector_max_new_tokens,
    ):
        print("[counter baseline repair] matching shared detector baselines already exist; skipping", flush=True)
        return
    runner_args = SimpleNamespace(
        attack=args.attack,
        budget=args.budget,
        detector_seed=args.detector_seed,
        detector_max_queries=args.detector_max_queries,
        detector_max_new_tokens=args.detector_max_new_tokens,
        detector_temperature=args.detector_temperature,
        probe_queries=None,
        adfp_batch_size=8,
        detector_extra_args="",
    )
    updated: dict[tuple[str, str], dict] = {}
    for defense in DEFENSES:
        paths = {counter: comparison_path(storage, args.attack, args.budget, counter, defense)
                 for counter in COUNTERS}
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing comparison reports: " + ", ".join(missing))
        baselines = load_shared_baselines(paths)
        reference = baselines["defense_only"]
        shared = {}
        shared_paths = {}
        for group, label in (("clean", "negative"), ("defense_only", "positive")):
            detected = run_shared_baseline_detection(
                defense=defense,
                args=runner_args,
                detection_reference=reference,
                student=baselines[group],
                group=group,
                label=label,
                factory=DETECTOR_FACTORIES[defense],
            )
            if detected is None or detected.report_path is None:
                raise RuntimeError(f"No {defense} detector report for {group}")
            shared[group] = read_json(detected.report_path)
            shared_paths[group] = str(detected.report_path.resolve())
        for counter, path in paths.items():
            report = read_json(path)
            if "counter" not in report.get("reports", {}):
                raise ValueError(f"Counter detector report missing: {path}")
            backup = path.with_name("comparison_report.pre_shared_detector_baselines.json")
            if not backup.exists():
                shutil.copy2(path, backup)
            report["reports"].update(shared)
            report.setdefault("report_paths", {}).update(shared_paths)
            report["baseline_detection_protocol"] = {
                "shared_across_countermeasures": True,
                "seed": args.detector_seed,
                "temperature": args.detector_temperature,
                "max_queries": args.detector_max_queries,
                "max_new_tokens": args.detector_max_new_tokens,
            }
            atomic_write(path, report)
            updated[counter, defense] = report
            print(f"[counter baseline repair] updated {path}", flush=True)

    evaluation = Path(args.evaluation_root).resolve() if args.evaluation_root else (
        storage / "results" / "counter_eval" / f"{args.attack}_b{args.budget}_m1_v2")
    if refresh_completed_evaluation(evaluation, updated, args.attack, args.budget):
        print(f"[counter baseline repair] refreshed completed evaluation: {evaluation / 'summary.csv'}", flush=True)
    else:
        print(f"[counter baseline repair] evaluation is not complete; its next finalization must read the updated reports: {evaluation}", flush=True)


if __name__ == "__main__":
    main()
