"""Evaluate DuFFin with one teacher reference, multiple controls, and clean attack students."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from defenses.core.io_utils import read_jsonl, write_jsonl
from defenses.duffin.detector import compute_knowledge_duffin
from defenses.duffin.probes import load_default_probes
from evaluation.defense_eval.evaluate import BASE, CLEAN, TEACHER, local_corrected_soda_checkpoint


DEFAULT_NEGATIVES = {
    "base_negative": BASE,
    "qwen25_instruct_negative": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2_instruct_negative": "Qwen/Qwen2-7B-Instruct",
    "mistral_instruct_negative": "mistralai/Mistral-7B-Instruct-v0.3",
}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def pairwise_auc(positives, negatives):
    comparisons = [1.0 if p > n else 0.5 if p == n else 0.0 for p in positives for n in negatives]
    return sum(comparisons) / len(comparisons)


def write_roc(root, rows):
    positives = [row["score"] for row in rows if row["label"] == "positive"]
    negatives = [row["score"] for row in rows if row["label"] == "negative"]
    if not positives or not negatives:
        raise ValueError("DuFFin ROC-AUC requires at least one positive and one negative model")
    auc = pairwise_auc(positives, negatives)
    write_json(root / "roc_auc.json", {
        "detector": "duffin", "metric": "roc_auc", "roc_auc": auc,
        "num_positive": len(positives), "num_negative": len(negatives),
        "score_source": "common_valid_probes", "rows": rows,
        "note": "Model-level ROC-AUC over clean attack students and independently selected negative controls.",
    })
    thresholds = [float("inf")] + sorted({row["score"] for row in rows}, reverse=True) + [float("-inf")]
    with (root / "roc_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["threshold", "tpr", "fpr", "true_positives", "false_positives"])
        writer.writeheader()
        for threshold in thresholds:
            tp = sum(row["label"] == "positive" and row["score"] >= threshold for row in rows)
            fp = sum(row["label"] == "negative" and row["score"] >= threshold for row in rows)
            writer.writerow({"threshold": threshold, "tpr": tp / len(positives), "fpr": fp / len(negatives),
                             "true_positives": tp, "false_positives": fp})
    return auc


def summarize(root, negative_names):
    names = tuple(negative_names) + tuple(CLEAN)
    reference_negative = negative_names[0]
    data = {name: read_jsonl(root / name / "probe_rows.jsonl") for name in names}
    ids = [row["probe_id"] for row in data[names[0]]]
    for rows in data.values():
        if [row["probe_id"] for row in rows] != ids:
            raise ValueError("DuFFin probe order mismatch")
        if [row["teacher_choice"] for row in rows] != [row["teacher_choice"] for row in data[names[0]]]:
            raise ValueError("DuFFin teacher choices mismatch")
    common = [i for i in range(len(ids)) if all(
        data[name][i]["teacher_choice"] is not None and data[name][i]["student_choice"] is not None
        for name in names
    )]
    if not common:
        raise ValueError("No common valid probes across evaluated DuFFin models")
    scores = {name: compute_knowledge_duffin([data[name][i] for i in common]) for name in names}
    summary = []
    for name in names:
        own = compute_knowledge_duffin(data[name])
        summary.append({
            "model": name, "label": "negative" if name in negative_names else "positive",
            "score": scores[name]["score"],
            "difference_from_reference_negative": scores[name]["score"] - scores[reference_negative]["score"],
            "reference_negative": reference_negative,
            "common_valid_probes": len(common), "individual_score": own["score"],
            "individual_valid_probes": own["valid_probes"], "total_probes": len(ids),
        })
    write_json(root / "summary.json", {"comparison": "common_valid_probes", "results": summary,
                                      "note": "Scores use the intersection of probes parsed by every evaluated model."})
    write_json(root / "common_valid_probe_ids.json", [ids[i] for i in common])
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    write_roc(root, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--checkpoint-root", required=True, type=Path,
                        help="Local directory containing mea-<attack>-defenses folders.")
    parser.add_argument("--max-probes", type=int, default=1000)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--min-valid-rate", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--probe-seed", type=int, default=42)
    parser.add_argument(
        "--negative-model", action="append", default=None, metavar="NAME=MODEL",
        help="Negative control; repeatable. Defaults to base Qwen, Qwen Instruct controls, and Mistral.",
    )
    args = parser.parse_args()
    if args.max_probes < 1 or args.max_new_tokens < 1 or args.batch_size < 1:
        parser.error("probe and token limits must be positive")
    if not 0.0 <= args.min_valid_rate <= 1.0:
        parser.error("--min-valid-rate must be between 0 and 1")
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    negative_models = dict(DEFAULT_NEGATIVES)
    if args.negative_model:
        negative_models = {}
        for value in args.negative_model:
            if "=" not in value:
                parser.error("--negative-model must use NAME=MODEL")
            name, model = value.split("=", 1)
            name, model = name.strip(), model.strip()
            if not name or not model or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in name):
                parser.error(f"invalid --negative-model: {value}")
            negative_models[name] = model
    protocol = {"teacher": TEACHER, "base": BASE, "max_probes": args.max_probes,
                "max_new_tokens": args.max_new_tokens, "seed": args.probe_seed,
                "source": "mmlu_pro", "split": "test",
                "categories": "biology,business,chemistry,computer_science,math,physics"}
    protocol["negative_models"] = negative_models
    protocol["batch_size"] = args.batch_size
    protocol_path = root / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Protocol changed; use a new output root")
    write_json(protocol_path, protocol)
    # Remove a previous final summary so a failed rerun cannot look complete.
    for name in ("summary.csv", "summary.json", "common_valid_probe_ids.json", "roc_auc.json", "roc_curve.csv"):
        (root / name).unlink(missing_ok=True)
    reference = root / "reference"
    reference.mkdir(exist_ok=True)
    probes_path = reference / "probes.jsonl"
    if not probes_path.exists():
        probes = load_default_probes(source=protocol["source"], split=protocol["split"],
                                    categories=protocol["categories"], max_probes=args.max_probes, seed=args.probe_seed)
        if not probes:
            raise ValueError("Empty probe set")
        temporary = probes_path.with_suffix(".tmp")
        write_jsonl(probes, temporary)
        temporary.replace(probes_path)
    shared = ["--teacher_model", TEACHER, "--probe_set", str(probes_path),
              "--teacher_reference", str(reference / "teacher_reference.json"),
              "--max_new_tokens", str(args.max_new_tokens),
              "--min_valid_rate", str(args.min_valid_rate),
              "--batch_size", str(args.batch_size),
              "--probe_categories", protocol["categories"]]

    def run(extra):
        subprocess.run([sys.executable, "-m", "defenses.duffin.runner", *shared, *extra], check=True)

    run(["--reference_only", "--output_dir", str(reference)])
    for name, model in negative_models.items():
        run(["--student_checkpoint", model, "--label", "negative", "--model_id", name,
             "--output_dir", str(root / name)])
    for attack, folder in CLEAN.items():
        source_path = root / f"{attack}_source.json"
        local = local_corrected_soda_checkpoint("clean") if attack == "soda" else None
        if local is not None:
            checkpoint, source = local
            if source_path.exists() and json.loads(source_path.read_text()) != source:
                raise ValueError(f"Checkpoint source mismatch: {source_path}")
            write_json(source_path, source)
            run(["--student_checkpoint", str(checkpoint), "--student_base_model", BASE,
                 "--label", "positive", "--model_id", attack, "--output_dir", str(root / attack)])
            continue
        artifact_root = args.checkpoint_root.expanduser().resolve() / f"mea-{attack}-defenses"
        checkpoint = artifact_root / folder
        source = {"source": "local", "root": str(artifact_root), "folder": folder}
        if source_path.exists() and json.loads(source_path.read_text()) != source:
            raise ValueError(f"Checkpoint source mismatch: {source_path}")
        write_json(source_path, source)
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Missing local checkpoint: {checkpoint}")
        config = json.loads((checkpoint / "adapter_config.json").read_text())
        if config.get("base_model_name_or_path") != BASE:
            raise ValueError(f"Unexpected adapter base: {checkpoint}")
        run(["--student_checkpoint", str(checkpoint), "--student_base_model", BASE,
             "--label", "positive", "--model_id", attack, "--output_dir", str(root / attack)])
    summarize(root, tuple(negative_models))
    print(f"[DuFFin evaluation] complete -> {root / 'summary.csv'}")


if __name__ == "__main__":
    main()
