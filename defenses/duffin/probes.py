"""Default probe-set builders for Knowledge-DuFFin."""

from __future__ import annotations

import random
from typing import Any, Iterable

from defenses.duffin.detector import CHOICES


def _parse_categories(categories: str | Iterable[str] | None) -> set[str] | None:
    if categories is None:
        return None
    if isinstance(categories, str):
        values = [item.strip() for item in categories.split(",")]
    else:
        values = [str(item).strip() for item in categories]
    values = [item for item in values if item]
    if not values or any(item.lower() == "all" for item in values):
        return None
    return set(values)


def _category_matches(category: str, selected_categories: set[str] | None) -> bool:
    if selected_categories is None:
        return True
    category_norm = category.strip().lower()
    for selected in selected_categories:
        selected_norm = selected.strip().lower()
        if category_norm == selected_norm:
            return True
        if category_norm.endswith("_" + selected_norm):
            return True
    return False


def _clean_options(options: Any) -> list[str]:
    if options is None:
        return []
    if isinstance(options, dict):
        values = list(options.values())
    else:
        values = list(options)
    return [str(value).strip() for value in values if value not in (None, "", "N/A")]


def _options_dict(options: list[str]) -> dict[str, str]:
    if len(options) > len(CHOICES):
        raise ValueError(f"DuFFin supports at most {len(CHOICES)} choices per probe.")
    return {CHOICES[i]: value for i, value in enumerate(options)}


def _answer_letter(raw_answer: Any, options: list[str]) -> str | None:
    if raw_answer is None:
        return None
    if isinstance(raw_answer, int):
        return CHOICES[raw_answer] if 0 <= raw_answer < len(options) else None
    answer = str(raw_answer).strip()
    if len(answer) == 1 and answer.upper() in CHOICES[: len(options)]:
        return answer.upper()
    for i, option in enumerate(options):
        if answer == option:
            return CHOICES[i]
    return None


def _sample(records: list[dict[str, Any]], *, max_probes: int | None, seed: int) -> list[dict[str, Any]]:
    if max_probes is None or max_probes >= len(records):
        return records
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(records)), max(0, int(max_probes))))
    return [records[i] for i in indices]


def load_mmlu_pro_probes(
    *,
    split: str = "test",
    categories: str | Iterable[str] | None = "biology",
    max_probes: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Load MMLU-Pro probes from Hugging Face and normalize to DuFFin JSONL rows."""
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency path
        raise RuntimeError("Default DuFFin probes require the `datasets` package.") from exc

    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    if split not in dataset:
        raise ValueError(f"MMLU-Pro split '{split}' not found; available={list(dataset.keys())}")
    selected_categories = _parse_categories(categories)
    examples_by_category: dict[str, list[dict[str, Any]]] = {}
    if "validation" in dataset:
        for row in dataset["validation"]:
            category = str(row.get("category", "unknown"))
            options = _clean_options(row.get("options"))
            if len(options) < 2:
                continue
            examples_by_category.setdefault(category, []).append(
                {
                    "question": str(row["question"]).strip(),
                    "options": _options_dict(options),
                    "answer": _answer_letter(row.get("answer_index", row.get("answer")), options),
                    "cot_content": str(row.get("cot_content") or "").strip(),
                }
            )

    records: list[dict[str, Any]] = []
    for i, row in enumerate(dataset[split]):
        category = str(row.get("category", "unknown"))
        if not _category_matches(category, selected_categories):
            continue
        options = _clean_options(row.get("options"))
        if len(options) < 2:
            continue
        answer = _answer_letter(row.get("answer_index", row.get("answer")), options)
        records.append(
            {
                "probe_id": str(row.get("question_id", f"mmlu_pro_{split}_{i}")),
                "source": "TIGER-Lab/MMLU-Pro",
                "split": split,
                "category": category,
                "question": str(row["question"]).strip(),
                "options": _options_dict(options),
                "answer": answer,
                "few_shot_examples": examples_by_category.get(category, [])[:5],
            }
        )
    return _sample(records, max_probes=max_probes, seed=seed)


def load_mmlu_probes(
    *,
    split: str = "test",
    categories: str | Iterable[str] | None = "biology",
    max_probes: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Load classic MMLU probes from Hugging Face and normalize to DuFFin JSONL rows."""
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency path
        raise RuntimeError("Default DuFFin probes require the `datasets` package.") from exc

    hf_split = "test" if split == "test" else split
    dataset = load_dataset("cais/mmlu", "all", split=hf_split)
    selected_categories = _parse_categories(categories)
    records: list[dict[str, Any]] = []
    for i, row in enumerate(dataset):
        category = str(row.get("subject", "unknown"))
        if not _category_matches(category, selected_categories):
            continue
        options = _clean_options(row.get("choices"))
        if len(options) < 2:
            continue
        answer = _answer_letter(row.get("answer"), options)
        records.append(
            {
                "probe_id": f"mmlu_{hf_split}_{category}_{i}",
                "source": "cais/mmlu",
                "split": hf_split,
                "category": category,
                "question": str(row["question"]).strip(),
                "options": _options_dict(options),
                "answer": answer,
            }
        )
    return _sample(records, max_probes=max_probes, seed=seed)


def load_default_probes(
    *,
    source: str = "mmlu_pro",
    split: str = "test",
    categories: str | Iterable[str] | None = "biology",
    max_probes: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    normalized = source.strip().lower().replace("-", "_")
    if normalized in {"mmlu_pro", "mmlupro"}:
        return load_mmlu_pro_probes(split=split, categories=categories, max_probes=max_probes, seed=seed)
    if normalized == "mmlu":
        return load_mmlu_probes(split=split, categories=categories, max_probes=max_probes, seed=seed)
    raise ValueError("DuFFin probe source must be one of: mmlu_pro, mmlu")
