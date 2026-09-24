from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from zipfile import ZipFile

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from huggingface_hub import hf_hub_download

from evaluation.core.config import DatasetSpec
from evaluation.tasks.parsing import Parser


class DatasetLoadError(RuntimeError):
    pass


@dataclass(frozen=True)
class SplitSelection:
    requested_split: str
    actual_split: str | None
    status: str
    fallback_used: bool
    note: str


@dataclass
class AdaptedExample:
    example_id: str
    example_index: int
    split: str
    raw_example: dict[str, Any]
    logical_prompt: str
    rendered_prompt: str
    gold_answer: Any
    normalized_gold_answer: str | None
    parser: Parser
    extra: dict[str, Any]


class DatasetAdapter:
    prompt_template_id = "m1_generic"
    prompt_template_version = "1"
    few_shot_setting = "zero_shot"
    preprocessing_rule = "none"
    label_space: list[str] = []
    label_mapping: dict[str, str] = {}

    def __init__(self, spec: DatasetSpec):
        self.spec = spec
        self._dataset: DatasetDict | None = None
        self._load_error: str | None = None

    @property
    def prompt_template_hash(self) -> str:
        material = f"{self.prompt_template_id}:{self.prompt_template_version}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def load(self) -> DatasetDict:
        if self._dataset is not None:
            return self._dataset
        try:
            if self.spec.local_path:
                path = Path(self.spec.local_path)
                if not path.exists():
                    raise DatasetLoadError(f"Configured local_path does not exist: {path}")
                loaded = load_from_disk(str(path))
            else:
                manual = _load_retired_script_dataset(self.spec)
                if manual is not None:
                    self._dataset = manual
                    return manual
                kwargs: dict[str, Any] = {}
                if self.spec.revision:
                    kwargs["revision"] = self.spec.revision
                loaded = load_dataset(self.spec.identifier, self.spec.config_name, **kwargs)
            if isinstance(loaded, Dataset):
                loaded = DatasetDict({"train": loaded})
            if not isinstance(loaded, DatasetDict):
                raise DatasetLoadError(f"Expected DatasetDict, found {type(loaded).__name__}")
            self._dataset = loaded
            return loaded
        except Exception as exc:
            self._load_error = f"{type(exc).__name__}: {exc}"
            raise DatasetLoadError(
                f"Unable to load {self.spec.key} from {self.spec.identifier}"
                f" config={self.spec.config_name!r}: {self._load_error}"
            ) from exc

    def available_splits(self) -> list[str]:
        return list(self.load().keys())

    def resolve_split(self, requested: str, allow_fallback: bool) -> SplitSelection:
        available = self.available_splits()
        if requested in available:
            return SplitSelection(requested, requested, "available", False, "exact official split")
        if allow_fallback and requested == "validation" and "test" in available:
            return SplitSelection(
                requested, "test", "fallback", True,
                "explicit --allow-split-fallback mapped validation to official test",
            )
        return SplitSelection(
            requested, None, "unavailable", False,
            f"requested exact split {requested!r}; available={available}",
        )

    def get_split(self, split: str) -> Dataset:
        dataset = self.load()
        if split not in dataset:
            raise DatasetLoadError(f"Split {split!r} not available for {self.spec.key}")
        return dataset[split]

    def make_example_id(self, raw: dict[str, Any], split: str, index: int) -> str:
        for field in ("task_id", "id", "idx", "example_id", "question_id"):
            value = raw.get(field)
            if value is not None and str(value).strip():
                return str(value)
        canonical = json.dumps(raw, sort_keys=True, ensure_ascii=False, default=str)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]
        return f"{self.spec.key}:{split}:{digest}"

    def adapt(self, raw: dict[str, Any], split: str, index: int) -> AdaptedExample:
        raise NotImplementedError

    def adapt_many(self, rows: Iterable[tuple[int, dict[str, Any]]], split: str) -> list[AdaptedExample]:
        return [self.adapt(raw, split, index) for index, raw in rows]

    def dataset_metadata(self, selections: list[SplitSelection]) -> dict[str, Any]:
        dataset = self.load()
        split_sizes = {name: len(value) for name, value in dataset.items()}
        fingerprints = {
            name: getattr(value, "_fingerprint", None)
            for name, value in dataset.items()
        }
        selected = [item.actual_split for item in selections if item.actual_split]
        return {
            "dataset_name": self.spec.display_name,
            "dataset_source": self.spec.source,
            "local_path_or_Hugging_Face_identifier": self.spec.local_path or self.spec.identifier,
            "dataset_revision_or_version_if_available": self.spec.revision,
            "dataset_fingerprints": fingerprints,
            "config_name": self.spec.config_name,
            "split": selected,
            "available_splits": list(dataset.keys()),
            "total_available_examples": split_sizes,
            "selected_example_count": {},
            "task_type": self.spec.task_type,
            "label_space": self.label_space,
            "label_mapping": self.label_mapping,
            "prompt_template_id": self.prompt_template_id,
            "prompt_template_version_or_hash": self.prompt_template_hash,
            "few_shot_setting": self.few_shot_setting,
            "preprocessing_or_filtering_rule": self.preprocessing_rule,
            "provenance_note": self.spec.provenance_note,
        }


def format_multiple_choice(question: str, options: list[tuple[str, str]]) -> tuple[str, str]:
    option_lines = "\n".join(f"{label}. {text}" for label, text in options)
    logical = f"Question: {question.strip()}\n{option_lines}"
    rendered = (
        f"{logical}\n\nSelect the best answer. "
        f"End your response with 'Final answer: <LABEL>', using one of: "
        f"{', '.join(label for label, _ in options)}."
    )
    return logical, rendered


def _load_retired_script_dataset(spec: DatasetSpec) -> DatasetDict | None:
    if spec.key == "medqa" and spec.identifier == "bigbio/med_qa":
        return _load_medqa_zip()
    if spec.key == "pubmedqa" and spec.identifier == "bigbio/pubmed_qa":
        return _load_pubmedqa_zip(spec.config_name)
    if spec.key == "fpb" and spec.identifier == "takala/financial_phrasebank":
        return _load_fpb_zip(spec.config_name)
    return None


def _jsonl_rows_from_zip(zip_path: str, member: str) -> list[dict[str, Any]]:
    rows = []
    with ZipFile(zip_path) as archive:
        with archive.open(member) as handle:
            for line in handle:
                text = line.decode("utf-8").strip()
                if text:
                    rows.append(json.loads(text))
    return rows


def _load_medqa_zip() -> DatasetDict:
    zip_path = hf_hub_download("bigbio/med_qa", "data_clean.zip", repo_type="dataset")
    members = {
        "train": "data_clean/questions/US/train.jsonl",
        "validation": "data_clean/questions/US/dev.jsonl",
        "test": "data_clean/questions/US/test.jsonl",
    }
    return DatasetDict({
        split: Dataset.from_list(_jsonl_rows_from_zip(zip_path, member))
        for split, member in members.items()
    })


def _rows_from_json_object_zip(zip_path: str, member: str) -> list[dict[str, Any]]:
    with ZipFile(zip_path) as archive:
        data = json.loads(archive.read(member).decode("utf-8"))
    rows = []
    for key, row in data.items():
        row = dict(row)
        row.setdefault("id", key)
        rows.append(row)
    return rows


def _load_pubmedqa_zip(config_name: str | None) -> DatasetDict:
    # Equivalent to the legacy script's default source config when config_name is null.
    if config_name is None or "artificial" in config_name:
        zip_path = hf_hub_download("bigbio/pubmed_qa", "pqaa.zip", repo_type="dataset")
        return DatasetDict({
            "train": Dataset.from_list(_rows_from_json_object_zip(zip_path, "pqaa_train_set.json")),
            "validation": Dataset.from_list(_rows_from_json_object_zip(zip_path, "pqaa_dev_set.json")),
        })
    if "labeled" in config_name:
        zip_path = hf_hub_download("bigbio/pubmed_qa", "pqal.zip", repo_type="dataset")
        return DatasetDict({
            "train": Dataset.from_list(_rows_from_json_object_zip(zip_path, "pqal_fold0/train_set.json")),
            "validation": Dataset.from_list(_rows_from_json_object_zip(zip_path, "pqal_fold0/dev_set.json")),
            "test": Dataset.from_list(_rows_from_json_object_zip(zip_path, "pqal_test_set.json")),
        })
    raise DatasetLoadError(f"Manual PubMedQA loader does not support config {config_name!r}")


def _load_fpb_zip(config_name: str | None) -> DatasetDict:
    config_to_member = {
        None: "FinancialPhraseBank-v1.0/Sentences_50Agree.txt",
        "sentences_50agree": "FinancialPhraseBank-v1.0/Sentences_50Agree.txt",
        "sentences_66agree": "FinancialPhraseBank-v1.0/Sentences_66Agree.txt",
        "sentences_75agree": "FinancialPhraseBank-v1.0/Sentences_75Agree.txt",
        "sentences_allagree": "FinancialPhraseBank-v1.0/Sentences_AllAgree.txt",
    }
    try:
        member = config_to_member[config_name]
    except KeyError as exc:
        raise DatasetLoadError(f"Manual FPB loader does not support config {config_name!r}") from exc
    zip_path = hf_hub_download(
        "takala/financial_phrasebank",
        "data/FinancialPhraseBank-v1.0.zip",
        repo_type="dataset",
    )
    rows = []
    with ZipFile(zip_path) as archive:
        for index, line in enumerate(archive.read(member).decode("iso-8859-1").splitlines()):
            if not line.strip():
                continue
            sentence, label = line.rsplit("@", 1)
            rows.append({"id": index, "sentence": sentence.strip(), "label": label.strip()})
    return DatasetDict({"train": Dataset.from_list(rows)})
