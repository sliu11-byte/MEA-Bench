from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from evaluation.tasks.adapters import DatasetAdapter, DatasetLoadError, SplitSelection, create_adapter
from evaluation.core.config import M1Config


@dataclass
class DatasetDiscovery:
    dataset_key: str
    adapter: DatasetAdapter
    available_splits: list[str]
    selections: list[SplitSelection]
    load_error: str | None = None


class DatasetRegistry:
    def __init__(self, config: M1Config):
        self.config = config
        self.adapters = {key: create_adapter(spec) for key, spec in config.datasets.items()}

    def discover(self, allow_split_fallback: bool = False) -> list[DatasetDiscovery]:
        discoveries: list[DatasetDiscovery] = []
        for key, adapter in self.adapters.items():
            try:
                available = adapter.available_splits()
                selections = [
                    adapter.resolve_split(split, allow_split_fallback)
                    for split in adapter.spec.requested_splits
                ]
                discoveries.append(DatasetDiscovery(key, adapter, available, selections))
            except DatasetLoadError as exc:
                selections = [
                    SplitSelection(split, None, "unavailable", False, str(exc))
                    for split in adapter.spec.requested_splits
                ]
                discoveries.append(DatasetDiscovery(key, adapter, [], selections, str(exc)))
        return discoveries


def deterministic_sample_indices(total: int, seed: int, requested_count: int) -> list[int]:
    indices = list(range(total))
    random.Random(seed).shuffle(indices)
    return indices[: min(total, requested_count)]


def build_sample_manifest(
    adapter: DatasetAdapter,
    split: str,
    selection_seed: int,
    requested_count: int,
) -> tuple[dict[str, Any], list[int]]:
    dataset = adapter.get_split(split)
    indices = deterministic_sample_indices(len(dataset), selection_seed, requested_count)
    example_ids = [adapter.make_example_id(dict(dataset[index]), split, index) for index in indices]
    revision = adapter.spec.revision or getattr(dataset, "_fingerprint", None)
    manifest = {
        "dataset": adapter.spec.key,
        "dataset_revision": revision,
        "split": split,
        "selection_seed": selection_seed,
        "requested_count": requested_count,
        "selected_count": len(indices),
        "example_ids": example_ids,
        "example_indices": indices,
    }
    return manifest, indices

