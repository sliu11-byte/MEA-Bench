from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import urllib.request
from typing import Any

from .manifest import stable_hash, write_json


DEFAULT_DATASET_ID = os.environ.get(
    "MEA_QUERY_POOL_DATASET",
    "watermarkproject/lord-mea-benchmark",
)
DEFAULT_QUERY_POOL_FILE = "query_pool_10000.json"
DEFAULT_HF_QUERY_POOL = f"hf://{DEFAULT_DATASET_ID}/{DEFAULT_QUERY_POOL_FILE}"
AVAILABLE_QUERY_POOL_TIERS = (100, 1000, 10000, 50000, 100000)


def hf_query_pool_for_budget(budget: int) -> str:
    """Return the benchmark query-pool file that should back a budget tier.

    Exact published HF tiers are preferred. For intermediate smoke budgets, use
    the smallest published tier that can cover the requested prefix.
    """
    if budget <= 0:
        raise HFQueryPoolError("budget must be positive")
    for tier in AVAILABLE_QUERY_POOL_TIERS:
        if budget <= tier:
            return f"hf://{DEFAULT_DATASET_ID}/query_pool_{tier}.json"
    raise HFQueryPoolError(
        f"budget={budget} exceeds largest published query pool tier={AVAILABLE_QUERY_POOL_TIERS[-1]}"
    )


class HFQueryPoolError(RuntimeError):
    pass


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_cache_root() -> Path:
    storage_root = os.environ.get("STORAGE_ROOT")
    if storage_root:
        return Path(storage_root).expanduser() / "cache" / "mea_benchmark" / "hf_datasets"
    return repo_root() / ".cache" / "mea_benchmark" / "hf_datasets"


def parse_hf_spec(spec: str) -> tuple[str, str]:
    if not spec.startswith("hf://"):
        raise HFQueryPoolError(f"not an hf query pool spec: {spec}")
    payload = spec[len("hf://") :].strip("/")
    parts = payload.split("/")
    if len(parts) < 3:
        raise HFQueryPoolError("HF spec must look like hf://owner/dataset/path/to/file.json")
    dataset_id = "/".join(parts[:2])
    filename = "/".join(parts[2:])
    return dataset_id, filename


def _download_with_huggingface_hub(dataset_id: str, filename: str, cache_dir: Path) -> Path | None:
    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        return None
    downloaded = hf_hub_download(
        repo_id=dataset_id,
        filename=filename,
        repo_type="dataset",
        cache_dir=str(cache_dir),
    )
    return Path(downloaded)


def _download_with_urllib(dataset_id: str, filename: str, output_path: Path) -> Path:
    url = f"https://huggingface.co/datasets/{dataset_id}/resolve/main/{filename}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        output_path.write_bytes(response.read())
    return output_path


def ensure_hf_file(spec: str, *, cache_root: Path | None = None) -> Path:
    dataset_id, filename = parse_hf_spec(spec)
    cache_root = cache_root or default_cache_root()
    local_path = cache_root / dataset_id.replace("/", "__") / filename
    if local_path.exists():
        return local_path
    downloaded = _download_with_huggingface_hub(dataset_id, filename, cache_root)
    if downloaded is not None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if downloaded.resolve() != local_path.resolve():
            local_path.write_bytes(downloaded.read_bytes())
        return local_path
    return _download_with_urllib(dataset_id, filename, local_path)


def load_query_pool_records(query_pool_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(query_pool_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        records = payload.get("data", [])
        meta = dict(payload.get("meta", {}))
    elif isinstance(payload, list):
        records = payload
        meta = {}
    else:
        raise HFQueryPoolError(f"query pool must be a JSON object or list: {query_pool_path}")
    if not isinstance(records, list):
        raise HFQueryPoolError("query pool data must be a list")
    return [dict(record) for record in records], meta


def record_prompt_id(record: dict[str, Any], index: int) -> str:
    # Match stage1_data/build_teacher_transcript normalization exactly.
    value = record.get("id") or record.get("prompt_id") or record.get("query_id") or record.get("example_id")
    return str(value) if value is not None else f"row_{index:08d}"


def record_prompt_text(record: dict[str, Any]) -> str:
    value = record.get("prompt") or record.get("prompt_text") or record.get("query") or ""
    return str(value)


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def ensure_ordering_for_query_pool(query_pool_path: Path, *, cache_root: Path | None = None) -> Path:
    records, meta = load_query_pool_records(query_pool_path)
    pool_hash = stable_hash(
        [
            (
                record_prompt_id(record, idx),
                record_prompt_text(record),
            )
            for idx, record in enumerate(records)
        ]
    )
    prompt_pool_id = str(
        meta.get("prompt_pool_id")
        or meta.get("tier_id")
        or "openllm6_master_pool"
    )
    ordered_prompt_ids = [record_prompt_id(record, idx) for idx, record in enumerate(records)]
    ordering_payload = {
        "schema_version": "query_ordering_v1",
        "prompt_pool_id": prompt_pool_id,
        "seed": int(meta.get("seed", 0) or 0),
        "prompt_pool_hash": pool_hash,
        "ordering_hash": stable_hash(
            {
                "prompt_pool_id": prompt_pool_id,
                "prompt_pool_hash": pool_hash,
                "ordered_prompt_ids": ordered_prompt_ids,
            }
        ),
        "ordered_prompt_ids": ordered_prompt_ids,
        "ordered_indices": list(range(len(ordered_prompt_ids))),
        "source_query_pool": str(query_pool_path),
        "source_query_pool_sha256": file_sha256(query_pool_path),
    }
    cache_root = cache_root or default_cache_root()
    ordering_dir = cache_root / "orderings"
    ordering_path = ordering_dir / f"{query_pool_path.stem}.{ordering_payload['ordering_hash'][:12]}.ordering.json"
    if not ordering_path.exists():
        write_json(ordering_path, ordering_payload)
    return ordering_path


def resolve_query_pool_and_ordering(query_pool: str, query_ordering: str | None) -> tuple[Path, Path | None]:
    if query_pool.startswith("hf://"):
        query_pool_path = ensure_hf_file(query_pool)
    else:
        query_pool_path = Path(query_pool).expanduser().resolve()

    if query_ordering is None or query_ordering == "auto":
        return query_pool_path, ensure_ordering_for_query_pool(query_pool_path)
    if query_ordering.startswith("hf://"):
        return query_pool_path, ensure_hf_file(query_ordering)
    return query_pool_path, Path(query_ordering).expanduser().resolve()
