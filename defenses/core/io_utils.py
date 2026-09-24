"""JSONL I/O helpers and query-pool adapters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

PathLike = Union[str, Path]

QUERY_ID_KEYS = ("query_id", "id", "qid", "uid", "example_id", "idx")
QUERY_TEXT_KEYS = ("query", "prompt", "question", "problem", "instruction", "text", "input")


def ensure_dir(path: PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_jsonl(path: PathLike) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object at {path}:{line_no}, got {type(obj)}")
            records.append(obj)
    return records


def write_jsonl(records: Iterable[Dict[str, Any]], path: PathLike) -> int:
    out = Path(path)
    ensure_dir(out.parent)
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def _pick(d: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def normalize_query_record(raw: Dict[str, Any], index: int) -> Dict[str, str]:
    """Normalize heterogeneous upstream query records to {query_id, query}."""
    qid = _pick(raw, QUERY_ID_KEYS)
    if qid is None:
        qid = str(index)
    else:
        qid = str(qid)

    query = _pick(raw, QUERY_TEXT_KEYS)
    if query is None:
        raise ValueError(
            f"Record {index} missing query text; tried keys {QUERY_TEXT_KEYS}. Got keys={list(raw.keys())}"
        )
    if not isinstance(query, str):
        query = str(query)
    return {"query_id": qid, "query": query}


def load_queries(path: PathLike, *, max_queries: Optional[int] = None) -> List[Dict[str, str]]:
    """
    Load a query pool from JSONL (or a JSON list) and normalize to
    ``[{query_id, query}, ...]``.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8-sig"))
        if isinstance(data, dict):
            # Real query-pool files (see D:\mpudi\stage1\data\query_pool_*.json) wrap
            # the record list as {"meta": {...}, "data": [...]}; some tools instead
            # use {"queries": [...]}. Accept either wrapper key.
            for key in ("data", "queries", "records", "items"):
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break
        if not isinstance(data, list):
            raise ValueError(f"JSON query pool must be a list (optionally wrapped in a dict), got {type(data)}")
        raw_records = data
    else:
        raw_records = read_jsonl(p)

    queries = [normalize_query_record(r, i) for i, r in enumerate(raw_records)]
    if max_queries is not None:
        queries = queries[: max(0, int(max_queries))]
    return queries
