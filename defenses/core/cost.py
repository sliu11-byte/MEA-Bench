"""Structured cost / resource metadata for defense & detector runs."""

from __future__ import annotations

import platform
import time
from typing import Any, Dict, List, Optional, Sequence


def _device_info(device: Optional[str] = None) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "requested_device": device,
    }
    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device_count"] = int(torch.cuda.device_count())
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["resolved_device"] = device or "cuda"
        else:
            info["resolved_device"] = device or "cpu"
    except Exception:
        info["resolved_device"] = device or "unknown"
    return info


def build_cost(
    *,
    wall_seconds: float,
    num_queries: Optional[int] = None,
    models: Optional[Sequence[str]] = None,
    extra_models: Optional[Sequence[str]] = None,
    artifacts: Optional[Dict[str, str]] = None,
    caches: Optional[Sequence[str]] = None,
    data_files: Optional[Sequence[str]] = None,
    tokens_generated: Optional[int] = None,
    tokens_scored: Optional[int] = None,
    device: Optional[str] = None,
    notes: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    stage: str = "defense_install",
    method: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the cost block expected by the defense-scope docs.

    Also stamps Matrix-M6-friendly dimension aliases (``query_count``,
    ``wall_clock_hours``, ``input_tokens``/``output_tokens`` when known).
    Monetary fields (currency / prices) are left for the outer M6 ledger —
    this block never invents USD prices.
    """
    wall = round(float(wall_seconds), 3)
    cost: Dict[str, Any] = {
        "wall_seconds": wall,
        "wall_clock_hours": round(wall / 3600.0, 6),
        "stage": stage,
        "device": _device_info(device),
        # Matrix M6 uses null = unknown; we do not invent prices here.
        "api_billing": "unknown",
        "currency": None,
        "price_as_of": None,
    }
    if method:
        cost["method"] = method
    if num_queries is not None:
        nq = int(num_queries)
        cost["num_queries"] = nq
        cost["query_count"] = nq
    if models:
        cost["models"] = list(models)
    if extra_models:
        cost["extra_models"] = list(extra_models)
    if artifacts:
        cost["artifacts"] = dict(artifacts)
    if caches:
        cost["caches"] = list(caches)
    if data_files:
        cost["data_files"] = list(data_files)
    if tokens_generated is not None:
        tg = int(tokens_generated)
        cost["tokens_generated"] = tg
        cost["output_tokens"] = tg
    if tokens_scored is not None:
        ts = int(tokens_scored)
        cost["tokens_scored"] = ts
        cost.setdefault("input_tokens", ts)
    if notes:
        cost["notes"] = notes
    if extra:
        cost.update(extra)
    return cost


def merge_costs(costs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Sum wall time / tokens / queries across defense + detector + train costs."""
    total_wall = 0.0
    total_queries = 0
    total_gen = 0
    total_scored = 0
    models: List[str] = []
    parts: List[Dict[str, Any]] = []
    for c in costs:
        if not c:
            continue
        parts.append(c)
        total_wall += float(c.get("wall_seconds") or 0.0)
        total_queries += int(c.get("num_queries") or 0)
        total_gen += int(c.get("tokens_generated") or 0)
        total_scored += int(c.get("tokens_scored") or 0)
        for m in c.get("models") or []:
            if m not in models:
                models.append(m)
        for m in c.get("extra_models") or []:
            if m not in models:
                models.append(m)
    return {
        "wall_seconds": round(total_wall, 3),
        "num_queries": total_queries,
        "tokens_generated": total_gen,
        "tokens_scored": total_scored,
        "models": models,
        "parts": parts,
    }


class Timer:
    def __init__(self) -> None:
        self.t0 = time.time()

    def elapsed(self) -> float:
        return time.time() - self.t0
