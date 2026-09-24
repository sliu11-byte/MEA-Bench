"""Radioactivity checkpoint detector (aggregate green-rate → p-value)."""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import torch

from defenses.core.generation import format_prompt
from defenses.core.io_utils import load_queries, read_jsonl
from defenses.core.student_loader import load_student_from_checkpoint, load_student_from_manifest
from defenses.radioactivity.watermark import MarylandWatermark

logger = logging.getLogger(__name__)
PathLike = Union[str, Path]

# Paper-style radioactive detection aggregates many outputs; fewer probes make
# p-values noisy. This is a usage warning only — detection still runs.
_LOW_PROBE_WARN_THRESHOLD = 200


def warn_if_low_probe_count(n: int, *, source: str = "probes") -> None:
    """Log an English warning when probe/sample count is below the soft threshold."""
    if n < _LOW_PROBE_WARN_THRESHOLD:
        msg = (
            f"Radioactivity detector: only {n} {source} "
            f"(recommended >= {_LOW_PROBE_WARN_THRESHOLD} for stable p-values). "
            "Results may be noisy; increase --max_queries / probe pool size for formal runs."
        )
        logger.warning(msg)
        # Also print so CLI runs see it without requiring logging.basicConfig.
        print(f"[warn] {msg}", flush=True)


def load_watermark_artifacts(artifacts_dir: PathLike) -> Tuple[MarylandWatermark, Dict[str, Any], Optional[Set[tuple]]]:
    artifacts_dir = Path(artifacts_dir)
    cfg = json.loads((artifacts_dir / "watermark_config.json").read_text(encoding="utf-8"))
    wm = MarylandWatermark(
        vocab_size=int(cfg["vocab_size"]),
        ngram=int(cfg.get("ngram", 1)),
        seed=int(cfg.get("seed", 0)),
        hash_key=int(cfg.get("hash_key", 35317)),
        seeding=str(cfg.get("seeding", "hash")),
        gamma=float(cfg.get("gamma", 0.5)),
        delta=float(cfg.get("delta", 1.0)),
        scoring_method=str(cfg.get("scoring_method", "none")),
    )
    data_filter = None
    filter_path = artifacts_dir / "filter.pkl"
    if filter_path.exists():
        with filter_path.open("rb") as f:
            labels, _values = pickle.load(f)
        data_filter = set(labels)
        logger.info("Loaded filter with %d ngrams", len(data_filter))
    return wm, cfg, data_filter


def score_from_outputs_jsonl(
    outputs_path: PathLike,
    wm: MarylandWatermark,
    *,
    data_filter: Optional[Set[tuple]] = None,
    response_key: str = "response",
    tokenizer: Any = None,
) -> Dict[str, Any]:
    """Black-box detector path (spec input 'B'): score radioactive watermark
    stats directly from a ``student_outputs.jsonl`` file, no checkpoint
    needed. Rows may carry pre-tokenized ``token_ids``/``output_ids``, or
    ``response`` text (re-tokenized with ``tokenizer`` if given; must match
    the vocab the watermark artifacts were built for).
    """
    rows = read_jsonl(outputs_path)
    token_lists: List[List[int]] = []
    for row in rows:
        ids = row.get("token_ids") or row.get("output_ids")
        if ids is None and tokenizer is not None:
            text = row.get(response_key)
            if text:
                ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if ids is None:
            continue
        token_lists.append([int(t) for t in ids])

    if not token_lists:
        warn_if_low_probe_count(0, source="scorable output samples")
        return {
            "score": 0.0,
            "green_rate": None,
            "p_value": 1.0,
            "p_value_z": 1.0,
            "p_value_binomial": 1.0,
            "direction": "lower",
            "score_direction": "lower",
            "label": None,
            "note": "outputs lacked token_ids/response(+tokenizer); provide checkpoint probe instead",
            "num_samples": 0,
            "scored_tokens": 0,
            "total_tokens": 0,
        }
    warn_if_low_probe_count(len(token_lists), source="scorable output samples")
    stats = wm.score_texts(token_lists, data_filter=data_filter, chunked=True)
    return {
        "score": float(stats["p_value_z"]),
        "green_score": float(stats["score"]),
        "green_rate": float(stats["green_rate"]),
        "p_value": float(stats["p_value_z"]),
        "p_value_z": float(stats["p_value_z"]),
        "p_value_binomial": float(stats["p_value_binomial"]),
        "direction": "lower",
        "score_direction": "lower",
        "label": None,
        "num_samples": stats["num_samples"],
        "scored_tokens": stats["scored_tokens"],
        "total_tokens": stats["total_tokens"],
    }


class RadioactivityDetector:
    def __init__(self, watermark_artifacts: PathLike, *, use_filter: bool = True):
        self.artifacts_dir = Path(watermark_artifacts)
        self.wm, self.config, data_filter = load_watermark_artifacts(self.artifacts_dir)
        self.data_filter = data_filter if use_filter else None

    @torch.inference_mode()
    def _generate_outputs(
        self,
        model,
        tokenizer,
        queries: Sequence[Dict[str, str]],
        *,
        max_new_tokens: int,
        temperature: float,
        system_prompt: Optional[str],
    ) -> Tuple[List[List[int]], List[Dict[str, Any]]]:
        token_lists: List[List[int]] = []
        student_outputs: List[Dict[str, Any]] = []
        for qi, q in enumerate(queries):
            prompt = format_prompt(tokenizer, q["query"], system_prompt=system_prompt)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            prompt_len = int(inputs["input_ids"].shape[1])
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=0.95 if temperature > 0 else None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            full_ids = gen[0].tolist()
            # Keep full sequence for watermark scoring (matches prior behavior).
            token_lists.append(full_ids)
            response_ids = full_ids[prompt_len:]
            response = tokenizer.decode(response_ids, skip_special_tokens=True)
            probe_id = q.get("query_id") or q.get("id") or f"probe_{qi}"
            student_outputs.append(
                {
                    "probe_id": probe_id,
                    "query": q.get("query", ""),
                    "response": response,
                    "token_ids": response_ids,
                }
            )
        return token_lists, student_outputs

    def detect_checkpoint(
        self,
        student_checkpoint: Optional[PathLike] = None,
        *,
        attack_manifest: Optional[PathLike] = None,
        probe_queries: Union[PathLike, Sequence[Dict[str, str]]],
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        max_queries: Optional[int] = None,
        label: Optional[str] = None,
        student_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        pvalue_method: str = "z",
    ) -> Dict[str, Any]:
        if attack_manifest is not None:
            model, tokenizer, _ = load_student_from_manifest(attack_manifest)
            ckpt_str = str(attack_manifest)
        elif student_checkpoint is not None:
            model, tokenizer = load_student_from_checkpoint(student_checkpoint)
            ckpt_str = str(student_checkpoint)
        else:
            raise ValueError("Provide student_checkpoint or attack_manifest")

        if isinstance(probe_queries, (str, Path)):
            queries = load_queries(probe_queries, max_queries=max_queries)
        else:
            queries = list(probe_queries)
            if max_queries is not None:
                queries = queries[:max_queries]

        warn_if_low_probe_count(len(queries), source="probe queries")

        token_lists, student_outputs = self._generate_outputs(
            model,
            tokenizer,
            queries,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
        )
        stats = self.wm.score_texts(
            token_lists,
            data_filter=self.data_filter,
            chunked=True,
        )
        p_value = stats["p_value_z"] if pvalue_method == "z" else stats["p_value_binomial"]

        return {
            "student": student_name or ckpt_str,
            "student_checkpoint": ckpt_str,
            "score": float(p_value),
            "green_score": float(stats["score"]),
            "green_rate": float(stats["green_rate"]),
            "p_value": float(p_value),
            "p_value_z": float(stats["p_value_z"]),
            "p_value_binomial": float(stats["p_value_binomial"]),
            "direction": "lower",
            "score_direction": "lower",
            "label": label,
            "num_probe_queries": len(queries),
            "num_samples": stats["num_samples"],
            "scored_tokens": stats["scored_tokens"],
            "total_tokens": stats["total_tokens"],
            "student_outputs": student_outputs,
        }

    def detect_outputs(
        self,
        outputs_path: PathLike,
        *,
        tokenizer: Any = None,
        label: Optional[str] = None,
        student_name: Optional[str] = None,
        response_key: str = "response",
    ) -> Dict[str, Any]:
        """Black-box detector path (spec input 'B'): score an existing
        ``student_outputs.jsonl`` without loading any checkpoint.
        """
        result = score_from_outputs_jsonl(
            outputs_path,
            self.wm,
            data_filter=self.data_filter,
            response_key=response_key,
            tokenizer=tokenizer,
        )
        result["student"] = student_name or str(outputs_path)
        result["student_checkpoint"] = None
        result["student_outputs"] = str(Path(outputs_path).resolve())
        result["label"] = label
        result.setdefault("num_probe_queries", result.get("num_samples"))
        return result
