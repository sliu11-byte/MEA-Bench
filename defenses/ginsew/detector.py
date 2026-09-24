"""GINSEW detector (checkpoint probing + Lomb-Scargle Psnr / green-hit stats).

Paper (Zhao et al., ICML 2023): probe a suspect model, collect
``(t, Q_G1)`` pairs, then compute signal-to-noise ratio ``P_snr`` at the
watermark frequency via Lomb-Scargle periodogram.

Also reports a green-token hit-rate z-score (text-only fallback / auxiliary).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import lombscargle

from defenses.core.generation import format_prompt
from defenses.core.io_utils import load_queries, read_jsonl
from defenses.core.student_loader import load_student_from_checkpoint, load_student_from_manifest
from defenses.ginsew.watermark import WatermarkBase, WatermarkLogitsWarper, build_warper

logger = logging.getLogger(__name__)
PathLike = Union[str, Path]


def load_watermark_artifacts(artifacts_dir: PathLike) -> Tuple[WatermarkLogitsWarper, Dict[str, Any]]:
    artifacts_dir = Path(artifacts_dir)
    cfg_path = artifacts_dir / "watermark_config.json"
    state_path = artifacts_dir / "watermark_state.npz"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}

    if state_path.exists():
        data = np.load(state_path, allow_pickle=False)
        state = {
            "fraction": float(data["fraction"]),
            "strength": float(data["strength"]),
            "vocab_size": int(data["vocab_size"]),
            "watermark_key": int(data["watermark_key"]),
            "freq": int(data["freq"]),
            "eps": float(data["eps"]),
            "green_list_mask": data["green_list_mask"],
            "vec": data["vec"],
            "key": data["key"],
        }
        warper = build_warper(vocab_size=state["vocab_size"], state=state)
        return warper, {**cfg, **warper.to_config()}

    warper = build_warper(
        vocab_size=int(cfg.get("vocab_size", 50257)),
        watermark_key=int(cfg.get("watermark_key", 0)),
        fraction=float(cfg.get("fraction", 0.5)),
        strength=float(cfg.get("strength", 2.0)),
        freq=int(cfg.get("freq", 16)),
        eps=float(cfg.get("eps", 0.2)),
    )
    return warper, {**cfg, **warper.to_config()}


def _context_t(warper: WatermarkBase, token_id: int) -> float:
    token_id = int(np.clip(token_id, 0, warper.vec.shape[0] - 1))
    x = torch.matmul(warper.vec[token_id], warper.key)
    x_ = torch.distributions.Normal(0, 1).cdf(x / np.sqrt(warper.key.shape[0] / 3))
    return float(x_.item())


@torch.inference_mode()
def collect_group_probs(
    model,
    tokenizer,
    queries: Sequence[Dict[str, str]],
    warper: WatermarkBase,
    *,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    qmin: float = 0.0,
    system_prompt: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Probe model: for each decoding step collect (t, Q_G1).

    Returns times, q_g1, aux stats (including ``student_outputs`` per probe).
    """
    device = model.device
    green = warper.green_list_mask.to(device)
    g1_idx = torch.where(green == 1)[0]
    times: List[float] = []
    qvals: List[float] = []
    green_hits = 0
    scored_tokens = 0
    total_tokens = 0
    student_outputs: List[Dict[str, Any]] = []

    for qi, q in enumerate(queries):
        prompt = format_prompt(tokenizer, q["query"], system_prompt=system_prompt)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        generated = input_ids.clone()
        probe_times: List[float] = []
        probe_q: List[float] = []
        out_token_ids: List[int] = []

        for _ in range(max_new_tokens):
            outputs = model(generated)
            logits = outputs.logits[:, -1, :]
            probs = F.softmax(logits[0], dim=0)
            q_g1 = float(probs[g1_idx].sum().item())

            ctx_idx = min(5, generated.shape[1] - 1)
            t = _context_t(warper, int(generated[0, ctx_idx].item()))

            if q_g1 > qmin:
                times.append(t)
                qvals.append(q_g1)
                probe_times.append(t)
                probe_q.append(q_g1)

            if temperature and temperature > 0:
                next_id = torch.multinomial(F.softmax(logits[0] / temperature, dim=0), 1)
            else:
                next_id = torch.argmax(logits[0], dim=-1, keepdim=True)

            tok = int(next_id.item())
            out_token_ids.append(tok)
            total_tokens += 1
            if green[tok] > 0.5:
                green_hits += 1
            scored_tokens += 1

            generated = torch.cat([generated, next_id.view(1, 1)], dim=1)
            if tok == tokenizer.eos_token_id:
                break

        response = tokenizer.decode(out_token_ids, skip_special_tokens=True)
        probe_id = q.get("query_id") or q.get("id") or f"probe_{qi}"
        student_outputs.append(
            {
                "probe_id": probe_id,
                "query": q.get("query", ""),
                "response": response,
                "token_ids": out_token_ids,
                "times": probe_times,
                "q_g1": probe_q,
            }
        )

    aux = {
        "num_pairs": len(times),
        "green_hits": green_hits,
        "scored_tokens": scored_tokens,
        "total_tokens": total_tokens,
        "green_rate": green_hits / max(1, scored_tokens),
        "student_outputs": student_outputs,
    }
    return np.asarray(times, dtype=np.float64), np.asarray(qvals, dtype=np.float64), aux


def compute_psnr(times: np.ndarray, qvals: np.ndarray, freq: float) -> float:
    """
    Lomb-Scargle periodogram SNR at watermark frequency.

    P_snr ≈ power(f_w) / mean(power at other frequencies).
    """
    if len(times) < 8:
        return 0.0
    # Detrend
    y = qvals - qvals.mean()
    # Candidate angular frequencies around watermark freq and a noise grid
    f_w = float(freq)
    # Convert paper freq to angular freq for lombscargle: w = 2*pi*f
    # Official warper uses cos(freq * x_) with x_ in [0,1], so period ≈ 2π/freq in t-space.
    # Use angular frequency = freq (as used inside cos(freq * x_)).
    w_target = np.array([f_w], dtype=np.float64)
    w_grid = np.linspace(max(0.5, f_w / 4), f_w * 4, 64)
    w_grid = w_grid[np.abs(w_grid - f_w) > 0.5]
    if len(w_grid) == 0:
        w_grid = np.linspace(1.0, 40.0, 64)

    p_target = lombscargle(times, y, w_target, normalize=True)
    p_noise = lombscargle(times, y, w_grid, normalize=True)
    noise = float(np.mean(np.atleast_1d(p_noise))) + 1e-12
    target = float(np.atleast_1d(p_target)[0])
    return target / noise


def green_zscore(green_rate: float, n: int, fraction: float) -> float:
    if n <= 0:
        return 0.0
    return (green_rate - fraction) / np.sqrt(fraction * (1 - fraction) / n)


def _extract_probe_pairs_from_row(row: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    times = row.get("times") or row.get("probe_times")
    q_g1 = row.get("q_g1") or row.get("probe_q_g1")
    if times is None or q_g1 is None:
        return [], []
    return [float(t) for t in times], [float(q) for q in q_g1]


def score_from_outputs_jsonl(
    outputs_path: PathLike,
    warper: WatermarkBase,
    *,
    response_key: str = "response",
    tokenizer: Any = None,
) -> Dict[str, Any]:
    """Black-box detector path (spec input 'B').

    Prefer paper PSNR when rows carry ``times``+``q_g1`` (or
    ``probe_times``/``probe_q_g1``). Otherwise fall back to green-token
    z-score from ``token_ids``/``response``.
    """
    rows = read_jsonl(outputs_path)

    all_times: List[float] = []
    all_q: List[float] = []
    for row in rows:
        t, q = _extract_probe_pairs_from_row(row)
        all_times.extend(t)
        all_q.extend(q)

    if len(all_times) >= 8 and len(all_times) == len(all_q):
        psnr = compute_psnr(
            np.asarray(all_times, dtype=np.float64),
            np.asarray(all_q, dtype=np.float64),
            float(warper.freq),
        )
        # Still compute green stats when token_ids available (auxiliary).
        green_hits = 0
        n = 0
        for row in rows:
            token_ids = row.get("token_ids") or row.get("output_ids")
            if token_ids is None and tokenizer is not None:
                text = row.get(response_key)
                if text:
                    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if token_ids is None:
                continue
            for tid in token_ids:
                tid = int(tid)
                if 0 <= tid < len(warper.green_list_mask):
                    n += 1
                    if warper.green_list_mask[tid] > 0.5:
                        green_hits += 1
        rate = green_hits / n if n else None
        z = green_zscore(rate, n, warper.fraction) if n else None
        return {
            "score": float(psnr),
            "psnr": float(psnr),
            "direction": "higher",
            "score_direction": "higher",
            "label": None,
            "green_rate": rate,
            "z_score": float(z) if z is not None else None,
            "num_tokens": n,
            "num_pairs": len(all_times),
            "score_note": "psnr_from_probe_pairs",
        }

    green_hits = 0
    n = 0
    for row in rows:
        token_ids = row.get("token_ids") or row.get("output_ids")
        if token_ids is None and tokenizer is not None:
            text = row.get(response_key)
            if text:
                token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if token_ids is None:
            continue
        for tid in token_ids:
            tid = int(tid)
            if 0 <= tid < len(warper.green_list_mask):
                n += 1
                if warper.green_list_mask[tid] > 0.5:
                    green_hits += 1
    if n == 0:
        return {
            "score": 0.0,
            "direction": "higher",
            "score_direction": "higher",
            "label": None,
            "green_rate": None,
            "z_score": None,
            "note": "outputs lacked token_ids/response(+tokenizer); provide checkpoint probe instead",
            "score_note": "no_scorable_outputs",
            "num_tokens": 0,
        }
    rate = green_hits / n
    z = green_zscore(rate, n, warper.fraction)
    return {
        "score": float(z),
        "direction": "higher",
        "score_direction": "higher",
        "label": None,
        "green_rate": rate,
        "z_score": float(z),
        "num_tokens": n,
        "score_note": "text_only_green_zscore_fallback",
    }


class GinsewDetector:
    def __init__(self, watermark_artifacts: PathLike, *, qmin: float = 0.0):
        self.artifacts_dir = Path(watermark_artifacts)
        self.warper, self.config = load_watermark_artifacts(self.artifacts_dir)
        self.qmin = qmin

    def detect_checkpoint(
        self,
        student_checkpoint: Optional[PathLike] = None,
        *,
        attack_manifest: Optional[PathLike] = None,
        probe_queries: Union[PathLike, Sequence[Dict[str, str]]],
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        max_queries: Optional[int] = None,
        label: Optional[str] = None,
        student_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
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

        times, qvals, aux = collect_group_probs(
            model,
            tokenizer,
            queries,
            self.warper,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            qmin=self.qmin,
            system_prompt=system_prompt,
        )
        psnr = compute_psnr(times, qvals, float(self.warper.freq))
        z = green_zscore(aux["green_rate"], aux["scored_tokens"], self.warper.fraction)
        student_outputs = aux.get("student_outputs") or []
        probe_pairs = {
            "times": times.tolist(),
            "q_g1": qvals.tolist(),
            "num_pairs": int(aux["num_pairs"]),
        }

        # Primary score = Psnr (paper); also surface z-score.
        return {
            "student": student_name or ckpt_str,
            "student_checkpoint": ckpt_str,
            "score": float(psnr),
            "psnr": float(psnr),
            "z_score": float(z),
            "green_rate": aux["green_rate"],
            "direction": "higher",
            "score_direction": "higher",
            "label": label,
            "num_probe_queries": len(queries),
            "num_pairs": aux["num_pairs"],
            "scored_tokens": aux["scored_tokens"],
            "total_tokens": aux["total_tokens"],
            "student_outputs": student_outputs,
            "probe_pairs": probe_pairs,
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
            outputs_path, self.warper, response_key=response_key, tokenizer=tokenizer
        )
        result["student"] = student_name or str(outputs_path)
        result["student_checkpoint"] = None
        result["student_outputs"] = str(Path(outputs_path).resolve())
        result["label"] = label
        result.setdefault("num_probe_queries", result.get("num_tokens"))
        return result
