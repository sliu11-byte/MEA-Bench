"""GINSEW LogitsWarper — adapted from XuandongZhao/Ginsew."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

try:
    from transformers import LogitsWarper as _LogitsBase
except ImportError:  # transformers>=4.46 folded warpers into LogitsProcessor
    from transformers import LogitsProcessor as _LogitsBase


class WatermarkBase:
    """
    Base class for GINSEW watermarking distributions.

    Args:
        fraction: Fraction of vocab assigned to green-list (group 1).
        strength: Legacy strength param (kept for API compatibility).
        vocab_size: Vocabulary size.
        watermark_key: Random seed / secret key for green-listing.
        freq: Sinusoidal watermark frequency.
        eps: Watermark amplitude (paper ε).
    """

    def __init__(
        self,
        fraction: float = 0.5,
        strength: float = 2.0,
        vocab_size: int = 50257,
        watermark_key: int = 0,
        freq: int = 16,
        eps: float = 0.2,
    ):
        rng = np.random.default_rng(self._hash_fn(watermark_key))
        mask = np.array(
            [True] * int(fraction * vocab_size)
            + [False] * (vocab_size - int(fraction * vocab_size))
        )
        rng.shuffle(mask)
        self.vec = torch.tensor(rng.normal(loc=0, scale=1, size=(vocab_size, 256)), dtype=torch.float32)
        self.key = torch.tensor(rng.random(256), dtype=torch.float32)
        self.freq = freq
        self.eps = eps
        self.green_list_mask = torch.tensor(mask, dtype=torch.float32)
        self.strength = strength
        self.fraction = fraction
        self.vocab_size = vocab_size
        self.watermark_key = watermark_key

    @staticmethod
    def _hash_fn(x: int) -> int:
        x = np.int64(x)
        return int.from_bytes(hashlib.sha256(x.tobytes()).digest()[:4], "little")

    def to_config(self) -> Dict[str, Any]:
        return {
            "fraction": self.fraction,
            "strength": self.strength,
            "vocab_size": self.vocab_size,
            "watermark_key": self.watermark_key,
            "freq": self.freq,
            "eps": self.eps,
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            **self.to_config(),
            "green_list_mask": self.green_list_mask.cpu().numpy(),
            "vec": self.vec.cpu().numpy(),
            "key": self.key.cpu().numpy(),
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "WatermarkBase":
        obj = cls(
            fraction=float(state["fraction"]),
            strength=float(state.get("strength", 2.0)),
            vocab_size=int(state["vocab_size"]),
            watermark_key=int(state["watermark_key"]),
            freq=int(state["freq"]),
            eps=float(state["eps"]),
        )
        # Restore exact masks / vectors if provided (detector reproducibility)
        if "green_list_mask" in state:
            obj.green_list_mask = torch.tensor(state["green_list_mask"], dtype=torch.float32)
        if "vec" in state:
            obj.vec = torch.tensor(state["vec"], dtype=torch.float32)
        if "key" in state:
            obj.key = torch.tensor(state["key"], dtype=torch.float32)
        return obj


class WatermarkLogitsWarper(WatermarkBase, _LogitsBase):
    """Official GINSEW logits warper: inject sinusoidal signal into group probs."""

    def __init__(self, *args, **kwargs):
        WatermarkBase.__init__(self, *args, **kwargs)
        # LogitsProcessor / LogitsWarper may or may not take args
        try:
            _LogitsBase.__init__(self)
        except TypeError:
            pass

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.FloatTensor:
        device = scores.device
        green_mask = self.green_list_mask.to(device)
        vec = self.vec.to(device)
        key = self.key.to(device)

        # Context hash from early tokens (official uses index 5)
        ctx_idx = min(5, input_ids.shape[1] - 1)
        token_idx = input_ids[0][ctx_idx].clamp(max=vec.shape[0] - 1)
        x = torch.matmul(vec[token_idx], key)
        x_ = torch.distributions.Normal(0, 1).cdf(x / np.sqrt(key.shape[0] / 3))

        probs = F.softmax(scores[0], dim=0)
        g0_idx = torch.where(green_mask == 0)[0]
        g1_idx = torch.where(green_mask == 1)[0]
        g0_prob = probs[g0_idx].sum()
        g1_prob = probs[g1_idx].sum()
        z0 = torch.cos(self.freq * x_)
        z1 = torch.cos(torch.add(self.freq * x_, torch.pi))
        new_g0_prob = (g0_prob + 1e-25 + self.eps * (1 + z0)) / (1 + 2 * self.eps + 1e-25)
        new_g1_prob = (g1_prob + 1e-25 + self.eps * (1 + z1)) / (1 + 2 * self.eps + 1e-25)
        new_probs = probs.clone()
        new_probs[g0_idx] = new_g0_prob / (g0_prob + 1e-25) * probs[g0_idx]
        new_probs[g1_idx] = new_g1_prob / (g1_prob + 1e-25) * probs[g1_idx]
        new_logits = torch.log(new_probs + 1e-25).view(1, -1)
        if scores.shape[0] > 1:
            # Broadcast first-row warping for batched decode (same key)
            out = scores.clone()
            out[0] = new_logits[0]
            for b in range(1, scores.shape[0]):
                # Recompute per-row with that row's context token
                token_idx_b = input_ids[b][ctx_idx].clamp(max=vec.shape[0] - 1)
                x_b = torch.matmul(vec[token_idx_b], key)
                x_b_ = torch.distributions.Normal(0, 1).cdf(x_b / np.sqrt(key.shape[0] / 3))
                probs_b = F.softmax(scores[b], dim=0)
                g0_prob_b = probs_b[g0_idx].sum()
                g1_prob_b = probs_b[g1_idx].sum()
                z0_b = torch.cos(self.freq * x_b_)
                z1_b = torch.cos(torch.add(self.freq * x_b_, torch.pi))
                new_g0_b = (g0_prob_b + 1e-25 + self.eps * (1 + z0_b)) / (1 + 2 * self.eps + 1e-25)
                new_g1_b = (g1_prob_b + 1e-25 + self.eps * (1 + z1_b)) / (1 + 2 * self.eps + 1e-25)
                new_probs_b = probs_b.clone()
                new_probs_b[g0_idx] = new_g0_b / (g0_prob_b + 1e-25) * probs_b[g0_idx]
                new_probs_b[g1_idx] = new_g1_b / (g1_prob_b + 1e-25) * probs_b[g1_idx]
                out[b] = torch.log(new_probs_b + 1e-25)
            return out
        return new_logits


def build_warper(
    vocab_size: int,
    *,
    watermark_key: int = 0,
    fraction: float = 0.5,
    strength: float = 2.0,
    freq: int = 16,
    eps: float = 0.2,
    state: Optional[Dict[str, Any]] = None,
) -> WatermarkLogitsWarper:
    if state is not None:
        base = WatermarkBase.from_state_dict(state)
        warper = WatermarkLogitsWarper(
            fraction=base.fraction,
            strength=base.strength,
            vocab_size=base.vocab_size,
            watermark_key=base.watermark_key,
            freq=base.freq,
            eps=base.eps,
        )
        warper.green_list_mask = base.green_list_mask
        warper.vec = base.vec
        warper.key = base.key
        return warper
    return WatermarkLogitsWarper(
        fraction=fraction,
        strength=strength,
        vocab_size=vocab_size,
        watermark_key=watermark_key,
        freq=freq,
        eps=eps,
    )
