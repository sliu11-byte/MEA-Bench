"""Maryland / KGW-style watermark (facebookresearch/radioactive-watermark)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from scipy import special


class MarylandWatermark:
    """
    KGW / Maryland green-list watermark.

    At each step, hash the previous ``ngram`` tokens to seed an RNG, partition
    the vocabulary into a green list of size ``gamma * V``, and add ``delta``
    to green-list logits.
    """

    def __init__(
        self,
        vocab_size: int,
        *,
        ngram: int = 1,
        seed: int = 0,
        hash_key: int = 35317,
        seeding: str = "hash",
        gamma: float = 0.5,
        delta: float = 1.0,
        scoring_method: str = "none",
    ):
        self.vocab_size = int(vocab_size)
        self.ngram = int(ngram)
        self.seed = int(seed)
        self.hash_key = int(hash_key)  # salt_key in official code
        self.seeding = seeding
        self.gamma = float(gamma)
        self.delta = float(delta)
        self.scoring_method = scoring_method
        self.hashtable = torch.randperm(1000003)
        self.rng = torch.Generator()
        self.rng.manual_seed(self.seed)

    def to_config(self) -> Dict[str, Any]:
        return {
            "method": "maryland",
            "vocab_size": self.vocab_size,
            "ngram": self.ngram,
            "seed": self.seed,
            "hash_key": self.hash_key,
            "seeding": self.seeding,
            "gamma": self.gamma,
            "delta": self.delta,
            "scoring_method": self.scoring_method,
        }

    def hashint(self, integer_tensor: torch.LongTensor) -> torch.LongTensor:
        return self.hashtable[integer_tensor.cpu() % len(self.hashtable)]

    def get_seed_rng(self, input_ids: Sequence[int]) -> int:
        if self.seeding == "hash":
            seed = self.seed
            for i in input_ids:
                seed = (seed * self.hash_key + int(i)) % (2**64 - 1)
            return seed
        if self.seeding == "additive":
            seed = self.hash_key * int(sum(input_ids))
            return int(self.hashint(torch.tensor(seed)).item())
        if self.seeding == "skip":
            seed = self.hash_key * int(input_ids[0])
            return int(self.hashint(torch.tensor(seed)).item())
        if self.seeding == "min":
            seed = self.hashint(torch.tensor([self.hash_key * int(i) for i in input_ids]))
            return int(torch.min(seed).item())
        raise ValueError(f"Unknown seeding={self.seeding}")

    def greenlist(self, ngram_tokens: Sequence[int]) -> torch.LongTensor:
        seed = self.get_seed_rng(ngram_tokens)
        self.rng.manual_seed(seed)
        vocab_permutation = torch.randperm(self.vocab_size, generator=self.rng)
        return vocab_permutation[: int(self.gamma * self.vocab_size)]

    def bias_logits(self, logits: torch.FloatTensor, ngram_tokens_batch: torch.LongTensor) -> torch.FloatTensor:
        """Add delta bias to green-list tokens for each batch row."""
        logits = logits.clone()
        bsz, vocab_size = logits.shape
        for ii in range(bsz):
            green = self.greenlist(ngram_tokens_batch[ii].tolist())
            bias = torch.zeros(vocab_size, device=logits.device, dtype=logits.dtype)
            bias[green] = self.delta
            logits[ii] = logits[ii] + bias
        return logits

    def score_token(self, ngram_tokens: Sequence[int], token_id: int) -> float:
        green = self.greenlist(ngram_tokens)
        return 1.0 if int(token_id) in set(green.tolist()) else 0.0

    def pvalue_binomial(self, score: float, ntoks: int, eps: float = 1e-200) -> float:
        """Exact MarylandDetector p-value via betainc / binomial CDF."""
        if ntoks <= 0:
            return 1.0
        pvalue = special.betainc(score, 1 + ntoks - score, self.gamma)
        return float(max(pvalue, eps))

    def pvalue_z(self, score: float, ntoks: int, eps: float = 1e-200) -> float:
        """MarylandDetectorZ p-value from normal approximation."""
        if ntoks <= 0:
            return 1.0
        zscore = (score - self.gamma * ntoks) / np.sqrt(self.gamma * (1 - self.gamma) * ntoks)
        pvalue = 0.5 * special.erfc(zscore / np.sqrt(2))
        return float(max(pvalue, eps))

    def score_texts(
        self,
        token_id_lists: List[List[int]],
        *,
        data_filter: Optional[set] = None,
        chunked: bool = True,
    ) -> Dict[str, Any]:
        """
        Score generated token sequences.

        When ``chunked=True``, aggregates all texts into one score (radioactivity
        paper style — needs large token volume for stable p-values).
        """
        rts: List[float] = []
        scored_tokens = 0
        total_tokens = 0
        num_samples = len(token_id_lists)
        seen = set()

        for tokens in token_id_lists:
            total_tokens += len(tokens)
            start_pos = self.ngram + 1
            for cur_pos in range(start_pos, len(tokens)):
                ngram_tokens = tokens[cur_pos - self.ngram : cur_pos]
                tup = tuple(ngram_tokens)
                if self.scoring_method == "v1":
                    if tup in seen or (data_filter is not None and tup not in data_filter):
                        continue
                    seen.add(tup)
                elif self.scoring_method == "v2":
                    tup2 = tuple(list(ngram_tokens) + [tokens[cur_pos]])
                    if tup2 in seen or (data_filter is not None and tup not in data_filter):
                        continue
                    seen.add(tup2)
                elif data_filter is not None and tup not in data_filter:
                    continue

                rt = self.score_token(ngram_tokens, tokens[cur_pos])
                rts.append(rt)
                scored_tokens += 1

        score_sum = float(np.sum(rts)) if rts else 0.0
        n = len(rts)
        p_bin = self.pvalue_binomial(score_sum, n)
        p_z = self.pvalue_z(score_sum, n)
        green_rate = score_sum / max(1, n)

        return {
            "score": score_sum,
            "green_rate": green_rate,
            "p_value": p_z,
            "p_value_binomial": p_bin,
            "p_value_z": p_z,
            "scored_tokens": scored_tokens,
            "total_tokens": total_tokens,
            "num_samples": num_samples,
            "score_direction": "lower",
        }
