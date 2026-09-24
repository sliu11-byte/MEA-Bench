"""Radioactivity watermarked teacher generator."""

from __future__ import annotations

import json
import pickle
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn.functional as F

from defenses.core.constants import DEFENSE_ROLE
from defenses.core.generation import format_prompt, load_causal_lm, make_transcript_record
from defenses.core.io_utils import ensure_dir
from defenses.radioactivity.watermark import MarylandWatermark

PathLike = Union[str, Path]


class RadioactivityGenerator:
    def __init__(
        self,
        teacher_model: str,
        *,
        device: Optional[str] = None,
        system_prompt: Optional[str] = None,
        watermark_config: Optional[Dict[str, Any]] = None,
    ):
        self.teacher_model_name = teacher_model
        self.system_prompt = system_prompt
        self.teacher, self.tokenizer, self.device = load_causal_lm(teacher_model, device=device)
        wm = dict(watermark_config or {})
        vocab_size = len(self.tokenizer)
        self.wm = MarylandWatermark(
            vocab_size=vocab_size,
            ngram=int(wm.get("ngram", 1)),
            seed=int(wm.get("seed", 0)),
            hash_key=int(wm.get("hash_key", wm.get("salt_key", 35317))),
            seeding=str(wm.get("seeding", "hash")),
            gamma=float(wm.get("gamma", 0.5)),
            delta=float(wm.get("delta", 1.0)),
            scoring_method=str(wm.get("scoring_method", "none")),
        )
        self.watermark_artifact_id = wm.get("watermark_artifact_id") or uuid.uuid4().hex[:12]
        self._generated_token_lists: List[List[int]] = []

    def save_artifacts(self, artifacts_dir: PathLike, *, build_filter: bool = True) -> Dict[str, str]:
        artifacts_dir = ensure_dir(artifacts_dir)
        cfg_path = Path(artifacts_dir) / "watermark_config.json"
        cfg = {
            **self.wm.to_config(),
            "watermark_artifact_id": self.watermark_artifact_id,
            "teacher_model": self.teacher_model_name,
        }
        cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        paths = {"watermark_config": str(cfg_path.resolve())}

        if build_filter:
            filter_path = Path(artifacts_dir) / "filter.pkl"
            labels, values = self._build_ngram_filter(self._generated_token_lists, self.wm.ngram)
            with filter_path.open("wb") as f:
                pickle.dump((labels, values), f)
            paths["filter"] = str(filter_path.resolve())
        return paths

    @staticmethod
    def _build_ngram_filter(token_lists: List[List[int]], ngram: int):
        counts: Counter = Counter()
        for toks in token_lists:
            for i in range(len(toks) - ngram + 1):
                counts[tuple(toks[i : i + ngram])] += 1
        if not counts:
            return [], []
        labels, values = zip(*sorted(counts.items(), key=lambda kv: kv[1], reverse=True))
        return list(labels), list(values)

    @torch.inference_mode()
    def _generate_one(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> tuple[str, List[int]]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.teacher.device)
        generated = inputs["input_ids"]
        prompt_len = generated.shape[1]
        past = None

        for _ in range(max_new_tokens):
            if past is None:
                out = self.teacher(generated, use_cache=True)
            else:
                out = self.teacher(generated[:, -1:], use_cache=True, past_key_values=past)
            past = out.past_key_values
            logits = out.logits[:, -1, :]

            # ngram context for green-list
            if generated.shape[1] >= self.wm.ngram:
                ngram_tokens = generated[:, -self.wm.ngram :]
            else:
                pad = torch.zeros(
                    (1, self.wm.ngram - generated.shape[1]),
                    dtype=generated.dtype,
                    device=generated.device,
                )
                ngram_tokens = torch.cat([pad, generated], dim=1)
            logits = self.wm.bias_logits(logits, ngram_tokens)

            if temperature and temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                # top-p
                sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum - sorted_probs > top_p
                sorted_probs = sorted_probs.masked_fill(mask, 0.0)
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                next_sorted = torch.multinomial(sorted_probs, 1)
                next_id = torch.gather(sorted_idx, -1, next_sorted)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_id], dim=1)
            if int(next_id.item()) == self.tokenizer.eos_token_id:
                break

        gen_ids = generated[0, prompt_len:].tolist()
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        full_ids = generated[0].tolist()
        return text, full_ids

    def generate(
        self,
        queries: Sequence[Dict[str, str]],
        config: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        cfg = {
            "temperature": 0.8,
            "top_p": 0.95,
            "max_new_tokens": 256,
            "system_prompt": self.system_prompt,
        }
        if config:
            cfg.update(config)

        self._generated_token_lists = []
        records: List[Dict[str, Any]] = []
        for q in queries:
            prompt = format_prompt(self.tokenizer, q["query"], system_prompt=cfg.get("system_prompt"))
            resp, full_ids = self._generate_one(
                prompt,
                max_new_tokens=int(cfg["max_new_tokens"]),
                temperature=float(cfg["temperature"]),
                top_p=float(cfg["top_p"]),
            )
            self._generated_token_lists.append(full_ids)
            records.append(
                make_transcript_record(
                    query_id=q["query_id"],
                    query=q["query"],
                    response=resp,
                    defense="radioactivity",
                    defense_role=DEFENSE_ROLE["radioactivity"],
                    teacher_model=self.teacher_model_name,
                    extra={
                        "watermark_artifact_id": self.watermark_artifact_id,
                        "radioactivity_config": self.wm.to_config(),
                    },
                )
            )
        return records
