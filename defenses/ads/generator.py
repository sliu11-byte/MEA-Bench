"""ADS: Antidistillation Sampling generator.

Core sampling formula (locuslab/antidistillation-sampling)::

    new_logits = teacher_logits + (lam / (2 * eps)) * (student_+eps - student_-eps)

Requires precomputed proxy-student gradients (``grad_path``) when ``lam > 0``.
When ``lam == 0``, falls back to ordinary teacher sampling (clean baseline).

``grad_path`` is produced by ``defenses.ads.save_grad`` (a simplified,
dependency-light port of the official ``save_grad.py``):

    python -m defenses.ads.save_grad \\
        --proxy_student <same model as --proxy_student below> \\
        --holdout_transcript <clean teacher traces, disjoint from the ADS query pool> \\
        --output_path outputs/defenses/ads/<run_id>/artifacts/student_grads.pt
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.nn import functional as F
from transformers import LogitsProcessor, LogitsProcessorList

from defenses.core.constants import DEFENSE_ROLE
from defenses.core.generation import (
    format_prompt,
    generate_responses,
    load_causal_lm,
    make_transcript_record,
)

logger = logging.getLogger(__name__)


class CachedModelWrapper:
    """KV-cache wrapper for incremental proxy-student forward passes."""

    def __init__(self, model):
        self.model = model
        self.past_key_values = None
        self.last_position = 0

    def __call__(self, input_ids, attention_mask=None):
        if self.past_key_values is None or input_ids.shape[1] <= self.last_position:
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
            self.past_key_values = outputs.past_key_values
            self.last_position = input_ids.shape[1]
            return outputs.logits

        new_token = input_ids[:, -1:]
        outputs = self.model(
            new_token,
            attention_mask=attention_mask,
            use_cache=True,
            past_key_values=self.past_key_values,
            return_dict=True,
        )
        self.past_key_values = outputs.past_key_values
        self.last_position += 1
        return outputs.logits

    def reset(self):
        self.past_key_values = None
        self.last_position = 0


def _reconcile_grad_shape(grad: torch.Tensor, target_shape: torch.Size, name: str) -> Optional[torch.Tensor]:
    """Fit a loaded gradient tensor to ``target_shape``.

    Grad files produced by the vendored official ``save_grad.py`` (see
    ``defenses/ads/save_grad_official.py``) come from a proxy student whose
    tokenizer had a fresh ``[PAD]`` token added (``add_special_tokens`` +
    ``resize_token_embeddings``), so embedding-like params can be a handful of
    rows larger than a plain ``from_pretrained`` load of the same checkpoint.
    Since only the *shape*, not identity, of extra rows matters here, we
    truncate/zero-pad along dim 0 rather than reject the whole tensor.
    """
    if grad.shape == target_shape:
        return grad
    if len(grad.shape) == len(target_shape) and grad.shape[1:] == target_shape[1:]:
        n_target, n_grad = target_shape[0], grad.shape[0]
        if n_grad > n_target:
            logger.warning(
                "Truncating grad for %s from %s to %s rows (likely resized proxy-student embedding)",
                name, tuple(grad.shape), tuple(target_shape),
            )
            return grad[:n_target]
        if n_grad < n_target:
            logger.warning(
                "Zero-padding grad for %s from %s to %s rows (likely resized proxy-student embedding)",
                name, tuple(grad.shape), tuple(target_shape),
            )
            pad = torch.zeros((n_target - n_grad, *grad.shape[1:]), dtype=grad.dtype)
            return torch.cat([grad, pad], dim=0)
    logger.warning("Skipping grad for %s: incompatible shape %s vs %s", name, tuple(grad.shape), tuple(target_shape))
    return None


def _perturb_from_grads(model, grads: Dict[str, torch.Tensor], eps: float, sign: float) -> None:
    used = set()
    for name, param in model.named_parameters():
        key = name if name in grads else f"module.{name}"
        if key not in grads:
            continue
        grad = grads[key].to(param.device, dtype=torch.float32)
        if grad.shape != param.data.shape:
            grad = _reconcile_grad_shape(grad, param.data.shape, name)
            if grad is None:
                continue
        param.data = (param.data.to(torch.float32) + sign * eps * grad.to(param.device)).to(param.data.dtype)
        used.add(key)
    missing = set(grads.keys()) - used
    if missing:
        logger.warning("Unused gradient keys (first 5): %s", list(missing)[:5])


class ADSLogitsProcessor(LogitsProcessor):
    """
    Antidistillation logits processor.

    ``new_scores = scores + (lam/(2*eps)) * (student_logits - dstudent_logits)``
    """

    def __init__(self, lam: float, eps: float, attention_mask: torch.Tensor, student, dstudent):
        super().__init__()
        self.lam = lam
        self.eps = eps
        self.attention_mask = attention_mask
        self.student = student
        self.dstudent = dstudent

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        attention_mask = F.pad(
            self.attention_mask,
            pad=(0, input_ids.shape[1] - self.attention_mask.shape[1]),
            value=1,
        )
        out_target = self.student(input_ids=input_ids, attention_mask=attention_mask)[:, -1]
        out_dtarget = self.dstudent(input_ids=input_ids, attention_mask=attention_mask)[:, -1]
        ad_term = (self.lam / (2 * self.eps)) * (out_target.float() - out_dtarget.float())
        return scores.float() + ad_term


class ADSGenerator:
    def __init__(
        self,
        teacher_model: str,
        *,
        proxy_student: Optional[str] = None,
        grad_path: Optional[str] = None,
        device: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        self.teacher_model_name = teacher_model
        self.proxy_student_name = proxy_student
        self.grad_path = grad_path
        self.system_prompt = system_prompt
        self.teacher, self.tokenizer, self.device = load_causal_lm(teacher_model, device=device)
        self._student = None
        self._dstudent = None
        self._ads_ready = False

    def _prepare_ads(self, lam: float, eps: float) -> None:
        if lam == 0:
            self._ads_ready = False
            return
        if not self.proxy_student_name or not self.grad_path:
            raise ValueError(
                "ADS with lam>0 requires proxy_student and grad_path "
                "(precomputed via official save_grad.py)."
            )
        if self._ads_ready:
            return

        logger.info("Loading proxy student ±eps for ADS (lam=%s, eps=%s)", lam, eps)
        student_model, _, _ = load_causal_lm(self.proxy_student_name, device=self.device)
        dstudent_model, _, _ = load_causal_lm(self.proxy_student_name, device=self.device)
        grads = torch.load(self.grad_path, map_location="cpu")
        _perturb_from_grads(student_model, grads, eps, sign=+1.0)
        _perturb_from_grads(dstudent_model, grads, eps, sign=-1.0)
        del grads
        self._student = CachedModelWrapper(student_model)
        self._dstudent = CachedModelWrapper(dstudent_model)
        self._ads_ready = True

    def generate(
        self,
        queries: Sequence[Dict[str, str]],
        config: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        cfg = {
            "lam": 0.0,
            "eps": 1e-2,
            "tau": 0.7,
            "top_p": 0.95,
            "max_new_tokens": 256,
            "batch_size": 1,
            "allow_lam_batch": False,
            "system_prompt": self.system_prompt,
        }
        if config:
            cfg.update(config)

        lam = float(cfg["lam"])
        eps = float(cfg["eps"])
        self._prepare_ads(lam, eps)

        prompts = [
            format_prompt(self.tokenizer, q["query"], system_prompt=cfg.get("system_prompt"))
            for q in queries
        ]

        records: List[Dict[str, Any]] = []
        batch_size = int(cfg["batch_size"])
        if lam != 0 and batch_size > 1 and not bool(cfg.get("allow_lam_batch", False)):
            logger.warning(
                "ADS lam>0 generation is using microbatch=1 instead of requested batch_size=%s. "
                "Batched ADS logits processing can hang with multiple device_map models on 72B teachers.",
                batch_size,
            )
            print(
                "[ADS] lam>0: using microbatch=1 for defended generation "
                f"instead of requested batch_size={batch_size}",
                flush=True,
            )
            batch_size = 1

        total_batches = math.ceil(len(queries) / batch_size) if queries else 0
        for batch_index, start in enumerate(range(0, len(queries), batch_size), start=1):
            q_batch = list(queries[start : start + batch_size])
            p_batch = prompts[start : start + batch_size]
            logger.info(
                "ADS generate batch %s/%s: lam=%s, size=%s, produced=%s",
                batch_index,
                total_batches,
                lam,
                len(q_batch),
                len(records),
            )
            print(
                f"[ADS] generator batch {batch_index}/{total_batches} "
                f"start: lam={lam}, size={len(q_batch)}, produced={len(records)}",
                flush=True,
            )

            logits_processor = None
            if lam != 0:
                inputs = self.tokenizer(
                    p_batch, return_tensors="pt", padding=True, truncation=True
                ).to(self.teacher.device)
                self._student.reset()
                self._dstudent.reset()
                logits_processor = LogitsProcessorList(
                    [
                        ADSLogitsProcessor(
                            lam=lam,
                            eps=eps,
                            attention_mask=inputs["attention_mask"],
                            student=self._student,
                            dstudent=self._dstudent,
                        )
                    ]
                )

            responses = generate_responses(
                self.teacher,
                self.tokenizer,
                p_batch,
                max_new_tokens=int(cfg["max_new_tokens"]),
                temperature=float(cfg["tau"]),
                top_p=float(cfg["top_p"]),
                logits_processor=logits_processor,
                batch_size=len(p_batch),
            )
            logger.info(
                "ADS generated batch %s/%s: total=%s",
                batch_index,
                total_batches,
                len(records) + len(responses),
            )
            print(
                f"[ADS] generator batch {batch_index}/{total_batches} "
                f"done: total={len(records) + len(responses)}",
                flush=True,
            )

            ads_config = {
                "lam": lam,
                "eps": eps,
                "tau": float(cfg["tau"]),
                "top_p": float(cfg["top_p"]),
                "max_new_tokens": int(cfg["max_new_tokens"]),
                "proxy_student": self.proxy_student_name,
                "grad_path": self.grad_path,
            }
            for q, resp in zip(q_batch, responses):
                records.append(
                    make_transcript_record(
                        query_id=q["query_id"],
                    query=q["query"],
                    response=resp,
                    defense="ads",
                    defense_role=DEFENSE_ROLE["ads"],
                    teacher_model=self.teacher_model_name,
                    extra={"ads_config": ads_config},
                    )
                )
        return records
