"""Trace Rewriting generator.

This defense first obtains a normal teacher answer, then asks a rewriter LM to
rewrite the full answer. It does not implement watermarking, detection, OPRO,
or trace/final-answer segmentation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from defenses.core.constants import DEFENSE_ROLE
from defenses.core.generation import (
    format_prompt,
    generate_openai_chat_responses,
    generate_responses,
    load_causal_lm,
    make_transcript_record,
)
from defenses.trace_rewriting.prompts import resolve_rewrite_instruction


BackendName = str


def _rewrite_prompt(
    *,
    query: str,
    original_response: str,
    rewrite_instruction: str,
) -> str:
    return (
        f"{rewrite_instruction}\n\n"
        "Original user query:\n"
        f"{query}\n\n"
        "Original answer to rewrite:\n"
        f"{original_response}\n\n"
        "Rewritten answer:"
    )


class TraceRewritingGenerator:
    def __init__(
        self,
        *,
        teacher_model: Optional[str],
        rewriter_model: str,
        device: Optional[str] = None,
        teacher_system_prompt: Optional[str] = None,
        rewriter_system_prompt: Optional[str] = None,
        load_teacher: bool = True,
        teacher_backend: BackendName = "local_hf",
        teacher_base_url: Optional[str] = None,
        teacher_api_key: str = "EMPTY",
        teacher_request_model: Optional[str] = None,
        rewriter_backend: BackendName = "local_hf",
        rewriter_base_url: Optional[str] = None,
        rewriter_api_key: str = "EMPTY",
        rewriter_request_model: Optional[str] = None,
    ):
        self.teacher_model_name = teacher_model or "unknown"
        self.rewriter_model_name = rewriter_model
        self.teacher_system_prompt = teacher_system_prompt
        self.rewriter_system_prompt = rewriter_system_prompt
        self.teacher_backend = teacher_backend
        self.teacher_base_url = teacher_base_url
        self.teacher_api_key = teacher_api_key
        self.teacher_request_model = teacher_request_model or teacher_model or "unknown"
        self.rewriter_backend = rewriter_backend
        self.rewriter_base_url = rewriter_base_url
        self.rewriter_api_key = rewriter_api_key
        self.rewriter_request_model = rewriter_request_model or rewriter_model

        self.teacher = None
        self.teacher_tokenizer = None
        self.device = device
        if load_teacher:
            if not teacher_model:
                raise ValueError("teacher_model is required when clean transcript responses are not supplied")
            if teacher_backend == "local_hf":
                self.teacher, self.teacher_tokenizer, self.device = load_causal_lm(teacher_model, device=device)
            elif teacher_backend == "openai_compatible":
                if not teacher_base_url:
                    raise ValueError("teacher_base_url is required when teacher_backend=openai_compatible")
            else:
                raise ValueError(f"unknown teacher_backend={teacher_backend!r}")

        self.rewriter = None
        self.rewriter_tokenizer = None
        self.rewriter_device = device
        if rewriter_backend == "local_hf":
            self.rewriter, self.rewriter_tokenizer, self.rewriter_device = load_causal_lm(
                rewriter_model, device=device
            )
        elif rewriter_backend == "openai_compatible":
            if not rewriter_base_url:
                raise ValueError("rewriter_base_url is required when rewriter_backend=openai_compatible")
        else:
            raise ValueError(f"unknown rewriter_backend={rewriter_backend!r}")

    def _teacher_responses(
        self,
        queries: Sequence[Dict[str, str]],
        cfg: Dict[str, Any],
    ) -> List[str]:
        if self.teacher_backend == "openai_compatible":
            return generate_openai_chat_responses(
                [q["query"] for q in queries],
                base_url=str(self.teacher_base_url),
                model=self.teacher_request_model,
                api_key=self.teacher_api_key,
                system_prompt=cfg.get("teacher_system_prompt"),
                max_new_tokens=int(cfg["teacher_max_new_tokens"]),
                temperature=float(cfg["teacher_temperature"]),
                top_p=float(cfg["teacher_top_p"]),
                timeout=int(cfg.get("teacher_timeout", 600)),
                max_retries=int(cfg.get("teacher_max_retries", 3)),
            )
        if self.teacher is None or self.teacher_tokenizer is None:
            return [str(q["original_response"]) for q in queries]
        prompts = [
            format_prompt(self.teacher_tokenizer, q["query"], system_prompt=cfg.get("teacher_system_prompt"))
            for q in queries
        ]
        return generate_responses(
            self.teacher,
            self.teacher_tokenizer,
            prompts,
            max_new_tokens=int(cfg["teacher_max_new_tokens"]),
            temperature=float(cfg["teacher_temperature"]),
            top_p=float(cfg["teacher_top_p"]),
            batch_size=int(cfg["batch_size"]),
        )

    def _rewrite_responses(self, rewrite_prompts: Sequence[str], cfg: Dict[str, Any]) -> List[str]:
        if self.rewriter_backend == "openai_compatible":
            extra_body = {}
            if cfg.get("rewriter_extra_body"):
                extra_body.update(dict(cfg["rewriter_extra_body"]))
            if cfg.get("af_reasoning_effort"):
                extra_body["reasoning_effort"] = cfg["af_reasoning_effort"]
            return generate_openai_chat_responses(
                rewrite_prompts,
                base_url=str(self.rewriter_base_url),
                model=self.rewriter_request_model,
                api_key=self.rewriter_api_key,
                system_prompt=cfg.get("rewriter_system_prompt"),
                max_new_tokens=int(cfg["max_new_tokens"]),
                temperature=float(cfg["temperature"]),
                top_p=float(cfg["top_p"]),
                timeout=int(cfg.get("rewriter_timeout", 600)),
                max_retries=int(cfg.get("rewriter_max_retries", 3)),
                extra_body=extra_body or None,
            )
        if self.rewriter is None or self.rewriter_tokenizer is None:
            raise RuntimeError("local HF rewriter was not loaded")
        formatted_rewrite_prompts = [
            format_prompt(self.rewriter_tokenizer, p, system_prompt=cfg.get("rewriter_system_prompt"))
            for p in rewrite_prompts
        ]
        return generate_responses(
            self.rewriter,
            self.rewriter_tokenizer,
            formatted_rewrite_prompts,
            max_new_tokens=int(cfg["max_new_tokens"]),
            temperature=float(cfg["temperature"]),
            top_p=float(cfg["top_p"]),
            batch_size=int(cfg["batch_size"]),
        )

    def generate(
        self,
        queries: Sequence[Dict[str, str]],
        config: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        cfg: Dict[str, Any] = {
            "rewrite_strategy": "optimized_prompt_official",
            "rewrite_scope": "full_answer",
            "teacher_temperature": 0.7,
            "teacher_top_p": 0.95,
            "teacher_max_new_tokens": 256,
            "temperature": 0.6,
            "top_p": 0.95,
            "max_new_tokens": 1024,
            "batch_size": 1,
            "teacher_system_prompt": self.teacher_system_prompt,
            "rewriter_system_prompt": self.rewriter_system_prompt,
        }
        if config:
            cfg.update(config)

        rewrite_instruction = resolve_rewrite_instruction(
            str(cfg["rewrite_strategy"]),
            cfg.get("rewrite_instruction"),
        )
        cfg["rewrite_instruction"] = rewrite_instruction
        cfg["rewrite_scope"] = "full_answer"

        originals = self._teacher_responses(queries, cfg)
        rewrite_prompts = [
            _rewrite_prompt(
                query=q["query"],
                original_response=orig,
                rewrite_instruction=rewrite_instruction,
            )
            for q, orig in zip(queries, originals)
        ]
        rewritten = self._rewrite_responses(rewrite_prompts, cfg)

        record_cfg = {
            "rewrite_strategy": cfg["rewrite_strategy"],
            "rewrite_instruction": rewrite_instruction,
            "temperature": float(cfg["temperature"]),
            "top_p": float(cfg["top_p"]),
            "max_new_tokens": int(cfg["max_new_tokens"]),
            "rewrite_scope": "full_answer",
            "teacher_temperature": float(cfg["teacher_temperature"]),
            "teacher_top_p": float(cfg["teacher_top_p"]),
            "teacher_max_new_tokens": int(cfg["teacher_max_new_tokens"]),
            "teacher_backend": self.teacher_backend,
            "rewriter_backend": self.rewriter_backend,
            "rewriter_request_model": self.rewriter_request_model,
        }
        records: List[Dict[str, Any]] = []
        for q, original_response, rewritten_response in zip(queries, originals, rewritten):
            records.append(
                make_transcript_record(
                    query_id=q["query_id"],
                    query=q["query"],
                    response=rewritten_response,
                    defense="trace_rewriting",
                    defense_role=DEFENSE_ROLE["trace_rewriting"],
                    teacher_model=self.teacher_model_name,
                    extra={
                        "original_response": original_response,
                        "rewriter_model": self.rewriter_model_name,
                        "trace_rewriting_config": record_cfg,
                    },
                )
            )
        return records
