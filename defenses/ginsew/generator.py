"""GINSEW watermarked teacher generator."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from defenses.core.constants import DEFENSE_ROLE
from defenses.core.generation import (
    format_prompt,
    generate_responses,
    load_causal_lm,
    make_transcript_record,
)
from defenses.core.io_utils import ensure_dir
from defenses.ginsew.watermark import WatermarkLogitsWarper, build_warper

PathLike = Union[str, Path]


class GinsewGenerator:
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
        vocab_size = len(self.tokenizer)
        wm = dict(watermark_config or {})
        self.warper: WatermarkLogitsWarper = build_warper(
            vocab_size=vocab_size,
            watermark_key=int(wm.get("watermark_key", wm.get("key", 0))),
            fraction=float(wm.get("fraction", wm.get("green_list_fraction", 0.5))),
            strength=float(wm.get("strength", 2.0)),
            freq=int(wm.get("freq", 16)),
            eps=float(wm.get("eps", 0.2)),
        )
        self.watermark_artifact_id = wm.get("watermark_artifact_id") or uuid.uuid4().hex[:12]

    def save_artifacts(self, artifacts_dir: PathLike) -> Dict[str, str]:
        artifacts_dir = ensure_dir(artifacts_dir)
        cfg_path = Path(artifacts_dir) / "watermark_config.json"
        state_path = Path(artifacts_dir) / "watermark_state.npz"

        cfg = {
            **self.warper.to_config(),
            "watermark_artifact_id": self.watermark_artifact_id,
            "teacher_model": self.teacher_model_name,
        }
        cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

        state = self.warper.state_dict()
        np.savez_compressed(
            state_path,
            green_list_mask=state["green_list_mask"],
            vec=state["vec"],
            key=state["key"],
            fraction=np.array(state["fraction"]),
            strength=np.array(state["strength"]),
            vocab_size=np.array(state["vocab_size"]),
            watermark_key=np.array(state["watermark_key"]),
            freq=np.array(state["freq"]),
            eps=np.array(state["eps"]),
        )
        return {
            "watermark_config": str(cfg_path.resolve()),
            "watermark_state": str(state_path.resolve()),
        }

    def generate(
        self,
        queries: Sequence[Dict[str, str]],
        config: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        cfg = {
            "temperature": 0.7,
            "top_p": 0.95,
            "max_new_tokens": 256,
            "batch_size": 1,
            "system_prompt": self.system_prompt,
        }
        if config:
            cfg.update(config)

        prompts = [
            format_prompt(self.tokenizer, q["query"], system_prompt=cfg.get("system_prompt"))
            for q in queries
        ]
        responses = generate_responses(
            self.teacher,
            self.tokenizer,
            prompts,
            max_new_tokens=int(cfg["max_new_tokens"]),
            temperature=float(cfg["temperature"]),
            top_p=float(cfg["top_p"]),
            logits_warper=self.warper,
            batch_size=int(cfg["batch_size"]),
        )

        records: List[Dict[str, Any]] = []
        for q, resp in zip(queries, responses):
            records.append(
                make_transcript_record(
                    query_id=q["query_id"],
                    query=q["query"],
                    response=resp,
                    defense="ginsew",
                    defense_role=DEFENSE_ROLE["ginsew"],
                    teacher_model=self.teacher_model_name,
                    extra={
                        "watermark_artifact_id": self.watermark_artifact_id,
                        "ginsew_config": self.warper.to_config(),
                    },
                )
            )
        return records
