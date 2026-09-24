"""Student model loading — prefers attacks.core.student_model when available."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


def _try_import_attacks():
    try:
        from attacks.core import student_model as sm  # type: ignore

        return sm
    except Exception:
        try:
            from attacks.core.student_model import (  # type: ignore
                load_student_from_checkpoint as _ck,
                load_student_from_manifest as _mf,
            )

            class _Shim:
                load_student_from_checkpoint = staticmethod(_ck)
                load_student_from_manifest = staticmethod(_mf)

            return _Shim()
        except Exception:
            return None


def _fallback_load_checkpoint(
    checkpoint: PathLike,
    *,
    device: Optional[str] = None,
    torch_dtype: Optional[str] = "auto",
    trust_remote_code: bool = True,
) -> Tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ckpt = str(checkpoint)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = torch_dtype
    if dtype == "auto":
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    elif isinstance(dtype, str):
        dtype = getattr(torch, dtype)

    tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        ckpt,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        device_map="auto" if device.startswith("cuda") else None,
    )
    if not device.startswith("cuda"):
        model = model.to(device)
    model.eval()
    return model, tokenizer


def _as_model_tokenizer(result: Any) -> Tuple[Any, Any]:
    """Normalize student loader return values to the legacy defense tuple API."""
    if isinstance(result, tuple):
        if len(result) < 2:
            raise TypeError(f"Student loader returned a tuple with {len(result)} item(s); expected at least 2.")
        return result[0], result[1]
    model = getattr(result, "model", None)
    tokenizer = getattr(result, "tokenizer", None)
    if model is not None and tokenizer is not None:
        return model, tokenizer
    raise TypeError(f"Unsupported student loader return type: {type(result).__name__}")


def load_student_from_checkpoint(
    checkpoint: PathLike,
    **kwargs: Any,
) -> Tuple[Any, Any]:
    """
    Load (model, tokenizer) from a student checkpoint.

    Prefer ``attacks.core.student_model`` if present; otherwise fall back to
    HuggingFace ``AutoModelForCausalLM`` / ``AutoTokenizer``.
    """
    sm = _try_import_attacks()
    if sm is not None and hasattr(sm, "load_student_from_checkpoint"):
        logger.info("Loading student via attacks.core.student_model")
        return _as_model_tokenizer(sm.load_student_from_checkpoint(checkpoint, **kwargs))

    logger.info("attacks.core.student_model unavailable; using transformers fallback")
    return _fallback_load_checkpoint(checkpoint, **kwargs)


def load_student_from_manifest(
    manifest_path: PathLike,
    *,
    checkpoint_key: str = "student_checkpoint",
    **kwargs: Any,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Load student from an attack/defense manifest JSON.

    Returns ``(model, tokenizer, manifest_dict)``.
    """
    path = Path(manifest_path)
    data = json.loads(path.read_text(encoding="utf-8"))

    sm = _try_import_attacks()
    if sm is not None and hasattr(sm, "load_student_from_manifest"):
        logger.info("Loading student via attacks.core.student_model.load_student_from_manifest")
        result = sm.load_student_from_manifest(manifest_path, **kwargs)
        if isinstance(result, tuple) and len(result) == 3:
            return result
        model, tokenizer = _as_model_tokenizer(result)
        return model, tokenizer, data

    ckpt = data.get(checkpoint_key) or data.get("checkpoint") or data.get("output_dir")
    if not ckpt:
        # common nested layouts
        for key in ("artifacts", "paths", "outputs"):
            nested = data.get(key) or {}
            if isinstance(nested, dict):
                ckpt = nested.get(checkpoint_key) or nested.get("student_checkpoint") or nested.get("checkpoint")
                if ckpt:
                    break
    if not ckpt:
        raise KeyError(
            f"Could not find student checkpoint in manifest {path}; "
            f"expected key '{checkpoint_key}' (or checkpoint/output_dir)"
        )

    model, tokenizer = load_student_from_checkpoint(ckpt, **kwargs)
    return model, tokenizer, data
