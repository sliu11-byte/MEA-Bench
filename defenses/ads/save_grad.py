"""Compute proxy-student gradients for ADS (lam > 0).

Adapted from locuslab/antidistillation-sampling's ``save_grad.py``, simplified
to a single-process, dependency-light version (no ``accelerate``/``trl``) that
fits this repo's conventions: it consumes a holdout teacher transcript JSONL
(``{query, response, ...}``) instead of a HF ``datasets`` disk cache + yaml
config, and produces the same artifact shape the official script does — a
``Dict[str, Tensor]`` keyed by ``model.named_parameters()`` name, averaged over
the holdout set.

That gradient file is exactly what ``ADSGenerator``/``defenses/ads/run.py``
expect via ``--grad_path`` when ``lam > 0``:

    grads = torch.load(grad_path, map_location="cpu")
    student_model += eps * grads   # "+eps" proxy student
    dstudent_model -= eps * grads  # "-eps" proxy student

Usage::

    python -m defenses.ads.save_grad \\
        --proxy_student sshleifer/tiny-gpt2 \\
        --holdout_transcript outputs/.../clean_teacher.jsonl \\
        --output_path outputs/defenses/ads/<run_id>/artifacts/student_grads.pt

The holdout transcript should be *disjoint* from the queries ADS will
actually defend (per the official method: gradients come from a held-out set
of teacher traces, not the traces being defended).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from defenses.core.generation import format_prompt
from defenses.core.io_utils import ensure_dir, read_jsonl

PathLike = Union[str, Path]


def _build_input_and_labels(
    tokenizer,
    prompt: str,
    response: str,
    *,
    max_length: int,
    completion_only: bool,
) -> Optional[Sequence[List[int]]]:
    """Tokenize ``prompt + response`` and mask the prompt span in ``labels``.

    Returns ``None`` if the response is truncated away entirely (nothing left
    to compute completion-only loss on).
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_text = prompt + response
    enc = tokenizer(full_text, truncation=True, max_length=max_length, add_special_tokens=False)
    input_ids = enc["input_ids"]
    if not input_ids:
        return None

    labels = list(input_ids)
    if completion_only:
        prompt_len = min(len(prompt_ids), len(input_ids))
        for i in range(prompt_len):
            labels[i] = -100
        if all(l == -100 for l in labels):
            return None
    return input_ids, labels


def compute_proxy_student_grads(
    proxy_student: str,
    holdout_records: Sequence[Dict[str, Any]],
    *,
    output_path: PathLike,
    max_samples: Optional[int] = None,
    max_length: int = 512,
    completion_only: bool = True,
    system_prompt: Optional[str] = None,
    seed: int = 0,
    device: Optional[str] = None,
    model_dtype: str = "auto",
    grad_dtype: str = "float32",
) -> Dict[str, Any]:
    """Compute and save averaged proxy-student gradients over a holdout transcript.

    Mirrors the official ``save_grad.py``: load the proxy student, run a
    completion-only forward/backward pass per holdout ``(query, response)``
    trace, accumulate ``param.grad`` across the set, and save the average.

    Returns a small manifest dict (also written next to ``output_path``).
    """
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if model_dtype == "auto":
        resolved_model_dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    else:
        resolved_model_dtype = getattr(torch, model_dtype)
    resolved_grad_dtype = getattr(torch, grad_dtype)

    tokenizer = AutoTokenizer.from_pretrained(proxy_student, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        proxy_student, trust_remote_code=True, dtype=resolved_model_dtype
    ).to(device)
    model.train()

    records = list(holdout_records)
    if max_samples is not None:
        records = records[: max(0, int(max_samples))]
    if not records:
        raise ValueError("Empty holdout transcript; cannot compute proxy-student gradients")

    grads: Dict[str, torch.Tensor] = {
        name: torch.zeros(param.shape, dtype=resolved_grad_dtype, device="cpu")
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    n_used = 0
    t0 = time.time()
    for rec in records:
        query = rec.get("query") or rec.get("prompt") or ""
        response = rec.get("response") or rec.get("output") or ""
        if not str(response).strip():
            continue

        prompt = format_prompt(tokenizer, query, system_prompt=system_prompt)
        built = _build_input_and_labels(
            tokenizer, prompt, str(response), max_length=max_length, completion_only=completion_only
        )
        if built is None:
            continue
        input_ids, labels = built

        input_tensor = torch.tensor([input_ids], device=device)
        label_tensor = torch.tensor([labels], device=device)

        model.zero_grad(set_to_none=True)
        out = model(input_ids=input_tensor, labels=label_tensor)
        out.loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grads[name].add_(param.grad.detach().to(device="cpu", dtype=resolved_grad_dtype))
        n_used += 1

    if n_used == 0:
        raise ValueError(
            "No usable holdout samples (all empty or truncated away); "
            "cannot compute proxy-student gradients"
        )

    for name in grads:
        grads[name] /= n_used

    out_path = Path(output_path)
    ensure_dir(out_path.parent)
    torch.save(grads, out_path)

    grad_norm = float(sum(torch.norm(g).item() ** 2 for g in grads.values()) ** 0.5)
    manifest = {
        "proxy_student": proxy_student,
        "output_path": str(out_path.resolve()),
        "num_samples_used": n_used,
        "num_samples_total": len(records),
        "completion_only": completion_only,
        "max_length": max_length,
        "seed": seed,
        "model_dtype": str(resolved_model_dtype).replace("torch.", ""),
        "grad_dtype": str(resolved_grad_dtype).replace("torch.", ""),
        "grad_device": "cpu",
        "grad_norm": grad_norm,
        "num_params": len(grads),
        "wall_seconds": round(time.time() - t0, 3),
    }
    manifest_path = out_path.with_suffix(out_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path.resolve())
    return manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute averaged proxy-student gradients for ADS (lam>0), "
        "consumed by defenses/ads/run.py via --grad_path."
    )
    p.add_argument("--proxy_student", required=True, help="HF model name/path; must match --proxy_student passed to ads/run.py")
    p.add_argument(
        "--holdout_transcript",
        required=True,
        help="JSONL of {query, response} teacher traces, disjoint from the ADS query pool",
    )
    p.add_argument("--output_path", required=True, help="Where to write the gradient .pt file")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--completion_only", action="store_true", default=True)
    p.add_argument("--no_completion_only", dest="completion_only", action="store_false")
    p.add_argument("--system_prompt", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--model_dtype",
        default="auto",
        choices=["auto", "float32", "bfloat16", "float16"],
        help="Proxy model dtype. Default auto uses bfloat16 on CUDA to avoid 7B fp32 OOM.",
    )
    p.add_argument(
        "--grad_dtype",
        default="float32",
        choices=["float32", "bfloat16", "float16"],
        help="CPU accumulation/storage dtype for averaged gradients.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.holdout_transcript)
    manifest = compute_proxy_student_grads(
        args.proxy_student,
        records,
        output_path=args.output_path,
        max_samples=args.max_samples,
        max_length=args.max_length,
        completion_only=args.completion_only,
        system_prompt=args.system_prompt,
        seed=args.seed,
        device=args.device,
        model_dtype=args.model_dtype,
        grad_dtype=args.grad_dtype,
    )
    print(
        f"[ADS save_grad] processed {manifest['num_samples_used']}/{manifest['num_samples_total']} "
        f"samples in {manifest['wall_seconds']}s"
    )
    print(f"[ADS save_grad] grad_norm={manifest['grad_norm']:.4e} -> {manifest['output_path']}")
    print(f"[ADS save_grad] manifest -> {manifest['manifest_path']}")


if __name__ == "__main__":
    main()
