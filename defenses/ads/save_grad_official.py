"""Run the vendored, unmodified locuslab/antidistillation-sampling `save_grad.py`
(``third_party/ads/``) against one of this repo's JSONL holdout transcripts.

This is the "byte-for-byte upstream parity" backend for proxy-student gradient
computation. It requires:

- ``third_party/ads/save_grad.py`` + ``third_party/ads/utils.py`` (vendored
  as-is from locuslab/antidistillation-sampling, BSD-3-Clause; see
  ``third_party/ads/LICENSE``)
- ``accelerate``, ``datasets``, ``pyyaml``, ``rich`` and a ``trl`` release
  that still exposes ``DataCollatorForCompletionOnlyLM``
  (``trl>=0.16,<0.20``; removed upstream in trl 0.20 — see
  https://github.com/huggingface/trl/commit/e102ac8)

We never modify the vendored files. Two narrow, in-process compatibility
patches are applied around the call (never touching the files on disk):

1. ``third_party/ads/utils.py::init`` unconditionally calls
   ``torch.cuda.get_device_capability()``, which raises when no GPU is
   present (an upstream assumption, not a bug we introduced) — patched to a
   harmless stub when ``torch.cuda.is_available()`` is False.
2. The vendored script's dataset preprocessing calls
   ``dataset.map(..., num_proc=96)``. On Linux this relies on ``fork()``, so
   worker processes inherit the already-populated module globals (e.g. the
   ``tokenizer`` variable the ``preprocessor`` closure reads). Windows only
   has ``spawn``, which re-imports the entry script fresh in each worker
   *without* re-running its ``if __name__ == "__main__":`` body — so
   ``tokenizer`` is undefined there and every worker crashes with
   ``NameError: name 'tokenizer' is not defined``. This is a genuine
   Windows-vs-Linux incompatibility in the upstream script, not something our
   wrapper introduces (`python third_party/ads/save_grad.py ...` run
   directly would hit the same crash on Windows). We patch
   ``datasets.Dataset.map`` to force ``num_proc=None`` (single-process) for
   the duration of the call, which sidesteps it entirely and, as a bonus, is
   far faster than spawning 6+ workers that each re-import torch/transformers.

For everyday use, ``defenses/ads/run.py --grad_backend lite`` (the default)
uses ``defenses/ads/save_grad.py`` instead — a dependency-light reimplementation
of the same completion-only gradient-accumulation algorithm that works on any
machine/trl version without this scaffolding. Use ``--grad_backend official``
when you specifically want upstream parity and have a compatible ``trl``.
"""

from __future__ import annotations

import runpy
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from defenses.core.io_utils import ensure_dir

PathLike = Union[str, Path]
THIRD_PARTY_ADS = Path(__file__).resolve().parents[2] / "third_party" / "ads"

# Matches the hardcoded `response_str` in third_party/ads/save_grad.py — the
# DeepSeek-style assistant marker its DataCollatorForCompletionOnlyLM splits on.
RESPONSE_MARKER = "<\uff5cAssistant\uff5c>"


def check_official_deps() -> Optional[str]:
    """Return a human-readable error string if the official backend can't run, else None."""
    missing: List[str] = []
    for mod in ("datasets", "accelerate", "yaml", "rich"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    try:
        from trl import DataCollatorForCompletionOnlyLM  # noqa: F401
    except ImportError:
        missing.append("trl<0.20,>=0.16 (exposes DataCollatorForCompletionOnlyLM)")
    if missing:
        return (
            "Missing/incompatible dependencies for the official ADS save_grad.py: "
            + ", ".join(missing)
            + ". Install them or use --grad_backend lite instead."
        )
    if not (THIRD_PARTY_ADS / "save_grad.py").exists() or not (THIRD_PARTY_ADS / "utils.py").exists():
        return (
            f"Vendored save_grad.py/utils.py not found under {THIRD_PARTY_ADS}. "
            "Pull them from locuslab/antidistillation-sampling into third_party/ads/."
        )
    return None


def _build_holdout_dataset(
    records: Sequence[Dict[str, Any]],
    tokenizer_name: str,
    *,
    save_dir: PathLike,
    system_prompt: Optional[str] = None,
) -> Path:
    """Build the HF ``datasets.Dataset`` the vendored script's preprocessor expects:
    one ``text`` column per holdout trace, formatted as
    ``<prompt><Assistant marker><response>``.
    """
    import datasets as hf_datasets
    from transformers import AutoTokenizer

    from defenses.core.generation import format_prompt

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    texts = []
    for rec in records:
        query = rec.get("query") or rec.get("prompt") or ""
        response = rec.get("response") or rec.get("output") or ""
        if not str(response).strip():
            continue
        prompt = format_prompt(tokenizer, query, system_prompt=system_prompt)
        texts.append(f"{prompt}{RESPONSE_MARKER}{response}")

    if not texts:
        raise ValueError("No usable holdout traces to build the official ADS dataset from")

    ds = hf_datasets.Dataset.from_dict({"text": texts})
    save_dir = ensure_dir(save_dir)
    ds.save_to_disk(str(save_dir))
    return save_dir


def _run_vendored_script(argv: List[str]) -> None:
    """Execute ``third_party/ads/save_grad.py`` in-process via ``runpy``,
    with the two narrow compatibility patches described in the module
    docstring. Both are restored in a ``finally`` block so they never leak
    into the rest of this process.
    """
    import datasets
    import torch

    old_argv = sys.argv
    old_path = list(sys.path)
    old_get_cap = torch.cuda.get_device_capability
    old_map = datasets.Dataset.map

    patched_cuda = not torch.cuda.is_available()
    if patched_cuda:
        torch.cuda.get_device_capability = lambda *a, **k: (0, 0)  # type: ignore[method-assign]

    def _map_single_process(self, *args, **kwargs):
        kwargs["num_proc"] = None
        return old_map(self, *args, **kwargs)

    datasets.Dataset.map = _map_single_process

    # The vendored script prints decoded token text straight to stdout, which
    # can contain characters the host console's legacy codepage (e.g. Windows
    # `gbk`) can't encode. Widen stdout/stderr to utf-8 for the duration of
    # the call rather than letting an incidental print() crash the run.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    sys.argv = argv
    sys.path.insert(0, str(THIRD_PARTY_ADS))
    try:
        runpy.run_path(str(THIRD_PARTY_ADS / "save_grad.py"), run_name="__main__")
    finally:
        sys.argv = old_argv
        sys.path = old_path
        datasets.Dataset.map = old_map
        if patched_cuda:
            torch.cuda.get_device_capability = old_get_cap


def run_official_save_grad(
    proxy_student: str,
    holdout_records: Sequence[Dict[str, Any]],
    *,
    exp_dir: PathLike,
    seed: int = 0,
    batch_size: int = 1,
    system_prompt: Optional[str] = None,
) -> Path:
    """Run the vendored ``save_grad.py`` against ``holdout_records``.

    Returns the path to the produced ``student_grads.pt`` (same
    ``Dict[str, Tensor]`` shape ``ADSGenerator`` expects via ``--grad_path``).
    """
    import yaml

    missing = check_official_deps()
    if missing:
        raise RuntimeError(missing)

    exp_dir = ensure_dir(exp_dir)
    with tempfile.TemporaryDirectory() as tmp:
        trace_dir = Path(tmp) / "holdout_traces"
        _build_holdout_dataset(
            holdout_records, proxy_student, save_dir=trace_dir, system_prompt=system_prompt
        )

        holdout_config = {
            "trace_path": str(trace_dir.resolve()),
            "trace_colname": "text",
            "exp_dir": str(exp_dir.resolve()),
        }
        config_path = Path(tmp) / "holdout_config.yaml"
        config_path.write_text(yaml.safe_dump(holdout_config), encoding="utf-8")

        argv = [
            str(THIRD_PARTY_ADS / "save_grad.py"),
            str(config_path),
            "--proxy_student", proxy_student,
            "--tokenizer", proxy_student,
            "--seed", str(seed),
            "--trace_colname", "text",
            "--batch_size", str(batch_size),
        ]
        _run_vendored_script(argv)

    grad_path = exp_dir / "student_grads.pt"
    if not grad_path.exists():
        raise RuntimeError(f"Official save_grad.py did not produce {grad_path}")
    return grad_path
