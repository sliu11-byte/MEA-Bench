from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def checkpoint_tensor_health(checkpoint: str | Path) -> dict[str, Any]:
    """Return tensor-level finite-value health for a saved HF/PEFT checkpoint."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    safetensor_files = sorted(checkpoint.glob("*.safetensors"))
    bin_files = sorted(checkpoint.glob("*.bin"))
    if not safetensor_files and not bin_files:
        raise RuntimeError(f"No checkpoint tensor files found in {checkpoint}")

    checked = 0
    nonfinite: list[str] = []
    if safetensor_files:
        from safetensors import safe_open

        for tensor_file in safetensor_files:
            with safe_open(tensor_file, framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    tensor = handle.get_tensor(name)
                    checked += 1
                    if not torch.isfinite(tensor).all().item():
                        nonfinite.append(f"{tensor_file.name}:{name}")
    else:
        for tensor_file in bin_files:
            payload = torch.load(tensor_file, map_location="cpu", weights_only=True)
            for name, tensor in payload.items():
                if not torch.is_tensor(tensor):
                    continue
                checked += 1
                if not torch.isfinite(tensor).all().item():
                    nonfinite.append(f"{tensor_file.name}:{name}")
    return {"tensor_count": checked, "nonfinite_tensors": nonfinite, "all_finite": not nonfinite}


__all__ = ["checkpoint_tensor_health"]
