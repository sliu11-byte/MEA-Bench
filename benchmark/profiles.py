from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping

import yaml


class ProfileError(ValueError):
    pass


@dataclass(frozen=True)
class BenchmarkProfile:
    name: str
    path: Path
    payload: Mapping[str, Any]

    @property
    def attack_budgets(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.payload["budgets"]["attacks"])

    @property
    def adaptive_budgets(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.payload["budgets"]["adaptive"])

    def section(self, name: str) -> Mapping[str, Any]:
        value = self.payload.get(name)
        if not isinstance(value, Mapping):
            raise ProfileError(f"profile {self.name!r} has no mapping section {name!r}")
        return value


def profiles_dir() -> Path:
    return Path(__file__).resolve().parent / "profiles"


def load_profile(name: str = "paper") -> BenchmarkProfile:
    if not re.fullmatch(r"[a-z][a-z0-9_-]*", name):
        raise ProfileError(f"invalid profile name: {name!r}")
    path = profiles_dir() / f"{name}.yaml"
    if not path.is_file():
        available = ", ".join(sorted(item.stem for item in profiles_dir().glob("*.yaml")))
        raise ProfileError(f"unknown profile {name!r}; available: {available}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ProfileError(f"profile must contain a mapping: {path}")
    required = ("schema_version", "budgets", "data", "attack", "defense", "adaptive", "evaluation")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ProfileError(f"profile {name!r} is missing sections: {', '.join(missing)}")
    profile = BenchmarkProfile(name=name, path=path, payload=payload)
    if not profile.attack_budgets:
        raise ProfileError(f"profile {name!r} defines no attack budgets")
    return profile
