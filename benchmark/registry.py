from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


class RegistryError(ValueError):
    pass


@dataclass(frozen=True)
class AttackSpec:
    name: str
    uses_shared_transcript: bool
    required_services: tuple[str, ...]
    prerequisites: tuple[str, ...] = ()


@dataclass(frozen=True)
class DefenseSpec:
    name: str
    category: str
    required_attack_artifacts: tuple[str, ...]


@dataclass(frozen=True)
class AdaptiveAttackSpec:
    name: str
    supported_defenses: frozenset[str]


ATTACKS: Mapping[str, AttackSpec] = {
    "seqkd": AttackSpec("seqkd", True, ("teacher",)),
    "lord": AttackSpec("lord", True, ("teacher",)),
    "soda": AttackSpec("soda", True, ("teacher", "student"), ("seqkd_warmup",)),
    "qedks": AttackSpec("qedks", False, ("teacher",)),
    "model_leeching": AttackSpec("model_leeching", False, ("teacher",)),
    "gad": AttackSpec("gad", True, ("teacher",)),
}

DEFENSES: Mapping[str, DefenseSpec] = {
    name: DefenseSpec(name, "defended_extraction", ())
    for name in ("ads", "doge", "trace_rewriting", "adfp", "ginsew", "radioactivity")
}
DEFENSES = {
    **DEFENSES,
    "duffin": DefenseSpec("duffin", "result_based", ("checkpoint", "model_outputs")),
    "mmd": DefenseSpec("mmd", "result_based", ("query_log",)),
    "prada": DefenseSpec("prada", "result_based", ("query_log",)),
    "seat": DefenseSpec("seat", "result_based", ("query_log",)),
}

_ADAPTIVE_DEFENSES = frozenset(("adfp", "ginsew", "radioactivity"))
ADAPTIVE_ATTACKS: Mapping[str, AdaptiveAttackSpec] = {
    "dipper": AdaptiveAttackSpec("dipper", _ADAPTIVE_DEFENSES),
    "translation": AdaptiveAttackSpec("translation", _ADAPTIVE_DEFENSES),
}


def attack_spec(name: str) -> AttackSpec:
    try:
        return ATTACKS[name]
    except KeyError as exc:
        raise RegistryError(f"unknown attack {name!r}; available: {', '.join(sorted(ATTACKS))}") from exc


def defense_spec(name: str) -> DefenseSpec:
    try:
        return DEFENSES[name]
    except KeyError as exc:
        raise RegistryError(f"unknown defense {name!r}; available: {', '.join(sorted(DEFENSES))}") from exc


def validate_attack_budget(attack: str, budget: int, supported_budgets: tuple[int, ...]) -> AttackSpec:
    spec = attack_spec(attack)
    if budget not in supported_budgets:
        choices = ", ".join(str(value) for value in supported_budgets)
        raise RegistryError(f"budget {budget} is not supported by the paper profile; choose one of: {choices}")
    return spec


def validate_adaptive_combination(
    adaptive_attack: str,
    defense: str,
    attack: str,
    budget: int,
    supported_budgets: tuple[int, ...] = (1000,),
) -> None:
    attack_spec(attack)
    defense_spec(defense)
    try:
        adaptive = ADAPTIVE_ATTACKS[adaptive_attack]
    except KeyError as exc:
        available = ", ".join(sorted(ADAPTIVE_ATTACKS))
        raise RegistryError(f"unknown adaptive attack {adaptive_attack!r}; available: {available}") from exc
    if defense not in adaptive.supported_defenses:
        choices = ", ".join(sorted(adaptive.supported_defenses))
        raise RegistryError(f"{adaptive_attack} does not support defense {defense!r}; choose one of: {choices}")
    if budget not in supported_budgets:
        choices = ", ".join(str(value) for value in supported_budgets)
        raise RegistryError(f"adaptive budget {budget} is not in the paper profile; choose one of: {choices}")
