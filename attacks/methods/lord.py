from __future__ import annotations

from pathlib import Path

from attacks.core.base import AttackResult, AttackRunConfig
from attacks.methods.stage1_budget import run_stage1_budget_attacker


class LoRDAttacker:
    name = "lord"

    def run(self, config: AttackRunConfig, *, run_id: str, run_dir: Path) -> AttackResult:
        return run_stage1_budget_attacker(self.name, config, run_id=run_id, run_dir=run_dir)
