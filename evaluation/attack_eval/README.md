# Attack evaluation

Run from the repository root and provide a local attack-results directory:

```bash
ATTACK_ROOT=/path/to/attack_runs/b100 bash runs/evaluation/evaluate_attack_b100.sh
```

The evaluator computes six-task macro accuracy, BERTScore, Rep-4, and optional
allocated GPU-hours. `EVAL_LIMIT` enables a small smoke run.
