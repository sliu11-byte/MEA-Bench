# Evaluation runs

All generated run artifacts and defense checkpoints are read from local paths.
Hugging Face remains available only for benchmark datasets and model downloads.

## Attack evaluation

```bash
ATTACK_ROOT=/path/to/attack_runs/b100 \
bash runs/evaluation/evaluate_attack_b100.sh

ATTACK_B1000_ROOT=/path/to/attack_runs/b1000 \
bash runs/evaluation/evaluate_attack_b1000.sh
```

The B=10000 entry point is `runs/evaluation/evaluate_attack_b10000.sh` and uses
`ATTACK_B10000_ROOT`. Optional smoke runs set `EVAL_LIMIT` and `EVAL_ATTACK`.

## Defense evaluation

Place local checkpoints under
`${DEFENSE_CHECKPOINT_ROOT:-$STORAGE_ROOT/inputs/defense_checkpoints}` using the
layout expected by `evaluation/defense_eval/evaluate.py`, then run:

```bash
bash runs/evaluation/build_defense_reference.sh
ATTACK=seqkd bash runs/evaluation/evaluate_defense_students.sh
bash runs/evaluation/evaluate_duffin.sh
```

Countermeasure evaluation is available through
`evaluate_counter_seqkd_b1000.sh` and `evaluate_counter_seqkd_b1000_v2.sh`.
