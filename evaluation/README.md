# Evaluation

This package is the reorganized evaluation layer for the MEA benchmark. It separates shared rollout infrastructure, task adapters, answer parsers, and metric-specific scoring.

## Layout

- `core/`: shared rollout infrastructure, including model clients, run metadata, writers, validation, code execution, and config dataclasses.
- `tasks/`: reusable task interfaces, dataset adapters, split discovery, prompt rendering, and answer parsers.
- `metrics/`: metric-specific summary and scoring logic for M1-M7.
- `rollouts/`: executable rollout protocols. The current executable rollout is `m1_capability.py`.
- `scripts/`: parameterized local entry points.
- `tests/`: unit tests for the migrated metric implementations.

## Current Entry Point

Run the migrated M1 capability rollout with:

```bash
python -m evaluation.rollouts.m1_capability \
  --config evaluation/configs/m1_rollout.yaml \
  --mode dry-run
```

The original `legacy_m1_rollout/evaluate/m1_rollout/` package is kept in place for compatibility while this package becomes the canonical evaluation layer.

## Design Note

Dataset adapters and answer parsers are shared task-level components. M1 is only the metric protocol that compares parsed model predictions with gold answers. Future M2-M7 rollouts should reuse `core/` and `tasks/`, then add their own metric logic under `metrics/` and executable protocol under `rollouts/`.

## Scripts

Use one parameterized M1 script instead of separate dry-run/smoke/pilot/full wrappers:

```bash
bash evaluation/scripts/run_m1.sh dry-run
bash evaluation/scripts/run_m1.sh smoke
bash evaluation/scripts/run_m1.sh smoke-10
M1_PREFLIGHT_RUN_ID=<passed_smoke_run_id> bash evaluation/scripts/run_m1.sh pilot
M1_ALLOW_FULL_RUN=1 bash evaluation/scripts/run_m1.sh full
```


Build shared held-out prompts for attack/defense/countermeasure evaluation with:

```bash
export STORAGE_ROOT=${STORAGE_ROOT:-/path/to/storage/$USER/A-Benchmark-for-Model-distillation-survey}
python3 -m evaluation.scripts.build_heldout_queries
```

By default this reads `hf://watermarkproject/lord-mea-benchmark/query_pool_100000.json`, skips the maximum attack training budget of 10000 queries, and selects complete 1000-query blocks 10, 11, and 12. Set `MEA_QUERY_POOL_DATASET` to override the dataset ID. When `STORAGE_ROOT` is set, the output defaults to `${STORAGE_ROOT}/outputs/heldout_queries/heldout_prompts.jsonl` and the downloaded query-pool cache goes under `${STORAGE_ROOT}/cache/`; otherwise it falls back to repo-local `outputs/heldout_queries` and `.cache/`. The output `heldout_prompts.jsonl` is therefore disjoint from the benchmark attack budgets 100/1000/10000 while preserving the query-pool bank mix. The script also writes `heldout_manifest.json` with block ids, source hashes, and bank counts.

Generate teacher answers for those prompts directly on allocated GPUs:

```bash
python3 -m evaluation.scripts.build_heldout_teacher_outputs \
  --prompts-jsonl ${STORAGE_ROOT}/outputs/heldout_queries/heldout_prompts.jsonl \
  --output-jsonl ${STORAGE_ROOT}/outputs/heldout_queries/heldout_teacher_outputs.jsonl \
  --teacher-model meta-llama/Llama-3.3-70B-Instruct \
  --backend local_hf \
  --mode chat \
  --torch-dtype bfloat16 \
  --device-map auto \
  --max-tokens 1536
```


```bash
```

The wrapper requests two GPUs by default, writes outputs under `${STORAGE_ROOT}/outputs/heldout_queries`, and sets Hugging Face caches under `${STORAGE_ROOT}/cache/huggingface` unless you override `HF_HOME`/`HF_HUB_CACHE`/`TRANSFORMERS_CACHE`. For a smoke test, submit with `HELDOUT_LIMIT=5`. If an OpenAI-compatible endpoint is preferred, set `HELDOUT_TEACHER_BACKEND=openai` and provide `TEACHER_ENDPOINT_URL`.

The teacher output rows contain `id`, `prompt`, and `teacher_response`, which can be passed to `evaluation.attack_eval.evaluate_attack_outputs` as `--teacher-jsonl`.

## Metric Modules

The 2026-07-15 handoff metric code has been integrated under `evaluation/metrics/`.

| Module | Metric family | Notes |
|---|---|---|
| `m1_capability.py` | M1 rollout summaries plus re-exported Matrix M1 report helpers | Keeps existing rollout summaries and exposes OpenLLM6/domain/accuracy helpers from `m1_capability_reports.py`. |
| `m1_capability_reports.py` | M1 standalone reports | OpenLLM6 six-task average, generic accuracy, and Matrix domain6 accuracy. |
| `m2_fidelity.py` | M2 fidelity | Agreement, ROUGE-L, optional BERTScore/MAUVE/cross-PPL. |
| `m3_quality.py` | M3 quality | rep-n, continuation-only PPL, and pairwise judge aggregation. |
| `m4_provenance.py` | M4 provenance | Detector TPR@FPR, ROC-AUC, KGW z/p, fingerprint FSR/FPR/uniqueness. |
| `m5_robustness.py` | M5 robustness | Before/after retention, loss, evasion/removal rate using locked M4 thresholds. |
| `m6_cost.py` | M6 cost | Query/token/GPU-hour/time/dollar costs and optional queries-to-target. |
| `m7_detection.py` | M7 active detection | TPR@FPR, queries-to-detection, detection curves, naive/evasive grouping. |

Examples are stored under `evaluation/metrics/examples/M1` through `M7`. Tests are under `evaluation/tests/metrics`.

Run all migrated metric tests with:

```bash
python3 -m unittest discover -s evaluation/tests/metrics -p 'test_*.py' -q
```

Most metrics use only the Python standard library. Optional model-based metrics require extra packages: M2 BERTScore/MAUVE/cross-PPL need `bert-score`, `mauve-text`, `torch`, and `transformers`; M3 PPL needs `torch` and `transformers`. M1 rollouts require `datasets`, `huggingface_hub`, `PyYAML`, and endpoint access to the configured models.
