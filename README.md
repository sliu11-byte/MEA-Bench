# A Benchmark for Model Extraction Attacks

This repository is being organized into a benchmark harness for model extraction attack reproduction and evaluation. The current cleaned-up code lives in three main areas:

```text
attacks/      unified Stage-1 attacker pipeline
evaluation/   shared rollout, dataset adapters, answer parsers, and M1-M7 metrics
infra/        vLLM serving scripts and runtime helpers
```

Older method-specific folders are kept as source/reference material while their useful pieces are migrated into the unified layout.

## Default Data Source

Stage-1 attacks now use the Hugging Face dataset by default:

```text
auto  # resolves to the published HF query-pool tier for the requested budget
```

The CLI resolves this `hf://...` path before running an attacker. On first use it downloads the file from `anonymous-mea-benchmark/mea-query-pools`, caches it under:

```text
.cache/mea_benchmark/hf_datasets/
```

and generates a deterministic query ordering automatically when `--query-ordering auto` is used. The attack pipeline therefore no longer depends on the old local `stage1-LoRD-SeqKD/data/query_pool.json` and `query_ordering.json` defaults.

## Attack Pipeline

Run one attacker:

```bash
python3 attacks/scripts/run_attack.py \
  --attack seqkd \
  --budget 1000 \
  --teacher-backend vllm_openai \
  --teacher-model Qwen/Qwen2.5-72B-Instruct \
  --teacher-endpoint-url http://127.0.0.1:8000/v1 \
  --teacher-request-model Qwen/Qwen2.5-72B-Instruct \
  --student-model Qwen/Qwen2.5-7B
```

Run all registered attackers:

```bash
bash attacks/scripts/run_all_attacks.sh
```

Run manifest-only checks:

```bash
DRY_RUN=1 bash attacks/scripts/run_all_attacks.sh
```

Currently registered attackers:

```text
seqkd
lord
soda
qedks
model_leeching
gad               placeholder only
```

Detailed attack implementation and running notes are in:

```text
attacks/README.md
```

## Evaluation Pipeline

`evaluation/` contains the shared rollout and metric-side code. The intent is that attack methods produce student checkpoints, and evaluation measures those checkpoints independently of how they were produced.

The shared evaluation pieces include dataset adapters, answer parsers, rollout helpers, and migrated M1-M7 metric code. M1 currently focuses on comparing model answers against reference answers; the surrounding structure is designed so other MEA metrics can reuse the same rollout layer and differ mainly in parsing/statistics.

## vLLM Serving

`infra/` contains scripts/configs for starting OpenAI-compatible vLLM endpoints for teacher and student models. Attack scripts assume endpoint-style access when `--teacher-backend vllm_openai` is used.

Typical default endpoints:

```text
teacher: http://127.0.0.1:8000/v1
student: http://127.0.0.1:8001/v1
```

## Outputs and Cache

Runtime artifacts are intentionally not tracked by git:

```text
outputs/      attack/evaluation run outputs
.cache/       downloaded HF query pools and generated orderings
```

Each attack run writes an `attack_manifest.json` under:

```text
outputs/attacks/{attack}/{run_id}/
```

The manifest records normalized config, query pool path/hash, ordering path/hash, method artifacts, and checkpoint path when available.
