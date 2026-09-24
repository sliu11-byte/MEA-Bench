# Defense Pipeline

This directory contains benchmark-side defense implementations. The formal defense benchmark treats a generator-style defense as a black-box teacher replacement:

```text
attack + clean teacher      -> extracted student (clean baseline)
attack + defended teacher   -> extracted student (defense setting)
```

The attack code should not know which defense is active. It only sees an OpenAI-compatible teacher endpoint.

## Public Entry

Use one runner per defense. Generator-style runners start a defended teacher, run the selected attack against that endpoint, stop the teacher, and then run the detector when the method has one. Detector-only methods expose the same runner convention but skip the generator step.

```bash
python -m defenses.ginsew.runner \
  --attack seqkd \
  --budget 1000 \
  --teacher-model Qwen/Qwen2.5-7B-Instruct \
  --student-model Qwen/Qwen2.5-0.5B-Instruct \
  --defense-config '{"gamma":0.25,"delta":2.0,"max_new_tokens":512}' \
  --output-dir outputs/defenses/ginsew/run_001
```

Attack-specific arguments can be appended after `--`; they are forwarded to `attacks/scripts/run_attack.py`.

Response-only countermeasures can be enabled directly in the same online runner
with `--countermeasure dipper` or `--countermeasure translation`. The runner
inserts an OpenAI-compatible proxy between the defended teacher and the attack,
so online attacks still query a live teacher endpoint while each returned
response is rewritten before attack training sees it.

```bash
python -m defenses.trace_rewriting.runner \
  --attack model_leeching \
  --budget 1000 \
  --teacher-model Qwen/Qwen2.5-7B-Instruct \
  --teacher-base-url http://127.0.0.1:8000/v1 \
  --rewriter-model Qwen/Qwen2.5-14B-Instruct \
  --rewriter-backend openai_compatible \
  --rewriter-base-url http://127.0.0.1:8001/v1 \
  --student-model Qwen/Qwen2.5-0.5B-Instruct \
  -- --model-leeching-epochs 1
```

## Output Layout

Generator-style runners write:

```text
<run_output>/
  defense_run_manifest.json
  oracle/
    oracle_manifest.json
    teacher_received_queries.jsonl
    defended_teacher_transcript.jsonl
    oracle_server.log
    artifacts/
  attack/
    <attack>/<attack_run_id>/attack_manifest.json
  detector/
```

`oracle/teacher_received_queries.jsonl` is the query stream actually sent by the attacker to the defended teacher. Query-traffic detectors should use this file, not a planned query pool.

## Formal SeqKD Runs

The first formal defense stage only needs this setting:

```text
attack: seqkd
budget: 1000
teacher: Qwen/Qwen2.5-72B-Instruct
student: Qwen/Qwen2.5-7B
```

Clean baselines are run by `attacks/` directly. The scripts here are only for actual defense runs. By default, `DEFENSE_EXECUTION_MODE=auto` uses batch offline defended transcript generation when the selected attack can consume a precomputed transcript (`seqkd`, `lord`, `gad`, `soda`), then passes that transcript to the attack with `--teacher-transcript-path`. Set `DEFENSE_EXECUTION_MODE=online` to force the defended-teacher server path for adaptive attacks.

Formal SeqKD scripts live under `defenses/scripts/seqkd/`:

```text
common.sh
run_ginsew_seqkd.sh
run_radioactivity_seqkd.sh
run_adfp_seqkd.sh
run_ads_seqkd.sh
run_trace_rewriting_seqkd.sh
run_doge_seqkd.sh
run_duffin_seqkd.sh
run_query_traffic_seqkd.sh
run_all_seqkd.sh
submit_seqkd_array.sh
```

Default formal output paths on HPG:

```text
large outputs/checkpoints/cache:
  /path/to/storage/$USER/A-Benchmark-for-Model-distillation-survey/outputs/defenses/seqkd_b1000

light log/manifest mirror:
  /home/$USER/project/A-Benchmark-for-Model-distillation-survey/logs/defenses/seqkd_b1000
```

The log mirror copies `*.log`, `*.out`, `*.err`, `*manifest.json`, `*report*.json`, `*metrics*.json`, and `*summary*.json`. Model checkpoints and large transcripts stay under `/blue`.

Run examples below assume the current directory is the repository root. If you are already inside `defenses/`, drop the leading `defenses/` from script paths, for example `bash scripts/seqkd/submit_seqkd_array.sh`.


### Formal Parameter Policy

Attack-facing parameters are kept aligned with the formal attack configuration and should not be tuned inside a defense run unless the experiment intentionally changes the attack setting:

```text
TEACHER_TEMPERATURE=0.0
TEACHER_TOP_P=1.0
TEACHER_MAX_TOKENS=1536
STAGE1_CONFIG=attacks/configs/formal_stage1_budget.yaml
```

Defense-specific defaults are set from the reproduction notes or official code defaults when available:

```text
GINSEW: fraction=0.5, strength=2.0, freq=16, eps=0.2
Radioactivity: method=maryland, ngram=4, seed=0, hash_key=35317, gamma=0.25, delta=2.0, scoring_method=v2
ADFP: gamma=0.5, window_size=2, strength_lambda=140.0, batch_size=4
ADS: lam=0.1, eps=0.01, tau=0.9, holdout=2880, batch_size=4; tau/lam/eps come from the official ADS sweep range, and holdout=2880 follows the official math pipeline
Trace Rewriting: rewrite_strategy=optimized_prompt_official, rewriter temperature=0.6, top_p=0.95
DOGe: anti_kd_coef=3e-5, kd_temperature=2.0, learning_rate=5e-5, epochs=2, batch_size=1, gradient_accumulation_steps=16, max_length=2048
```

Known parameter decisions that may need adjustment before final reporting:

```text
Radioactivity: the official repo also reports a closed/supervised example with ngram=2, delta=3, gamma=0.25. Current formal default follows the main script defaults: ngram=4, delta=2.
ADFP: strength_lambda=140.0 is the current implementation default. If the paper's preferred strength differs for our model family, override DEFENSE_CONFIG.
ADFP and ADS holdout generation use batch_size=4 by default for transcript-capable attacks. ADS lam>0 defended generation uses microbatch=1 by default to avoid hangs from batched ADS logits processing with multiple device_map models; set ADS_ALLOW_LAM_BATCH=1 only for controlled small-model tests.
ADS: the paper/code sweep many tau/lam/eps combinations. Current formal default is a single representative point; a final ADS report should either justify this point or run a sweep.
DOGe: DOGE_MAX_STEPS is intentionally unset by default. Setting it is a pilot/speed override, not paper-matched formal behavior.
```


Submit multiple generator-style defenses:

```bash
METHODS="ginsew radioactivity adfp trace_rewriting ads doge" \
bash defenses/scripts/seqkd/submit_seqkd_array.sh
```


```bash
```


```bash
bash defenses/scripts/seqkd/run_ginsew_seqkd.sh
```

Common formal overrides:

```bash
BUDGET=1000 \
TEACHER_MODEL=Qwen/Qwen2.5-72B-Instruct \
STUDENT_MODEL=Qwen/Qwen2.5-7B \
QUERY_POOL=auto \
DEFENSE_EXECUTION_MODE=auto \
TEACHER_MAX_TOKENS=1536 \
OUTPUT_ROOT=/path/to/storage/$USER/A-Benchmark-for-Model-distillation-survey/outputs/defenses/seqkd_b1000 \
LOG_ROOT=/home/$USER/project/A-Benchmark-for-Model-distillation-survey/logs/defenses/seqkd_b1000 \
METHODS="ginsew radioactivity adfp" \
bash defenses/scripts/seqkd/submit_seqkd_array.sh
```

Trace Rewriting is also a generator-style defense; it is listed separately only because its formal script needs a clean teacher endpoint via `TEACHER_BASE_URL`.

Trace Rewriting required parameter:

```text
TEACHER_BASE_URL: OpenAI-compatible clean teacher endpoint, e.g. http://host:8000/v1
```

Trace Rewriting optional parameters:

```text
TEACHER_REQUEST_MODEL: model name served by the clean teacher endpoint
REWRITER_BACKEND: local_hf or openai_compatible; default openai_compatible
REWRITER_BASE_URL: defaults to TEACHER_BASE_URL
REWRITER_MODEL / REWRITER_REQUEST_MODEL: defaults to TEACHER_MODEL
DEFENSE_CONFIG: JSON config; default uses optimized_prompt_official
```

Trace Rewriting example:

```bash
METHODS="trace_rewriting" \
TEACHER_BASE_URL=http://127.0.0.1:8000/v1 \
TEACHER_REQUEST_MODEL=Qwen/Qwen2.5-72B-Instruct \
bash defenses/scripts/seqkd/submit_seqkd_array.sh
```

### ADS

ADS no longer requires the user to provide `GRAD_PATH`. The formal runner treats the gradient file as an internal defense artifact:

```text
$OUTPUT_ROOT/ads/prep/student_grads.pt
$OUTPUT_ROOT/ads/prep/holdout_teacher.jsonl
```

The script automatically takes disjoint holdout queries from the query pool, generates clean holdout teacher responses with `lam=0`, computes proxy-student gradients, and then starts the ADS defended teacher.

Useful optional overrides:

```text
ADS_HOLDOUT: number of holdout queries for gradient computation; default 2880 for formal runs
ADS_BATCH_SIZE: batch size for holdout generation and outer checkpoint chunks; default 4
ADS_ALLOW_LAM_BATCH: opt-in flag for batched lam>0 ADS logits processing; default 0
ADS_GRAD_MAX_LENGTH: max sequence length for gradient computation; default 512
ADS_GRAD_MODEL_DTYPE: proxy dtype for gradient computation; default auto, bf16 on CUDA
ADS_GRAD_DTYPE: CPU gradient accumulation/storage dtype; default float32
PROXY_MODEL: defaults to STUDENT_MODEL
GRAD_PATH: optional reuse path for debugging/resuming only
DEFENSE_CONFIG: optional ADS JSON config
```

Example:

```bash
METHODS="ads" \
ADS_HOLDOUT=2880 \
bash defenses/scripts/seqkd/submit_seqkd_array.sh
```

### DOGe

DOGe no longer requires the user to provide `DOGE_CHECKPOINT`. The formal runner treats the defended teacher checkpoint as an internal defense artifact:

```text
$OUTPUT_ROOT/doge/prep/doge_train.jsonl
$OUTPUT_ROOT/doge/prep/doge_teacher/
```

DOGe first needs clean teacher responses to build its defensive-training set, so it requires a clean teacher endpoint.

Required:

```text
TEACHER_BASE_URL: OpenAI-compatible clean teacher endpoint
```

Useful optional overrides:

```text
DOGE_TRAIN_EXAMPLES: number of training examples; default BUDGET
DOGE_MAX_STEPS: optional pilot/speed cap; unset by default
DOGE_NUM_TRAIN_EPOCHS: default 2
DOGE_BATCH_SIZE: default 1
DOGE_GRADIENT_ACCUMULATION_STEPS: default 16
DOGE_CHECKPOINT: optional reuse checkpoint path for debugging/resuming only
```

Example:

```bash
METHODS="doge" \
TEACHER_BASE_URL=http://127.0.0.1:8000/v1 \
TEACHER_REQUEST_MODEL=Qwen/Qwen2.5-72B-Instruct \
bash defenses/scripts/seqkd/submit_seqkd_array.sh
```

### Detector-Only Methods

`DuFFin` and `query_traffic` do not run a defended teacher. They consume existing attack outputs.

DuFFin requires a student checkpoint to test:

```bash
METHODS="duffin" \
STUDENT_CHECKPOINT=/blue/.../outputs/attacks_full/seqkd/<run_id>/training/checkpoint \
bash defenses/scripts/seqkd/submit_seqkd_array.sh
```

Common DuFFin overrides:

```text
DUFFIN_MAX_PROBES: default 1000
DUFFIN_MAX_NEW_TOKENS: default 1024 (official Knowledge-DuFFin generation length)
DUFFIN_MIN_VALID_RATE: default 0.9; lower parse coverage marks the run invalid
DUFFIN_BATCH_SIZE: default 4; reduce on OOM or increase after checking peak GPU memory

The complete DuFFin evaluation is run by
reference, evaluates the base negative and clean attack students, and writes the
model-level ROC-AUC summary in one job.

The default negative controls are `Qwen2.5-7B`, `Qwen2.5-7B-Instruct`,
`Qwen2-7B-Instruct`, and `Mistral-7B-Instruct-v0.3`. Override the set with
repeatable `--negative-model NAME=MODEL` arguments to the Python evaluator.
DUFFIN_PROBE_SOURCE: default mmlu_pro
DUFFIN_PROBE_CATEGORIES: default biology,business,chemistry,computer_science,math,physics
```

`query_traffic` needs the queries actually sent by an attack to the teacher. It can consume either a prepared `TEACHER_QUERY_LOG` or an attack-output source root.

Use a prepared query log:

```bash
METHODS="query_traffic" \
TEACHER_QUERY_LOG=/blue/.../teacher_received_queries.jsonl \
bash defenses/scripts/seqkd/submit_seqkd_array.sh
```

Collect from an explicit attack-output source, local or HF:

```bash
METHODS="query_traffic" \
ATTACK_NAME=seqkd \
bash defenses/scripts/seqkd/submit_seqkd_array.sh
```

The collector can also be run manually:

```bash
python3 -m defenses.query_traffic.collect_attack_queries \
  --output-dir /path/to/storage/$USER/A-Benchmark-for-Model-distillation-survey/outputs/query_traffic_inputs \
  --limit 1000
```

It writes:

```text
<output-dir>/<attack>/<run_id>/teacher_received_queries.jsonl
<output-dir>/<attack>/<run_id>/attack_query_collection_manifest.json
<output-dir>/attack_query_collection_index.json
```

Common query-traffic overrides:

```text
ATTACK_NAME: which collected attack to test; default seqkd
QT_DETECTORS: default "mmd prada seat"
QT_BENIGN_MODE: default matched_mix
QT_NUM_QUERIES: default BUDGET
BENIGN_QUERY_LOG: optional custom benign reference queries
QT_EMBEDDING_MODEL: default sentence-transformers/all-MiniLM-L6-v2
```

## SeqKD Smoke Tests

Online SeqKD smoke scripts live under `defenses/scripts/seqkd_smoke/`. They use Qwen same-family small models by default:

```text
teacher: Qwen/Qwen2.5-0.5B-Instruct
student: Qwen/Qwen2.5-0.5B
budget: 100
```

Run one method:

```bash
bash defenses/scripts/seqkd_smoke/run_ginsew_seqkd_smoke.sh
bash defenses/scripts/seqkd_smoke/run_radioactivity_seqkd_smoke.sh
bash defenses/scripts/seqkd_smoke/run_adfp_seqkd_smoke.sh
bash defenses/scripts/seqkd_smoke/run_ads_seqkd_smoke.sh
bash defenses/scripts/seqkd_smoke/run_trace_rewriting_seqkd_smoke.sh
bash defenses/scripts/seqkd_smoke/run_doge_seqkd_smoke.sh
bash defenses/scripts/seqkd_smoke/run_duffin_seqkd_smoke.sh
bash defenses/scripts/seqkd_smoke/run_query_traffic_seqkd_smoke.sh
```

Run all smoke scripts:

```bash
bash defenses/scripts/seqkd_smoke/run_all_seqkd_smokes.sh
```

Common overrides:

```bash
BUDGET=100 \
MAX_NEW_TOKENS=16 \
DETECTOR_MAX_QUERIES=5 \
OUTPUT_ROOT=outputs/defenses/seqkd_smoke_online \
bash defenses/scripts/seqkd_smoke/run_ginsew_seqkd_smoke.sh
```


```bash
mkdir -p logs
```


```bash
bash defenses/scripts/seqkd_smoke/submit_seqkd_smoke_array.sh
```

Limit the array to selected methods:

```bash
METHODS="ginsew trace_rewriting query_traffic" \
bash defenses/scripts/seqkd_smoke/submit_seqkd_smoke_array.sh
```

## Online Oracle

The lower-level server is available for debugging:

```bash
python -m defenses.oracle.serve_defended_teacher \
  --defense ginsew \
  --teacher-model Qwen/Qwen2.5-7B-Instruct \
  --defense-config '{"gamma":0.25,"delta":2.0,"max_new_tokens":512}' \
  --output-dir outputs/defenses/ginsew/oracle_001 \
  --host 127.0.0.1 \
  --port 9000
```

Then any attack can target it exactly like a normal teacher:

```bash
python attacks/scripts/run_attack.py \
  --attack seqkd \
  --budget 1000 \
  --teacher-backend vllm_openai \
  --teacher-endpoint-url http://127.0.0.1:9000/v1 \
  --teacher-request-model defended-ginsew \
  --teacher-model Qwen/Qwen2.5-7B-Instruct \
  --student-model Qwen/Qwen2.5-0.5B-Instruct
```

The oracle exposes:

```text
/v1/chat/completions
/v1/completions
/v1/models
```

## Supported Methods

Generator + detector:

```text
ginsew
radioactivity
adfp
```

Generator only:

```text
ads
trace_rewriting
doge
```

Detector only:

```text
duffin
query_traffic
```

ADS and DOGe generate their method-internal artifacts inside their formal runners by default. `GRAD_PATH` and `DOGE_CHECKPOINT` are optional reuse/debug overrides, not required public inputs. Trace Rewriting needs an upstream OpenAI-compatible clean teacher via `--teacher-base-url` when using the formal script.

## Core Layout

```text
defenses/core/generator/   start defended teacher, run attack, collect attack outputs
defenses/core/detector/    run detector commands and normalize detector results
defenses/core/runner/      full defense run orchestration and manifests
defenses/oracle/           OpenAI-compatible defended teacher server
```

Legacy offline transcript scripts were removed from the default tree so formal runs do not accidentally bypass the defended-teacher boundary.
