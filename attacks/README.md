# Unified Stage-1 Attack Pipeline

This directory contains the unified Stage-1 model extraction attack pipeline. The design is intentionally lightweight: the shared pipeline handles run directories, common arguments, validation, and `attack_manifest.json`; each attacker owns its own data construction and training logic.

## Quick Start

For the HPG formal run with teacher/student vLLM endpoints, shared teacher transcript, and two attack batches, see [FULL_RUN_MANUAL.md](FULL_RUN_MANUAL.md).

Run one attack:

```bash
python3 attacks/scripts/run_attack.py \
  --attack seqkd \
  --budget 1000 \
  --teacher-backend vllm_openai \
  --teacher-model meta-llama/Llama-3.3-70B-Instruct \
  --teacher-endpoint-url http://127.0.0.1:8000/v1 \
  --teacher-request-model meta-llama/Llama-3.3-70B-Instruct \
  --student-model meta-llama/Llama-3.1-8B-Instruct
```

Run all registered attacks:

```bash
bash attacks/scripts/run_all_attacks.sh
```

Run a subset:

```bash
ATTACKS="seqkd lord model_leeching" BUDGET=1000 bash attacks/scripts/run_all_attacks.sh
```

Run manifest-only checks:

```bash
DRY_RUN=1 bash attacks/scripts/run_all_attacks.sh
```


Run after starting the required teacher/student endpoints:

```bash
ATTACKS="seqkd lord soda qedks model_leeching" \
BUDGET=1000 \
TEACHER_ENDPOINT_URL=http://127.0.0.1:8000/v1 \
STUDENT_ENDPOINT_URL=http://127.0.0.1:8001/v1 \
STUDENT_MODEL=meta-llama/Llama-3.1-8B-Instruct \
bash attacks/scripts/run_all_attacks.sh
```

## Common Parameters

Use one shared vocabulary across attackers:

- `--attack`: one of `seqkd`, `lord`, `soda`, `qedks`, `model_leeching`, `gad`.
- `--budget`: number of query examples or planned attack queries.
- `--query-pool`: JSON query pool. Default: `auto`, which selects the published HF query-pool tier for `--budget`. The CLI downloads it from Hugging Face on first use and caches it under `.cache/mea_benchmark/hf_datasets/`.
- `--query-ordering`: deterministic ordering file. Default: `auto`, which creates an ordering file beside the cached HF dataset. Pass a concrete JSON path only when reproducing a custom ordering.
- `--output-dir`: root for attack runs. Default: `outputs/attacks`.
- `--teacher-model`: teacher model id or local path.
- `--teacher-endpoint-url`: OpenAI-compatible teacher endpoint base URL.
- `--teacher-request-model`: model name sent to the teacher endpoint.
- `--student-model`: student/base/warmup model id, local path, or checkpoint path.
- `--student-endpoint-url`: OpenAI-compatible student endpoint base URL for attacks that query the student.
- `--student-request-model`: model name sent to the student endpoint.
- `--warmup-model`: warmup/reference checkpoint for attacks that need one, such as SODA.
- `--dry-run`: build a manifest and planned artifact paths without importing training dependencies or running model calls.

Method-specific parameters use method prefixes, for example `--soda-beta`, `--qedks-use-ppl-schedule`, `--model-leeching-learning-rate`, and `--gad-group-size`.


## Query Pool Source

The default Stage-1 query pool comes from the Hugging Face dataset `anonymous-mea-benchmark/mea-query-pools`. When `--query-pool auto` is used, the CLI selects the published tier for the requested budget. Override it with `MEA_QUERY_POOL_DATASET` when needed:

```text
hf://anonymous-mea-benchmark/mea-query-pools/query_pool_100.json
hf://anonymous-mea-benchmark/mea-query-pools/query_pool_1000.json
hf://anonymous-mea-benchmark/mea-query-pools/query_pool_10000.json
hf://anonymous-mea-benchmark/mea-query-pools/query_pool_50000.json
hf://anonymous-mea-benchmark/mea-query-pools/query_pool_100000.json
```

`attacks/scripts/run_attack.py` and `attacks/scripts/build_shared_teacher_transcript.py` resolve `auto` and `hf://...` specs before constructing the attacker config. On first use it tries `huggingface_hub` and falls back to direct HTTPS download, then stores the file under `.cache/mea_benchmark/hf_datasets/`. With `--query-ordering auto`, it also writes a deterministic ordering JSON for the cached pool. This means the current attack pipeline no longer depends on `stage1-LoRD-SeqKD/data/query_pool.json` or `query_ordering.json` as default inputs.

Budget tiers are treated as benchmark data tiers, not arbitrary prefixes of `query_pool_10000.json`. For `100`, `1000`, `10000`, `50000`, and `100000`, `auto` selects the exact published HF file. For intermediate smoke budgets such as `500`, `auto` selects the smallest published tier that can cover the requested prefix, currently `query_pool_1000.json`. The HF dataset does not currently publish `query_pool_200000.json`, so a `200000` run must pass an explicit larger query pool once that file exists.

## Environment

For real attack smoke tests and LoRA/DPO training, install the attack environment dependencies inside the target conda environment:

```bash
pip install -r attacks/requirements-smoke.txt
python3 attacks/scripts/check_attack_env.py --require-trl --require-vllm --strict-versions
```

## Output Layout

Every run writes under:

```text
outputs/attacks_full/{attack}/{run_id}/
  attack_manifest.json
  ... method-owned artifacts ...
```

`attack_manifest.json` is written by `attacks/core/pipeline.py` and contains:

- normalized run config
- query pool hash and ordering hash
- attack result status
- checkpoint path, if available
- method artifact paths and lightweight metrics

Method-owned subdirectories vary by attack. Common names include `transcripts/`, `teacher_transcripts/`, `train_data/`, `checkpoints/`, `query_plans/`, and `logs/`.

### Checkpoint Outputs by Attack

The final student checkpoint path is always recorded in `attack_manifest.json` as `checkpoint_dir`. Evaluation code should read that field instead of guessing method-specific subdirectories.

| attack | checkpoint kind | typical `checkpoint_dir` target | can load directly? | notes |
|---|---|---|---|---|
| `seqkd` | full HF causal-LM checkpoint | `training/seqkd/budget_<N>/seqkd/checkpoint-final` | yes | saved by `Trainer.save_model`; includes tokenizer files. |
| `lord` | full HF causal-LM checkpoint | `training/lord/budget_<N>/lord/checkpoint-final` | yes | saved by `model.save_pretrained`; also writes `lord_state.json`. |
| `soda` | DPO-trained HF checkpoint | `checkpoints/dpo` | usually yes | saved by `DPOTrainer.save_model`; loader still checks whether it is a PEFT adapter. |
| `qedks` | LoRA adapter checkpoint | `checkpoints/lora_sft` | needs base model | use `meta-llama/Llama-3.1-8B-Instruct` unless `adapter_config.json` records another base. |
| `model_leeching` | LoRA adapter checkpoint | `checkpoints/lora_sft` | needs base model | same loader path as QEDKS. |
| `gad` | full HF generator checkpoint | `training/gad/checkpoint-final` | yes | discriminator is saved separately as a training artifact and is not the exported student. |

Useful inspection commands:

```bash
RUN=$(ls -td outputs/attacks_full/seqkd/* | head -n 1)
cat "$RUN/attack_manifest.json"

python3 - <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
print(manifest["checkpoint_dir"])
PY "$RUN/attack_manifest.json"
```

## Load and Query a Trained Student Checkpoint

Attack outputs are usable student checkpoints. Full checkpoints can be loaded directly; LoRA adapter checkpoints need their base model. Evaluation code should use the shared loader in `attacks.core.student_model` so every metric calls students through the same interface:

```python
from attacks.core.student_model import GenerationConfig, load_student_from_manifest

student = load_student_from_manifest(
    "outputs/attacks_full/seqkd/<run_id>/attack_manifest.json",
    dtype="bfloat16",
    device_map="auto",
)

response = student.generate_one(
    "Explain photosynthesis in one paragraph.",
    generation_config=GenerationConfig(max_new_tokens=256, use_chat_template=True),
)

responses = student.generate(
    ["What is model extraction?", "Define knowledge distillation."],
    generation_config=GenerationConfig(max_new_tokens=128, use_chat_template=True),
)
```

For a direct checkpoint path, including LoRA adapters:

```python
from attacks.core.student_model import load_student_from_checkpoint

student = load_student_from_checkpoint(
    "/path/to/lora_adapter_or_full_checkpoint",
    base_model="meta-llama/Llama-3.1-8B-Instruct",  # only needed when LoRA metadata cannot infer it
    dtype="bfloat16",
    device_map="auto",
)
print(student.generate_one("What is model extraction?", use_chat_template=True))
```

The command-line wrapper uses the same interface:

```bash
python3 attacks/scripts/query_student_checkpoint.py \
  --manifest outputs/attacks_full/model_leeching/<run_id>/attack_manifest.json \
  --query-file queries.txt \
  --output-jsonl outputs/student_generations/model_leeching.jsonl \
  --use-chat-template
```

## Implementation Structure

```text
attacks/
  core/
    base.py          # AttackRunConfig, AttackResult, BaseAttacker protocol
    manifest.py      # JSON writing and hashing helpers
    pipeline.py      # attacker registry, run directory creation, manifest writing
    student_model.py # reusable checkpoint loader/generation interface for evaluation
  methods/
    seqkd.py         # thin attacker class for SeqKD
    lord.py          # thin attacker class for LoRD
    soda.py          # SODA orchestration
    qedks.py         # QEDKS orchestration
    model_leeching.py# Model Leeching orchestration
    gad.py           # GAD orchestration
    gad_impl/        # GAD discriminator and adversarial training loop
    stage1_budget.py # shared SeqKD/LoRD wrapper
    *_impl/          # method-specific implementation modules
  configs/
    stage1_attacks.yaml
    stage1_budget.yaml
  scripts/
    run_attack.py
    run_all_attacks.sh
    query_student_checkpoint.py # CLI wrapper around attacks.core.student_model
```

The key abstraction is in `attacks/core/base.py`:

```python
class BaseAttacker(Protocol):
    name: st

    def run(self, config: AttackRunConfig, *, run_id: str, run_dir: Path) -> AttackResult:
        ...
```

The pipeline does not know each method's training data schema. It only passes common config into the selected attacker and expects an `AttackResult`. This keeps method-specific data construction inside each attacker.

## Attacker Flows

### SeqKD

Files:

- `attacks/methods/seqkd.py`
- `attacks/methods/stage1_budget.py`
- `attacks/methods/stage1_budget_impl/`

Flow:

```text
query_pool + budget
  -> build/reuse teacher transcript
  -> response-only SFT
  -> checkpoint
```

SeqKD and LoRD share the migrated Stage-1 implementation in `stage1_budget_impl`. The public attacker class is intentionally thin; the algorithm split happens in `stage1_budget_impl/run_stage1_budget.py`.

### LoRD

Files:

- `attacks/methods/lord.py`
- `attacks/methods/stage1_budget.py`
- `attacks/methods/stage1_budget_impl/`

Flow:

```text
query_pool + budget
  -> build/reuse teacher transcript
  -> LoRD training loop
  -> checkpoint
```

LoRD uses the same CLI shape as SeqKD with `--attack lord`.

### SODA

Files:

- `attacks/methods/soda.py`
- `attacks/methods/soda_impl/`

Flow:

```text
teacher transcript
  -> collect student rejected responses
  -> build preference pairs
  -> DPO from warmup/reference model
  -> checkpoint
```

SODA uses a completed SeqKD checkpoint `q_w` as both the DPO initialization and frozen reference policy. When both `--warmup-model` and `--transcript-dir` are omitted, SODA automatically selects the newest compatible completed SeqKD run under `<output-dir>/seqkd` and reuses both its checkpoint and teacher transcript. Compatibility requires the same budget, base student, and query pool/order. If a transcript is supplied explicitly, it must also match the selected SeqKD run. If no compatible run exists, SODA stops with an error instead of falling back to the untrained base student.

`--warmup-model` remains available as an explicit override. A local override must be a complete full-model or LoRA checkpoint and must not point to the unchanged base student.

Student negatives are generated once from the untrained base student `q_0`; they are not generated from the SeqKD warmup model:

```text
base student q_0 -> frozen student negatives
teacher transcript + q_0 -> existing SeqKD checkpoint q_w
q_w + preference pairs -> SODA DPO checkpoint
```

### QEDKS

Files:

- `attacks/methods/qedks.py`
- `attacks/methods/qedks_impl/`

Flow:

```text
seed query plan
  -> collect teacher seed responses
  -> template query plan
  -> optional PPL scheduling
  -> collect teacher template responses
  -> follow-up query plan
  -> collect teacher follow-up responses
  -> combined teacher transcript
  -> SFT JSONL
  -> LoRA SFT checkpoint
```

Optional PPL-guided scheduling can be enabled with `--qedks-use-ppl-schedule`.

### Model Leeching

Files:

- `attacks/methods/model_leeching.py`
- `attacks/methods/model_leeching_impl/`

Flow:

```text
query_pool + budget
  -> template wrapper before teacher query
  -> collect teacher raw JSON-like responses
  -> parse/clean/validate
  -> SFT JSONL
  -> LoRA SFT checkpoint
```

This implementation follows the reproduction scope: compared with SeqKD, Model Leeching adds two thin layers, prompt templating before querying and parse/clean/validate before training. Artifacts include `templated_queries.jsonl`, `teacher_raw.jsonl`, `clean_records.jsonl`, `rejected_records.jsonl`, `cleaning_stats.json`, and `model_leeching_sft.jsonl`.

### GAD

Files:

- `attacks/methods/gad.py`
- `attacks/methods/gad_impl/train_gad.py`

Flow:

```text
query_pool + budget
  -> build/reuse offline teacher transcript
  -> SeqKD SFT warmup for generator G
  -> initialize discriminator D as LM backbone + scalar score head
  -> D warmup with Bradley-Terry teacher > student pairs
  -> repeat: student rollout K responses, D scores rewards, GRPO-style group-normalized actor update, BT discriminator update
  -> checkpoint for G plus discriminator artifact
```

The implementation follows the reproduction scope and the official `YTianZHU/verl` GAD branch at the algorithm level: D is a sequence-level scorer, BT loss is `-logsigmoid(D_teacher - D_student)`, default rollout group size is `K=8`, and default KL regularization is `0.001`. It does not vendor the official `verl` fork; it reuses this benchmark's teacher transcript and SeqKD warmup infrastructure.


## Adding a New Attacke

1. Add `attacks/methods/{name}.py` with a class implementing `run(config, run_id, run_dir) -> AttackResult`.
2. Put method-owned helper modules under `attacks/methods/{name}_impl/` if needed.
3. Register the class in `attacks/core/pipeline.py`.
4. Add the name to `--attack` choices in `attacks/scripts/run_attack.py`.
5. Add method-specific CLI arguments only when they are genuinely method-specific. Reuse common teacher/student/budget/output parameters whenever possible.
6. Add a short entry in `attacks/configs/stage1_attacks.yaml` and this README.
7. Verify with:

```bash
python3 -m py_compile attacks/core/*.py attacks/methods/*.py attacks/scripts/run_attack.py
python3 attacks/scripts/run_attack.py --attack {name} --budget 10 --dry-run
```

For implementation sanity, keep this boundary: the shared pipeline manages orchestration and manifests; the attacker class manages method-specific data and training.

## Memory and Staged Execution Notes

Some attacks are substantially heavier than plain SeqKD SFT. The code includes several implementation-level memory optimizations that do not change the intended attack algorithms.

### Default-enabled optimizations

These are enabled by default in `run_attack.py` and/or the formal HPG config:

- **bf16 loading/training**
  - CLI default: `--bf16`
  - Used by SODA and GAD through the shared CLI flags. LoRD uses `LORD_BF16=true` from `attacks/configs/formal_stage1_budget.yaml`.

- **LoRA training for heavy methods**
  - CLI default: `--use-lora`
  - SODA DPO and GAD use the shared LoRA flags.
  - LoRD defaults to `LORD_USE_LORA=true` in the formal config.
  - SeqKD formal config intentionally keeps `STAGE1_SEQKD_USE_LORA=false` unless overridden, because SeqKD full checkpoint is the baseline student output.

- **Gradient checkpointing**
  - CLI default: `--gradient-checkpointing`
  - LoRD defaults to `LORD_GRADIENT_CHECKPOINTING=true`.
  - SeqKD formal config defaults to `STAGE1_SEQKD_GRADIENT_CHECKPOINTING=true`.

- **LoRD LoRA snapshot optimization**
  - Enabled automatically when LoRD runs with `use_lora=true`.
  - Instead of `copy.deepcopy(model)`, LoRD stores only the trainable LoRA snapshot state on CPU and temporarily swaps it into the single live model when it needs snapshot generation/scoring.
  - If `LORD_USE_LORA=false`, LoRD falls back to the older full-model snapshot path.

- **GAD batched rollout log-prob calculation**
  - Always enabled.
  - GAD computes old/new log probabilities for all rollout responses from the same prompt in one padded batch forward instead of one forward per response.

- **GAD memory trace**
  - Enabled by default with interval `25`.
  - CLI: `--gad-memory-log-interval 25`
  - The trace is written into `gad_manifest.json` as `memory_trace`, with peak values also summarized as `peak_cuda_allocated_gib` and `peak_cuda_reserved_gib`.

### Optional optimizations

These are not enabled by default because they may slow training or change run logistics:

- **GAD inactive-model CPU offload**
  - Default: off.
  - CLI:

```bash
--gad-offload-inactive-models
```


```bash
GAD_OFFLOAD_INACTIVE_MODELS=1
```

  - Effect: during GAD discriminator updates, generator `G` and its optimizer state can be moved to CPU; during generator updates, discriminator `D` and its optimizer state can be moved to CPU. This reduces peak GPU residency at the cost of extra CPU/GPU transfer time.

Example:

```bash
ATTACKS="gad" \
GAD_OFFLOAD_INACTIVE_MODELS=1 \
GAD_MEMORY_LOG_INTERVAL=1 \
```

- **GAD G/D split across two GPUs**
  - Default: off. If no device is provided, both models use the default CUDA device.
  - CLI:

```bash
--gad-generator-device cuda:0 \
--gad-discriminator-device cuda:1
```


```bash
GAD_GENERATOR_DEVICE=cuda:0
GAD_DISCRIMINATOR_DEVICE=cuda:1
```


```bash
ATTACKS="gad" \
GAD_GENERATOR_DEVICE=cuda:0 \
GAD_DISCRIMINATOR_DEVICE=cuda:1 \
```

  - Effect: generator/student `G` stays on `cuda:0`, discriminator `D` stays on `cuda:1`, and the training loop only moves small scalar tensors between devices. This is usually preferable to CPU offload when 2 GPUs are available.

### SODA staged execution

SODA has two separable phases:

```text
student vLLM generates rejected responses
  -> build teacher/student preference pairs
  -> DPO training
```

To avoid keeping a student vLLM server alive during DPO training, SODA can reuse precomputed intermediate files.

Reuse existing student negatives:

```bash
SODA_STUDENT_NEGATIVES_JSONL=/path/to/student_negatives.jsonl \
ATTACKS="soda" \
```

This skips `collect_student_negatives`, rebuilds `preferences.jsonl` from the shared teacher transcript and the supplied student negatives, then runs DPO.

Reuse existing preference pairs:

```bash
SODA_PREFERENCES_JSONL=/path/to/preferences.jsonl \
ATTACKS="soda" \
```

This skips both `collect_student_negatives` and `build_preferences`, then runs DPO directly. In this mode SODA no longer needs a live student endpoint for training.


The formal budget entry points under `runs/attack/budget_*/` automatically stage QEDKS, Model Leeching, and SODA:

```text
vLLM endpoint allocation
  -> concurrent CPU collection/preparation
  -> endpoint cancellation
  -> one-GPU LoRA/DPO allocation
```

The query and training phases therefore never reserve each other's GPUs. QEDKS and Model Leeching no longer request three GPUs in one job, and SODA no longer requests two GPUs in one job. SODA's endpoint loads the auto-discovered SeqKD LoRA warmup with vLLM's LoRA serving support; the same checkpoint is then used to initialize DPO.

Collection concurrency is controlled by:

```bash
ATTACK_QUERY_CONCURRENCY=8
```


### SODA numerical safety

SODA DPO treats finite loss as insufficient evidence of a valid update. The trainer checks every trainable gradient after each micro-batch. If LoRA dropout produces a transient non-finite backward pass, it restores the accumulated gradients from before that micro-batch and retries with a new dropout mask. The default retry limit is three:

```bash
SODA_NONFINITE_GRADIENT_RETRIES=3
SODA_MAX_GRAD_NORM=1.0
```

If all retries fail, the process exits nonzero. Before saving, all trainable parameters are checked; after saving, every checkpoint tensor is checked again. A successful run writes `training_safety.json` beside the adapter with optimizer-step learning rates, retry counts, and checkpoint health. No attack or defense completion manifest is written after a failed check.

The linear learning-rate scheduler intentionally reaches zero after the final optimizer update. Consequently, a final Trainer log entry containing `learning_rate: 0.0` does not by itself mean training used a zero rate. `training_safety.json` records the rate immediately before every optimizer step and rejects a run that never used a positive rate.

The pinned environment combines TRL 1.9.2 with PEFT 0.15.2. TRL reads the newer optional `LoraConfig.target_parameters` field while cloning the warmup adapter into its frozen `ref` adapter, but PEFT 0.15.2 predates that field. SODA explicitly supplies the equivalent legacy default `None`, reports the compatibility path in the log, and records it in `training_safety.json`; it does not remove or rename any DPO hyperparameter.

Before allocating an endpoint or loading model weights, the staged SODA entry point runs the complete pinned-runtime API audit. It checks all DPOConfig fields used by this project, DPOTrainer initialization, PEFT load/add-adapter calls, the optimizer callback hook, exact package versions, and whether the PEFT compatibility path is required. Run the same preflight manually with:

```bash
python3 attacks/scripts/check_soda_runtime.py
```

Existing checkpoints can be audited independently:

```bash
python3 attacks/scripts/validate_checkpoint_finite.py /path/to/checkpoint
```

The command exits with status `1` when any tensor contains NaN or infinity. Student checkpoint loading and defense detector entry points also reject non-finite SODA adapters before generation.
