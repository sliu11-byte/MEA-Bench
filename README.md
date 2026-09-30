# MEA-Bench: A Benchmark for Model Extraction Attacks

This repository provides a unified benchmark for model extraction attacks,
defenses, adaptive attacks, and evaluation. The public interface is organized
around a small number of portable commands. Method-specific Python modules and
legacy run scripts are implementation details rather than user-facing entry
points.

The four portable entry points below validate their public arguments and then
dispatch to the method implementations. Slurm resource wrappers are intentionally
excluded. Method-owned manifests remain authoritative for resume behavior and
artifact provenance.

## Paper and Authors

**Do Defenses Against LLM Extraction Work Across Attacks? A Lifecycle Benchmark of Black-Box Model Extraction**

Shuze Liu (Florida State University), Kaixiang Zhao (Brigham Young University),
Runyang Xu (University of Michigan, Ann Arbor), Jingzhi Chen (State University
of New York at Buffalo), Nathan Wu (Wake Forest University), Yu Wang (University
of Georgia), and Yushun Dong (Florida State University).

Paper source: https://github.com/sliu11-byte/MEA_benchmark_arXiv

This repository provides the implementations and reproduction interface for the
paper. Query pools are hosted under the authors' Hugging Face project:
https://huggingface.co/datasets/watermarkproject/lord-mea-benchmark

## Repository Layout

```text
attacks/          attack implementations and shared attack pipeline
defenses/         defense implementations and detector adapters
countermeasures/  adaptive-attack pipeline
evaluation/       shared tasks, rollout code, and metrics
infra/            portable model-serving helpers
runs/             canonical public entry points
```

## Canonical Public Interface

The benchmark exposes four top-level commands:

| Experiment | Entry point | Required experiment arguments |
|---|---|---|
| Attack | `runs/run_attack.sh` | `--attack`, `--budget` |
| Defense | `runs/run_defense.sh` | defended extraction: `--defense`, `--attack`, `--budget`; result-based defense: `--defense`, `--attack-run` |
| Adaptive attack | `runs/run_adaptive_attack.sh` | `--adaptive-attack`, `--defense`, `--attack`, `--budget` |
| Evaluation | `runs/run_evaluation.sh` | `--manifest` |

All four commands:

- work as ordinary shell commands without Slurm;
- accept `--help` and `--dry-run`;
- validate the requested experiment before starting;
- automatically create, discover, validate, and reuse all prerequisite artifacts;
- use deterministic benchmark defaults under `--profile paper`;
- preserve each method's existing resume and artifact-reuse behavior;
- write or preserve machine-readable run metadata and input provenance;
- avoid embedded usernames, cluster accounts, email addresses, or site-specific paths.

The runners coordinate experiment logic in the current shell process. They do
not submit cluster jobs or choose a scheduler partition. The caller is responsible
for allocating sufficient CPU, memory, and GPUs before invoking a runner. A user
may provide already running OpenAI-compatible teacher and student endpoints; the
attack dispatcher can also start its existing method-local vLLM services.

The commands shown below are the complete public reproduction interface. A
reader should not need to prepare transcripts, warmup checkpoints, adaptive
baselines, held-out teacher outputs, or checkpoint bundles manually.
Those are internal dependencies owned by the runners. Manual configuration is
limited to resources or credentials that cannot be inferred, such as GPU
allocation, access to gated Hugging Face models, and optional external model
endpoints.

## Supported Methods

### Attacks

The attack registry contains six methods:

```text
seqkd
lord
soda
qedks
model_leeching
gad
```

The paper protocol uses attack budgets `100`, `1000`, and `10000`. The published
query dataset also contains 50,000- and 100,000-record pools for deterministic
subsets and held-out construction, but these are not advertised as primary attack
budgets.

### Defenses

Defenses are separated by the artifacts they require.

**Defended-extraction defenses** modify responses, training, or the extraction
process and therefore run a selected attack under the defense:

```text
ads
doge
trace_rewriting
adfp
ginsew
radioactivity
```

**Result-based defenses and detectors** consume artifacts from a completed attack
run:

```text
duffin
mmd
prada
seat
```

MMD, PRADA, and SEAT primarily consume attack query traffic. DuFFin consumes the
completed attack artifacts required by its detector. The defense registry, not
the shell wrapper, defines the exact artifact requirements for each method.

### Adaptive Attacks

The adaptive-attack registry contains:

```text
dipper
translation
```

`translation` denotes the benchmark's back-translation adaptive attack. The
registry rejects attack-defense-adaptation combinations that are not
implemented or not part of the benchmark protocol.

## Setup

Use Python 3.10 and a CUDA/PyTorch environment compatible with the versions
recorded in the paper. The unified benchmark environment is:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the commands below from the repository root. The scripts locate their
method implementations relative to that root and write all default artifacts
there.

The single top-level requirements file covers attacks, defenses, adaptive
attacks, local vLLM serving, and evaluation. It uses the CUDA 12.8 PyTorch wheel
index recorded by the paper environment. On a machine with a different CUDA
stack, install the matching PyTorch build first and then install the remaining
requirements. Activate the desired environment before invoking a runner; the
portable scripts do not load environment modules or activate Conda automatically.

Verify the installation before a full run:

```bash
python3 attacks/scripts/check_attack_env.py \
  --require-trl --require-vllm --strict-versions
```

The Llama models used by attacks, adaptive attacks, and their evaluation are
gated on Hugging Face. Request access to both Llama model repositories, then
authenticate once before running the benchmark:

```bash
hf auth login
```

On a non-interactive machine, set `HF_TOKEN` instead. The token is read by the
Hugging Face libraries and must never be written into a manifest, script, or
published repository. The query-pool dataset itself is public.

### Models Used in the Paper

| Experiment | Role | Hugging Face model ID |
|---|---|---|
| Clean attacks | victim/teacher | `meta-llama/Llama-3.3-70B-Instruct` |
| Clean attacks | surrogate/student | `meta-llama/Llama-3.1-8B-Instruct` |
| Defenses | victim/teacher | `Qwen/Qwen2.5-72B-Instruct` |
| Defenses | base surrogate/student | `Qwen/Qwen2.5-7B` |
| Adaptive attacks | victim/teacher | `meta-llama/Llama-3.3-70B-Instruct` |
| Adaptive attacks | surrogate/student | `meta-llama/Llama-3.1-8B-Instruct` |
| DIPPER adaptive attack | response rewriter | `kalpeshk2011/dipper-paraphraser-xxl` |
| DIPPER adaptive attack | tokenizer | `google/t5-v1_1-xxl` |
| Back-translation adaptive attack | translation model | `facebook/seamless-m4t-v2-large` |

The attack runner can serve its large models through local OpenAI-compatible
vLLM endpoints. Portable defaults are:

```text
teacher endpoint: http://127.0.0.1:8000/v1
student endpoint: http://127.0.0.1:8001/v1
```

For `runs/run_attack.sh`, the serving mode determines who owns the server
process:

| Mode | Server lifecycle |
|---|---|
| `local` (default) | The script starts vLLM in the current allocation, waits for it to become healthy, and stops only the process that it started. |
| `external` | The user starts an OpenAI-compatible endpoint before running the script. The script connects to it and never stops it. |

The defense and adaptive-attack entrypoints use their method-specific runners;
their required model IDs are listed above. GPU allocation, distributed launch,
and cluster scheduling remain the caller's responsibility. Sampling, training,
and evaluation defaults are loaded from the paper profile rather than
duplicated in individual shell scripts.

### Artifact Locations

All run commands interpret `--output-root` as a common artifact root and append
their experiment-type directory automatically:

| Command | Default primary output | Manifest passed to evaluation |
|---|---|---|
| `run_attack.sh` | `outputs/attacks/<attack>/<run-id>/` | `attack_manifest.json` |
| `run_defense.sh` (defended extraction) | `outputs/defenses/<defense>/<attack>/b<budget>/` | `defense_run_manifest.json` |
| `run_defense.sh` (result-based) | `outputs/defenses/<defense>/<attack>/b<budget>/` | generated `detector_manifest.json` |
| `run_adaptive_attack.sh` | `outputs/adaptive/<adaptive>/<defense>/<attack>/b<budget>/` | `defense_run_manifest.json` |
| `run_evaluation.sh` | `outputs/evaluation/<run-type>/<source-run-id>/` | not applicable |

For example, adding `--output-root /data/mea-runs` to an attack command changes
`outputs/attacks/...` to `/data/mea-runs/attacks/...`; the same rule applies to
`defenses/` and `adaptive/`. Evaluation reads the source selected by
`--manifest` and then discovers its local checkpoint and related artifacts
automatically. Its own `--output-root` controls only the destination for metrics.

The recommended workflow is:

1. run the desired command once with `--dry-run` to validate its arguments;
2. run it without `--dry-run` and wait for a zero exit status;
3. locate the manifest shown in the final command output; and
4. pass that exact manifest to `runs/run_evaluation.sh`.

## 1. Run an Attack

Run exactly one attack and one budget per invocation:

```bash
bash runs/run_attack.sh \
  --attack seqkd \
  --budget 1000 \
  --profile paper
```

Common optional arguments are:

```text
--output-root outputs
--seed 20260701
--resume
--dry-run
```

The paper profile supplies the query pool, model IDs, generation settings, and
local serving mode. Existing OpenAI-compatible endpoints can be selected as an
advanced override with `--teacher-serving external`, `--teacher-endpoint`,
`--student-serving external`, and `--student-endpoint`.

The runner is responsible for:

1. resolving the exact query-pool tier and deterministic ordering;
2. validating the teacher and student configuration;
3. creating or reusing protocol-compatible teacher transcripts;
4. generating method-specific prerequisites such as the matched SeqKD
   initialization needed by SODA;
5. running the selected method without changing its defining acquisition or
   training procedure;
6. writing a completed `attack_manifest.json`.

SeqKD, LoRD, SODA, and GAD reuse the shared offline teacher transcript under the
paper profile. QEDKS and Model Leeching retain their method-specific acquisition
procedures. Transcript reuse is accepted only when the budget, query-pool hash,
ordering hash, victim model, and generation settings match.

Serving defaults to `local`, restoring the former Slurm orchestrator's lifecycle
inside the caller's allocation. For offline attacks the runner starts teacher
vLLM, builds the transcript, stops vLLM, and then trains; for QEDKS and Model
Leeching it keeps a method-local teacher for acquisition. SODA also starts its
required student service automatically. Pass `--teacher-serving external` or
`--student-serving external` only to reuse an endpoint that is already running.
GPU placement and capacity remain the caller's responsibility.

The existing attack pipeline owns the timestamped run directory. Expected
output:

```text
outputs/attacks/<attack>/<timestamp>_<attack>_b<budget>/
  attack_manifest.json
  transcripts/
  train_data/
  checkpoint-final/
  logs/
```

Method-owned directories differ for some attacks, but their paths are recorded
in `attack_manifest.json`. Downstream code should read the manifest instead of
guessing a checkpoint directory. QEDKS resumes compatible incomplete query state
automatically; other methods reuse explicitly supplied transcripts, prepared
data, or warmup checkpoints rather than treating `--resume` as a universal
checkpoint restart mechanism.

`--output-root` sets the artifact root. For example,
`--output-root /data/mea-runs` changes the manifest location to:

```text
/data/mea-runs/attacks/<attack>/<run-id>/attack_manifest.json
```

## 2. Run a Defense

### Defended extraction

For ADS, DOGe, Trace Rewriting, ADFP, GINSEW, and Radioactivity, specify the
defense, attack, and budget:

```bash
bash runs/run_defense.sh \
  --defense adfp \
  --attack seqkd \
  --budget 1000 \
  --profile paper
```

The default `--execution-mode auto` selects the supported local execution path,
loads the required models, and starts and stops any internal oracle or student
service used by that path. `--teacher-endpoint` and `--student-endpoint` are
optional overrides for already running compatible services; they are not
prerequisites for the command above.

The runner constructs the defended oracle or training path, executes the attack,
and records the defended transcript, defense artifacts, detector outputs, and
student checkpoint in `defense_run_manifest.json`.

The runner does not silently reuse an incompatible clean attack. Any reused
warmup checkpoint or transcript must pass the same model, budget, query-pool,
ordering, and generation-config checks used by the attack pipeline.

The paper's defense benchmark uses `Qwen/Qwen2.5-72B-Instruct` as teacher and
`Qwen/Qwen2.5-7B` as base student for every attack and defense condition. This
includes defended extraction and result-based defenses. When SODA is the
extraction attack, the runner resolves its matched SeqKD initialization as an
internal training prerequisite; this is not a separate defense experiment or a
reader-facing replay protocol. The runner rejects an accidental Llama/Qwen
mixture. Adaptive/countermeasure experiments are a separate protocol and retain
their recorded Llama teacher/student pair.

### Result-based defenses and detectors

For DuFFin, MMD, PRADA, and SEAT, provide a completed attack manifest:

```bash
bash runs/run_defense.sh \
  --defense mmd \
  --attack-run outputs/attacks/seqkd/<timestamp>_seqkd_b1000/attack_manifest.json
```

The runner infers the attack, budget, query log, checkpoint, and other available
artifacts from the manifest. `--attack` or `--budget` may be supplied as
assertions, but conflicting values are an error.

Defended-extraction output:

```text
outputs/defenses/<defense>/<attack>/b<budget>/
  defense_run_manifest.json
  artifacts/
  detector/
  logs/
```

Result-based methods may omit checkpoint directories, but they must record the
upstream attack manifest and all consumed query/output hashes. Their output is:

```text
outputs/defenses/<defense>/<attack>/b<budget>/
  [<detector>/]detector_manifest.json
  [<detector>/]detector_report.json
  logs/
```

The optional extra detector directory is method-owned; use the manifest printed
by the completed command rather than constructing this path in another script.

For both defended-extraction and result-based runs, `--output-root /data/mea-runs`
places the selected defense under:

```text
/data/mea-runs/defenses/<defense>/<attack>/b<budget>/
```

The manifest to evaluate is `defense_run_manifest.json` for defended extraction
or the generated `detector_manifest.json` for DuFFin, MMD, PRADA, and SEAT.

## 3. Run an Adaptive Attack

Specify the complete experimental condition:

```bash
bash runs/run_adaptive_attack.sh \
  --adaptive-attack dipper \
  --defense adfp \
  --attack seqkd \
  --budget 1000 \
  --profile paper
```

The runner starts its defended oracle and response-rewriting stages itself and
prepares the required clean and defense-only baselines by default. The endpoint
options are only overrides for compatible services already running in the same
allocation. Use `--reuse-baselines` only when previously generated baselines are
present and should be required rather than prepared again.

The original adaptive orchestrator runs the selected attack against the defended
oracle with DIPPER or back-translation applied. It does not consume a completed
ordinary defense run. By default it prepares the clean and defense-only detector
baselines needed by the countermeasure experiment and reuses them only after
compatibility validation.

Expected output:

```text
outputs/adaptive/<adaptive_attack>/<defense>/<attack>/b<budget>/
  defense_run_manifest.json
  countermeasure/
  oracle/
  attack/
  detector/
  logs/
```

With `--output-root /data/mea-runs`, the same layout is rooted at
`/data/mea-runs/adaptive/`. The manifest passed to evaluation is:

```text
/data/mea-runs/adaptive/<adaptive_attack>/<defense>/<attack>/b<budget>/defense_run_manifest.json
```

## 4. Evaluate a Completed Run

Evaluation is independent of training and accepts any supported run manifest:

```bash
bash runs/run_evaluation.sh \
  --manifest outputs/attacks/seqkd/<timestamp>_seqkd_b1000/attack_manifest.json
```

`--manifest` is also the manual way to choose where evaluation reads its input.
The evaluator does not select a latest run or scan unrelated output directories.
After receiving the manifest, it automatically follows the checkpoint, report,
baseline, and transcript paths recorded in that manifest. For example, a run
written to a custom artifact root is evaluated with:

```bash
bash runs/run_evaluation.sh \
  --manifest /data/mea-runs/defenses/adfp/seqkd/b1000/defense_run_manifest.json
```

Useful optional arguments are:

```text
--metrics all
--limit 2
--output-root outputs/evaluation
--storage-root PATH
--dry-run
```

The evaluation `--output-root` controls only where evaluation metrics are
written; it does not change the source experiment. `--storage-root` is normally
unnecessary. Use it when a snapshot has moved between machines and its manifest
contains an old absolute path below `outputs/`:

```bash
bash runs/run_evaluation.sh \
  --manifest /data/copied-repo/outputs/attacks/seqkd/<run-id>/attack_manifest.json \
  --storage-root /data/copied-repo \
  --output-root /data/mea-evaluation
```

In this example, paths formerly recorded as `/old/machine/outputs/...` are
resolved as `/data/copied-repo/outputs/...`. If the paths stored in the manifest
already exist, no storage-root override is needed.

`--limit` is a smoke-test limit and is reflected in the resolved plan and output
location. A smoke result must never overwrite or be presented as a full benchmark
result.

The evaluator determines the source run type, discovers checkpoints from the
manifest, prepares the shared held-out prompts and teacher outputs when they are
missing, and dispatches to the corresponding evaluation driver. Detector-only
defense runs already produce their detector reports during `run_defense.sh`; the
evaluation command validates and collects those reports instead of rerunning the
detector.

Evaluation behavior depends on the selected manifest:

| Source manifest | Evaluation behavior |
|---|---|
| Attack | Loads that run's student checkpoint and computes the implemented attack metrics. M6 is marked unavailable when allocation records are not present. |
| Defended extraction | Builds or reuses the shared Qwen teacher/base reference, then evaluates the selected defended student. |
| Adaptive attack | Follows the comparison and baseline manifests, then evaluates the clean, defense-only, and selected adaptive checkpoints. |
| Result-based detector | Validates and collects the existing detector report; it does not rerun the detector. |

Evaluation artifacts are written under
`outputs/evaluation/<run_type>/<source_run_id>/`. Their internal filenames follow
the selected existing evaluator. A smoke run uses
`<source_run_id>_smoke_<limit>` so it cannot overwrite the full evaluation.

### Successful Completion

A zero command exit status is required. Before treating a run as complete, also
check the following machine-readable artifacts:

| Run | Completion check |
|---|---|
| Attack | `attack_manifest.json` contains `result.status: "completed"` and `result.checkpoint_dir` points to an existing checkpoint. |
| Defended extraction | `defense_run_manifest.json` contains `status: "ok"`, `generator_result.status: "ok"`, and a local `generator_result.student_checkpoint_path`. |
| Adaptive attack | `defense_run_manifest.json` contains `status: "ok"`; its generator metadata records the countermeasure and comparison report. |
| Result-based detector | `detector_manifest.json` exists and its `output_report` points to the generated detector report. |
| Evaluation | The run directory contains its summary CSV/JSON; adaptive and detector evaluation additionally write `complete.json` with `complete: true`. |

The final console output prints the authoritative manifest or summary path. Use
that path rather than selecting a similarly named historical run.

## Data

The extraction query pools are published at:

```text
https://huggingface.co/datasets/watermarkproject/lord-mea-benchmark
```

`--query-pool auto` selects the exact file for the requested budget and caches it
under `cache/mea_benchmark/hf_datasets/` when using the canonical runners. The
available files are:

```text
query_pool_100.json
query_pool_1000.json
query_pool_10000.json
query_pool_50000.json
query_pool_100000.json
```

The 100- and 1,000-record tiers are deterministic nested subsets by record ID,
not literal prefixes of the larger JSON files. Reproduction therefore uses the
exact published tier rather than slicing another file.

## Dependency and Resume Rules

Every runner follows the same rules:

1. The public command accepts experiment identity, not internal artifact paths.
2. Missing prerequisites are generated automatically in dependency order.
3. Existing artifacts are reused only after model, budget, query, configuration,
   and content-hash validation.
4. Ambiguous or incompatible artifacts are never selected silently.
5. A failed or partial run is never reported as completed.
6. `--dry-run` validates the experiment identity and displays its dispatch plan
   without model calls or training.

These rules preserve the experiment dependencies from the former Slurm
orchestrators while executing them directly in the caller's allocation.

## Manifest Contract

Attack planning uses `mea_experiment_manifest_v1`; completed attacks retain the
established `attack_manifest_v1` format with `run_config`, input hashes, and a
method-owned `result`. Defense and adaptive runs retain
`defense_run_manifest.json`; adaptive identity is recorded in its countermeasure
metadata and artifacts. Result-based defenses record the consumed attack
manifest or query log in their detector output.

Paths in newly published planning metadata should be relative to the repository
or configured output root whenever possible. Existing method manifests may use
runtime paths because checkpoints and transcripts are large local artifacts.
Published manifests must be checked before release and must not contain home
directories, cluster paths, usernames, access tokens, or endpoint credentials.

## Development Entry Points

The repository still contains method-level scripts under `attacks/scripts/`,
`defenses/scripts/`, `countermeasures/scripts/`, and the existing `runs/`
subdirectories. They remain useful for development and for checking parity with
earlier experiments, but they are not part of the public reproduction interface.

Reproduction instructions should use only the four canonical runners so model,
budget, seed, query-pool, generation, and output defaults remain consistent.

Runtime artifacts and caches are intentionally not tracked by Git:

```text
outputs/
.cache/
cache/
logs/
```

## License

Original MEA-Bench code and documentation are licensed under the
[MIT License](LICENSE). Copyright (c) 2026 MEA-Bench contributors.

Third-party code, dependencies, datasets, and model weights remain subject to
their respective upstream licenses and terms; the MIT License does not relicense
those materials. Preserve applicable upstream copyright and license notices.
