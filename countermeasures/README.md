# Countermeasures

This directory contains post-defense countermeasure modules. The current
WaterPark integration keeps only response-only, defense-agnostic attacks:

- `dipper`: DIPPER paraphrasing with MEA default `lex=40, order=0`.
- `translation`: English -> French -> English back-translation.

Both methods consume a defended teacher transcript and emit a cleaned teacher
transcript that can be passed to the existing attack pipeline with
`--teacher-transcript-path`.

```bash
python3 -m countermeasures.waterpark_response \
  --method dipper \
  --input outputs/defenses/adfp/<run_id>/transcripts/defended_teacher.jsonl \
  --output outputs/countermeasures/waterpark_dipper/<run_id>/transcripts/cleaned_teacher.jsonl
```

```bash
python3 -m countermeasures.waterpark_response \
  --method translation \
  --input outputs/defenses/adfp/<run_id>/transcripts/defended_teacher.jsonl \
  --output outputs/countermeasures/waterpark_translation/<run_id>/transcripts/cleaned_teacher.jsonl
```

The output rows preserve the original metadata and set:

- `source_response`: the defended teacher response before the countermeasure.
- `response` / `teacher_response`: the rewritten response used for student training.
- `countermeasure`: `waterpark_dipper` or `waterpark_translation`.

These modules intentionally do not include WaterPark copy-paste/text mixing,
because that attack requires a clean/no-watermark output that is unavailable to
the attacker under the defended-API threat model.

## Method Status

We currently track five candidate countermeasures. Only the first two are part
of the main black-box benchmark because they operate directly on the defended
response available to the attacker.

| Method | Target defense family | Status | Reason |
|---|---|---|---|
| DIPPER paraphrasing | Text watermark defenses | Implemented | Response-only rewriting; no clean teacher access or defense secret is required. |
| Back-translation | Text watermark defenses | Implemented | Response-only rewriting; no clean teacher access or defense secret is required. |
| Copy-paste / text mixing | Text watermark defenses | Excluded from main benchmark | Requires clean/no-watermark text to mix with defended text, which conflicts with the black-box defended-API threat model. |
| The Distillation Game `strategic_fd` | Anti-distillation defenses | Not included in main benchmark | Official method relies on clean/standard holdout teacher traces to compute the useful learning direction, which gives the attacker clean-teacher prior access. |
| Trace Rewriting `Paraphrased` adaptive attack | Trace Rewriting anti-distillation defense | Negative result only | The paper reports it as a failed adaptive attack: paraphrasing OPT traces does not recover student performance and can amplify degradation. |

### DIPPER Paraphrasing

DIPPER is a response-level paraphrasing countermeasure. It takes each defended
teacher response as input and emits a paraphrased response for student training.
Our implementation uses `kalpeshk2011/dipper-paraphraser-xxl` with the MEA
default `lex=40, order=0`. This is included because the attacker only needs the
responses it already collected from the defended API.

### Back-Translation

Back-translation rewrites a defended response by translating it through an
intermediate language and then translating it back to English. Our implementation
uses English -> French -> English via `facebook/seamless-m4t-v2-large`. Like
DIPPER, this is included because it is response-only and defense-agnostic.

### Copy-Paste / Text Mixing

WaterPark also evaluates copy-paste or text-mixing attacks, but these require a
clean/no-watermark text source to splice into the defended response. Under our
black-box setting, the attacker only queries the defended API and does not know
which responses are clean or watermarked. If the attacker already had clean
teacher responses, those responses would themselves be a stronger training
source, so this method is excluded from the main benchmark.

### The Distillation Game `strategic_fd`

The Distillation Game proposes an adaptive weighted-SFT attack. It computes a
student/proxy gradient on clean standard holdout teacher traces, estimates which
defended training traces align with that useful learning direction, and then
trains with per-example weights. The clean holdout traces are essential to the
method: replacing them with defended holdout traces would make the reference
direction already distorted by the defense. Because this clean-teacher access
conflicts with our no-clean-teacher black-box threat model, we do not include it
in the main countermeasure runs. It can still be treated as a separate
strong-prior/oracle-assisted experiment if we later want that comparison.

### Trace Rewriting `Paraphrased` Adaptive Attack

The Trace Rewriting paper evaluates an adaptive attacker that paraphrases the
defense's optimized traces before fine-tuning. This is not an effective
countermeasure in the paper: the reported result is that paraphrasing fails to
restore distillation performance and may further damage the structure of the
reasoning traces. Therefore we record it as a negative robustness example rather
than a main benchmark countermeasure. The public Trace Rewriting code also does
not include a standalone implementation of this adaptive attack.

## Formal SeqKD Scripts

The countermeasure formal-run scripts mirror `defenses/scripts/seqkd/` under
`countermeasures/scripts/seqkd/`, but intentionally include only the three
watermark defenses: GINSEW, Radioactivity, and ADFP. They call the existing
defense runners with an online countermeasure proxy enabled, so online attacks
still receive one rewritten response at a time.

Run one defense-countermeasure pair locally:

```bash
COUNTERMEASURE=dipper \
COUNTERMEASURE_LEX=40 \
COUNTERMEASURE_ORDER=0 \
bash countermeasures/scripts/seqkd/run_ginsew_seqkd.sh
```

Run the three watermark defenses for both response-only countermeasures:

```bash
COUNTERMEASURES="dipper translation" \
METHODS="ginsew radioactivity adfp" \
bash countermeasures/scripts/seqkd/run_all_seqkd.sh
```


```bash
COUNTERMEASURES="dipper translation" \
METHODS="ginsew radioactivity adfp" \
bash countermeasures/scripts/seqkd/submit_seqkd_array.sh
```

ADS, DOGe, and Trace Rewriting are anti-distillation defenses, not text-watermark defenses, so they are not part of this WaterPark countermeasure script set.

Default output roots are separated from the clean defense runs:

```text
outputs/countermeasures/seqkd_b<BUDGET>/<COUNTERMEASURE>/<defense>/
logs/countermeasures/seqkd_b<BUDGET>/<COUNTERMEASURE>/<defense>/
```

## Online Runner Integration

For attacks that need online teacher access, use the existing defense runners
with `--countermeasure`. The runner starts the defended teacher first, then
starts an OpenAI-compatible countermeasure proxy, and points the attack at that
proxy. Each teacher response is rewritten one by one before the attack sees it.

```bash
python3 -m defenses.ginsew.runner \
  --attack seqkd \
  --budget 1000 \
  --teacher-model Qwen/Qwen2.5-72B-Instruct \
  --student-model Qwen/Qwen2.5-7B \
  --defense-config '{"fraction":0.5,"strength":2.0,"freq":16,"eps":0.2}' \
  --countermeasure dipper \
  --countermeasure-lex 40 \
  --countermeasure-order 0
```

The same path works for online attacks such as `qedks` and `model_leeching`,
because they continue to call an OpenAI-compatible teacher endpoint.

```bash
python3 -m defenses.radioactivity.runner \
  --attack model_leeching \
  --budget 1000 \
  --teacher-model Qwen/Qwen2.5-72B-Instruct \
  --student-model Qwen/Qwen2.5-7B \
  --countermeasure translation
```

The formal SeqKD shell scripts expose the same switch through environment
variables:

```bash
COUNTERMEASURE=dipper \
COUNTERMEASURE_LEX=40 \
COUNTERMEASURE_ORDER=0 \
bash defenses/scripts/seqkd/run_ginsew_seqkd.sh
```

Runner outputs include:

- `countermeasure/waterpark_<method>/countermeasure_teacher_transcript.jsonl`
- `countermeasure/waterpark_<method>/countermeasure_manifest.json`
- `countermeasure/waterpark_<method>/countermeasure_proxy.log`
- `attack/<attack>/<run_id>/attack_manifest.json` with `run_config.extra.countermeasure_enabled=true`

## Full Pipeline

Use `countermeasures.pipeline` when the experiment should run the defended
teacher transcript, apply a response-only countermeasure, and then train the
attacker from the cleaned transcript in one command. The attack manifest records
`countermeasure_enabled: true` under `run_config.extra`.

```bash
python3 -m countermeasures.pipeline \
  --defense ginsew \
  --countermeasure dipper \
  --attack seqkd \
  --budget 1000 \
  --teacher-model Qwen/Qwen2.5-72B-Instruct \
  --student-model Qwen/Qwen2.5-7B \
  --defense-config '{"fraction":0.5,"strength":2.0,"freq":16,"eps":0.2}' \
  --lex 40 \
  --order 0
```

Back-translation uses the same wrapper:

```bash
python3 -m countermeasures.pipeline \
  --defense radioactivity \
  --countermeasure translation \
  --attack seqkd \
  --budget 1000 \
  --teacher-model Qwen/Qwen2.5-72B-Instruct \
  --student-model Qwen/Qwen2.5-7B
```

To reuse an already generated defended transcript, skip the defense stage:

```bash
python3 -m countermeasures.pipeline \
  --defended-transcript outputs/defenses/ginsew/<run_id>/transcripts/defended_teacher.jsonl \
  --countermeasure dipper \
  --attack seqkd \
  --budget 1000 \
  --query-pool auto \
  --student-model Qwen/Qwen2.5-7B
```

The online defense runners in `defenses/*/runner.py` query the defended teacher
live during attack training. This wrapper instead calls the transcript-producing
`defenses/*/run.py` scripts, because DIPPER and back-translation need a complete
response text before they can rewrite it.

