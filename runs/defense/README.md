# Defense runs

Use the portable top-level runner for defended extraction:

```bash
bash runs/run_defense.sh --defense ginsew --attack seqkd --budget 1000
```

DuFFin, MMD, PRADA, and SEAT instead consume a completed attack manifest:

```bash
bash runs/run_defense.sh --defense mmd --attack-run /path/to/attack_manifest.json
```

The top-level wrapper dispatches to the retained `runs/defense/common.sh` and
`runs/defense/detector_common.sh` implementations. Scripts under
`defenses/scripts/` are development and smoke-test controls.

When SODA is selected as the extraction attack, the runner automatically
generates the protected responses, trains or reuses the matched SeqKD
initialization, prepares the preference data, and then trains SODA. No separate
artifact-preparation command is required.

Query-traffic runs accept a local attack output directory through
`ATTACK_OUTPUT_SOURCE`. Benchmark query pools may still be read from Hugging
Face because they are experimental datasets rather than generated run results.
