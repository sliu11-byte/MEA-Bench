# Defense runs

Defense experiments use the ordinary shell entry points under
`defenses/scripts/seqkd/` and `defenses/scripts/seqkd_smoke/`.

```bash
STORAGE_ROOT=/path/to/storage METHOD=ginsew \
bash defenses/scripts/seqkd/run_seqkd.sh
```

SODA replay inputs are local-only. Set `SODA_SEQKD_LOCAL_ROOT`,
`SODA_STUDENT_NEGATIVES_JSONL`, `SODA_CLEAN_SEQKD_CHECKPOINT`, or
`SODA_CLEAN_PREFERENCES` when the defaults under `STORAGE_ROOT` do not match
your layout.

Query-traffic runs accept a local attack output directory through
`ATTACK_OUTPUT_SOURCE`. Benchmark query pools may still be read from Hugging
Face because they are experimental datasets rather than generated run results.
