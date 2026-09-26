# Attack runs

Use the portable top-level runner for one method and budget:

```bash
bash runs/run_attack.sh --attack seqkd --budget 1000 --profile paper
```

This wrapper validates the paper profile and delegates to the retained
`attacks/scripts/run_attacks.sh` implementation. The default query pools are
benchmark datasets hosted on Hugging Face. Generated transcripts, checkpoints,
and manifests stay local; this snapshot does not upload run artifacts.

The scripts under `attacks/scripts/` remain internal controls for staged or
service-separated execution. They are useful when an infrastructure operator
needs separate endpoint and training allocations, but they are not the public
reproduction interface.

If a SODA run already produced `student_negatives.jsonl`, reuse it instead of
querying the student again:

```bash
ATTACK=soda \
SHARED_TRANSCRIPT_DIR=/path/to/shared_transcript \
SODA_STUDENT_NEGATIVES_JSONL=/path/to/student_negatives.jsonl \
bash attacks/scripts/prepare_attack_data.sh
```

The preparation script compares prompt IDs. Complete negatives skip the student
endpoint; incomplete negatives resume only the missing records. Empty teacher or
student responses are excluded and recorded in `preference_stats.json`.
