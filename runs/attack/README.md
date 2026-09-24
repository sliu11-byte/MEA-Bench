# Attack runs

Attack experiments are launched directly through `attacks/scripts/run_attacks.sh`.
Select one method with `ATTACK_INDEX` or provide a space-separated `ATTACKS` list.

```bash
STORAGE_ROOT=/path/to/storage \
ATTACKS="seqkd lord soda qedks model_leeching gad" \
bash attacks/scripts/run_attacks.sh
```

The default query pools are benchmark datasets hosted on Hugging Face. Generated
transcripts, checkpoints, and manifests stay under `STORAGE_ROOT`; this snapshot
does not upload run artifacts.

For staged execution, use `prepare_attack_data.sh`, `train_prepared_attack.sh`,
`serve_teacher.sh`, and `serve_student.sh` directly.
