# Countermeasure runs

Run a countermeasure through the shared SeqKD shell entry point:

```bash
STORAGE_ROOT=/path/to/storage \
COUNTERMEASURE=dipper METHOD=ginsew BUDGET=1000 \
bash countermeasures/scripts/seqkd/run_seqkd.sh
```

Supported countermeasures are `dipper` and `translation`. Supported detector
methods are `ginsew`, `radioactivity`, and `adfp`. Outputs remain local under
`STORAGE_ROOT`; this snapshot contains no artifact uploader or scheduler wrapper.
