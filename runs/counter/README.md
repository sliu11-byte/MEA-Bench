# Countermeasure runs

Run an adaptive attack through the portable top-level entry point:

```bash
bash runs/run_adaptive_attack.sh \
  --adaptive-attack dipper --defense ginsew --attack seqkd --budget 1000
```

Supported countermeasures are `dipper` and `translation`. Supported detector
methods are `ginsew`, `radioactivity`, and `adfp`. The wrapper delegates to the
retained `runs/counter/common.sh` implementation. Outputs remain local; this
snapshot contains no artifact uploader or scheduler wrapper.
