# Evaluation runs

Use the top-level entry point for every public evaluation:

```bash
bash runs/run_evaluation.sh \
  --manifest outputs/attacks/seqkd/<run-id>/attack_manifest.json
```

The manifest may belong to an attack, defended extraction run, adaptive attack,
or result-based detector. The entry point determines the run type and evaluates
only that selected run.

It automatically:

- resolves checkpoints and reports recorded by the manifest;
- remaps historical absolute paths below `outputs/` to the current local
  `STORAGE_ROOT`;
- builds the held-out prompt split when it is absent;
- generates and validates held-out teacher outputs when required;
- builds the shared Qwen reference evaluation for a defended extraction run;
- follows adaptive baseline manifests to their local attack checkpoints; and
- validates and collects an existing result-based detector report.

Useful optional arguments are:

```text
--limit N
--output-root outputs/evaluation
--storage-root PATH
--dry-run
```

`--limit` creates a smoke evaluation. The lower-level scripts in this directory
are implementation drivers used by the top-level command; their environment
variables and historical bundle layouts are not part of the public reproduction
interface.

`--manifest` selects the source run and is the only input location normally
needed. `--output-root` selects where metrics are written. If a copied snapshot
contains obsolete absolute paths below `outputs/`, pass the snapshot's current
artifact root through `--storage-root`; the evaluator remaps those recorded paths
without requiring checkpoint arguments.

Hugging Face is used only for benchmark datasets and model downloads. Evaluation
results, checkpoints, generated teacher references, and detector reports are
read from or written to local storage.
