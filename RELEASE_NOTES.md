# Paper release preparation

The release is based on anonymous snapshot commit 740034f, retaining its portable
entry points, paper profiles, method implementations, dependency checks, and
resume behavior. The older development repository is used as the reference for
non-anonymous project identifiers; its code was not copied over the newer harness.

## Restored project information

- Added the paper title, author list, affiliations, and paper-source link.
- Restored the public query-pool dataset to watermarkproject/lord-mea-benchmark.
- Confirmed identical bytes for the 100, 1000, and 10000 query tiers.
- Confirmed matching Hugging Face LFS SHA-256 identifiers for the 50000 and 100000 tiers.
- Restored the held-out manifest's dataset URL; all hashes and records are retained.
- Kept the snapshot's credential redaction in held-out prompts.
- Replaced anonymous local-path examples with portable placeholders.
- Made the optional cluster log-sync utility require the caller's username and
  remote repository path instead of using anonymous account defaults.

## Remote setup

This is an independent local clone. The snapshot-source remote is retained for
traceability and its push URL is disabled. The official origin URL is
https://github.com/sliu11-byte/MEA-Bench.git.

## Validation

- Nine existing benchmark contract and local-run resolver tests pass.
- All four canonical runners successfully display their help.
- Shell syntax and Git whitespace checks pass.
- The default query-pool resolver now selects the non-anonymous dataset.
- Full GPU training and evaluation were not run during release preparation.
