# Shared held-out prompts

`heldout_prompts.jsonl` contains 3,000 evaluation prompts (~1.37 MiB), tracked as
ordinary Git text. Attack and defense evaluation use this file by default.
Teacher responses and generated student outputs remain under STORAGE_ROOT.

The file was reconstructed from the cached public query pool using
`evaluation.scripts.build_heldout_queries`: indices 10000 through 12999, blocks
10, 11, 12. The source SHA-256 and content hash are in `heldout_manifest.json`.
The sorted prompt-ID hash matches completed teacher job 42187817:
`abf6ff15bebba52d9253e1f1a4f6a2be38889c924bec495b5191d0f9f377b73a`.
ID equality alone does not establish prompt-text equality with the cluster copy;
attack evaluation checks every prompt text against its teacher reference.

Source data: https://huggingface.co/datasets/anonymous-mea-benchmark/mea-query-pools
This redistribution does not override the original source datasets' terms.
Per-row bank/source/split fields preserve source attribution.

To rebuild from the same source revision, first obtain a query pool whose SHA-256
matches the manifest, then pass its local path to the builder with
`--output-dir evaluation/data/heldout`. Do not overwrite this shared file from
a changed upstream pool without validating and versioning the new protocol.
