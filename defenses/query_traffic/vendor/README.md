# Vendored Query-Traffic Detectors

This directory vendors the self-contained MMD, PRADA, and SEAT detector entry
scripts from the local `model_extraction_detection` reference implementation.
The benchmark wrapper `python -m defenses.query_traffic.run` uses these files by
default so HPG runs do not depend on a developer-local path such as
`/home/anonymous/projects/model_extraction_detection`.

The wrapper still accepts `--source_repo` or `MODEL_EXTRACTION_DETECTION_REPO`
for comparison against an external checkout.
