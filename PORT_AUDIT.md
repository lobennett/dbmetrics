# Computation ownership

| Component | Owner |
| --- | --- |
| Probe, sampled RGB/audio, optical flow, FFmpeg cut candidates | `dbmetrics` |
| Legacy manifest extractor, PySceneDetect adapter and sparse semantic VLM | `dbmetrics.legacy`, `_vlm`, `_semantics` |
| Dense Paddle OCR and OCR-guided image selection | `dbmetrics._ocr`, `overlay` |
| Compact caption/overlay classification and cached model validation | `dbmetrics.captions`, `_local_model` |
| Worker supervision, deadlines and checkpoint protocol | `dbmetrics.batch` |
| Classification confusion counts and exclusions | `dbmetrics.evaluation` |
| Half-video similarity, continuity and affect CSV helpers | `dbmetrics.experimental` |
| Offline research-sidecar CLI | `dbmetrics.cli.research` |
| Catalog selection, staging and metadata import/indexing | Website adapters |
| Review exports, fixed-cohort evaluation bindings and review UI | Review/application adapters |
| Media serving, authentication and study/trial tracking | Website |
| GPU availability, Slurm submission and fleet queues | Cluster controllers |

Website filenames remain as thin imports for compatibility. They contain no
CPU/OCR/VLM formulas. Experimental adapters also delegate to the package.
Historical remote run snapshots remain unchanged and can still contain the old code.

Before switching a cluster: install the pinned package in a new private worker
environment, run a small CPU/OCR/VLM canary, check artifact import, then submit
new runs. Do not install into another person's shared environment or resume an
old journal with new package hashes. No GPU canary has been executed for this port.
