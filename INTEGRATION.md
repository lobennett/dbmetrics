# Worker integration

The website reads metric artifacts; worker processes perform computation.
Existing command names are compatibility wrappers around `dbmetrics`.
The package preserves legacy artifact schemas and worker protocols.

Install the pinned package in each **new worker environment**, not a shared
environment serving existing jobs. Record the package's source hashes with the
run configuration. Package changes require a new output directory/run identity;
do not resume an old run with different computation code.

CPU: `dbmetrics.legacy` preserves the manifest/worker CLI and uses package CPU modules.
GPU: `dbmetrics.batch` retains the supervised worker protocol; `_vlm` contains
legacy inference and uniform decoded-frame selection. Both Qwen protocols share
the local snapshot verifier. OCR uses `dbmetrics._ocr`.

Artifact import, study selection, review UI, media serving and cluster submission
remain application responsibilities. No active remote snapshot is rewritten.

New CPU run configurations include the complete package code hashes. GPU workers
check the same hashes before loading weights. The separated controller preflights
its GPU and OCR interpreters; a missing or mismatched package stops the run.
The website dependency is pinned to a Git commit, not a moving branch.

Next: a small canary in each new worker environment, then expand the new queues.
The similarity notebook is a separate exploration; it is not a bulk job or website feature.
