# CPU extraction

Source: `Stanford-Digital-Brain-Project/dbp-dataset-browser` local latest worktree, 2026-09-29.
CPU source files are unchanged from commit `4b8e1edb3363e233d95a20d684844fd0f310b95d`.
The audio, sampled RGB, technical-prefix and FFmpeg-cut computations are copied without formula changes.
Imports are package-relative. Shared bounded IO/probe functions are extracted from `pilot_extract.py`.
The public wrapper supplies a single file path and returns only requested CPU records.

Each stage runs in an isolated, timeout-bounded subprocess. Inputs are hashed and checked for changes;
symlink media are rejected. Outputs retain processing status, evidence and media identity.
Package provenance has its own code hash and does not impersonate existing production run fingerprints.

Both notebooks and the demo asset were copied from website commit
`14f73a895ad00a90fc1b0b76bef3880a296351df`, preserving their existing outputs.
The old copies are retained until the new repository is published and checked.

OCR/Qwen code is not extracted in this first batch. Experimental affect, embedding and review-page code is not promoted.

## Local verification

The 102-test suite includes exact original/package record comparisons on 50 real videos,
the bundled video, a constant-gray clip and a black/white-cut clip. Only runtime timeout metadata,
diagnostic text and the intentionally new package provenance are excluded from equality.
The same records are checked by the website's measurement-artifact validator.
This establishes extraction equivalence, not agreement with human scientific judgments.

The preserved computation commands were tested with FFmpeg 4.4. The explanatory notebook
uses modern `-fps_mode` and was executed separately with the system FFmpeg.

See `VERIFICATION.md` for the larger comparison and measured coverage.
