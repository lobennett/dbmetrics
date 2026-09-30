# Scope

This package computes offline metrics. It does not serve the website or submit cluster jobs.

- Follow `SCIENCE.md`: define, inspect, manually evaluate, then integrate.
- Work on one scientific experiment at a time. Do not add speculative website features.
- Preserve established formulas, sampling, missingness and evidence during extraction.
- Algorithm or model changes require separate evaluation; passing code tests is not scientific validation.
- Do not switch running jobs, alter shared environments, or launch large runs as part of refactoring.
- Keep weights, datasets, credentials, private reviews and generated metadata out of Git.
- The existing `notebooks/data/nsd_subj01.mp4` is the sole approved demo-media exception.
- Preserve exploratory notebook readability. Do not run the no-mistakes pipeline on these notebooks.
- Test with a compatible FFmpeg. Record the binary/version rather than changing formulas to hide failures.
