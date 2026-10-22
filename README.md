# dbmetrics

Offline video measurements, separate from the DBP website.
Existing CPU, OCR and Qwen computations are preserved; exploratory methods are separate.

## Use

Install FFmpeg and `ffprobe`. The preserved pipeline is tested with FFmpeg 4.2.2;
newer versions can change frame timing and diagnostics, not just CLI flags. Then:

```sh
uv sync --extra technical
uv run python -c "from dbmetrics import measure_video; print(measure_video('example.mp4'))"
DBP_TEST_FFMPEG=/path/to/compatible/ffmpeg uv run python -m unittest discover -s tests -v
```

`measure_video` returns 16 records with values, processing status, media hash and computational evidence.
Use `stages=['probe']` for duration, dimensions, frame rate and audio-track presence.
Other stages are `visual`, `audio`, `technical` and `ffmpeg_cuts`.
Failed/missing values remain unknown. Source files are never modified.
Pass the compatible binary via `ffmpeg='/path/to/ffmpeg'` or the CLI's `--ffmpeg`;
do not interpret decode failures as measurements. Verify a canary on your worker runtime.

## Notebooks

- `notebooks/website-metrics.ipynb`: existing computations.
- `notebooks/exploratory-metrics.ipynb`: candidate methods, not production metrics.
- `notebooks/api-walkthrough.ipynb`: website access and experiment workflow.

```sh
cd notebooks
uv sync
uv run jupyter lab
```

The metric walkthrough includes a demo MP4; the API walkthrough connects to a running DBP website.
See `VALIDATION.md` for evaluation requirements.

## Text and logos

`dbmetrics.ocr` detects text regions using a supplied local Paddle checkpoint.
`dbmetrics.evidence.EvidenceVideo` verifies an OCR report against its video.
`dbmetrics.overlay.OverlayRuntime` uses OCR-guided images and the pinned Qwen model.
`dbmetrics.captions.CaptionWatermarkRuntime.infer_images` uses the compact observation prompt.
See `TEXT.md` for dependencies and usage. Neither runtime downloads weights.

`dbmetrics.experimental` contains half-video similarity, continuity diagnostics
and affect-rating CSV helpers. These are not automatically included in CPU results.
Website, review UI and cluster scheduling code remain outside this package.

Existing worker commands delegate through compatibility wrappers; see `INTEGRATION.md`.
Explore candidate metrics in `notebooks/exploratory-metrics.ipynb`.

## Catalog → measurements → website

The catalog identifies clips; the MP4 files supply the pixels and audio.
Export a fixed selection from the catalog into a manifest:

```json
{"schema_version":"dbt1-evidence-manifest-v1","clips":[{"clip_id":"--42OoF8WJM_part_00000011","source_id":"--42OoF8WJM","part":11}]}
```

Run the package's existing batch interface on that selection:

```sh
uv run python -m dbmetrics.legacy \
  --manifest /data/manifest.json --media-root /data/videos \
  --media-kind rendition --stage probe \
  --output /data/results/probe.json --run-id probe-v1
```

Use separate output files and run IDs for `visual`, `audio`, `technical` and
`ffmpeg_cuts`. For renditions, media paths are the first two hex characters of
SHA-1(clip ID), followed by `<clip_id>.mp4`; originals are directly under the
media root. Do not assume archived catalog paths are current locations.

Each output record identifies the clip and metric, value or unknown, processing
status, media hash and evidence. Failed measurements are not zero. The website
imports these artifacts into its metric index; this package never rewrites the
catalog or website index. Manifest export, cluster scheduling and artifact import
remain outside this computation package. Start with a small canary, then shard
the manifest using `--offset` and `--limit` with distinct outputs.

For Python use, `measure_video(path, clip_id=...)` computes the sixteen CPU records
for one MP4; save its returned records with your run's provenance. OCR and Qwen
need their separately configured local checkpoints; they are not implicit CPU stages.

## Source statistics

`dbmetrics.source_statistics` defines source views, likes and comments. These
belong to the original YouTube video, not individual clips; join by `source_id`.
Missing counts stay null. Archived views have an unknown collection date.

```sh
uv sync --extra youtube
uv run python -m dbmetrics.cli.youtube_sources --catalog /path/to/catalog.sqlite3 --output /path/to/source-statistics.jsonl --provider youtube --limit 10
```

The metadata-only Python API downloads no media and requires no API key. Results
include retrieval time, upload date and extractor version. Re-running resumes
without duplicating source IDs; use `--all` for the remaining sources. Failed
lookups stay unknown and are retained; use a new output for a deliberate retry or
refresh. Access/rate-limit responses stop the run. For archived views only, use
`--provider catalog --all`. Source files and the website are unchanged.
Counts are extractor-reported and may be rounded. They describe source popularity,
not a participant's engagement; use the official YouTube Data API when exact
public counters are required.
