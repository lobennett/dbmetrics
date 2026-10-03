# dbp-metrics

Offline video measurements, separate from the DBP website.
Existing CPU, OCR and Qwen computations are preserved; exploratory methods are separate.

## Use

Install FFmpeg (including `ffprobe`; the preserved pipeline requires a version accepting `-vsync 0`), then:

```sh
uv sync --extra technical
uv run python -c "from dbp_metrics import measure_video; print(measure_video('example.mp4'))"
DBP_TEST_FFMPEG=/path/to/compatible/ffmpeg uv run python -m unittest discover -s tests -v
```

`measure_video` returns 16 records with values, processing status, media hash and computational evidence.
Use `stages=['probe']` for duration, dimensions, frame rate and audio-track presence.
Other stages are `visual`, `audio`, `technical` and `ffmpeg_cuts`.
Failed/missing values remain unknown. Source files are never modified.
If your default FFmpeg has removed `-vsync`, pass a compatible binary via `ffmpeg='/path/to/ffmpeg'`; do not interpret decode failures as measurements.

## Notebooks

```sh
cd notebooks
uv sync
uv run jupyter lab
```

The metric walkthrough includes a demo MP4; the API walkthrough connects to a running DBP website.
See `VALIDATION.md` for evaluation requirements.

## Text and logos

`dbp_metrics.ocr` detects text regions using a supplied local Paddle checkpoint.
`dbp_metrics.evidence.EvidenceVideo` verifies an OCR report against its video.
`dbp_metrics.overlay.OverlayRuntime` uses OCR-guided images and the pinned Qwen model.
`dbp_metrics.captions.CaptionWatermarkRuntime.infer_images` uses the compact observation prompt.
See `TEXT.md` for dependencies and usage. Neither runtime downloads weights.

`dbp_metrics.experimental` contains half-video similarity, continuity diagnostics
and affect-rating CSV helpers. These are not automatically included in CPU results.
Website, review UI and cluster scheduling code remain outside this package.

Existing worker commands delegate through compatibility wrappers; see `INTEGRATION.md`.
Explore candidate metrics in `notebooks/exploratory-metrics.ipynb`.

## Source statistics

`dbp_metrics.source_statistics` defines source views, likes and comments. These
belong to the original YouTube video, not individual clips; join by `source_id`.
Missing counts stay null. Archived views have an unknown collection date.

```sh
uv sync --extra youtube
uv run python -m dbp_metrics.cli.youtube_sources --catalog /path/to/catalog.sqlite3 --output /path/to/source-statistics.jsonl --provider youtube --limit 10
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
