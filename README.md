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

## Source statistics

Source views, likes, and comments describe the original YouTube video. Join them
to clips by `source_id`; they are not measurements of clip engagement.

```sh
uv sync --extra youtube
uv run python -m dbp_metrics.cli.youtube_sources --catalog /data/catalog.sqlite3 --provider youtube --output /data/source-statistics.jsonl --limit 10
```

This uses yt-dlp's Python API without downloading media or requiring an API key.
It records retrieval times; counts may be rounded and unavailable counts remain
null. Use `--provider catalog` for archived views, whose collection date is unknown.
After inspecting the pilot, use `--all` for all remaining sources (default: two
seconds between requests). Rerunning resumes the same output, skipping recorded
sources, including failures. Use a new output for retries or refreshed counts.
The collector stops on access/rate-limit responses. Neither the original catalog
nor the website is modified. Freeze a copy before importing it into the website.

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
Start similarity exploration with `notebooks/half-video-similarity.ipynb`.
