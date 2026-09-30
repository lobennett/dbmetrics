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
