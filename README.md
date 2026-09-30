# dbp-metrics

Offline video measurements, separate from the DBP website.
This first extraction preserves existing CPU algorithms; it does not establish scientific validity.

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
New scientific methods belong in small exploratory notebooks first, then independent manual evaluation,
then tested package code. Website integration follows review—not notebook execution.

OCR/VLM extraction is the next batch. Running cluster jobs and website imports are unchanged.
