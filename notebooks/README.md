# Metric computation walkthrough

`source-statistics.ipynb` demonstrates archived source counts, an optional live
YouTube lookup, and how clips inherit those values. Set `DBP_CATALOG` to a local
catalog path. For live retrieval, install the root package's `youtube` extra in
the notebook environment: `uv pip install --python .venv/bin/python -e '..[youtube]'`.

This notebook follows one bundled MP4 from decoded frames to the metrics used by the DBP browser. It keeps the calculations explicit so each value can be inspected rather than treating the production pipeline as a black box.

## Requirements

- Python 3.11–3.12
- [`uv`](https://docs.astral.sh/uv/)
- `ffmpeg` and `ffprobe`

Install FFmpeg with `brew install ffmpeg` on macOS or `sudo apt install ffmpeg` on Ubuntu.

## Run

```bash
cd notebooks
uv sync
uv run jupyter lab metric-computation-walkthrough.ipynb
```

To execute every cell without opening Jupyter:

```bash
uv run jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  metric-computation-walkthrough.ipynb
```

The notebook uses `data/nsd_subj01.mp4`, so no path edits are required. CPU metrics run locally. The final text-label section explains the OCR/Qwen pipeline and uses reviewed demonstration labels; it does not download model weights or run the production VLM.

`api-interface-walkthrough.ipynb` connects to a running DBP website. The environment includes a pinned API client and the local metrics package. Run its cells individually: creating experiments and tracking trials changes account data.

In VS Code, select `notebooks/.venv/bin/python` as the kernel.

`half-video-similarity.ipynb` shows the sampled halves, RGB histogram comparison
and an optional pinned V-JEPA comparison. The default run is local and downloads nothing.
