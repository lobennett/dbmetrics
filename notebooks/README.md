# Notebooks

| Notebook | Purpose |
| --- | --- |
| [website-metrics.ipynb](website-metrics.ipynb) | Existing website metrics, step by step on one video. |
| [exploratory-metrics.ipynb](exploratory-metrics.ipynb) | Similarity, transitions, event changes, affect and interest before integration. |
| [api-walkthrough.ipynb](api-walkthrough.ipynb) | Sign in, inspect metrics, create assignments, download stimuli and track trials. |

## Run

Requires Python 3.11–3.12, `uv`, `ffmpeg` and `ffprobe`.

```sh
cd notebooks
uv sync
uv run jupyter lab
```

In VS Code, select `notebooks/.venv/bin/python`. Run cells individually: API creation and trial tracking change account data; previews do not. A running website and your account are required for the API notebook.

The website notebook uses bundled `data/nsd_subj01.mp4`; text labels are demonstrations, not inference. The exploratory notebook uses a local gardening clip if available; otherwise supply your own video. Helpers live in `src/dbmetrics/experimental/`. Keep generated data and private reviews out of Git.

## Optional V-JEPA

```sh
uv sync --group vjepa
uv run --group vjepa python -c 'from huggingface_hub import snapshot_download; from dbmetrics.experimental.half_similarity import MODEL_ID, MODEL_REVISION; snapshot_download(MODEL_ID, revision=MODEL_REVISION, allow_patterns=["*.json", "*.safetensors"])'
```

The explicit weight download is large. Restart the kernel, then set `RUN_VJEPA = True`. Other model runs also require cached weights and explicit opt-in. Exploratory results are not validated website metrics.
