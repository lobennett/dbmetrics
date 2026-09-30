# Text computation

Paddle detects candidate text polygons on decoded frames. The overlay runtime
selects at most eight full frames: four temporal anchors plus OCR-rich frames.
Qwen receives the images, timestamps and a fixed prompt—not cropped OCR polygons.
Outputs distinguish added text/logos from physical scene text/logos.
The compact prompt additionally distinguishes captions, watermarks and other overlays.
Sampled absence is not proof that the entire clip is clear; scores are not calibrated confidence.

## OCR

Use a separate worker environment. The recorded CPU OCR environment used
`paddleocr==3.0.3`, `paddlex==3.0.3`, `paddlepaddle==3.0.0`, `numpy==1.26.4`.
GPU Paddle requires its corresponding platform-specific build; these CPU versions
do not establish GPU compatibility. FFmpeg must support `-vsync 0`.

```sh
python -m dbp_metrics.cli.ocr --media example.mp4 --clip-id example \
  --output-dir reports --model-dir /path/to/checkpoint \
  --model-source approved-source --model-revision approved-revision
```

The report contains media/model hashes, frame timestamps, polygons and completion status.

## Qwen

Reuse a compatible CUDA worker environment with Transformers 4.57.x, paired
PyTorch/torchvision, Pillow and safetensors. Supply an already-cached pinned snapshot.
The reviewed 8B model revision is `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`.

```python
import time
from dbp_metrics.evidence import EvidenceVideo
from dbp_metrics.overlay import OverlayRuntime, extract_evidence_frames, prepare_views
from dbp_metrics.captions import CaptionWatermarkRuntime

video = EvidenceVideo("example.mp4", "reports/report.jsonl")
deadline = time.monotonic() + 600
runtime = OverlayRuntime(snapshot, revision, "cuda:0", deadline, model_size="8b")
prediction = runtime.infer(video, video.clip_id, "ffmpeg", deadline)
```

For the separate compact protocol, instantiate `CaptionWatermarkRuntime` and call
`infer_images`, not its inherited `infer` method:

```python
runtime = CaptionWatermarkRuntime(snapshot, revision, "cuda:0", deadline)
frames, sampling = extract_evidence_frames(video, video.clip_id, "ffmpeg", deadline)
images, sampling = prepare_views(frames, sampling)
prediction = runtime.infer_images(images, sampling, deadline)
```

The caller supplies a deadline and saves the returned prediction. This package
does not reserve GPUs, schedule batches or import results into the website.
