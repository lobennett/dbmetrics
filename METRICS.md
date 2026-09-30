# CPU measurements

| Metric | Computation / units |
| --- | --- |
| Duration | Container duration, seconds (`ffprobe`). |
| Width / height | Encoded non-attached video dimensions, pixels. |
| Frame rate | Average frame-rate rational, frames/second. |
| Audio track present | Whether an audio stream exists; does not imply audible sound. |
| Brightness | Mean encoded Rec.709-weighted RGB luma / 255. |
| Contrast | Mean within-frame population luma standard deviation / 255. |
| Colorfulness | Hasler–Suesstrunk RGB opponent proxy / 255; not bounded to 1. |
| Temporal change | Mean absolute luma difference between adjacent sampled frames / 255. |
| Audio RMS | Root mean square decoded mono PCM amplitude / 32768; not loudness. |
| Audio silence fraction | Sample-weighted fraction in 20 ms windows with RMS ≤ 0.001. |
| RGB chroma | Mean RGB max-minus-min / 255. |
| Near-gray fraction | Fraction of pixels with RGB max-minus-min ≤ 2 levels. |
| Frame interval CV | Population standard deviation / mean of adjacent presentation intervals. |
| Optical flow | Mean Farneback magnitude, resized pixels per adjacent frame pair. |
| Hard-cut candidates | Count of FFmpeg scene scores ≥ 0.3; not ground-truth semantic cuts. |

Sampled visual metrics use at most 120 frames, ≥0.5 seconds apart, in the first 60 seconds,
area-resized to 160×160 encoded RGB. Technical metrics use up to 120 consecutive frames in
the first 10 seconds. These are prefix/sample measurements, not whole-video coverage.

Audio is mono, 16 kHz signed 16-bit PCM; clips exceeding the 60-second audio budget are unknown,
not silently represented by a prefix. Cuts require a complete bounded indexed decode.
Every result carries its actual sampling evidence and processing status.
