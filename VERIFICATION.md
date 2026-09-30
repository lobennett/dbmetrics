# Extraction verification — 2026-09-29

**102 tests passed.** All 848 measurement records matched the original implementation
across 53 videos: 50 locally staged dataset clips, the bundled demo, and two controlled
fixtures (constant gray and a black/white cut). The website's artifact validator accepted every set.

The real clips were selected by stable SHA-256 ordering of filenames, deduplicated by video ID.
This is a reproducible regression sample, not a population accuracy benchmark.
Values, processing status, media hashes and deterministic evidence were compared exactly.
Only execution timeouts, worker diagnostics and intentionally new package provenance were excluded.

| Stage | Completed values / requested records |
| --- | --- |
| Probe | 265 / 265 |
| Sampled visual | 208 / 212 |
| Audio | 100 / 106 |
| Technical prefix | 204 / 212 |
| Cut candidates | 53 / 53 |
| Total | 830 / 848 |

The 18 unknown values matched the original; they were not converted into zeros or negative labels.
Tests used Python 3.12, NumPy 2.5.3, OpenCV 4.14.0.94 and FFmpeg 4.4.
The package wheel/source distribution built, and the metric notebook executed separately
with modern system FFmpeg. API notebook structure/imports were checked without modifying account data.

## Repeat

```sh
uv sync --extra technical
DBP_ORIGINAL_CPU_ROOT=/path/to/dbp-dataset-browser \
DBP_EQUIVALENCE_MEDIA_ROOT=/path/to/videos \
DBP_EQUIVALENCE_LIMIT=50 \
DBP_TEST_FFMPEG=/path/to/compatible/ffmpeg \
uv run python -m unittest discover -s tests -v
```

This verifies faithful CPU extraction—not scientific validity of a proxy or OCR/VLM accuracy.
No production jobs or website imports have been switched to this package.

## OCR/Qwen and exploratory extraction

The expanded suite ran 233 tests: 232 passed; the optional original-CPU comparison
was skipped in this run (its previous 53-video comparison is recorded above).
Tests include real FFmpeg decoding, report transport, response parsing, sampling,
mocked generation, deadlines and experimental helper/CLI behavior.
Pillow crop tests ran in the notebook environment; no GPU model was loaded.

Twenty existing real OCR reports were accepted against their hash-matched media,
and their selected frame sets were bounded to eight frames.
The OCR, half-similarity, continuity and affect modules are byte-identical to their
original implementations. Every function/class in both Qwen modules has an
identical AST after removing imports; only module paths changed. Frozen prompt
hashes and model revisions are tested.

This port has not rerun Paddle or Qwen inference on a GPU. It does not provide a
new classification-accuracy estimate or promote experimental methods to validated ones.

## Worker integration, 0.3.0

330 package tests ran: 329 passed; the optional original-CPU comparison was skipped.
The run included website artifact validators via an explicitly supplied website
test tree. Another 129 website extraction/scoring/research tests passed with
the compatibility wrappers and installed package. OCR adapter tests passed (25),
caption batch tests passed (23), persistent worker tests passed (5), controller
tests passed (6), and evaluation adapter tests passed (20).

The new similarity notebook executed on the bundled MP4: both halves decoded,
sampled frames displayed and the RGB histogram cosine computed. V-JEPA was
disabled; no weights were downloaded. The executed notebook was saved outside Git.
No cluster job, live deployment or GPU inference was started or changed.
