#!/usr/bin/env python3
"""Validate or benchmark one local clip; optional dependencies are never installed."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


from dbp_metrics._ocr import Limits, MODEL, package_versions, resolve_ffmpeg, run_benchmark, validate_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media", required=True, help="local MOV/MP4 file, at most 512 MiB")
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--output-dir", help="private 0700 directory; one new 0600 JSONL per invocation")
    parser.add_argument("--dry-run", action="store_true", help="validate only; no imports, decoding, hashing or writes")
    parser.add_argument("--detector", choices=("paddle", "decode-only"), default="paddle")
    parser.add_argument("--model-dir", help=f"explicit pre-provisioned {MODEL} static inference directory")
    parser.add_argument("--model-source", help="operator-supplied checkpoint origin; never fetched")
    parser.add_argument("--model-revision", help="operator-supplied immutable revision or release identifier")
    parser.add_argument("--device", default="cpu", help="cpu or one explicit gpu:N; never auto-select")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="existing local executable; never downloaded")
    parser.add_argument("--max-frames", type=int, default=600, help="decoded frame budget, 1..1800; one extra EOF sentinel")
    parser.add_argument("--max-seconds", type=float, default=15, help="relative presentation window (0,60], exclusive end")
    parser.add_argument("--timeout", type=float, default=120, help="whole worker wall deadline (0,600] seconds")
    parser.add_argument("--threads", type=int, default=1, help="CPU/FFmpeg thread controls, 1..4")
    parser.add_argument("--every-nth", type=int, default=1, help="1=dense; larger values are explicitly sampled")
    parser.add_argument("--max-pixels", type=int, default=8_847_360, help="per decoded frame pixel ceiling, at most 8847360")
    parser.add_argument("--detector-side", type=int, default=960, help="Paddle maximum input side, at most 1920; no crop")
    arguments = parser.parse_args(argv)
    try:
        limits = Limits(**{key: getattr(arguments, key) for key in asdict(Limits())})
        binary = resolve_ffmpeg(arguments.ffmpeg)
        config = {"media": str(Path(arguments.media).expanduser().absolute()), "clip_id": arguments.clip_id,
                  "detector": arguments.detector, "device": arguments.device,
                  "ffmpeg": binary or arguments.ffmpeg, "limits": asdict(limits),
                  "model_dir": str(Path(arguments.model_dir).expanduser().absolute()) if arguments.model_dir else None,
                  "model_source": arguments.model_source, "model_revision": arguments.model_revision}
        validate_config(config, dry_run=arguments.dry_run)
        if arguments.dry_run:
            versions = package_versions()
            ready = bool(binary and config["model_dir"] and versions["paddleocr"] and
                         (versions["paddlepaddle"] or versions["paddlepaddle-gpu"]))
            print(json.dumps({"status": "dry_run", "inference_executed": False,
                              "ready_for_inference": ready, "readiness_is_dependency_metadata_only": True,
                              "ffmpeg_available": bool(binary), "packages": versions, "config": config}))
            return 0
        if not arguments.output_dir:
            raise ValueError("--output-dir is required for a benchmark")
        if not binary:
            raise ValueError("FFmpeg unavailable; supply --ffmpeg /existing/path")
        result = run_benchmark(config, arguments.output_dir)
        print(json.dumps(result, allow_nan=False))
        return 0 if result["processing_status"] == "complete" else 3
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
