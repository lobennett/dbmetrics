"""Measure a manifest shard into per-video research sidecars; no paid services."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile


from dbp_metrics.experimental.half_similarity import Config as SimilarityConfig, VJEPABackend, measure as measure_similarity, media_hash, measure_scene_window
from dbp_metrics.experimental.scene_continuity import Config as SceneConfig, measure as measure_scene


def complete(document, vjepa, scene_windows):
    scene = document.get("scene_continuity")
    windows = document.get("scene_windows")
    if (not isinstance(scene, dict) or scene.get("status") not in ("ok", "insufficient_evidence")
            or not isinstance(scene.get("coverage"), dict)
            or scene["coverage"].get("kind") not in ("full", "prefix")
            or not isinstance(scene.get("trace"), list) or not isinstance(scene.get("candidates"), list)
            or not isinstance(windows, list) or len(windows) != min(scene_windows, len(scene["candidates"]))):
        return False
    fields = ("visual_histogram_cosine", "vjepa_cosine") if vjepa else ("visual_histogram_cosine",)
    for result in [document.get("half_similarity"), *windows]:
        if not isinstance(result, dict) or any(
                not isinstance(result.get(field), dict) or result[field].get("status") != "ok"
                for field in fields):
            return False
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="JSONL: clip_id, media_path")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--vjepa", action="store_true", help="Requires cached pinned weights and optional dependencies")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--scene-windows", type=int, default=0, help="Compare up to this many detected boundaries per clip (0–10)")
    args = parser.parse_args(argv)
    if not 0 <= args.shard_index < args.shard_count or not 0 <= args.scene_windows <= 10:
        parser.error("Invalid shard or scene-window bounds")
    args.output_root.mkdir(parents=True, exist_ok=True)
    config = SimilarityConfig(vjepa=args.vjepa, device=args.device)
    backend = VJEPABackend(args.device) if args.vjepa else None
    code_root = Path(__file__).resolve().parents[1]
    policy = dict(vjepa=args.vjepa, device=args.device, scene_windows=args.scene_windows, revision="research-pilot-v1",
                  code_sha256={name: media_hash(code_root / name) for name in
                               ("experimental/scene_continuity.py", "experimental/half_similarity.py", "cli/research.py")})
    seen = set()
    failed = 0
    for line in args.manifest.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("Manifest rows must be objects")
        clip_id = row.get("clip_id")
        if type(clip_id) is not str or not re.fullmatch(r"[A-Za-z0-9_-]+_part_\d{8}", clip_id) or clip_id in seen:
            raise ValueError("Manifest requires unique normalized clip IDs")
        seen.add(clip_id)
        if int(hashlib.sha256(clip_id.encode()).hexdigest(), 16) % args.shard_count != args.shard_index:
            continue
        try:
            media = Path(row.get("media_path"))
            if not media.is_absolute():
                media = args.manifest.parent / media
            digest = media_hash(media)
            target = args.output_root / f"{clip_id}.json"
            if target.exists():
                saved = json.loads(target.read_text())
                if (not isinstance(saved, dict) or saved.get("schema_version") != "dbp-research-pilot-v1"
                        or saved.get("clip_id") != clip_id or saved.get("media_sha256") != digest
                        or saved.get("policy") != policy):
                    raise ValueError("Existing result has different identity, media or policy; choose a new output directory")
                if not complete(saved, args.vjepa, args.scene_windows):
                    raise ValueError("Existing result is incomplete; address the failure and use a new output directory")
                print(json.dumps(dict(clip_id=clip_id, status="already_recorded")), flush=True)
                continue
            scene = measure_scene(media, SceneConfig())
            similarity = measure_similarity(media, config, backend=backend)
            windows = [measure_scene_window(media, candidate["timestamp_seconds"] - scene["coverage"]["first_timestamp_seconds"], config=config, backend=backend)
                       for candidate in scene["candidates"][:args.scene_windows]]
            if media_hash(media) != digest:
                raise ValueError("Media changed while computing diagnostics")
            document = dict(schema_version="dbp-research-pilot-v1", clip_id=clip_id, media_sha256=digest,
                            policy=policy, scene_continuity=scene, half_similarity=similarity, scene_windows=windows)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", dir=args.output_root, delete=False) as output:
                    temporary = Path(output.name)
                    json.dump(document, output, allow_nan=False)
                    output.flush()
                    os.fsync(output.fileno())
                os.link(temporary, target)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            print(json.dumps(dict(clip_id=clip_id, status="recorded", vjepa=similarity["vjepa_cosine"]["status"])), flush=True)
            if not complete(document, args.vjepa, args.scene_windows):
                failed += 1
        except (OSError, ValueError, TypeError, TimeoutError) as error:
            failed += 1
            print(json.dumps(dict(clip_id=clip_id, status="error", error=str(error))), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
