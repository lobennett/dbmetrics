"""Opt-in Qwen local inference, called only inside the extractor's timed worker.

Reuse a separately approved ML environment with Transformers 4.57.x, compatible
PyTorch/torchvision builds, Pillow and safetensors. Qwen's required VideoProcessor
needs torchvision v2 even for image-only inputs. Preflight checks the official
torch/torchvision release pairing and importability; it is not a security audit.
This module installs/downloads nothing. Imports are lazy;
the web environment never needs ML dependencies. Only a pre-approved local HF
cache snapshot for an explicitly selected fixed identity is accepted, not a loose
checkpoint directory. Qwen2.5 remains the default; Qwen3 requires its approved pin.
Snapshot paths establish local revision metadata, not remote authenticity: the
operator must approve the cache's origin and keep it immutable during execution.
All files are fingerprinted; HF SHA-256 blob names are checked when present.

The isolated extractor loads once per clip; the supervised batch runner may reuse
a verified runtime on one explicitly selected logical CUDA index. Its parent kills
the process group at each deadline. No auto-sharding, CPU fallback or URL loading.
Eight or fewer ordered images carry actual, first-frame-relative decoded PTS as
text labels; this is sparse multi-image evidence, not native full-video coverage.
Private raw responses/provenance stay in the exclusive output outside Git.

Verified interfaces:
https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/qwen2_5_vl
https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/config.json
https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct
https://ffmpeg.org/ffmpeg-filters.html#showinfo
https://github.com/pytorch/vision#installation
"""

from fractions import Fraction
import math
import os
from pathlib import Path
import re

from .legacy import (MODEL_ID, SEMANTIC_FIELDS, bounded_number, hash_fd,
                           parse_vlm_response, remaining, result, run_bounded,
                           signature, strict_json)
from ._semantics import PROMPT


from ._local_model import (MODEL_VARIANTS, SUPPORTED_MODEL_IDS, CHAT_TEMPLATE,
    FRAME_SIDE, FRAME_BYTES, MAX_NEW_TOKENS, MAX_MODEL_BYTES, TEXT_SHAPE, VISION_SHAPE,
    QWEN3_MODEL_ID, QWEN3_REVISION, QWEN3_8B_MODEL_ID, QWEN3_8B_REVISION,
    METADATA_FILES, snapshot_path, snapshot_files, validate_device,
    validate_local_model, check_model_unchanged)


def sample_times(duration, frame_count):
    bounded_number(duration, 0.001, 600, "VLM clip duration")
    if type(frame_count) is not int or not 1 <= frame_count <= 8:
        raise ValueError("request 1 to 8 uniformly spaced samples")
    return [duration * (index + 0.5) / frame_count for index in range(frame_count)]


class InsufficientFrames(ValueError):
    pass


def parse_frames(pixels, diagnostics, requested, duration):
    if len(diagnostics) > 131072 or len(pixels) != len(requested) * FRAME_BYTES:
        raise ValueError("incomplete/oversized ffmpeg frame output")
    text = diagnostics.decode("utf-8", "strict")
    if re.search(r"\[(?:error|fatal|panic)\]", text, re.IGNORECASE):
        raise ValueError("ffmpeg reported a decode error")
    time_bases = re.findall(r"showinfo[^\n]*config in time_base:\s*([0-9]+/[0-9]+)", text)
    if len(time_bases) != 1:
        raise ValueError("missing or ambiguous showinfo time base")
    try:
        time_base = Fraction(time_bases[0])
    except ZeroDivisionError as error:
        raise ValueError("invalid showinfo time base") from error
    if time_base <= 0:
        raise ValueError("invalid showinfo time base")
    lines = [line for line in text.splitlines() if "showinfo" in line and re.search(r"\bn:\s*", line)]
    if len(lines) != len(requested):
        raise ValueError("decoded frames cannot be paired with actual PTS")
    frames = []
    for index, line in enumerate(lines):
        match = re.search(r"\bn:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:(\S+).*?fmt:(\S+).*?\bs:(\d+)x(\d+)", line)
        if not match or int(match.group(1)) != index or match.group(4, 5, 6) != ("rgb24", "448", "448"):
            raise ValueError("malformed or reordered showinfo frame")
        pts = int(match.group(2))
        actual = float(pts * time_base)
        reported = float(match.group(3))
        upper = requested[index + 1] if index + 1 < len(requested) else duration
        if (not math.isfinite(reported) or not math.isclose(actual, reported, rel_tol=1e-5, abs_tol=1e-6)
                or not requested[index] <= actual < upper):
            raise ValueError("missing/nonfinite/out-of-window sampled PTS")
        frames.append({"sample_index": index, "requested_seconds": requested[index], "actual_seconds": actual,
                       "pts": pts, "time_base": time_bases[0],
                       "rgb": pixels[index * FRAME_BYTES:(index + 1) * FRAME_BYTES]})
    return frames


def decode_frames(descriptor, duration, frame_count, ffmpeg, deadline):
    requested = sample_times(duration, frame_count)
    targets = [format(timestamp, ".12g") for timestamp in requested]
    crossings = "+".join(f"gte(t,{target})*lt(prev_selected_t,{target})" for target in targets)
    selection = f"if(isnan(prev_selected_t),gte(t,{targets[0]}),{crossings})"
    filters = (f"setpts=PTS-STARTPTS,select='{selection}',"
               "scale=w='min(448,iw)':h='min(448,ih)':force_original_aspect_ratio=decrease,"
               "pad=448:448:(ow-iw)/2:(oh-ih)/2,setsar=1,format=rgb24,showinfo")
    command = [str(ffmpeg), "-nostdin", "-hide_banner", "-nostats", "-loglevel", "level+info", "-xerror",
               "-max_alloc", "67108864", "-protocol_whitelist", "file,pipe", "-threads", "1",
               "-filter_threads", "1", "-err_detect", "explode", "-f", "mov", "-i", "/dev/fd/" + str(descriptor),
               "-map", "0:V:0", "-an", "-sn", "-dn", "-vf", filters, "-frames:v", str(frame_count),
               "-vsync", "0", "-threads", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
    os.lseek(descriptor, 0, os.SEEK_SET)
    pixels, diagnostics = run_bounded(command, deadline, frame_count * FRAME_BYTES + 131072,
                                      (descriptor,), process_group=False, capture_stderr=True)
    if (len(pixels) < frame_count * FRAME_BYTES and len(pixels) % FRAME_BYTES == 0
            and not re.search(rb"\[(?:error|fatal|panic)\]", diagnostics, re.IGNORECASE)):
        raise InsufficientFrames("not enough distinct decoded frames at requested sample positions")
    frames = parse_frames(pixels, diagnostics, requested, duration)
    return frames, {"ffmpeg_binary": str(ffmpeg), "filter": filters, "strategy": "uniform_bin_midpoints_next_frame",
                    "timestamp_origin": "first_decoded_frame_pts", "frame_side_px": FRAME_SIDE,
                    "samples": [{key: value for key, value in frame.items() if key != "rgb"} for frame in frames]}


class VLMRuntime:
    def __init__(self):
        self.identity = None
        self.verified = None
        self.processor = None
        self.model = None


def infer_clip(descriptor, probe, local_model_dir, revision, device, ffmpeg, deadline, frame_count=8, model_id=MODEL_ID,
               runtime=None, expected_file_sha256=None):
    validate_device(device)
    if runtime is None:
        runtime = VLMRuntime()
    if not isinstance(runtime, VLMRuntime):
        raise ValueError("invalid reusable VLM runtime")
    identity = (str(Path(local_model_dir).resolve()), revision, model_id, device,
                os.environ.get("CUDA_VISIBLE_DEVICES"))
    if runtime.identity is not None and runtime.identity != identity:
        raise ValueError("reusable model identity cannot change")
    verified = runtime.verified or validate_local_model(local_model_dir, revision, deadline, model_id=model_id)
    if expected_file_sha256 is not None and verified["file_sha256"] != expected_file_sha256:
        raise ValueError("model hashes differ from the batch identity")
    check_model_unchanged(verified)
    variant = MODEL_VARIANTS[verified["model_id"]]
    duration = probe["duration_seconds"]["value"]
    sample_times(duration, frame_count)
    width, height = probe["width_px"]["value"], probe["height_px"]["value"]
    if (type(width) is not int or type(height) is not int or not 0 < width <= 4096
            or not 0 < height <= 4096 or width * height > 8388608):
        raise ValueError("VLM input dimensions are missing or exceed the decode bound")
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                       "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false"})
    import torch
    import transformers
    from PIL import Image
    try:
        import torchvision
        from torchvision.transforms.v2 import functional

        torch_series = ".".join(torch.__version__.split(".")[:2])
        vision_series = ".".join(torchvision.__version__.split(".")[:2])
        compatible = {"2.6": "0.21", "2.7": "0.22", "2.8": "0.23", "2.9": "0.24",
                      "2.10": "0.25", "2.11": "0.26", "2.12": "0.27", "2.13": "0.28"}
        if compatible.get(torch_series) != vision_series or not callable(functional.resize):
            raise ValueError("incompatible torch/torchvision release pair")
    except (ImportError, RuntimeError, OSError, AttributeError, ValueError) as error:
        raise ValueError("approved compatible torchvision with working v2 transforms is required") from error

    if not re.fullmatch(r"4\.57\.[0-9]+", transformers.__version__):
        raise ValueError("use a separately approved Transformers 4.57.x environment")
    try:
        processor_class = getattr(transformers, variant["processor_class"])
        model_class = getattr(transformers, variant["model_class"])
    except (AttributeError, ImportError, RuntimeError, OSError) as error:
        raise ValueError("approved runtime classes are unavailable for " + verified["model_id"]) from error
    if not torch.cuda.is_available() or int(device.split(":")[1]) >= torch.cuda.device_count():
        raise ValueError("selected CUDA device is unavailable; no fallback")
    remaining(deadline)
    torch.cuda.set_device(device)
    if variant["dtype"] == "bfloat16":
        try:
            if torch.cuda.is_bf16_supported(including_emulation=False) is not True:
                raise ValueError("native support not established")
        except (AttributeError, TypeError, RuntimeError, OSError, ValueError) as error:
            raise ValueError("Qwen3 requires verified native bfloat16 support on " + device + "; no fallback") from error
    try:
        frames, sampling = decode_frames(descriptor, duration, frame_count, ffmpeg, deadline)
    except InsufficientFrames:
        return {field: result(status="missing", reason="insufficient_distinct_frames") for field in SEMANTIC_FIELDS}
    remaining(deadline)
    loading = {"local_files_only": True, "trust_remote_code": False}
    if runtime.model is None:
        processor = processor_class.from_pretrained(str(verified["local_model_dir"]),
            revision=revision, **loading, min_pixels=(variant["vision_shape"]["patch_size"] * 4) ** 2,
            max_pixels=FRAME_SIDE ** 2, use_fast=variant["use_fast"])
        model = model_class.from_pretrained(str(verified["local_model_dir"]),
            revision=revision, **loading, use_safetensors=True, device_map=None,
            dtype=getattr(torch, variant["dtype"]), attn_implementation="eager")
        check_model_unchanged(verified)
        model.to(device)
        model.eval()
        runtime.identity, runtime.verified = identity, verified
        runtime.processor, runtime.model = processor, model
    processor, model = runtime.processor, runtime.model
    content = [{"type": "text", "text": PROMPT}]
    for frame in frames:
        content.extend([{"type": "text", "text": f"\nsample {frame['sample_index']}, observed PTS "
                         f"{frame['actual_seconds']:.12g} seconds relative to the first decoded frame:\n"},
                        {"type": "image"}])
    conversation = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True,
                                            chat_template=CHAT_TEMPLATE)
    if not isinstance(prompt, str) or len(prompt.encode("utf-8")) > 16384:
        raise ValueError("processor produced an oversized prompt")
    images = [Image.frombytes("RGB", (FRAME_SIDE, FRAME_SIDE), frame["rgb"]) for frame in frames]
    inputs = processor(text=[prompt], images=images, padding=True, return_tensors="pt")
    if len(inputs["input_ids"]) != 1 or not 1 <= len(inputs["input_ids"][0]) <= 4096:
        raise ValueError("VLM input token or batch limit exceeded")
    inputs = inputs.to(device)
    generation = {"max_new_tokens": MAX_NEW_TOKENS, "num_beams": 1, "num_return_sequences": 1,
                  "do_sample": False, "return_dict_in_generate": False, "output_scores": False,
                  "output_logits": False, "max_time": remaining(deadline)}
    with torch.inference_mode():
        generated = model.generate(**inputs, **generation)
    remaining(deadline)
    if len(generated) != 1:
        raise ValueError("model returned an unexpected batch")
    suffix = generated[0][len(inputs["input_ids"][0]):]
    if not 1 <= len(suffix) <= MAX_NEW_TOKENS:
        raise ValueError("model output token limit exceeded")
    decoded = processor.batch_decode([suffix], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    if len(decoded) != 1 or not isinstance(decoded[0], str):
        raise ValueError("processor returned an invalid decoded response")
    raw = decoded[0]
    truncated = len(raw.encode("utf-8")) > 32768
    retained = raw.encode("utf-8")[:32768].decode("utf-8", "ignore")
    provenance = {"model_snapshot": {key: value for key, value in verified.items() if not key.startswith("_")},
                  "model_dtype": variant["dtype"],
                  "device": device, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                  "transformers_version": transformers.__version__, "torch_version": torch.__version__,
                  "torchvision_version": torchvision.__version__,
                  "sampling": sampling, "input_mode": "ordered_timestamped_images_not_native_video",
                  "prompt": prompt, "chat_template": CHAT_TEMPLATE, "generation": generation,
                  "raw_response": retained, "raw_response_truncated": truncated}
    try:
        if truncated:
            raise ValueError("model response exceeds byte limit")
        parsed = parse_vlm_response(raw, [frame["actual_seconds"] for frame in frames])["fields"]
    except ValueError as error:
        parsed = {field: result(status="error", reason=str(error)) for field in SEMANTIC_FIELDS}
    for field in SEMANTIC_FIELDS:
        parsed.setdefault(field, result(reason="optional_field_not_reported"))
        parsed[field].update({"model_id": verified["model_id"], "model_revision": revision})
    parsed["human_present"]["evidence"][0]["run_provenance"] = provenance
    check_model_unchanged(verified)
    remaining(deadline)
    return parsed
