from fractions import Fraction
import math
import os
from pathlib import Path
import re

from ._runtime import hash_fd, signature, strict_json

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


FRAME_SIDE = 448
FRAME_BYTES = FRAME_SIDE * FRAME_SIDE * 3
MAX_NEW_TOKENS = 1024
MAX_MODEL_BYTES = 12 * 1024 ** 3
TEXT_SHAPE = {"hidden_size": 2048, "intermediate_size": 11008, "num_hidden_layers": 36,
              "num_attention_heads": 16, "num_key_value_heads": 2, "vocab_size": 151936}
VISION_SHAPE = {"depth": 32, "hidden_size": 1280, "intermediate_size": 3420, "num_heads": 16,
                "out_hidden_size": 2048, "patch_size": 14, "spatial_merge_size": 2,
                "temporal_patch_size": 2}
QWEN3_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
QWEN3_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
QWEN3_8B_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
QWEN3_8B_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
MODEL_VARIANTS = {
    MODEL_ID: {"revision": None, "model_type": "qwen2_5_vl", "model_class": "Qwen2_5_VLForConditionalGeneration",
               "processor_class": "Qwen2_5_VLProcessor", "image_processor_type": "Qwen2VLImageProcessor",
               "text_shape": TEXT_SHAPE, "vision_shape": VISION_SHAPE, "use_fast": False, "dtype": "float16"},
    QWEN3_MODEL_ID: {"revision": QWEN3_REVISION, "model_type": "qwen3_vl",
                     "model_class": "Qwen3VLForConditionalGeneration", "processor_class": "Qwen3VLProcessor",
                     "image_processor_type": "Qwen2VLImageProcessorFast", "use_fast": True, "dtype": "bfloat16",
                     "text_shape": {"hidden_size": 2048, "intermediate_size": 6144, "num_hidden_layers": 28,
                                    "num_attention_heads": 16, "num_key_value_heads": 8, "vocab_size": 151936,
                                    "head_dim": 128},
                     "vision_shape": {"depth": 24, "hidden_size": 1024, "intermediate_size": 4096,
                                      "num_heads": 16, "out_hidden_size": 2048, "patch_size": 16,
                                      "spatial_merge_size": 2, "temporal_patch_size": 2,
                                      "in_channels": 3, "num_position_embeddings": 2304}},
}
MODEL_VARIANTS[QWEN3_MODEL_ID]["deepstack"] = [5, 11, 17]
MODEL_VARIANTS[QWEN3_8B_MODEL_ID] = {
    **MODEL_VARIANTS[QWEN3_MODEL_ID], "revision": QWEN3_8B_REVISION,
    "max_model_bytes": 24 * 1024 ** 3, "deepstack": [8, 16, 24],
    "text_shape": {**MODEL_VARIANTS[QWEN3_MODEL_ID]["text_shape"],
                   "hidden_size": 4096, "intermediate_size": 12288,
                   "num_hidden_layers": 36, "num_attention_heads": 32},
    "vision_shape": {**MODEL_VARIANTS[QWEN3_MODEL_ID]["vision_shape"],
                     "depth": 27, "hidden_size": 1152, "intermediate_size": 4304,
                     "out_hidden_size": 4096},
}
SUPPORTED_MODEL_IDS = tuple(MODEL_VARIANTS)
METADATA_FILES = {"config.json", "preprocessor_config.json", "tokenizer_config.json", "tokenizer.json",
                  "vocab.json", "merges.txt", "generation_config.json", "chat_template.json",
                  "chat_template.jinja", "special_tokens_map.json", "added_tokens.json",
                  "video_preprocessor_config.json", "model.safetensors.index.json", "README.md", ".gitattributes"}
CHAT_TEMPLATE = (
    "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' }}"
    "{% for item in message['content'] %}{% if item['type'] == 'image' %}"
    "{{ '<|vision_start|><|image_pad|><|vision_end|>' }}{% else %}{{ item['text'] }}{% endif %}{% endfor %}"
    "{{ '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def snapshot_path(local_model_dir, revision, model_id=MODEL_ID):
    if not isinstance(model_id, str) or model_id not in MODEL_VARIANTS:
        raise ValueError("select an explicitly approved VLM model identity")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("model-revision must be a lowercase 40-hex commit")
    approved_revision = MODEL_VARIANTS[model_id]["revision"]
    if approved_revision is not None and revision != approved_revision:
        raise ValueError("model-revision does not match the approved VLM identity pin")
    root = Path(local_model_dir).resolve(strict=True)
    if (not root.is_dir() or root.name != revision or root.parent.name != "snapshots"
            or root.parent.parent.name != "models--" + model_id.replace("/", "--")):
        raise ValueError("local-vlm-dir must be the matching Qwen HF cache snapshots/<revision> directory")
    return root


def validate_device(device):
    if not isinstance(device, str) or not re.fullmatch(r"cuda:(?:[0-9]|[1-5][0-9]|6[0-3])", device):
        raise ValueError("select one explicit logical CUDA index, cuda:0 through cuda:63")


def snapshot_files(root):
    files = {}
    with os.scandir(root) as entries:
        for entry in entries:
            if len(files) >= 64:
                raise ValueError("too many model snapshot files")
            name = entry.name
            if name not in METADATA_FILES and not re.fullmatch(r"model(?:-[0-9]{5}-of-[0-9]{5})?\.safetensors", name):
                raise ValueError("unapproved model snapshot file: " + name)
            target = Path(entry.path).resolve(strict=True)
            if target.parent not in (root, root.parent.parent / "blobs") or not target.is_file():
                raise ValueError("model file escapes snapshot/cache blobs or is not regular")
            files[name] = target
    return files


def validate_local_model(local_model_dir, revision, deadline, model_id=MODEL_ID):
    root = snapshot_path(local_model_dir, revision, model_id=model_id)
    variant = MODEL_VARIANTS[model_id]
    files = snapshot_files(root)
    required = {"config.json", "preprocessor_config.json", "tokenizer_config.json", "tokenizer.json"}
    weights = {name for name in files if name.endswith(".safetensors")}
    if not required <= set(files) or not weights:
        raise ValueError("incomplete local tokenizer/config/safetensors snapshot")
    digests, signatures, metadata = {}, {}, {}
    total = 0
    for name, target in files.items():
        limit = 8 * 1024 ** 3 if name in weights else 32 * 1024 ** 2 if name == "tokenizer.json" else 8 * 1024 ** 2
        descriptor = os.open(target, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            digest, before = hash_fd(descriptor, deadline, limit)
            total += before[2]
            if total > variant.get("max_model_bytes", MAX_MODEL_BYTES):
                raise ValueError("model snapshot exceeds byte budget")
            if target.parent.name == "blobs" and re.fullmatch(r"[0-9a-f]{64}", target.name) and digest != target.name:
                raise ValueError("HF blob fingerprint mismatch")
            if name.endswith(".json") and name not in ("tokenizer.json", "vocab.json"):
                with os.fdopen(os.dup(descriptor), "rb") as stream:
                    payload = strict_json(stream.read(limit + 1))
                if signature(os.fstat(descriptor)) != before or not isinstance(payload, dict):
                    raise ValueError("invalid or changing model metadata")
                metadata[name] = payload
            digests[name], signatures[name] = digest, before
        finally:
            os.close(descriptor)
    for payload in metadata.values():
        pending = [payload]
        while pending:
            value = pending.pop()
            if isinstance(value, dict):
                if set(value) & {"auto_map", "quantization_config", "custom_pipelines"}:
                    raise ValueError("remote-code/quantization/custom pipeline configuration is forbidden")
                pending.extend(value.values())
            elif isinstance(value, list):
                pending.extend(value)
    config = metadata["config.json"]
    if (config.get("model_type") != variant["model_type"]
            or config.get("architectures") != [variant["model_class"]]
            or config.get("_commit_hash", revision) != revision):
        raise ValueError("local model identity/architecture/revision mismatch")
    text_config = config.get("text_config", config) if model_id == MODEL_ID else config.get("text_config")
    for shape, expected in ((text_config, variant["text_shape"]), (config.get("vision_config"), variant["vision_shape"])):
        if not isinstance(shape, dict) or any(type(shape.get(key)) is not int or shape[key] != value for key, value in expected.items()):
            raise ValueError("local configuration is not the approved bounded model architecture")
    if "deepstack" in variant:
        deepstack = config["vision_config"].get("deepstack_visual_indexes")
        if (not isinstance(deepstack, list) or deepstack != variant["deepstack"]
                or any(type(index) is not int for index in deepstack)):
            raise ValueError("unexpected Qwen3 deepstack configuration")
    preprocessor = metadata["preprocessor_config.json"]
    processor_shape = {"patch_size": variant["vision_shape"]["patch_size"], "temporal_patch_size": 2, "merge_size": 2}
    if (preprocessor.get("processor_class") != variant["processor_class"]
            or preprocessor.get("image_processor_type") != variant["image_processor_type"]
            or metadata["tokenizer_config.json"].get("tokenizer_class") not in ("Qwen2Tokenizer", "Qwen2TokenizerFast")
            or any(type(preprocessor.get(key)) is not int or preprocessor[key] != value
                   for key, value in processor_shape.items())):
        raise ValueError("unexpected local processor/tokenizer configuration")
    if "deepstack" in variant and "video_preprocessor_config.json" in metadata:
        video_processor = metadata["video_preprocessor_config.json"]
        if (video_processor.get("processor_class") != variant["processor_class"]
                or video_processor.get("video_processor_type") != "Qwen3VLVideoProcessor"
                or any(type(video_processor.get(key)) is not int or video_processor[key] != value
                       for key, value in processor_shape.items())):
            raise ValueError("unexpected local Qwen3 video processor configuration")
    if "model.safetensors.index.json" in metadata:
        weight_map = metadata["model.safetensors.index.json"].get("weight_map")
        if (not isinstance(weight_map, dict) or not weight_map
                or any(not isinstance(name, str) or name not in weights for name in weight_map.values())
                or set(weight_map.values()) != weights):
            raise ValueError("invalid safetensors shard index")
    elif weights != {"model.safetensors"}:
        raise ValueError("sharded safetensors require a complete local index")
    verified = {"model_id": model_id, "model_revision": revision, "local_model_dir": str(root),
                "revision_source": "local_hf_snapshot_path", "file_sha256": digests,
                "_paths": files, "_signatures": signatures}
    check_model_unchanged(verified)
    return verified


def check_model_unchanged(verified):
    current = snapshot_files(Path(verified["local_model_dir"]))
    if current != verified["_paths"] or any(signature(target.stat()) != verified["_signatures"][name]
                                             for name, target in current.items()):
        raise ValueError("local model snapshot changed during extraction")
