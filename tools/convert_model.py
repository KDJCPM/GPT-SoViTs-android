#!/usr/bin/env python3
"""Build a lossless canonical package for GPT-SoVITS V1 through V4.

This is a serialization conversion, not graph quantization. Tensor dtype, shape, strides and
values are verified after round-trip. The source files are model checkpoints and therefore use
pickle; only convert checkpoints you trust.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator
from collections.abc import Mapping

import torch
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors
from model_profiles import HEADER_PROFILES, PROFILES, detect_sovits


FORMAT_VERSION = 2
SUPPORTED_DEPLOYMENT_PROFILES = {"v2ProPlus", "v4"}


def complete_runtime_assets(upstream: Path, model_version: str) -> list[tuple[Path, str]]:
    """Resources required to reproduce the full upstream frontend/conditioning path."""
    roots = [
        (upstream / "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large", "runtime_assets/bert"),
        (upstream / "GPT_SoVITS/pretrained_models/chinese-hubert-base", "runtime_assets/hubert"),
        (upstream / "GPT_SoVITS/pretrained_models/sv", "runtime_assets/speaker_encoder"),
        (upstream / "GPT_SoVITS/text", "runtime_assets/text_source"),
    ]
    files = []
    if model_version == "v4":
        files.append((upstream / "GPT_SoVITS/pretrained_models/gsv-v4-pretrained/vocoder.pth", "runtime_assets/v4/vocoder.pth"))
    result=[]
    for root,prefix in roots:
        for source in sorted(root.rglob('*')):
            if source.is_file() and '__pycache__' not in source.parts and source.suffix not in {'.pyc'}:
                result.append((source,f"{prefix}/{source.relative_to(root).as_posix()}"))
    result.extend(files)
    try:
        import jieba_fast
        jieba_root=Path(jieba_fast.__file__).resolve().parent
        for relative in ("dict.txt","finalseg/prob_emit.p","finalseg/prob_start.p","finalseg/prob_trans.p",
                         "posseg/char_state_tab.p","posseg/prob_emit.p","posseg/prob_start.p","posseg/prob_trans.p"):
            result.append((jieba_root/relative,f"runtime_assets/jieba/{relative}"))
    except ImportError as error:
        raise RuntimeError('jieba_fast is required to create a complete package') from error
    missing=[str(source) for source,_ in result if not source.is_file()]
    if missing: raise FileNotFoundError(f"complete runtime assets missing: {missing}")
    return result


def load_checkpoint(path: Path, sovits: bool = False) -> Any:
    if not sovits:
        return torch.load(path, map_location="cpu", weights_only=False)
    with path.open("rb") as stream:
        head = stream.read(2)
        if head == b"PK":
            return torch.load(path, map_location="cpu", weights_only=False)
        # Newer GPT-SoVITS models store version metadata in place of the first ZIP bytes.
        if head not in HEADER_PROFILES:
            raise ValueError(f"Unsupported SoVITS header {head!r}")
        return torch.load(BytesIO(b"PK" + stream.read()), map_location="cpu", weights_only=False)


def tensors(value: Any, prefix: str = "") -> Iterator[tuple[str, torch.Tensor]]:
    if torch.is_tensor(value):
        yield prefix, value
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            yield from tensors(value[key], f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from tensors(item, f"{prefix}[{index}]")


def tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    raw = value.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def tensor_index(value: Any) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "dtype": str(t.dtype).removeprefix("torch."),
            "shape": list(t.shape),
            "sha256": tensor_digest(t),
        }
        for name, t in tensors(value)
    }


def state_tensors(value: Any) -> dict[str, torch.Tensor]:
    """Return the checkpoint's deployable state dict without serializing Python objects."""
    state = value.get("weight", value) if isinstance(value, dict) else value
    if not isinstance(state, dict):
        raise ValueError("Checkpoint state must be a dictionary")
    result = {}
    for name, tensor in state.items():
        if torch.is_tensor(tensor):
            result[str(name)] = tensor.detach().cpu().contiguous()
    if not result:
        raise ValueError("Checkpoint state does not contain tensors")
    return result


def json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping) or callable(getattr(value, "items", None)):
        return {str(k): json_value(v) for k, v in value.items() if not torch.is_tensor(v)}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value if not torch.is_tensor(v)]
    return str(value)


def checkpoint_metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return json_value({k: v for k, v in value.items() if k not in {"weight", "state_dict"}})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_and_verify(value: dict[str, torch.Tensor], path: Path) -> dict[str, Any]:
    source_index = tensor_index(value)
    # Safetensors is runtime-neutral: no pickle/Python class dependency and mmap-friendly.
    save_safetensors(value, path, metadata={"format": "gsvm-canonical-v2"})
    target_index = tensor_index(load_safetensors(path, device="cpu"))
    if source_index != target_index:
        missing = sorted(set(source_index) ^ set(target_index))[:10]
        changed = sorted(k for k in set(source_index) & set(target_index) if source_index[k] != target_index[k])[:10]
        raise RuntimeError(f"Round-trip tensor verification failed; missing={missing}, changed={changed}")
    return target_index


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def graph_abi(model_version: str, sample_rate: int) -> dict[str, Any]:
    """Stable boundary that every deployment executor must implement."""
    return {
        "abi": "gsv-synthesis",
        "abi_version": 1,
        "model_version": model_version,
        "tensor_layout": "row-major",
        "components": [
            {"id": "text_frontend", "executor": "host", "inputs": ["utf8_text", "language"], "outputs": ["phone_ids:int64[T]", "bert_features:float32[T,1024]"]},
            {"id": "ssl_encoder", "inputs": ["reference_pcm:float32[1,S16]"], "outputs": ["ssl:float32[1,768,F]"]},
            {"id": "speaker_encoder", "inputs": ["reference_pcm_32k:float32[1,S32]"], "outputs": ["speaker_embedding:float32[1,E]"]},
            {"id": "semantic_encoder", "weights": "weights/gpt.safetensors", "inputs": ["prompt_phone_ids:int64[P]", "target_phone_ids:int64[T]", "bert_features:float32[P+T,1024]"], "outputs": ["semantic_context:float32[L,512]"]},
            {"id": "semantic_decoder_step", "weights": "weights/gpt.safetensors", "stateful": True, "inputs": ["semantic_context:float32[L,512]", "previous_tokens:int64[N]", "kv_cache"], "outputs": ["logits:float32[1025]", "kv_cache"]},
            {"id": "sovits_decoder", "weights": "weights/sovits.safetensors", "inputs": ["semantic_tokens:int64[N]", "phone_ids:int64[T]", "reference_spectrogram:float32[1025,R]", "speaker_embedding:float32[1,E]"], "outputs": ["pcm:float32[S]"]},
        ],
        "audio": {"sample_rate": sample_rate, "channels": 1, "output_range": [-1.0, 1.0]},
        "backend_rules": {"precision": "source", "allow_partition": True, "required_fallback": "cpu"},
    }


def convert(gpt: Path, sovits: Path, output: Path, name: str, forced_version: str | None = None,
            upstream: Path | None = None, include_complete_assets: bool = True) -> Path:
    detected_profile, is_lora, source_header = detect_sovits(sovits)
    profile = PROFILES[forced_version] if forced_version else detected_profile
    if forced_version and forced_version != detected_profile.id:
        raise ValueError(f"Forced version {forced_version} conflicts with detected {detected_profile.id}")
    if profile.id not in SUPPORTED_DEPLOYMENT_PROFILES:
        raise ValueError(f"Only {sorted(SUPPORTED_DEPLOYMENT_PROFILES)} are supported by this mobile baseline, got {profile.id}")
    upstream=(upstream or Path(__file__).resolve().parents[2]).resolve()
    gpt_value = load_checkpoint(gpt)
    sovits_value = load_checkpoint(sovits, sovits=True)
    gpt_state, sovits_state = state_tensors(gpt_value), state_tensors(sovits_value)

    with tempfile.TemporaryDirectory(prefix="gsvm-") as temp_name:
        temp = Path(temp_name)
        gpt_weights = temp / "weights/gpt.safetensors"
        sovits_weights = temp / "weights/sovits.safetensors"
        gpt_weights.parent.mkdir(parents=True)
        gpt_index = save_and_verify(gpt_state, gpt_weights)
        sovits_index = save_and_verify(sovits_state, sovits_weights)
        write_json(temp / "config/gpt.json", checkpoint_metadata(gpt_value))
        write_json(temp / "config/sovits.json", checkpoint_metadata(sovits_value))
        write_json(temp / "abi.json", graph_abi(profile.id, profile.sample_rate))
        report = {
            "verification": "bit-exact-tensors",
            "gpt_tensor_count": len(gpt_index),
            "sovits_tensor_count": len(sovits_index),
            "gpt_parameter_elements": sum(int(torch.tensor(v["shape"]).prod()) for v in gpt_index.values()),
            "sovits_parameter_elements": sum(int(torch.tensor(v["shape"]).prod()) for v in sovits_index.values()),
        }
        write_json(temp / "verification.json", report)
        files = []
        relative_files = [
            "weights/gpt.safetensors", "weights/sovits.safetensors",
            "config/gpt.json", "config/sovits.json", "abi.json", "verification.json",
        ]
        if include_complete_assets:
            for source,relative in complete_runtime_assets(upstream,profile.id):
                target=temp/relative; target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(source,target)
                relative_files.append(relative)
        for relative in relative_files:
            file = temp / relative
            files.append({"path": relative, "size": file.stat().st_size, "sha256": sha256(file)})
        manifest = {
            "format": "gsvm",
            "format_version": FORMAT_VERSION,
            "name": name,
            "model_version": profile.id,
            "checkpoint_family": profile.checkpoint_family,
            "sample_rate": profile.sample_rate,
            "model_abi": "gsv-synthesis@1",
            "canonical_weights": "safetensors",
            "runtime": "canonical",
            "cpu_exporter": profile.cpu_exporter,
            "vocoder": profile.vocoder,
            "lora": is_lora,
            "source_header": source_header.decode("ascii", errors="replace"),
            "weight_policy": "preserve-source-dtype-no-quantization",
            "scope": ["v2ProPlus", "v4"],
            "completeness": "full-upstream-runtime-assets" if include_complete_assets else "weights-and-config-only",
            "artifacts": [],
            "target_soc": "any",
            "target_soc_family": "canonical",
            "backend_artifact": "",
            "files": files,
        }
        write_json(temp / "manifest.json", manifest)
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_suffix(output.suffix + ".partial")
        with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            archive.write(temp / "manifest.json", "manifest.json")
            for file in files:
                archive.write(temp / file["path"], file["path"])
        os.replace(partial, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpt", required=True, type=Path, help="GPT .ckpt")
    parser.add_argument("--sovits", required=True, type=Path, help="SoVITS .pth")
    parser.add_argument("--output", required=True, type=Path, help="Output .gsvm")
    parser.add_argument("--name", default="GPT-SoVITS voice")
    parser.add_argument("--version", choices=sorted(PROFILES), help="Normally auto-detected from SoVITS")
    parser.add_argument("--upstream", type=Path, default=Path(".."))
    parser.add_argument("--weights-only", action="store_true", help="Omit full frontend/runtime assets (not the quality baseline)")
    args = parser.parse_args()
    result = convert(args.gpt.resolve(), args.sovits.resolve(), args.output.resolve(), args.name, args.version,
                     args.upstream.resolve(), not args.weights_only)
    print(f"Created {result} ({result.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
