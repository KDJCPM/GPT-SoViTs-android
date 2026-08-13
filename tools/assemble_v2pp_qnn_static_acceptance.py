#!/usr/bin/env python3
"""Assemble one SM8750 V2PP static-bucket QNN acceptance artifact.

This deliberately emits a directory rather than a production .gsvm. It proves the complete neural
path on HTP for one conversion-time-specialized UTF-8 request without advertising general text or
runtime-reference support that has not yet been compiled into equivalent NPU buckets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from safetensors.torch import load_file


GRAPHS = (
    "bert_tokens_6",
    "t2s_prefill_p8",
    "t2s_step_c512",
    "vits_p8_s69",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrapped-graphs", required=True, type=Path)
    parser.add_argument("--conditioning", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    wrapped = args.wrapped_graphs.resolve()
    conditioning_path = args.conditioning.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    if not conditioning_path.is_file():
        raise SystemExit(f"conditioning file does not exist: {conditioning_path}")
    for graph in GRAPHS:
        for suffix in (".onnx", ".bin"):
            if not (wrapped / f"{graph}{suffix}").is_file():
                raise SystemExit(f"missing wrapped graph: {wrapped / f'{graph}{suffix}'}")

    conditioning = load_file(conditioning_path, device="cpu")
    prompt_semantic = conditioning["prompt_semantic"].flatten().tolist()
    pending = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for graph in GRAPHS:
            shutil.copy2(wrapped / f"{graph}.onnx", pending / f"{graph}.onnx")
            shutil.copy2(wrapped / f"{graph}.bin", pending / f"{graph}.bin")
        executor = {
            "format": "gsv-qnn-v2pp-static-acceptance",
            "format_version": 1,
            "operation": "synthesize_utf8_to_pcm16",
            "acceptance_only": True,
            "text": "你好世界",
            "normalized_text": "你好世界",
            "target_soc": "snapdragon_8_elite",
            "target_soc_family": "qualcomm_snapdragon_8",
            "target_asic": "SM8750",
            "target_soc_model": 69,
            "htp_arch": "V79",
            "qairt_version": "2.48.0.260626",
            "qnn_runtime_version": "2.48.0",
            "precision": "fp16",
            "cpu_neural_fallback": False,
            "token_ids": [101, 872, 1962, 686, 4518, 102],
            "phone_ids": [227, 167, 158, 119, 251, 214, 221, 194],
            "word2ph": [2, 2, 2, 2],
            "prompt_semantic": prompt_semantic,
            "prompt_semantic_length": len(prompt_semantic),
            "prefill_cache_length": 300,
            "cache_capacity": 512,
            "semantic_length": 69,
            "noise_seed": 1234,
            "sample_rate": 32000,
            "graphs": {
                "bert": "bert_tokens_6.onnx",
                "t2s_prefill": "t2s_prefill_p8.onnx",
                "t2s_step": "t2s_step_c512.onnx",
                "vits": "vits_p8_s69.onnx",
            },
        }
        executor_path = pending / "executor.json"
        executor_path.write_text(
            json.dumps(executor, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        files = []
        for path in sorted(pending.iterdir()):
            files.append(
                {
                    "path": path.name,
                    "size": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
        manifest = {
            "format": "gsv-qnn-static-acceptance-directory",
            "format_version": 1,
            "deployable": False,
            "acceptance_only": True,
            "executor": "qnn-htp",
            "entrypoint": "synthesize_utf8_to_pcm16",
            "target_soc": "snapdragon_8_elite",
            "target_soc_family": "qualcomm_snapdragon_8",
            "target_asic": "SM8750",
            "target_soc_model": 69,
            "htp_arch": "V79",
            "qairt_version": "2.48.0.260626",
            "backend_artifact": "executor.json",
            "precision": "fp16",
            "quantization": "none",
            "cpu_neural_fallback": False,
            "files": files,
        }
        (pending / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        pending.rename(output)
    except Exception:
        shutil.rmtree(pending, ignore_errors=True)
        raise
    print(f"Created {output} ({sum(path.stat().st_size for path in output.iterdir())} bytes)")


if __name__ == "__main__":
    main()
