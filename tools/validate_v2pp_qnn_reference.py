#!/usr/bin/env python3
"""Validate V2 Pro Plus runtime-reference ONNX graphs against saved PyTorch outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from partition_onnx_contiguous import read_partition_manifest, run_partitioned_onnx


def compare(actual: np.ndarray, expected: torch.Tensor) -> dict[str, object]:
    reference = expected.detach().cpu().numpy()
    if actual.shape != reference.shape:
        raise ValueError(f"shape mismatch: ONNX {actual.shape}, PyTorch {reference.shape}")
    difference = np.abs(actual - reference)
    return {
        "shape": list(actual.shape),
        "max_abs": float(difference.max()) if difference.size else 0.0,
        "mean_abs": float(difference.mean()) if difference.size else 0.0,
    }


def session(path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def validate(
    root: Path,
    tolerance: float,
    vits_onnx: Path | None = None,
    vits_partitions: Path | None = None,
) -> dict[str, object]:
    root = root.resolve()
    vits_path = (vits_onnx or root / "vits_reference_pc128_sc512.onnx")
    if not vits_path.is_absolute():
        vits_path = root / vits_path
    vits_path = vits_path.resolve()
    if not vits_path.is_file():
        raise ValueError(f"runtime-reference VITS ONNX does not exist: {vits_path}")
    graph_paths = {
        "reference_ssl": root / "reference_ssl_5s.onnx",
        "reference_prompt_semantic": root / "reference_prompt_semantic_5s.onnx",
        "reference_conditioning": root / "reference_conditioning_5s.onnx",
        "t2s_reference_prefill": root / "t2s_reference_prefill_pc128.onnx",
        "vits_reference": vits_path,
    }
    missing = [str(path) for path in graph_paths.values() if not path.is_file()]
    if missing:
        raise ValueError(f"runtime-reference ONNX graphs are missing: {', '.join(missing)}")
    result: dict[str, object] = {
        "format": "gsv-v2pp-qnn-reference-onnx-validation",
        "format_version": 1,
        "absolute_tolerance": tolerance,
        "graphs": {
            name: {"path": str(path.resolve()), "sha256": digest(path)}
            for name, path in graph_paths.items()
        },
    }
    partition_manifest = vits_partitions.resolve() if vits_partitions is not None else None
    partition_paths: list[Path] = []
    if partition_manifest is not None:
        partition_document, partition_paths = read_partition_manifest(partition_manifest)
        if partition_document["source"]["sha256"] != digest(vits_path):
            raise ValueError("runtime-reference VITS partitions do not match the source graph")
        result["graphs"]["vits_reference_partitions"] = {
            "manifest": {"path": str(partition_manifest), "sha256": digest(partition_manifest)},
            "parts": [
                {"path": str(path), "sha256": digest(path)} for path in partition_paths
            ],
        }

    saved = torch.load(root / "reference_ssl_5s.pt", map_location="cpu", weights_only=False)
    outputs = session(graph_paths["reference_ssl"]).run(
        None, {"reference_pcm_16k": saved["reference_pcm_16k"].numpy()}
    )
    result["ssl"] = compare(outputs[0], saved["ssl_content"])

    saved = torch.load(
        root / "reference_prompt_semantic_5s.pt", map_location="cpu", weights_only=False
    )
    outputs = session(graph_paths["reference_prompt_semantic"]).run(
        None, {"ssl_content": saved["ssl_content"].numpy()}
    )
    prompt_equal = bool(np.array_equal(outputs[0], saved["prompt_semantic"].numpy()))
    result["prompt_semantic"] = {
        "shape": list(outputs[0].shape),
        "exact": prompt_equal,
    }
    if not prompt_equal:
        raise ValueError("prompt semantic ONNX tokens differ from PyTorch")

    saved = torch.load(
        root / "reference_conditioning_5s.pt", map_location="cpu", weights_only=False
    )
    outputs = session(graph_paths["reference_conditioning"]).run(
        None,
        {
            "reference_pcm_16k": saved["reference_pcm_16k"].numpy(),
            "reference_pcm_32k_reflected": saved["reference_pcm_32k_reflected"].numpy(),
        },
    )
    result["reference_conditioning"] = {
        "reference_spectrogram": compare(outputs[0], saved["reference_spectrogram"]),
        "speaker_embedding": compare(outputs[1], saved["speaker_embedding"]),
    }

    saved = torch.load(
        root / "t2s_reference_prefill.pt", map_location="cpu", weights_only=False
    )
    input_names = [
        "text_seq",
        "text_bert",
        "text_valid",
        "prompt_semantic",
        "prompt_phone_ids",
        "prompt_bert",
        "prompt_phone_valid",
    ]
    outputs = session(graph_paths["t2s_reference_prefill"]).run(
        None, {name: saved[name].numpy() for name in input_names}
    )
    result["t2s_reference_prefill"] = {
        "logits": compare(outputs[0], saved["logits"]),
        "k_cache": compare(outputs[1], saved["k_cache"]),
        "v_cache": compare(outputs[2], saved["v_cache"]),
    }

    saved = torch.load(root / "vits_reference.pt", map_location="cpu", weights_only=False)
    input_names = [
        "pred_semantic",
        "text_seq",
        "noise",
        "semantic_valid",
        "text_valid",
        "reference_spectrogram",
        "speaker_embedding",
    ]
    vits_inputs = {name: value.numpy() for name, value in zip(input_names, saved["inputs"])}
    outputs = session(vits_path).run(None, vits_inputs)
    result["vits_reference"] = compare(outputs[0], saved["audio"])
    if partition_manifest is not None:
        partition_outputs = run_partitioned_onnx(partition_manifest, vits_inputs)
        result["vits_reference_partition"] = compare(partition_outputs[0], saved["audio"])
        result["vits_reference_partition_equivalence"] = compare(
            partition_outputs[0], torch.from_numpy(outputs[0])
        )

    def measurements(value: object):
        if isinstance(value, dict):
            if "max_abs" in value:
                yield float(value["max_abs"])
            for item in value.values():
                yield from measurements(item)

    maximum = max(measurements(result), default=0.0)
    result["maximum_absolute_error"] = maximum
    if maximum > tolerance:
        raise ValueError(f"runtime-reference ONNX error {maximum} exceeds {tolerance}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tolerance", type=float, default=2e-4)
    parser.add_argument(
        "--vits-onnx",
        type=Path,
        help="Validated VITS graph; defaults to vits_reference_pc128_sc512.onnx under --root",
    )
    parser.add_argument(
        "--vits-partitions",
        type=Path,
        help="Contiguous VITS partition manifest to validate against the source ONNX.",
    )
    args = parser.parse_args()
    document = validate(args.root, args.tolerance, args.vits_onnx, args.vits_partitions)
    output = args.output or args.root / "reference-validation.json"
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"Validated runtime-reference ONNX graphs: max_abs={document['maximum_absolute_error']}")


if __name__ == "__main__":
    main()
