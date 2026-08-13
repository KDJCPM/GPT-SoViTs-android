#!/usr/bin/env python3
"""Validate exported V2PP QNN ONNX graphs against their PyTorch references."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from partition_onnx_contiguous import read_partition_manifest, run_partitioned_onnx


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def compare(
    name: str,
    actual: np.ndarray,
    expected: torch.Tensor,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, object]:
    expected_array = expected.detach().cpu().numpy()
    if actual.shape != expected_array.shape:
        raise AssertionError(f"{name} shape {actual.shape} != {expected_array.shape}")
    difference = np.abs(actual.astype(np.float64) - expected_array.astype(np.float64))
    result: dict[str, object] = {
        "shape": list(actual.shape),
        "max_abs": float(difference.max()) if difference.size else 0.0,
        "mean_abs": float(difference.mean()) if difference.size else 0.0,
    }
    if not np.allclose(
        actual,
        expected_array,
        atol=absolute_tolerance,
        rtol=relative_tolerance,
    ):
        raise AssertionError(
            f"{name} differs: max_abs={result['max_abs']:.9g} "
            f"mean_abs={result['mean_abs']:.9g}"
        )
    return result


def compare_arrays(
    name: str,
    actual: np.ndarray,
    expected: np.ndarray,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, object]:
    return compare(
        name,
        actual,
        torch.from_numpy(expected),
        absolute_tolerance,
        relative_tolerance,
    )


def run_graph(
    graph: Path,
    inputs: dict[str, torch.Tensor],
) -> list[np.ndarray]:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(graph),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    try:
        feed = {name: value.detach().cpu().numpy() for name, value in inputs.items()}
        return session.run(None, feed)
    finally:
        del session
        gc.collect()


def validate_bert(
    root: Path,
    token_capacity: int,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, object]:
    reference = torch.load(
        root / f"bert_tokens_{token_capacity}_reference.pt",
        map_location="cpu",
        weights_only=True,
    )
    outputs = run_graph(
        root / f"bert_tokens_{token_capacity}.onnx",
        {
            "input_ids": reference["input_ids"],
            "token_type_ids": reference["token_type_ids"],
            "attention_mask": reference["attention_mask"],
        },
    )
    return {
        "hidden_features": compare(
            "BERT hidden_features",
            outputs[0],
            reference["hidden_features"],
            absolute_tolerance,
            relative_tolerance,
        )
    }


def validate_t2s(
    root: Path,
    phone_capacity: int,
    cache_capacity: int,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, object]:
    reference = torch.load(
        root / f"t2s_pc{phone_capacity}_c{cache_capacity}_reference.pt",
        map_location="cpu",
        weights_only=True,
    )
    prefill = run_graph(
        root / f"t2s_prefill_pc{phone_capacity}.onnx",
        {
            "text_seq": reference["text_seq"],
            "text_bert": reference["text_bert"],
            "text_valid": reference["text_valid"],
        },
    )
    step = run_graph(
        root / f"t2s_step_c{cache_capacity}.onnx",
        {
            "last_token": reference["last_token"],
            "position_embedding": reference["position_embedding"],
            "k_cache": reference["k_cache"],
            "v_cache": reference["v_cache"],
            "write_mask": reference["write_mask"],
            "attention_bias": reference["attention_bias"],
        },
    )
    return {
        "prefill_logits": compare(
            "T2S prefill logits",
            prefill[0],
            reference["prefill_logits"],
            absolute_tolerance,
            relative_tolerance,
        ),
        "prefill_k_cache": compare(
            "T2S prefill k_cache",
            prefill[1],
            reference["prefill_k_cache"],
            absolute_tolerance,
            relative_tolerance,
        ),
        "prefill_v_cache": compare(
            "T2S prefill v_cache",
            prefill[2],
            reference["prefill_v_cache"],
            absolute_tolerance,
            relative_tolerance,
        ),
        "step_logits": compare(
            "T2S step logits",
            step[0],
            reference["step_logits"],
            absolute_tolerance,
            relative_tolerance,
        ),
        "step_new_keys": compare(
            "T2S step new_keys",
            step[1],
            reference["step_new_keys"],
            absolute_tolerance,
            relative_tolerance,
        ),
        "step_new_values": compare(
            "T2S step new_values",
            step[2],
            reference["step_new_values"],
            absolute_tolerance,
            relative_tolerance,
        ),
    }


def validate_vits(
    root: Path,
    graph: Path,
    partitions: Path | None,
    phone_capacity: int,
    semantic_capacity: int,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, object]:
    reference = torch.load(
        root / f"vits_pc{phone_capacity}_sc{semantic_capacity}_reference.pt",
        map_location="cpu",
        weights_only=True,
    )
    inputs = {
        "pred_semantic": reference["pred_semantic"],
        "text_seq": reference["text_seq"],
        "noise": reference["noise"],
        "semantic_valid": reference["semantic_valid"],
        "text_valid": reference["text_valid"],
    }
    outputs = run_graph(graph, inputs)
    result = {
        "audio": compare(
            "VITS audio",
            outputs[0],
            reference["audio"],
            absolute_tolerance,
            relative_tolerance,
        )
    }
    if partitions is not None:
        partition_outputs = run_partitioned_onnx(
            partitions,
            {name: value.detach().cpu().numpy() for name, value in inputs.items()},
        )
        result["partition_audio"] = compare(
            "partitioned VITS audio",
            partition_outputs[0],
            reference["audio"],
            absolute_tolerance,
            relative_tolerance,
        )
        result["partition_equivalence"] = compare_arrays(
            "partitioned VITS versus source ONNX",
            partition_outputs[0],
            outputs[0],
            absolute_tolerance,
            relative_tolerance,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--token-capacity", type=int, default=130)
    parser.add_argument("--phone-capacity", type=int, default=128)
    parser.add_argument("--semantic-capacity", type=int, default=512)
    parser.add_argument("--cache-capacity", type=int, default=1024)
    parser.add_argument(
        "--vits-graph",
        type=Path,
        help="Validate a rewritten VITS graph; defaults to the graph under --root.",
    )
    parser.add_argument(
        "--vits-partitions",
        type=Path,
        help="Contiguous partition manifest to validate sequentially against --vits-graph.",
    )
    parser.add_argument("--absolute-tolerance", type=float, default=2.0e-4)
    parser.add_argument("--relative-tolerance", type=float, default=2.0e-4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    vits_graph = (
        args.vits_graph.resolve()
        if args.vits_graph is not None
        else root / f"vits_pc{args.phone_capacity}_sc{args.semantic_capacity}.onnx"
    )
    graph_paths = {
        "bert": root / f"bert_tokens_{args.token_capacity}.onnx",
        "t2s_prefill": root / f"t2s_prefill_pc{args.phone_capacity}.onnx",
        "t2s_step": root / f"t2s_step_c{args.cache_capacity}.onnx",
        "vits": vits_graph,
    }
    partition_manifest = args.vits_partitions.resolve() if args.vits_partitions else None
    if partition_manifest is not None:
        partition_document, partition_paths = read_partition_manifest(partition_manifest)
        if partition_document["source"]["sha256"] != digest(vits_graph):
            raise ValueError("VITS partition source does not match --vits-graph")
    missing = [str(path) for path in graph_paths.values() if not path.is_file()]
    if missing:
        raise ValueError(f"QNN ONNX graphs are missing: {', '.join(missing)}")
    result = {
        "format": "gsv-v2pp-qnn-onnx-fp32-validation",
        "format_version": 1,
        "absolute_tolerance": args.absolute_tolerance,
        "relative_tolerance": args.relative_tolerance,
        "graphs": {
            name: {"path": str(path.resolve()), "sha256": digest(path)}
            for name, path in graph_paths.items()
        },
        "bert": validate_bert(
            root,
            args.token_capacity,
            args.absolute_tolerance,
            args.relative_tolerance,
        ),
        "t2s": validate_t2s(
            root,
            args.phone_capacity,
            args.cache_capacity,
            args.absolute_tolerance,
            args.relative_tolerance,
        ),
        "vits": validate_vits(
            root,
            vits_graph,
            partition_manifest,
            args.phone_capacity,
            args.semantic_capacity,
            args.absolute_tolerance,
            args.relative_tolerance,
        ),
    }
    if partition_manifest is not None:
        result["graphs"]["vits_partitions"] = {
            "manifest": {"path": str(partition_manifest), "sha256": digest(partition_manifest)},
            "parts": [
                {"path": str(path), "sha256": digest(path)} for path in partition_paths
            ],
        }
    encoded = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
