#!/usr/bin/env python3
"""Align G2PW's descriptor table for QAIRT FP16 static-tensor serialization.

QAIRT 2.48 rounds the expected size of an FP16 static tensor down to an
eight-byte boundary. G2PW's 39402x1305 descriptor is four bytes short under
that calculation. Appending unused zero rows preserves every legal lookup and
makes the serialized tensor length exact without changing model precision.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


WEIGHT_NAME = "second_order_descriptor.weight"
CHAR_WEIGHT_NAME = "char_descriptor.weight"
POSITION_WEIGHT_NAME = "pos_classifier.weight"
QNN_STATIC_ALIGNMENT_BYTES = 8
FP16_BYTES = 2


def normalize_descriptor(model: onnx.ModelProto) -> dict[str, object]:
    initializers = {value.name: value for value in model.graph.initializer}
    weight = initializers.get(WEIGHT_NAME)
    char_weight = initializers.get(CHAR_WEIGHT_NAME)
    position_weight = initializers.get(POSITION_WEIGHT_NAME)
    if weight is None or char_weight is None or position_weight is None:
        raise ValueError("G2PW descriptor, character, or position initializer is missing")
    values = numpy_helper.to_array(weight)
    char_values = numpy_helper.to_array(char_weight)
    position_values = numpy_helper.to_array(position_weight)
    if values.ndim != 2 or char_values.ndim != 2 or position_values.ndim != 2:
        raise ValueError("G2PW descriptor initializers must be two-dimensional")
    if values.dtype != np.float32:
        raise ValueError(f"{WEIGHT_NAME} must remain FP32 before QAIRT conversion, got {values.dtype}")
    used_rows = int(char_values.shape[0] * position_values.shape[0])
    columns = int(values.shape[1])
    row_alignment = QNN_STATIC_ALIGNMENT_BYTES // math.gcd(
        QNN_STATIC_ALIGNMENT_BYTES, columns * FP16_BYTES
    )
    padded_rows = ((used_rows + row_alignment - 1) // row_alignment) * row_alignment
    actual_rows = int(values.shape[0])
    if actual_rows not in {used_rows, padded_rows}:
        raise ValueError(
            f"{WEIGHT_NAME} has {actual_rows} rows, expected either "
            f"{char_values.shape[0]}*{position_values.shape[0]}={used_rows} used rows "
            f"or {padded_rows} aligned rows"
        )
    gathers = [
        node
        for node in model.graph.node
        if node.op_type == "Gather" and node.input and node.input[0] == WEIGHT_NAME
    ]
    if len(gathers) != 1:
        raise ValueError(f"expected one Gather consuming {WEIGHT_NAME}, found {len(gathers)}")

    if actual_rows == padded_rows:
        padding = values[used_rows:]
        if padding.size and np.any(padding):
            raise ValueError(f"{WEIGHT_NAME} aligned padding rows must remain zero")
        return {
            "weight": WEIGHT_NAME,
            "used_rows": used_rows,
            "padded_rows": padded_rows,
            "added_rows": 0,
            "existing_padding_rows": padded_rows - used_rows,
            "columns": columns,
            "fp16_bytes": padded_rows * columns * FP16_BYTES,
        }
    added_rows = padded_rows - used_rows
    if added_rows == 0:
        return {
            "weight": WEIGHT_NAME,
            "used_rows": used_rows,
            "padded_rows": padded_rows,
            "added_rows": 0,
            "existing_padding_rows": 0,
            "columns": columns,
            "fp16_bytes": padded_rows * columns * FP16_BYTES,
        }
    padded = np.zeros((padded_rows, columns), dtype=np.float32)
    padded[:used_rows] = values
    if not np.array_equal(padded[:used_rows], values) or np.any(padded[used_rows:]):
        raise AssertionError("descriptor alignment changed used weights or added non-zero rows")
    replacement = numpy_helper.from_array(padded, WEIGHT_NAME)
    index = list(model.graph.initializer).index(weight)
    model.graph.initializer[index].CopyFrom(replacement)
    onnx.checker.check_model(model)
    return {
        "weight": WEIGHT_NAME,
        "used_rows": used_rows,
        "padded_rows": padded_rows,
        "added_rows": added_rows,
        "existing_padding_rows": 0,
        "columns": columns,
        "fp16_bytes": padded_rows * columns * FP16_BYTES,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    if not source.is_file():
        raise SystemExit(f"input ONNX does not exist: {source}")
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    model = onnx.load(str(source), load_external_data=True)
    summary = normalize_descriptor(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output))
    print(f"Created {output}: {summary}")


if __name__ == "__main__":
    main()
