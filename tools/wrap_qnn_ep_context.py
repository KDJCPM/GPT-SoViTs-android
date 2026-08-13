#!/usr/bin/env python3
"""Wrap an offline QAIRT context binary in an ONNX Runtime EPContext model."""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path

import onnx
from onnx import TensorProto, helper


QNN_TO_ONNX_DTYPE = {
    "QNN_DATATYPE_INT_8": TensorProto.INT8,
    "QNN_DATATYPE_INT_16": TensorProto.INT16,
    "QNN_DATATYPE_INT_32": TensorProto.INT32,
    "QNN_DATATYPE_INT_64": TensorProto.INT64,
    "QNN_DATATYPE_UINT_8": TensorProto.UINT8,
    "QNN_DATATYPE_UINT_16": TensorProto.UINT16,
    "QNN_DATATYPE_UINT_32": TensorProto.UINT32,
    "QNN_DATATYPE_UINT_64": TensorProto.UINT64,
    "QNN_DATATYPE_FLOAT_16": TensorProto.FLOAT16,
    "QNN_DATATYPE_FLOAT_32": TensorProto.FLOAT,
    "QNN_DATATYPE_BOOL_8": TensorProto.BOOL,
}


def qnn_normalized_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def match_tensor_names(
    source_names: list[str],
    context_names: list[str],
    label: str,
) -> dict[str, str]:
    if len(source_names) != len(set(source_names)):
        raise ValueError(f"source {label} tensor names are not unique")
    if len(context_names) != len(set(context_names)):
        raise ValueError(f"context {label} tensor names are not unique")
    if len(source_names) != len(context_names):
        raise ValueError(
            f"context {label} count does not match source ONNX: "
            f"source={len(source_names)} context={len(context_names)}"
        )
    remaining = set(source_names)
    result: dict[str, str] = {}
    for context_name in context_names:
        if context_name in remaining:
            result[context_name] = context_name
            remaining.remove(context_name)
    for context_name in context_names:
        if context_name in result:
            continue
        candidates = [
            source_name
            for source_name in remaining
            if qnn_normalized_name(source_name) == context_name
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"context {label} tensor {context_name!r} cannot be mapped uniquely to "
                f"source ONNX; candidates={candidates}"
            )
        result[context_name] = candidates[0]
        remaining.remove(candidates[0])
    if remaining:
        raise ValueError(f"source {label} tensors were not compiled into the context: {sorted(remaining)}")
    return result


def source_element_count(value: onnx.ValueInfoProto) -> int | None:
    dimensions = value.type.tensor_type.shape.dim
    if any(dimension.dim_value <= 0 for dimension in dimensions):
        return None
    return math.prod(dimension.dim_value for dimension in dimensions)


def validate_tensor_interface(
    source_values: list[onnx.ValueInfoProto],
    context_values: list[dict],
    label: str,
) -> dict[str, str]:
    source_by_name = {value.name: value for value in source_values}
    mapping = match_tensor_names(
        list(source_by_name),
        [value.get("name", "") for value in context_values],
        label,
    )
    compatible_types = {
        TensorProto.FLOAT: {TensorProto.FLOAT, TensorProto.FLOAT16},
        TensorProto.FLOAT16: {TensorProto.FLOAT16},
        TensorProto.INT64: {TensorProto.INT64, TensorProto.INT32},
        TensorProto.INT32: {TensorProto.INT32},
        TensorProto.BOOL: {TensorProto.BOOL},
    }
    for context_value in context_values:
        context_name = context_value["name"]
        source = source_by_name[mapping[context_name]]
        context_info = make_value_info(context_value)
        source_dtype = source.type.tensor_type.elem_type
        context_dtype = context_info.type.tensor_type.elem_type
        if context_dtype not in compatible_types.get(source_dtype, {source_dtype}):
            raise ValueError(
                f"context {label} tensor {context_name!r} changed dtype incompatibly: "
                f"source={TensorProto.DataType.Name(source_dtype)} "
                f"context={TensorProto.DataType.Name(context_dtype)}"
            )
        count = source_element_count(source)
        context_count = math.prod(context_value["dimensions"])
        if count is not None and count != context_count:
            raise ValueError(
                f"context {label} tensor {context_name!r} changed element count: "
                f"source={count} context={context_count}"
            )
    return mapping


def load_graph_io(path: Path) -> tuple[list[dict], list[dict]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    graphs = document.get("info", {}).get("graphs", [])
    if len(graphs) != 1:
        raise ValueError(f"expected exactly one graph in {path}, found {len(graphs)}")
    graph = graphs[0].get("info", {})
    inputs = [value["info"] for value in graph.get("graphInputs", [])]
    outputs = [value["info"] for value in graph.get("graphOutputs", [])]
    if not inputs or not outputs:
        raise ValueError(f"context metadata has no graph inputs or outputs: {path}")
    return inputs, outputs


def make_value_info(value: dict) -> onnx.ValueInfoProto:
    name = value.get("name")
    dimensions = value.get("dimensions")
    qnn_dtype = value.get("dataType")
    if not isinstance(name, str) or not name:
        raise ValueError("QNN tensor metadata has no name")
    if not isinstance(dimensions, list) or not dimensions or any(
        not isinstance(item, int) or item <= 0 for item in dimensions
    ):
        raise ValueError(f"QNN tensor {name} has invalid dimensions: {dimensions!r}")
    try:
        onnx_dtype = QNN_TO_ONNX_DTYPE[qnn_dtype]
    except KeyError as error:
        raise ValueError(f"QNN tensor {name} uses unsupported dtype {qnn_dtype!r}") from error
    return helper.make_tensor_value_info(name, onnx_dtype, dimensions)


def wrap_context(
    source_path: Path,
    context_path: Path,
    context_info_path: Path,
    output: Path,
) -> tuple[Path, Path]:
    source_path = source_path.resolve()
    context_path = context_path.resolve()
    context_info_path = context_info_path.resolve()
    output = output.resolve()
    if not source_path.is_file() or not context_path.is_file() or not context_info_path.is_file():
        raise ValueError("source ONNX, context binary and context metadata must all exist")
    source = onnx.load(source_path, load_external_data=False)
    initializer_names = {value.name for value in source.graph.initializer}
    source_inputs = [value for value in source.graph.input if value.name not in initializer_names]
    context_inputs, context_outputs = load_graph_io(context_info_path)
    validate_tensor_interface(source_inputs, context_inputs, "inputs")
    validate_tensor_interface(list(source.graph.output), context_outputs, "outputs")
    inputs = [make_value_info(value) for value in context_inputs]
    outputs = [make_value_info(value) for value in context_outputs]
    output.parent.mkdir(parents=True, exist_ok=True)
    deployed_context = output.with_suffix(".bin")
    shutil.copy2(context_path, deployed_context)
    node = helper.make_node(
        "EPContext",
        inputs=[value.name for value in inputs],
        outputs=[value.name for value in outputs],
        name="QNNExecutionProvider_QNN_0",
        domain="com.microsoft",
        embed_mode=0,
        ep_cache_context=deployed_context.name,
        main_context=1,
        partition_name="QNNExecutionProvider_QNN_0",
        source="QNNExecutionProvider",
    )
    graph = helper.make_graph([node], source.graph.name or "qnn_context", inputs, outputs)
    model = helper.make_model(
        graph,
        producer_name="gsv-qairt-context-wrapper",
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, output)
    return output, deployed_context


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-onnx", required=True, type=Path)
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument(
        "--context-info",
        required=True,
        type=Path,
        help="JSON emitted by qnn-context-binary-utility for the exact context binary.",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    output, deployed_context = wrap_context(
        args.source_onnx,
        args.context,
        args.context_info,
        args.output,
    )
    print(f"Created {output} -> {deployed_context.name}")


if __name__ == "__main__":
    main()
