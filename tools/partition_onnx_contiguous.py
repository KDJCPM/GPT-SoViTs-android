#!/usr/bin/env python3
"""Partition one static ONNX graph into ordered contiguous node ranges.

The generated manifest is a conversion artifact. It records every graph boundary and allows a
backend builder to compile each range independently while a thin executor carries named tensors
between the prepared sessions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


FORMAT = "gsv-onnx-contiguous-partitions"
FORMAT_VERSION = 1


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def clone(message):
    result = type(message)()
    result.CopyFrom(message)
    return result


def inferred_value_info(
    model: onnx.ModelProto,
    dimension_values: dict[str, int] | None = None,
) -> dict[str, onnx.ValueInfoProto]:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    fixed = dimension_values or {}
    values: dict[str, onnx.ValueInfoProto] = {}
    for value in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]:
        copied = clone(value)
        for dimension in copied.type.tensor_type.shape.dim:
            if dimension.HasField("dim_param") and dimension.dim_param in fixed:
                dimension.dim_value = fixed[dimension.dim_param]
        values[value.name] = copied
    for initializer in inferred.graph.initializer:
        if initializer.name not in values:
            values[initializer.name] = helper.make_tensor_value_info(
                initializer.name,
                initializer.data_type,
                list(initializer.dims),
            )
    return values


def tensor_description(value: onnx.ValueInfoProto) -> dict[str, object]:
    tensor = value.type.tensor_type
    if not tensor.elem_type:
        raise ValueError(f"tensor {value.name!r} has no element type")
    if not tensor.HasField("shape"):
        raise ValueError(f"tensor {value.name!r} has no static shape")
    shape: list[int] = []
    for dimension in tensor.shape.dim:
        if not dimension.HasField("dim_value") or dimension.dim_value <= 0:
            raise ValueError(f"tensor {value.name!r} has a dynamic or invalid shape")
        shape.append(dimension.dim_value)
    return {
        "name": value.name,
        "data_type": int(tensor.elem_type),
        "data_type_name": TensorProto.DataType.Name(tensor.elem_type),
        "shape": shape,
    }


def boundary_indices(model: onnx.ModelProto, boundaries_before: Iterable[str]) -> list[int]:
    by_name: dict[str, list[int]] = {}
    for index, node in enumerate(model.graph.node):
        by_name.setdefault(node.name, []).append(index)
    result: list[int] = []
    for name in boundaries_before:
        matches = by_name.get(name, [])
        if len(matches) != 1:
            raise ValueError(f"boundary node {name!r} must occur exactly once; found {len(matches)}")
        result.append(matches[0])
    if result != sorted(set(result)):
        raise ValueError("boundary nodes must be unique and supplied in graph order")
    if any(index <= 0 for index in result):
        raise ValueError("a boundary cannot create an empty first partition")
    return result


def reject_nested_graphs(model: onnx.ModelProto) -> None:
    for node in model.graph.node:
        for attribute in node.attribute:
            if attribute.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS):
                raise ValueError(
                    f"node {node.name!r} contains a nested graph; contiguous partitioning is not safe"
                )


def fold_constant_nodes(model: onnx.ModelProto) -> int:
    initializer_names = {value.name for value in model.graph.initializer}
    retained: list[onnx.NodeProto] = []
    added: list[onnx.TensorProto] = []
    folded = 0
    for node in model.graph.node:
        if node.op_type != "Constant":
            retained.append(clone(node))
            continue
        if node.input or len(node.output) != 1 or not node.output[0]:
            raise ValueError(f"Constant node {node.name!r} has an unsupported interface")
        output = node.output[0]
        if output in initializer_names:
            raise ValueError(f"Constant output {output!r} conflicts with an initializer")
        attributes = {value.name: helper.get_attribute_value(value) for value in node.attribute}
        if len(attributes) != 1:
            raise ValueError(f"Constant node {node.name!r} must carry exactly one value")
        name, value = next(iter(attributes.items()))
        if name == "value" and isinstance(value, onnx.TensorProto):
            tensor = clone(value)
            tensor.name = output
        elif name in {"value_float", "value_floats"}:
            tensor = numpy_helper.from_array(np.asarray(value, dtype=np.float32), output)
        elif name in {"value_int", "value_ints"}:
            tensor = numpy_helper.from_array(np.asarray(value, dtype=np.int64), output)
        else:
            raise ValueError(f"Constant node {node.name!r} uses unsupported attribute {name!r}")
        initializer_names.add(output)
        added.append(tensor)
        folded += 1
    if folded:
        del model.graph.node[:]
        model.graph.node.extend(retained)
        model.graph.initializer.extend(added)
        model.metadata_props.add(key="gsv.folded_constant_nodes", value=str(folded))
        onnx.checker.check_model(model)
    return folded


def make_partition(
    model: onnx.ModelProto,
    value_info: dict[str, onnx.ValueInfoProto],
    start: int,
    end: int,
    index: int,
) -> tuple[onnx.ModelProto, dict[str, object]]:
    nodes = list(model.graph.node[start:end])
    if not nodes:
        raise ValueError(f"partition {index} is empty")
    initializer_by_name = {value.name: value for value in model.graph.initializer}
    sparse_names = {value.values.name for value in model.graph.sparse_initializer}
    if sparse_names:
        raise ValueError("sparse initializers are not supported by the contiguous partitioner")
    producer: dict[str, int] = {}
    consumers: dict[str, list[int]] = {}
    for node_index, node in enumerate(model.graph.node):
        for name in node.output:
            if not name:
                continue
            if name in producer:
                raise ValueError(f"tensor {name!r} has more than one producer")
            producer[name] = node_index
        for name in node.input:
            if name:
                consumers.setdefault(name, []).append(node_index)

    live_inputs: list[str] = []
    used_initializers: set[str] = set()
    seen_inputs: set[str] = set()
    for node_index in range(start, end):
        node = model.graph.node[node_index]
        for name in node.input:
            if not name:
                continue
            if name in initializer_by_name:
                used_initializers.add(name)
                continue
            source_index = producer.get(name)
            if source_index is not None and source_index >= node_index:
                raise ValueError(
                    f"graph is not topological: {name!r} is produced at {source_index} "
                    f"but consumed at {node_index}"
                )
            if source_index is None or source_index < start:
                if name not in seen_inputs:
                    seen_inputs.add(name)
                    live_inputs.append(name)

    graph_outputs = {value.name for value in model.graph.output}
    live_outputs: list[str] = []
    seen_outputs: set[str] = set()
    for node in nodes:
        for name in node.output:
            if not name or name in seen_outputs:
                continue
            needed_later = any(consumer >= end for consumer in consumers.get(name, ()))
            if needed_later or name in graph_outputs:
                seen_outputs.add(name)
                live_outputs.append(name)
    if not live_outputs:
        raise ValueError(f"partition {index} has no live outputs")

    missing = [name for name in [*live_inputs, *live_outputs] if name not in value_info]
    if missing:
        raise ValueError(f"shape inference did not describe boundary tensors: {', '.join(missing)}")
    inputs = [clone(value_info[name]) for name in live_inputs]
    outputs = [clone(value_info[name]) for name in live_outputs]
    input_names = set(live_inputs)
    output_names = set(live_outputs)
    relevant = {
        name
        for node in nodes
        for name in [*node.input, *node.output]
        if name and name not in used_initializers
    }
    internal_info = [
        clone(value)
        for value in value_info.values()
        if value.name in relevant and value.name not in input_names and value.name not in output_names
    ]
    initializers = [
        clone(value) for value in model.graph.initializer if value.name in used_initializers
    ]
    graph = helper.make_graph(
        [clone(node) for node in nodes],
        f"{model.graph.name or 'graph'}_partition_{index:02d}",
        inputs,
        outputs,
        initializer=initializers,
        value_info=internal_info,
        doc_string=model.graph.doc_string,
    )
    partition = clone(model)
    partition.graph.CopyFrom(graph)
    partition.metadata_props.add(
        key="gsv.contiguous_partition",
        value=f"index={index};nodes={start}:{end}",
    )
    onnx.checker.check_model(partition, full_check=True)
    return partition, {
        "index": index,
        "node_start": start,
        "node_end": end,
        "node_count": end - start,
        "first_node": nodes[0].name,
        "last_node": nodes[-1].name,
        "inputs": [tensor_description(value) for value in inputs],
        "outputs": [tensor_description(value) for value in outputs],
        "initializer_count": len(initializers),
    }


def partition_model(
    source: Path,
    output_dir: Path,
    boundaries_before: list[str],
    prefix: str = "partition",
    dimension_values: dict[str, int] | None = None,
) -> dict[str, object]:
    source = source.resolve()
    output_dir = output_dir.resolve()
    if not source.is_file():
        raise ValueError(f"source ONNX does not exist: {source}")
    if output_dir.exists():
        raise ValueError(f"partition output already exists: {output_dir}")
    model = onnx.load(source, load_external_data=True)
    onnx.checker.check_model(model)
    reject_nested_graphs(model)
    folded_constants = fold_constant_nodes(model)
    indices = boundary_indices(model, boundaries_before)
    fixed_dimensions = dimension_values or {}
    if any(not name or value <= 0 for name, value in fixed_dimensions.items()):
        raise ValueError("fixed symbolic dimensions need a name and a positive value")
    values = inferred_value_info(model, fixed_dimensions)
    ranges = list(zip([0, *indices], [*indices, len(model.graph.node)]))
    output_dir.mkdir(parents=True)
    partitions: list[dict[str, object]] = []
    for index, (start, end) in enumerate(ranges):
        partition, metadata = make_partition(model, values, start, end, index)
        filename = f"{prefix}_{index:02d}.onnx"
        path = output_dir / filename
        onnx.save_model(partition, path)
        metadata.update({"name": f"{prefix}_{index:02d}", "path": filename, "sha256": digest(path)})
        partitions.append(metadata)
    manifest = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "source": {"path": str(source), "sha256": digest(source)},
        "boundaries_before": boundaries_before,
        "fixed_dimensions": fixed_dimensions,
        "folded_constant_nodes": folded_constants,
        "partitions": partitions,
        "graph_inputs": [tensor_description(values[value.name]) for value in model.graph.input],
        "graph_outputs": [tensor_description(values[value.name]) for value in model.graph.output],
    }
    (output_dir / "partitions.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def read_partition_manifest(path: Path) -> tuple[dict[str, object], list[Path]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("format") != FORMAT or document.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported contiguous partition manifest")
    root = path.resolve().parent
    partitions = document.get("partitions")
    if not isinstance(partitions, list) or len(partitions) < 2:
        raise ValueError("contiguous partition manifest must contain at least two partitions")
    paths: list[Path] = []
    for expected_index, item in enumerate(partitions):
        if not isinstance(item, dict) or item.get("index") != expected_index:
            raise ValueError("contiguous partition indices are not ordered")
        part = (root / str(item.get("path", ""))).resolve()
        part.relative_to(root)
        if not part.is_file() or digest(part) != item.get("sha256"):
            raise ValueError(f"partition {expected_index} payload verification failed")
        paths.append(part)
    source = document.get("source")
    if not isinstance(source, dict):
        raise ValueError("partition manifest has no source provenance")
    source_path = Path(str(source.get("path", ""))).resolve()
    if not source_path.is_file() or digest(source_path) != source.get("sha256"):
        raise ValueError("partition source provenance verification failed")
    return document, paths


def run_partitioned_onnx(path: Path, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
    import onnxruntime as ort

    document, partitions = read_partition_manifest(path)
    values = dict(inputs)
    for partition in partitions:
        session = ort.InferenceSession(str(partition), providers=["CPUExecutionProvider"])
        try:
            feed = {item.name: values[item.name] for item in session.get_inputs()}
            output = session.run(None, feed)
            values.update({item.name: value for item, value in zip(session.get_outputs(), output)})
        finally:
            del session
    return [values[item["name"]] for item in document["graph_outputs"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--boundary-before", required=True, action="append")
    parser.add_argument("--prefix", default="partition")
    parser.add_argument(
        "--dim-param",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Resolve one symbolic boundary dimension to a positive static value.",
    )
    args = parser.parse_args()
    fixed_dimensions: dict[str, int] = {}
    for raw in args.dim_param:
        name, separator, raw_value = raw.partition("=")
        if not separator:
            raise SystemExit("--dim-param must use NAME=VALUE")
        try:
            value = int(raw_value)
        except ValueError as error:
            raise SystemExit("--dim-param VALUE must be an integer") from error
        if not name or value <= 0 or name in fixed_dimensions:
            raise SystemExit("--dim-param names must be unique and values must be positive")
        fixed_dimensions[name] = value
    document = partition_model(
        args.source,
        args.output_dir,
        args.boundary_before,
        args.prefix,
        fixed_dimensions,
    )
    print(
        f"Created {len(document['partitions'])} contiguous ONNX partitions under "
        f"{args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
