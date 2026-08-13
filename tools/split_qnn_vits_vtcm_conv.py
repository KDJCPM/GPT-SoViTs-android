#!/usr/bin/env python3
"""Split one long VITS Conv along time so each prepared HTP op fits the target VTCM.

The rewrite is exact: each chunk includes the convolution halo, uses boundary padding only at the
original tensor edges, and the chunk outputs are concatenated back to the original tensor name.
It is intentionally conversion-time and shape-specific; Android never sees this compatibility rule.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


DEFAULT_NODE = "/dec/resblocks.9/convs1.1/Conv"


def attributes(node: onnx.NodeProto) -> dict[str, object]:
    return {value.name: helper.get_attribute_value(value) for value in node.attribute}


def split_conv_along_time(
    model: onnx.ModelProto,
    node_name: str,
    time_length: int,
    chunks: int,
) -> None:
    if time_length <= 0:
        raise ValueError("time length must be positive")
    if chunks < 2 or chunks > time_length:
        raise ValueError("chunks must be between 2 and the time length")
    matches = [node for node in model.graph.node if node.name == node_name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {node_name!r} node, found {len(matches)}")
    node = matches[0]
    if node.op_type != "Conv" or len(node.input) not in (2, 3) or len(node.output) != 1:
        raise ValueError(f"{node_name} is not a supported one-output Conv")
    config = attributes(node)
    kernel = tuple(config.get("kernel_shape", ()))
    dilation = tuple(config.get("dilations", (1,)))
    stride = tuple(config.get("strides", (1,)))
    pads = tuple(config.get("pads", (0, 0)))
    if len(kernel) != 1 or len(dilation) != 1 or stride != (1,):
        raise ValueError("only stride-1 one-dimensional Conv can be split")
    receptive_width = dilation[0] * (kernel[0] - 1)
    if receptive_width % 2 or pads != (receptive_width // 2,) * 2:
        raise ValueError("Conv must use symmetric same-length padding")
    radius = receptive_width // 2
    index = next(i for i, value in enumerate(model.graph.node) if value is node)
    prefix = f"gsv_vtcm_split_{index}"
    if any(value.name.startswith(prefix) for value in model.graph.initializer):
        raise ValueError("model already contains the VTCM split initializer prefix")

    axes = f"{prefix}_axes"
    steps = f"{prefix}_steps"
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(np.array([2], dtype=np.int64), axes),
            numpy_helper.from_array(np.array([1], dtype=np.int64), steps),
        ]
    )
    replacement: list[onnx.NodeProto] = []
    chunk_outputs: list[str] = []
    for chunk in range(chunks):
        output_start = time_length * chunk // chunks
        output_end = time_length * (chunk + 1) // chunks
        input_start = max(0, output_start - radius)
        input_end = min(time_length, output_end + radius)
        start_name = f"{prefix}_{chunk}_start"
        end_name = f"{prefix}_{chunk}_end"
        slice_output = f"{prefix}_{chunk}_input"
        conv_output = f"{prefix}_{chunk}_output"
        model.graph.initializer.extend(
            [
                numpy_helper.from_array(np.array([input_start], dtype=np.int64), start_name),
                numpy_helper.from_array(np.array([input_end], dtype=np.int64), end_name),
            ]
        )
        replacement.append(
            helper.make_node(
                "Slice",
                [node.input[0], start_name, end_name, axes, steps],
                [slice_output],
                name=f"{node_name}/VtcmSlice{chunk}",
            )
        )
        chunk_config = dict(config)
        chunk_config["pads"] = [radius if input_start == 0 else 0, radius if input_end == time_length else 0]
        replacement.append(
            helper.make_node(
                "Conv",
                [slice_output, *node.input[1:]],
                [conv_output],
                name=f"{node_name}/VtcmConv{chunk}",
                **chunk_config,
            )
        )
        chunk_outputs.append(conv_output)
    replacement.append(
        helper.make_node(
            "Concat",
            chunk_outputs,
            list(node.output),
            name=f"{node_name}/VtcmConcat",
            axis=2,
        )
    )
    model.graph.node.remove(node)
    for offset, value in enumerate(replacement):
        model.graph.node.insert(index + offset, value)
    model.metadata_props.add(
        key=f"gsv.qnn.vtcm_split.{index}",
        value=f"{node_name};time={time_length};chunks={chunks};halo={radius}",
    )


def infer_time_lengths(model: onnx.ModelProto, node_names: list[str]) -> dict[str, int]:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    shapes: dict[str, tuple[int | str, ...]] = {}
    for value in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]:
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        shapes[value.name] = tuple(
            dimension.dim_value if dimension.HasField("dim_value") else dimension.dim_param
            for dimension in tensor_type.shape.dim
        )
    result: dict[str, int] = {}
    for node_name in node_names:
        matches = [node for node in inferred.graph.node if node.name == node_name]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one {node_name!r} node, found {len(matches)}")
        shape = shapes.get(matches[0].input[0])
        if shape is None or len(shape) != 3 or not isinstance(shape[2], int) or shape[2] <= 0:
            raise ValueError(f"cannot infer a static NCL input shape for {node_name}: {shape}")
        result[node_name] = shape[2]
    return result


def select_node_names(
    model: onnx.ModelProto,
    exact_names: list[str],
    patterns: list[str],
) -> list[str]:
    compiled = [re.compile(pattern) for pattern in patterns]
    available = {node.name for node in model.graph.node}
    missing = set(exact_names).difference(available)
    if missing:
        raise ValueError(f"exact nodes are missing: {', '.join(sorted(missing))}")
    selected = set(exact_names)
    for pattern, expression in zip(patterns, compiled):
        matches = {node.name for node in model.graph.node if expression.fullmatch(node.name)}
        if not matches:
            raise ValueError(f"node regex matched no nodes: {pattern}")
        selected.update(matches)
    return [node.name for node in model.graph.node if node.name in selected]


def parse_split_group(value: str) -> tuple[int, str]:
    raw_chunks, separator, pattern = value.partition(":")
    if not separator or not pattern:
        raise ValueError("split groups must use CHUNKS:REGEX")
    try:
        chunks = int(raw_chunks)
    except ValueError as error:
        raise ValueError("split-group chunks must be an integer") from error
    if chunks < 2:
        raise ValueError("split-group chunks must be at least 2")
    re.compile(pattern)
    return chunks, pattern


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--node",
        action="append",
        default=[],
        help="Exact Conv node name; repeat to split multiple nodes.",
    )
    parser.add_argument(
        "--node-regex",
        action="append",
        default=[],
        help="Full-match Conv node regex; repeat to select multiple audited node groups.",
    )
    parser.add_argument(
        "--split-group",
        action="append",
        default=[],
        metavar="CHUNKS:REGEX",
        help="Split one full-match node regex with its own chunk count; repeat for multiple groups.",
    )
    parser.add_argument(
        "--time-length",
        type=int,
        help="Override shape inference for a single --node.",
    )
    parser.add_argument("--chunks", type=int, default=2)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_file():
        raise SystemExit(f"source ONNX does not exist: {source}")
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    model = onnx.load(source, load_external_data=False)
    if args.split_group and (args.node or args.node_regex or args.time_length is not None):
        raise SystemExit("--split-group cannot be combined with --node, --node-regex, or --time-length")
    if args.split_group:
        groups: list[tuple[int, list[str]]] = []
        selected: set[str] = set()
        for raw_group in args.split_group:
            chunks, pattern = parse_split_group(raw_group)
            names = select_node_names(model, [], [pattern])
            duplicate = selected.intersection(names)
            if duplicate:
                raise SystemExit(
                    f"split-group regexes select duplicate nodes: {', '.join(sorted(duplicate))}"
                )
            selected.update(names)
            groups.append((chunks, names))
        ordered_names = [node.name for node in model.graph.node if node.name in selected]
        lengths = infer_time_lengths(model, ordered_names)
        for chunks, names in groups:
            for node_name in names:
                split_conv_along_time(model, node_name, lengths[node_name], chunks)
        onnx.checker.check_model(model)
        output.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, output)
        summary = ", ".join(f"{len(names)}x{chunks}" for chunks, names in groups)
        print(
            f"Created {output} with {len(ordered_names)} Conv nodes split by groups {summary}"
        )
        return
    exact_names = args.node
    if not exact_names and not args.node_regex:
        exact_names = [DEFAULT_NODE]
    node_names = select_node_names(model, exact_names, args.node_regex)
    if args.time_length is not None:
        if len(node_names) != 1:
            raise SystemExit("--time-length can only be used with one --node")
        lengths = {node_names[0]: args.time_length}
    else:
        lengths = infer_time_lengths(model, node_names)
    for node_name in node_names:
        split_conv_along_time(model, node_name, lengths[node_name], args.chunks)
    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)
    print(
        f"Created {output} with {len(node_names)} Conv nodes split into "
        f"{args.chunks} exact time-axis chunks"
    )


if __name__ == "__main__":
    main()
