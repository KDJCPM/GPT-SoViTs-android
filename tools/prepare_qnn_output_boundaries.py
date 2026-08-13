#!/usr/bin/env python3
"""Rewrite fragile output projections and scalar/index boundaries before QAIRT compilation.

The learned operations and weights are unchanged. T2S prediction is partitioned into its own HTP
graph. Projection kernels that return zeros on QAIRT 2.48/V79 are expanded into equivalent
elementwise reductions, and SoVITS is exposed before two constant ``[0, 0]`` Gather operations.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def producer_map(model: onnx.ModelProto) -> dict[str, onnx.NodeProto]:
    return {output: node for node in model.graph.node for output in node.output}


def replace_t2s_output(source: Path, output: Path) -> tuple[str, onnx.TensorProto]:
    model = onnx.load(source, load_external_data=False)
    producers = producer_map(model)
    logits = model.graph.output[0].name
    projection = producers[logits]
    if projection.op_type != "MatMul" or len(projection.input) != 2:
        raise ValueError(f"expected final T2S MatMul in {source}, found {projection.op_type}")
    hidden, weight_name = projection.input
    weights = next((value for value in model.graph.initializer if value.name == weight_name), None)
    if weights is None or tuple(weights.dims) != (512, 1025):
        raise ValueError(f"unexpected T2S prediction weight {weight_name}")
    model.graph.node.remove(projection)
    model.graph.node.append(helper.make_node("Identity", [hidden], ["hidden_state"], name="QnnHiddenOutput"))
    model.graph.output[0].CopyFrom(
        helper.make_tensor_value_info("hidden_state", TensorProto.FLOAT, [1, 512])
    )
    model.graph.name = f"{model.graph.name}_hidden"
    onnx.checker.check_model(model)
    onnx.save(model, output)
    return weight_name, weights


def make_predict(weight: onnx.TensorProto, output: Path) -> None:
    # HTP 2.48 on V79 returns zeros for both the rank-2 MatMul and an equivalent 1x1 Conv.
    # Broadcasted multiplication followed by ReduceSum preserves the exact projection while
    # avoiding those broken kernels. This small head is only 512 * 1025 multiply-accumulates.
    reduction_weight = numpy_helper.from_array(
        numpy_helper.to_array(weight).T[None, :, :].copy(),
        name="prediction_weight",
    )
    reduction_axes = numpy_helper.from_array(
        np.array([2], dtype=np.int64),
        name="prediction_reduction_axes",
    )
    graph = helper.make_graph(
        [
            helper.make_node(
                "Mul", ["hidden_state", reduction_weight.name], ["weighted"], name="PredictMul"
            ),
            helper.make_node(
                "ReduceSum",
                ["weighted", reduction_axes.name],
                ["logits"],
                name="PredictReduce",
                keepdims=1,
            ),
        ],
        "t2s_predict",
        [helper.make_tensor_value_info("hidden_state", TensorProto.FLOAT, [1, 1, 512])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 1025, 1])],
        [reduction_weight, reduction_axes],
    )
    model = helper.make_model(graph, producer_name="gsv-qnn-output-boundary", opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, output)


def replace_vits_output(source: Path, output: Path) -> None:
    model = onnx.load(source, load_external_data=False)
    producers = producer_map(model)
    audio = model.graph.output[0].name
    final_gather = producers[audio]
    first_gather = producers[final_gather.input[0]]
    if final_gather.op_type != "Gather" or first_gather.op_type != "Gather":
        raise ValueError("expected the VITS [0, 0] output boundary to contain two Gather nodes")
    waveform_3d = first_gather.input[0]
    tanh = producers[waveform_3d]
    projection = producers[tanh.input[0]]
    if tanh.op_type != "Tanh" or projection.op_type != "Conv":
        raise ValueError("expected the VITS waveform boundary to end in Conv followed by Tanh")
    projection_weight = next(
        (value for value in model.graph.initializer if value.name == projection.input[1]), None
    )
    if projection_weight is None:
        raise ValueError("VITS output projection has no constant weight")
    projection_index = next(
        index for index, node in enumerate(model.graph.node) if node.name == projection.name
    )
    weight_array = numpy_helper.to_array(projection_weight)
    if weight_array.shape != (1, 24, 7) or len(projection.input) != 2:
        raise ValueError(f"unexpected VITS output projection shape {weight_array.shape}")

    # The V79 Conv projection produces all zeros even though its input and FP16 weights are valid.
    # Expand this one static 24x7 convolution so HTP executes only elementwise/reduction kernels.
    length = 88320
    expanded_initializers = [
        numpy_helper.from_array(np.array([0, 0, 3, 0, 0, 3], dtype=np.int64), "audio_pad"),
        numpy_helper.from_array(np.array(0.0, dtype=np.float32), "audio_pad_value"),
        numpy_helper.from_array(np.array([2], dtype=np.int64), "audio_slice_axes"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "audio_slice_steps"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "audio_reduction_axes"),
    ]
    expanded_nodes = [
        helper.make_node(
            "Pad",
            [projection.input[0], "audio_pad", "audio_pad_value"],
            ["audio_padded"],
            name="QnnAudioProjectionPad",
            mode="constant",
        )
    ]
    products = []
    for kernel_index in range(weight_array.shape[2]):
        start_name = f"audio_slice_start_{kernel_index}"
        end_name = f"audio_slice_end_{kernel_index}"
        weight_name = f"audio_projection_weight_{kernel_index}"
        slice_name = f"audio_slice_{kernel_index}"
        product_name = f"audio_product_{kernel_index}"
        expanded_initializers.extend(
            [
                numpy_helper.from_array(np.array([kernel_index], dtype=np.int64), start_name),
                numpy_helper.from_array(np.array([kernel_index + length], dtype=np.int64), end_name),
                numpy_helper.from_array(
                    weight_array[:, :, kernel_index : kernel_index + 1].copy(), weight_name
                ),
            ]
        )
        expanded_nodes.extend(
            [
                helper.make_node(
                    "Slice",
                    ["audio_padded", start_name, end_name, "audio_slice_axes", "audio_slice_steps"],
                    [slice_name],
                    name=f"QnnAudioProjectionSlice{kernel_index}",
                ),
                helper.make_node(
                    "Mul",
                    [slice_name, weight_name],
                    [product_name],
                    name=f"QnnAudioProjectionMul{kernel_index}",
                ),
            ]
        )
        products.append(product_name)
    accumulated = products[0]
    for kernel_index, product_name in enumerate(products[1:], start=1):
        next_accumulated = f"audio_accumulated_{kernel_index}"
        expanded_nodes.append(
            helper.make_node(
                "Add",
                [accumulated, product_name],
                [next_accumulated],
                name=f"QnnAudioProjectionAdd{kernel_index}",
            )
        )
        accumulated = next_accumulated
    expanded_nodes.append(
        helper.make_node(
            "ReduceSum",
            [accumulated, "audio_reduction_axes"],
            [projection.output[0]],
            name="QnnAudioProjectionReduce",
            keepdims=1,
        )
    )

    model.graph.node.remove(final_gather)
    model.graph.node.remove(first_gather)
    model.graph.node.remove(projection)
    model.graph.initializer.remove(projection_weight)
    model.graph.initializer.extend(expanded_initializers)
    for offset, node in enumerate(expanded_nodes):
        model.graph.node.insert(projection_index + offset, node)
    model.graph.node.append(
        helper.make_node("Identity", [waveform_3d], ["audio_3d"], name="QnnAudioOutput")
    )
    model.graph.output[0].CopyFrom(
        helper.make_tensor_value_info("audio_3d", TensorProto.FLOAT, [1, 1, 88320])
    )
    model.graph.name = f"{model.graph.name}_audio3d"
    onnx.checker.check_model(model)
    onnx.save(model, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    prefill_source = source / "t2s_prefill_p8.onnx"
    step_source = source / "t2s_step_c512.onnx"
    prefill_output = output / "t2s_prefill_hidden_p8.onnx"
    step_output = output / "t2s_step_hidden_c512.onnx"
    _, prefill_weight = replace_t2s_output(prefill_source, prefill_output)
    _, step_weight = replace_t2s_output(step_source, step_output)
    if prefill_weight.raw_data != step_weight.raw_data:
        raise ValueError("prefill and step prediction weights differ")
    make_predict(prefill_weight, output / "t2s_predict.onnx")
    replace_vits_output(source / "vits_p8_s69.onnx", output / "vits_audio3d_p8_s69.onnx")
    print(f"Created QNN output-boundary graphs in {output}")


if __name__ == "__main__":
    main()
