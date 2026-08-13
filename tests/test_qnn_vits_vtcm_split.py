import sys
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from split_qnn_vits_vtcm_conv import (
    infer_time_lengths,
    parse_split_group,
    select_node_names,
    split_conv_along_time,
)


class QnnVitsVtcmSplitTest(unittest.TestCase):
    def make_model(self) -> onnx.ModelProto:
        random = np.random.default_rng(7)
        weights = random.normal(size=(3, 2, 3)).astype(np.float32)
        bias = random.normal(size=(3,)).astype(np.float32)
        graph = helper.make_graph(
            [
                helper.make_node(
                    "Conv",
                    ["input", "weights", "bias"],
                    ["output"],
                    name="long_conv",
                    kernel_shape=[3],
                    dilations=[3],
                    pads=[3, 3],
                    strides=[1],
                )
            ],
            "test",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2, 17])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 17])],
            [
                numpy_helper.from_array(weights, "weights"),
                numpy_helper.from_array(bias, "bias"),
            ],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 9
        return model

    def test_split_is_numerically_equivalent_at_chunk_boundaries(self):
        model = self.make_model()
        source = onnx.ModelProto()
        source.CopyFrom(model)
        split_conv_along_time(model, "long_conv", time_length=17, chunks=2)
        onnx.checker.check_model(model)
        values = np.random.default_rng(11).normal(size=(1, 2, 17)).astype(np.float32)
        expected = ReferenceEvaluator(source).run(None, {"input": values})[0]
        actual = ReferenceEvaluator(model).run(None, {"input": values})[0]
        np.testing.assert_array_equal(expected, actual)
        self.assertEqual(["Slice", "Conv", "Slice", "Conv", "Concat"], [n.op_type for n in model.graph.node])

    def test_rejects_non_same_padding(self):
        model = self.make_model()
        conv = model.graph.node[0]
        next(value for value in conv.attribute if value.name == "pads").ints[:] = [0, 0]
        with self.assertRaisesRegex(ValueError, "symmetric same-length padding"):
            split_conv_along_time(model, "long_conv", time_length=17, chunks=2)

    def test_infers_static_time_length(self):
        self.assertEqual({"long_conv": 17}, infer_time_lengths(self.make_model(), ["long_conv"]))

    def test_selects_exact_and_regex_nodes_in_graph_order(self):
        model = self.make_model()
        self.assertEqual(["long_conv"], select_node_names(model, [], [r"long_.*"]))
        with self.assertRaisesRegex(ValueError, "matched no nodes"):
            select_node_names(model, [], [r"missing_.*"])

    def test_parses_per_group_chunk_count(self):
        pattern = r"/dec/resblocks\.(9|10|11)/convs1\.[12]/Conv"
        self.assertEqual((8, pattern), parse_split_group(f"8:{pattern}"))
        with self.assertRaisesRegex(ValueError, "CHUNKS:REGEX"):
            parse_split_group("8")
        with self.assertRaisesRegex(ValueError, "at least 2"):
            parse_split_group("1:.*")


if __name__ == "__main__":
    unittest.main()
