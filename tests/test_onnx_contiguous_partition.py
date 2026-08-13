import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from partition_onnx_contiguous import (
    fold_constant_nodes,
    partition_model,
    read_partition_manifest,
    run_partitioned_onnx,
)


class OnnxContiguousPartitionTests(unittest.TestCase):
    def make_model(self) -> onnx.ModelProto:
        graph = helper.make_graph(
            [
                helper.make_node("Add", ["input", "bias"], ["first"], name="first"),
                helper.make_node("Relu", ["first"], ["second"], name="second"),
                helper.make_node("Mul", ["second", "scale"], ["third"], name="third"),
                helper.make_node("Add", ["third", "first"], ["output"], name="fourth"),
            ],
            "branch",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])],
            [
                numpy_helper.from_array(np.array([[1.0, -2.0, 3.0]], np.float32), "bias"),
                numpy_helper.from_array(np.array([[2.0, 2.0, 2.0]], np.float32), "scale"),
            ],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 9
        return model

    def test_partitions_preserve_named_live_values_and_weights(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source.onnx"
            onnx.save(self.make_model(), source)
            output = root / "parts"
            document = partition_model(source, output, ["second", "fourth"], "stage")
            self.assertEqual(3, len(document["partitions"]))
            self.assertEqual(["first"], [x["name"] for x in document["partitions"][0]["outputs"]])
            self.assertEqual(["first"], [x["name"] for x in document["partitions"][1]["inputs"]])
            self.assertEqual(
                ["third", "first"],
                [x["name"] for x in document["partitions"][2]["inputs"]],
            )
            self.assertEqual([1, 1, 0], [x["initializer_count"] for x in document["partitions"]])
            values = {"input": np.array([[2.0, 1.0, -4.0]], np.float32)}
            expected = ReferenceEvaluator(self.make_model()).run(None, values)[0]
            actual = run_partitioned_onnx(output / "partitions.json", values)[0]
            np.testing.assert_array_equal(expected, actual)

    def test_manifest_verification_rejects_tampered_partition(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source.onnx"
            onnx.save(self.make_model(), source)
            output = root / "parts"
            partition_model(source, output, ["second"], "stage")
            (output / "stage_00.onnx").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "payload verification failed"):
                read_partition_manifest(output / "partitions.json")

    def test_rejects_boundaries_out_of_graph_order(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source.onnx"
            onnx.save(self.make_model(), source)
            with self.assertRaisesRegex(ValueError, "graph order"):
                partition_model(source, root / "parts", ["fourth", "second"])

    def test_explicitly_resolves_a_symbolic_boundary_dimension(self):
        model = self.make_model()
        model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "batch"
        model.graph.output[0].type.tensor_type.shape.dim[0].dim_param = "batch"
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source.onnx"
            onnx.save(model, source)
            with self.assertRaisesRegex(ValueError, "dynamic or invalid shape"):
                partition_model(source, root / "invalid", ["second"])
            document = partition_model(
                source,
                root / "parts",
                ["second"],
                dimension_values={"batch": 1},
            )
            self.assertEqual([1, 3], document["partitions"][0]["outputs"][0]["shape"])

    def test_constant_nodes_become_partition_local_initializers(self):
        graph = helper.make_graph(
            [
                helper.make_node(
                    "Constant",
                    [],
                    ["constant"],
                    name="constant_node",
                    value=helper.make_tensor("", TensorProto.FLOAT, [], [2.0]),
                ),
                helper.make_node("Add", ["input", "constant"], ["middle"], name="middle"),
                helper.make_node("Mul", ["middle", "constant"], ["output"], name="final"),
            ],
            "constant_branch",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 9
        copied = onnx.ModelProto()
        copied.CopyFrom(model)
        self.assertEqual(1, fold_constant_nodes(copied))
        self.assertEqual(["middle", "final"], [node.name for node in copied.graph.node])
        self.assertIn("constant", {value.name for value in copied.graph.initializer})
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source.onnx"
            onnx.save(model, source)
            document = partition_model(source, root / "parts", ["final"])
            self.assertEqual(1, document["folded_constant_nodes"])
            self.assertNotIn(
                "constant",
                [item["name"] for item in document["partitions"][0]["outputs"]],
            )
            self.assertEqual([1, 1], [item["initializer_count"] for item in document["partitions"]])


if __name__ == "__main__":
    unittest.main()
