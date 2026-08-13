import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from normalize_qnn_g2pw_descriptor import (
    CHAR_WEIGHT_NAME,
    POSITION_WEIGHT_NAME,
    QNN_STATIC_ALIGNMENT_BYTES,
    WEIGHT_NAME,
    normalize_descriptor,
)


def make_model() -> onnx.ModelProto:
    values = np.arange(10 * 5, dtype=np.float32).reshape(10, 5)
    graph = helper.make_graph(
        [helper.make_node("Gather", [WEIGHT_NAME, "index"], ["output"], axis=0)],
        "descriptor-test",
        [helper.make_tensor_value_info("index", TensorProto.INT64, [1])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 5])],
        [
            numpy_helper.from_array(values, WEIGHT_NAME),
            numpy_helper.from_array(np.zeros((2, 5), dtype=np.float32), CHAR_WEIGHT_NAME),
            numpy_helper.from_array(np.zeros((5, 5), dtype=np.float32), POSITION_WEIGHT_NAME),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model


class QnnG2pwDescriptorTest(unittest.TestCase):
    def test_alignment_preserves_every_legal_lookup(self):
        original = make_model()
        rewritten = onnx.ModelProto()
        rewritten.CopyFrom(original)
        summary = normalize_descriptor(rewritten)
        self.assertEqual(10, summary["used_rows"])
        self.assertEqual(12, summary["padded_rows"])
        self.assertEqual(0, summary["fp16_bytes"] % QNN_STATIC_ALIGNMENT_BYTES)
        weight = next(value for value in rewritten.graph.initializer if value.name == WEIGHT_NAME)
        aligned = numpy_helper.to_array(weight)
        np.testing.assert_array_equal(aligned[:10], np.arange(50, dtype=np.float32).reshape(10, 5))
        np.testing.assert_array_equal(aligned[10:], np.zeros((2, 5), dtype=np.float32))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original_path = root / "original.onnx"
            rewritten_path = root / "rewritten.onnx"
            onnx.save(original, original_path)
            onnx.save(rewritten, rewritten_path)
            original_session = ort.InferenceSession(str(original_path), providers=["CPUExecutionProvider"])
            rewritten_session = ort.InferenceSession(str(rewritten_path), providers=["CPUExecutionProvider"])
            for index in range(10):
                inputs = {"index": np.asarray([index], dtype=np.int64)}
                np.testing.assert_array_equal(
                    original_session.run(None, inputs)[0],
                    rewritten_session.run(None, inputs)[0],
                )

    def test_rejects_descriptor_row_mismatch(self):
        model = make_model()
        char_weight = next(value for value in model.graph.initializer if value.name == CHAR_WEIGHT_NAME)
        char_weight.CopyFrom(
            numpy_helper.from_array(np.zeros((3, 5), dtype=np.float32), CHAR_WEIGHT_NAME)
        )
        with self.assertRaisesRegex(ValueError, "rows, expected"):
            normalize_descriptor(model)

    def test_accepts_an_already_aligned_descriptor_idempotently(self):
        model = make_model()
        first = normalize_descriptor(model)
        self.assertEqual(2, first["added_rows"])
        second = normalize_descriptor(model)
        self.assertEqual(0, second["added_rows"])
        self.assertEqual(2, second["existing_padding_rows"])

    def test_rejects_nonzero_alignment_padding(self):
        model = make_model()
        normalize_descriptor(model)
        weight = next(value for value in model.graph.initializer if value.name == WEIGHT_NAME)
        values = numpy_helper.to_array(weight).copy()
        values[-1, -1] = 1.0
        weight.CopyFrom(numpy_helper.from_array(values, WEIGHT_NAME))
        with self.assertRaisesRegex(ValueError, "padding rows must remain zero"):
            normalize_descriptor(model)


if __name__ == "__main__":
    unittest.main()
