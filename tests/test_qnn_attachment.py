import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from build_qnn_attachment import QNN_SUFFIX, build_attachment, digest


class QnnAttachmentTest(unittest.TestCase):
    def make_component(self, root: Path) -> Path:
        component = root / "component"
        backend = component / "backend/context.bin"
        backend.parent.mkdir(parents=True)
        backend.write_bytes(b"qnn-context")
        info = component / "context-info.json"
        info.write_text("{}", encoding="utf-8")
        manifest = {
            "format": "gsv-qnn-compiled-component",
            "format_version": 1,
            "deployable": False,
            "executor": "qnn-htp",
            "precision": "fp16",
            "quantization": "none",
            "target_soc": "snapdragon_8_elite",
            "target_soc_family": "qualcomm_snapdragon_8",
            "target_asic": "SM8750",
            "target_soc_model": 69,
            "htp_arch": "V79",
            "qairt_version": "2.48.0.260626",
            "qnn_runtime_version": "2.48.0",
            "source_onnx_sha256": "source-hash",
            "static_inputs": {"tokens": [1, 8]},
            "files": [
                {"path": "backend/context.bin", "size": backend.stat().st_size, "sha256": digest(backend)},
                {"path": "context-info.json", "size": info.stat().st_size, "sha256": digest(info)},
            ],
        }
        (component / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return component

    def make_executor(self, root: Path) -> Path:
        executor = root / "executor.json"
        executor.write_text(
            json.dumps(
                {
                    "format": "gsv-qnn-executor",
                    "format_version": 1,
                    "operation": "synthesize_utf8_to_pcm16",
                    "runtime_abi_version": 1,
                    "runtime_options_version": 1,
                    "reference_input_version": 1,
                    "complete": True,
                    "utf8_text_input": True,
                    "pcm16_output": True,
                    "cpu_neural_fallback": False,
                    "reference": {
                        "duration_policy": "exact_samples",
                        "pcm_16k_samples": 80000,
                        "pcm_32k_samples": 160000,
                    },
                }
            ),
            encoding="utf-8",
        )
        return executor

    def test_pipeline_attachment_uses_required_suffix_and_exact_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            component = self.make_component(root)
            graph = root / "bert.onnx"
            graph.write_bytes(b"ep-context")
            output = root / f"v2pp-sm8750{QNN_SUFFIX}"
            manifest = build_attachment(
                output=output,
                role="pipeline",
                name="V2PP SM8750 QNN pipeline attachment",
                version="v2ProPlus",
                frontend_profile="portable-char-v1",
                target_soc="snapdragon_8_elite",
                components={"bert_p8": component},
                payload_mappings=[(graph, "runtime/qnn/bert_p8.onnx")],
                executor_descriptor=None,
                base_model=None,
                deployable=False,
            )
            self.assertEqual("qnn-pipeline-attachment", manifest["artifact_role"])
            self.assertEqual(69, manifest["target_soc_model"])
            self.assertFalse(manifest["cpu_neural_fallback"])
            with zipfile.ZipFile(output) as package:
                self.assertEqual("manifest.json", package.infolist()[0].filename)
                self.assertIn("runtime/qnn/attachment.json", package.namelist())

    def test_deployable_pipeline_does_not_require_voice_executor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / f"pipeline{QNN_SUFFIX}"
            manifest = build_attachment(
                output=output,
                role="pipeline",
                name="pipeline",
                version="v2ProPlus",
                frontend_profile="portable-char-v1",
                target_soc="snapdragon_8_elite",
                components={"bert": self.make_component(root)},
                payload_mappings=[],
                executor_descriptor=None,
                base_model=None,
                deployable=True,
            )
            self.assertTrue(manifest["deployable"])
            self.assertEqual("runtime/qnn/attachment.json", manifest["backend_artifact"])

    def test_wrong_suffix_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "must end"):
                build_attachment(
                    output=root / "attachment.gsvm",
                    role="pipeline",
                    name="invalid",
                    version="v2ProPlus",
                    frontend_profile="portable-char-v1",
                    target_soc="snapdragon_8_elite",
                    components={"bert": self.make_component(root)},
                    payload_mappings=[],
                    executor_descriptor=None,
                    base_model=None,
                    deployable=False,
                )

    def test_deployable_model_requires_complete_high_level_executor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            component = self.make_component(root)
            with self.assertRaisesRegex(ValueError, "high-level executor"):
                build_attachment(
                    output=root / f"voice{QNN_SUFFIX}",
                    role="model",
                    name="voice",
                    version="v2ProPlus",
                    frontend_profile="portable-char-v1",
                    target_soc="snapdragon_8_elite",
                    components={"vits": component},
                    payload_mappings=[],
                    executor_descriptor=None,
                    base_model=None,
                    deployable=True,
                )

    def test_model_manifest_exports_executor_option_abis(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_model = root / "voice.gsvm"
            base_model.write_bytes(b"cpu-voice")
            manifest = build_attachment(
                output=root / f"voice{QNN_SUFFIX}",
                role="model",
                name="voice",
                version="v2ProPlus",
                frontend_profile="portable-char-v1",
                target_soc="snapdragon_8_elite",
                components={"vits": self.make_component(root)},
                payload_mappings=[],
                executor_descriptor=self.make_executor(root),
                base_model=base_model,
                deployable=True,
            )
            self.assertEqual(1, manifest["runtime_options_version"])
            self.assertEqual(1, manifest["reference_input_version"])
            self.assertEqual("exact_samples", manifest["reference_input"]["duration_policy"])
            self.assertEqual(
                [(16000, 80000), (32000, 160000)],
                [
                    (item["sample_rate"], item["samples"])
                    for item in manifest["reference_input"]["pcm"]
                ],
            )


if __name__ == "__main__":
    unittest.main()
