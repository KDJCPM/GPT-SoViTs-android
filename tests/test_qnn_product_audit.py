import hashlib
import json
import sys
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path

import onnx
from onnx import TensorProto, helper


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from audit_v2pp_qnn_product import (
    FRONTEND_FILES,
    MODEL_GRAPHS,
    PIPELINE_GRAPHS,
    audit_product,
)
from build_qnn_htp_context import ANDROID_QNN_RUNTIME_VERSION, TARGETS, TARGET_SOC_FAMILY


class QnnProductAuditTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def sha256_bytes(value: bytes) -> str:
        return hashlib.sha256(value).hexdigest()

    @staticmethod
    def sha256_file(path: Path) -> str:
        value = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                value.update(block)
        return value.hexdigest()

    def write_package(self, path: Path, manifest: dict, payloads: dict[str, bytes]) -> None:
        document = json.loads(json.dumps(manifest))
        document["files"] = [
            {
                "path": name,
                "size": len(value),
                "sha256": self.sha256_bytes(value),
            }
            for name, value in sorted(payloads.items())
        ]
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("manifest.json", json.dumps(document))
            for name, value in sorted(payloads.items()):
                archive.writestr(name, value)

    @staticmethod
    def read_package(path: Path) -> tuple[dict, dict[str, bytes]]:
        with zipfile.ZipFile(path) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            payloads = {
                name: archive.read(name)
                for name in archive.namelist()
                if name != "manifest.json"
            }
        return manifest, payloads

    @staticmethod
    def ep_context(name: str) -> tuple[bytes, str, str]:
        input_name = f"{name}_input"
        output_name = f"{name}_output"
        node = helper.make_node(
            "EPContext",
            [input_name],
            [output_name],
            domain="com.microsoft",
            ep_cache_context=f"{name}.bin",
            embed_mode=0,
            main_context=1,
            source="QNNExecutionProvider",
        )
        graph = helper.make_graph(
            [node],
            name,
            [helper.make_tensor_value_info(input_name, TensorProto.FLOAT16, [1])],
            [helper.make_tensor_value_info(output_name, TensorProto.FLOAT16, [1])],
        )
        model = helper.make_model(
            graph,
            opset_imports=[
                helper.make_opsetid("", 17),
                helper.make_opsetid("com.microsoft", 1),
            ],
        )
        model.ir_version = 9
        return model.SerializeToString(), input_name, output_name

    @staticmethod
    def attachment(role: str, names: tuple[str, ...]) -> bytes:
        document = {
            "format": "gsv-qnn-attachment",
            "format_version": 1,
            "role": role,
            "components": {
                name: {
                    "source_onnx_sha256": hashlib.sha256(name.encode()).hexdigest(),
                    "static_inputs": {"input": [1]},
                }
                for name in names
            },
        }
        return json.dumps(document).encode()

    @staticmethod
    def tensor(name: str, logical_name: str) -> dict:
        return {
            "name": name,
            "data_type": int(TensorProto.FLOAT16),
            "data_type_name": "FLOAT16",
            "shape": [1],
            "logical_name": logical_name,
        }

    def make_product(self) -> tuple[Path, Path, Path]:
        base = self.root / "voice-cpu.gsvm"
        self.write_package(
            base,
            {
                "artifact_role": "model",
                "format": "gsvm-deploy",
                "format_version": 1,
                "model_version": "v2ProPlus",
                "sample_rate": 32000,
                "deployable": True,
                "executor": "torchscript-cpu-staged",
                "entrypoint": "synthesize_utf8_to_pcm16",
                "api_version": 1,
                "frontend_profile": "full-zh-en-g2pw-v3",
                "reference_input_version": 1,
            },
            {},
        )
        base_sha256 = self.sha256_file(base)

        graph_contracts: dict[str, tuple[str, str]] = {}
        graph_payloads: dict[str, bytes] = {}
        for name in (*PIPELINE_GRAPHS, *MODEL_GRAPHS):
            model, input_name, output_name = self.ep_context(name)
            graph_contracts[name] = (input_name, output_name)
            graph_payloads[f"runtime/qnn/{name}.onnx"] = model
            graph_payloads[f"runtime/qnn/{name}.bin"] = f"context:{name}".encode()

        def stages(prefix: str) -> list[dict]:
            result = []
            for index in range(6):
                name = f"{prefix}_{index:02d}"
                input_name, output_name = graph_contracts[name]
                input_logical = f"{prefix}_runtime_input" if index == 0 else f"{prefix}_value_{index - 1}"
                output_logical = "audio" if index == 5 else f"{prefix}_value_{index}"
                result.append(
                    {
                        "name": name,
                        "path": f"runtime/qnn/{name}.onnx",
                        "inputs": [self.tensor(input_name, input_logical)],
                        "outputs": [self.tensor(output_name, output_logical)],
                    }
                )
            return result

        executor = {
            "format": "gsv-qnn-executor",
            "format_version": 1,
            "operation": "synthesize_utf8_to_pcm16",
            "runtime_abi_version": 1,
            "complete": True,
            "utf8_text_input": True,
            "pcm16_output": True,
            "cpu_neural_fallback": False,
            "runtime_options_version": 0,
            "reference_input_version": 1,
            "sample_rate": 32000,
            "languages": ["auto", "zh", "en"],
            "engine": "gpt-sovits-v2pp-qnn-buckets",
            "engine_version": 2,
            "frontend": {
                "root": "runtime/frontend",
                "g2pw_model": "runtime/qnn/g2pw.onnx",
                "g2pw_sequence_length": 130,
            },
            "graphs": {
                "bert": "runtime/qnn/bert.onnx",
                "t2s_prefill": "runtime/qnn/t2s_prefill.onnx",
                "t2s_step": "runtime/qnn/t2s_step.onnx",
                "vits": stages("vits"),
                "reference_ssl": "runtime/qnn/reference_ssl.onnx",
                "reference_prompt_semantic": "runtime/qnn/reference_prompt_semantic.onnx",
                "reference_conditioning": "runtime/qnn/reference_conditioning.onnx",
                "t2s_reference_prefill": "runtime/qnn/t2s_reference_prefill.onnx",
                "vits_reference": stages("vits_reference"),
            },
            "shapes": {
                "token_capacity": 130,
                "phone_capacity": 128,
                "semantic_capacity": 512,
                "preset_prompt_phone_length": 65,
                "prefill_cache_length": 200,
                "cache_capacity": 1024,
                "layers": 24,
                "hidden_size": 512,
                "samples_per_semantic": 1280,
                "eos_token": 1024,
                "padding_mask_inputs": True,
            },
            "reference": {
                "duration_policy": "exact_samples",
                "pcm_16k_samples": 80000,
                "pcm_32k_samples": 160000,
                "spectrogram_reflect_pad": 704,
                "ssl_frames": 249,
                "prompt_semantic_length": 124,
                "prompt_phone_capacity": 128,
                "prefill_cache_length": 380,
                "reference_spectrogram_bins": 1025,
                "reference_spectrogram_frames": 250,
                "speaker_embedding_size": 20480,
            },
            "preset": {"prompt_semantic": list(range(7))},
            "max_text_codepoints": 4000,
            "inter_segment_silence_ms": 150,
        }

        target_soc = "snapdragon_8_elite"
        target = TARGETS[target_soc]
        qairt_version = "2.48.0.260626"
        attachment_for = "gsvm:v2ProPlus:full-zh-en-g2pw-v3:api1"
        bundle_id = f"{attachment_for}:qnn-htp:{target.asic}:qairt-{qairt_version}"
        common = {
            "format": "gsvm-deploy",
            "format_version": 1,
            "model_version": "v2ProPlus",
            "sample_rate": 32000,
            "executor": "qnn-htp",
            "entrypoint": "synthesize_utf8_to_pcm16",
            "api_version": 1,
            "deployable": True,
            "frontend_profile": "full-zh-en-g2pw-v3",
            "target_soc": target_soc,
            "target_soc_family": TARGET_SOC_FAMILY,
            "target_asic": target.asic,
            "target_soc_model": target.soc_model,
            "supported_target_socs": [target_soc],
            "htp_arch": target.htp_arch,
            "qairt_version": qairt_version,
            "qnn_runtime_version": ANDROID_QNN_RUNTIME_VERSION,
            "precision": "fp16",
            "quantization": "none",
            "cpu_neural_fallback": False,
            "attachment_for": attachment_for,
            "bundle_id": bundle_id,
        }

        pipeline = self.root / "pipeline.qnn.gsvm"
        pipeline_payloads = {
            name: value
            for name, value in graph_payloads.items()
            if any(f"/{graph}." in name for graph in PIPELINE_GRAPHS)
        }
        pipeline_payloads.update(
            {f"runtime/frontend/{name}": name.encode() for name in FRONTEND_FILES}
        )
        pipeline_payloads["runtime/qnn/attachment.json"] = self.attachment(
            "pipeline", PIPELINE_GRAPHS
        )
        self.write_package(
            pipeline,
            {
                **common,
                "artifact_role": "qnn-pipeline-attachment",
                "requires_role": "qnn-model-attachment",
                "backend_artifact": "runtime/qnn/attachment.json",
            },
            pipeline_payloads,
        )

        model = self.root / "voice.qnn.gsvm"
        model_payloads = {
            name: value
            for name, value in graph_payloads.items()
            if any(f"/{graph}." in name for graph in MODEL_GRAPHS)
        }
        model_payloads["runtime/qnn/attachment.json"] = self.attachment("model", MODEL_GRAPHS)
        model_payloads["runtime/qnn/executor.json"] = json.dumps(executor).encode()
        self.write_package(
            model,
            {
                **common,
                "artifact_role": "qnn-model-attachment",
                "requires_role": "qnn-pipeline-attachment",
                "backend_artifact": "runtime/qnn/executor.json",
                "base_model_sha256": base_sha256,
                "runtime_options_version": 0,
                "reference_input_version": 1,
                "reference_input": {
                    "preset_when_omitted": True,
                    "duration_policy": "exact_samples",
                    "pcm": [
                        {
                            "sample_rate": 16000,
                            "channels": 1,
                            "dtype": "float32",
                            "samples": 80000,
                        },
                        {
                            "sample_rate": 32000,
                            "channels": 1,
                            "dtype": "float32",
                            "samples": 160000,
                        },
                    ],
                },
            },
            model_payloads,
        )
        return pipeline, model, base

    def test_valid_paired_product_passes(self):
        pipeline, model, base = self.make_product()
        report = audit_product(pipeline, model, base)
        self.assertTrue(report["verified"])
        self.assertEqual(4, report["pipeline"]["ep_contexts"])
        self.assertEqual(16, report["model"]["ep_contexts"])
        self.assertEqual(19, report["graph_references"])

    def test_duplicate_zip_entry_fails(self):
        pipeline, model, base = self.make_product()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(pipeline, "a") as archive:
                archive.writestr("runtime/frontend/frontend.json", b"duplicate")
        with self.assertRaisesRegex(ValueError, "duplicate ZIP entries"):
            audit_product(pipeline, model, base)

    def test_tampered_payload_fails(self):
        pipeline, model, base = self.make_product()
        manifest, payloads = self.read_package(pipeline)
        name = "runtime/frontend/frontend.json"
        payloads[name] = bytes([payloads[name][0] ^ 1]) + payloads[name][1:]
        with zipfile.ZipFile(pipeline, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("manifest.json", json.dumps(manifest))
            for payload_name, value in sorted(payloads.items()):
                archive.writestr(payload_name, value)
        with self.assertRaisesRegex(ValueError, "payload SHA-256 mismatch"):
            audit_product(pipeline, model, base)

    def test_tampered_manifest_hash_fails(self):
        pipeline, model, base = self.make_product()
        manifest, payloads = self.read_package(pipeline)
        manifest["files"][0]["sha256"] = "0" * 64
        with zipfile.ZipFile(pipeline, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("manifest.json", json.dumps(manifest))
            for payload_name, value in sorted(payloads.items()):
                archive.writestr(payload_name, value)
        with self.assertRaisesRegex(ValueError, "payload SHA-256 mismatch"):
            audit_product(pipeline, model, base)

    def test_wrong_base_model_binding_fails(self):
        pipeline, model, base = self.make_product()
        manifest, payloads = self.read_package(model)
        manifest["base_model_sha256"] = "0" * 64
        self.write_package(model, manifest, payloads)
        with self.assertRaisesRegex(ValueError, "base_model_sha256"):
            audit_product(pipeline, model, base)

    def test_extra_production_payload_fails(self):
        pipeline, model, base = self.make_product()
        manifest, payloads = self.read_package(pipeline)
        payloads["runtime/unneeded-checkpoint.bin"] = b"not a runtime dependency"
        self.write_package(pipeline, manifest, payloads)
        with self.assertRaisesRegex(ValueError, "package file set differs"):
            audit_product(pipeline, model, base)

    def test_partition_tensor_contract_mismatch_fails(self):
        pipeline, model, base = self.make_product()
        manifest, payloads = self.read_package(model)
        executor = json.loads(payloads["runtime/qnn/executor.json"])
        executor["graphs"]["vits"][0]["inputs"][0]["shape"] = [2]
        payloads["runtime/qnn/executor.json"] = json.dumps(executor).encode()
        self.write_package(model, manifest, payloads)
        with self.assertRaisesRegex(ValueError, "shape"):
            audit_product(pipeline, model, base)

    def test_missing_executor_graph_fails(self):
        pipeline, model, base = self.make_product()
        manifest, payloads = self.read_package(model)
        executor = json.loads(payloads["runtime/qnn/executor.json"])
        del executor["graphs"]["t2s_step"]
        payloads["runtime/qnn/executor.json"] = json.dumps(executor).encode()
        self.write_package(model, manifest, payloads)
        with self.assertRaisesRegex(ValueError, "unexpected graph set"):
            audit_product(pipeline, model, base)


if __name__ == "__main__":
    unittest.main()
