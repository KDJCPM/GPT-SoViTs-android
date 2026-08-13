import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import torch
import onnx
from onnx import TensorProto, helper
from safetensors.torch import save_file


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from assemble_v2pp_qnn_attachments import (
    FRONTEND_FILES,
    build_executor,
    read_base_manifest,
    runtime_partition_contract,
    wrap_component,
)
from wrap_qnn_ep_context import match_tensor_names, wrap_context


class QnnProductAssemblerTest(unittest.TestCase):
    def test_executor_requires_complete_runtime_reference_graph_set(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            conditioning = root / "conditioning.safetensors"
            save_file({"prompt_semantic": torch.arange(7).reshape(1, 7)}, conditioning)
            bert = root / "bert.json"
            bert.write_text(json.dumps({"token_length": 130}), encoding="utf-8")
            t2s = root / "t2s.json"
            t2s.write_text(
                json.dumps(
                    {
                        "phone_capacity": 128,
                        "prompt_phone_length": 65,
                        "prompt_semantic_length": 7,
                        "prefill_cache_length": 420,
                        "cache_capacity": 1024,
                        "layers": 24,
                        "hidden_size": 512,
                    }
                ),
                encoding="utf-8",
            )
            vits = root / "vits.json"
            vits.write_text(
                json.dumps(
                    {
                        "phone_capacity": 128,
                        "semantic_capacity": 512,
                        "samples_per_semantic": 1280,
                        "sample_rate": 32000,
                    }
                ),
                encoding="utf-8",
            )
            reference_ssl = root / "reference_ssl.json"
            reference_ssl.write_text(
                json.dumps({"pcm_samples": 80000, "ssl_frames": 249}), encoding="utf-8"
            )
            reference_prompt = root / "reference_prompt.json"
            reference_prompt.write_text(json.dumps({"semantic_length": 124}), encoding="utf-8")
            reference_conditioning = root / "reference_conditioning.json"
            reference_conditioning.write_text(
                json.dumps(
                    {
                        "pcm_16k_samples": 80000,
                        "pcm_32k_samples": 160000,
                        "spectrogram_reflect_pad": 704,
                        "spectrogram_bins": 1025,
                        "speaker_embedding_size": 20480,
                    }
                ),
                encoding="utf-8",
            )
            reference_t2s = root / "reference_t2s.json"
            reference_t2s.write_text(
                json.dumps(
                    {
                        "phone_capacity": 128,
                        "prompt_phone_capacity": 128,
                        "prompt_semantic_length": 124,
                        "prefill_cache_length": 380,
                        "cache_capacity": 1024,
                    }
                ),
                encoding="utf-8",
            )
            reference_vits = root / "reference_vits.json"
            reference_vits.write_text(
                json.dumps(
                    {
                        "phone_capacity": 128,
                        "semantic_capacity": 512,
                        "reference_spectrogram_frames": 250,
                        "sample_rate": 32000,
                    }
                ),
                encoding="utf-8",
            )
            executor = build_executor(
                conditioning=conditioning,
                bert_metadata=bert,
                t2s_metadata=t2s,
                vits_metadata=vits,
                reference_ssl_metadata=reference_ssl,
                reference_prompt_metadata=reference_prompt,
                reference_conditioning_metadata=reference_conditioning,
                reference_t2s_metadata=reference_t2s,
                reference_vits_metadata=reference_vits,
                g2pw_sequence_length=130,
                vits_partitions=[
                    {
                        "name": f"vits_{index:02d}",
                        "path": f"runtime/qnn/vits_{index:02d}.onnx",
                        "inputs": [],
                        "outputs": [],
                    }
                    for index in range(2)
                ],
                reference_vits_partitions=[
                    {
                        "name": f"vits_reference_{index:02d}",
                        "path": f"runtime/qnn/vits_reference_{index:02d}.onnx",
                        "inputs": [],
                        "outputs": [],
                    }
                    for index in range(2)
                ],
            )
            self.assertTrue(executor["complete"])
            self.assertEqual(0, executor["runtime_options_version"])
            self.assertEqual(1, executor["reference_input_version"])
            self.assertEqual(["auto", "zh", "en"], executor["languages"])
            self.assertEqual(512, executor["shapes"]["semantic_capacity"])
            self.assertEqual(65, executor["shapes"]["preset_prompt_phone_length"])
            self.assertEqual(list(range(7)), executor["preset"]["prompt_semantic"])
            self.assertEqual(80000, executor["reference"]["pcm_16k_samples"])
            self.assertEqual("exact_samples", executor["reference"]["duration_policy"])
            self.assertEqual(1025, executor["reference"]["reference_spectrogram_bins"])
            self.assertEqual(2, executor["engine_version"])
            self.assertEqual(
                "runtime/qnn/vits_reference_00.onnx",
                executor["graphs"]["vits_reference"][0]["path"],
            )

    def test_product_pipeline_contains_complete_zh_en_frontend_assets(self):
        self.assertTrue(
            {
                "frontend.json",
                "english.json",
                "english-lexicon.tsv",
                "english-homographs.tsv",
                "english-tagger.bin",
                "english-g2p.bin",
                "english-unigrams.tsv",
                "english-bigrams.tsv",
            }.issubset(FRONTEND_FILES)
        )

    def test_base_model_profile_is_read_from_the_actual_package(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "voice.gsvm"
            manifest = {
                "artifact_role": "model",
                "model_version": "v2ProPlus",
                "deployable": True,
                "executor": "torchscript-cpu-single",
                "frontend_profile": "full-g2pw-v2",
                "reference_input_version": 1,
            }
            with zipfile.ZipFile(package, "w") as archive:
                archive.writestr("manifest.json", json.dumps(manifest))
            self.assertEqual("full-g2pw-v2", read_base_manifest(package)["frontend_profile"])

    def test_base_model_without_runtime_reference_abi_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "voice.gsvm"
            manifest = {
                "artifact_role": "model",
                "model_version": "v2ProPlus",
                "deployable": True,
                "executor": "torchscript-cpu-single",
                "frontend_profile": "full-g2pw-v2",
            }
            with zipfile.ZipFile(package, "w") as archive:
                archive.writestr("manifest.json", json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "reference_input_version=1"):
                read_base_manifest(package)

    def test_wrapper_rejects_an_onnx_that_did_not_build_the_component(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            component = root / "component"
            component.mkdir()
            source = root / "source.onnx"
            source.write_bytes(b"wrong graph")
            (component / "manifest.json").write_text(
                json.dumps(
                    {
                        "source_onnx_sha256": "0" * 64,
                        "backend_artifact": "backend.bin",
                    }
                ),
                encoding="utf-8",
            )
            (component / "backend.bin").write_bytes(b"context")
            (component / "context-info.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                wrap_component(
                    name="test",
                    source=source,
                    component=component,
                    output=root / "wrapped",
                )

    def test_runtime_partition_contract_uses_compiled_io_types_and_shapes(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            wrapped = root / "wrapped.onnx"
            graph = helper.make_graph(
                [helper.make_node("Identity", ["_input"], ["_output"])],
                "wrapped",
                [helper.make_tensor_value_info("_input", TensorProto.FLOAT16, [1, 4])],
                [helper.make_tensor_value_info("_output", TensorProto.FLOAT16, [1, 4])],
            )
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
            model.ir_version = 9
            onnx.save(model, wrapped)
            stage = {
                "name": "vits_00",
                "path": "runtime/qnn/vits_00.onnx",
                "inputs": [{"name": "/input"}],
                "outputs": [{"name": "/output"}],
            }
            contract = runtime_partition_contract(stage, wrapped)
            self.assertEqual("FLOAT16", contract["inputs"][0]["data_type_name"])
            self.assertEqual([1, 4], contract["outputs"][0]["shape"])
            self.assertEqual("_input", contract["inputs"][0]["name"])
            self.assertEqual("/input", contract["inputs"][0]["logical_name"])
            self.assertEqual("/output", contract["outputs"][0]["logical_name"])

    def test_epcontext_wrapper_preserves_compiled_names_and_layout(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source.onnx"
            graph = helper.make_graph(
                [helper.make_node("Identity", ["/input"], ["/output"])],
                "source",
                [helper.make_tensor_value_info("/input", TensorProto.FLOAT, [1, 2, 3])],
                [helper.make_tensor_value_info("/output", TensorProto.FLOAT, [1, 2, 3])],
            )
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
            model.ir_version = 9
            onnx.save(model, source)
            context = root / "context.bin"
            context.write_bytes(b"context")
            info = root / "context-info.json"
            tensor = lambda name: {
                "info": {
                    "name": name,
                    "dataType": "QNN_DATATYPE_FLOAT_16",
                    "dimensions": [1, 3, 2],
                }
            }
            info.write_text(
                json.dumps(
                    {
                        "info": {
                            "graphs": [
                                {
                                    "info": {
                                        "graphInputs": [tensor("_input")],
                                        "graphOutputs": [tensor("_output")],
                                    }
                                }
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            wrapped, deployed = wrap_context(source, context, info, root / "wrapped.onnx")
            result = onnx.load(wrapped, load_external_data=False)
            self.assertEqual("_input", result.graph.input[0].name)
            self.assertEqual("_output", result.graph.output[0].name)
            self.assertEqual(TensorProto.FLOAT16, result.graph.input[0].type.tensor_type.elem_type)
            self.assertEqual([1, 3, 2], [item.dim_value for item in result.graph.input[0].type.tensor_type.shape.dim])
            self.assertEqual(b"context", deployed.read_bytes())
            changed = json.loads(info.read_text(encoding="utf-8"))
            changed["info"]["graphs"][0]["info"]["graphInputs"][0]["info"]["dimensions"] = [1, 4, 2]
            info.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed element count"):
                wrap_context(source, context, info, root / "invalid.onnx")

    def test_qnn_name_mapping_rejects_ambiguous_or_changed_interfaces(self):
        self.assertEqual(
            {"_dec_Mul_output_0": "/dec/Mul_output_0"},
            match_tensor_names(["/dec/Mul_output_0"], ["_dec_Mul_output_0"], "outputs"),
        )
        with self.assertRaisesRegex(ValueError, "cannot be mapped uniquely"):
            match_tensor_names(["/a", "_a"], ["_a", "_a_1"], "outputs")


if __name__ == "__main__":
    unittest.main()
