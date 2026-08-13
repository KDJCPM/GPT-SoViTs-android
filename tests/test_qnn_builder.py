import json
import sys
import tempfile
import unittest
from pathlib import Path

import onnx
from onnx import TensorProto, helper


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from build_qnn_htp_context import (
    ANDROID_QNN_RUNTIME_VERSION,
    TARGETS,
    build_commands,
    parse_dimensions,
    sdk_version,
    validate_context_metadata,
    validate_preserved_io_layout,
    write_htp_backend_configs,
)


class QnnBuilderTest(unittest.TestCase):
    def test_product_allowlist_and_htp_architectures_are_exact(self):
        self.assertEqual(
            {
                "snapdragon_8_gen_3": ("SM8650", "V75"),
                "snapdragon_8_elite": ("SM8750", "V79"),
                "snapdragon_8_elite_gen_5": ("SM8850", "V81"),
            },
            {name: (value.asic, value.htp_arch) for name, value in TARGETS.items()},
        )
        self.assertEqual(
            {"SM8650": 57, "SM8750": 69, "SM8850": 87},
            {value.asic: value.soc_model for value in TARGETS.values()},
        )
        self.assertNotIn("snapdragon_8_gen_5", TARGETS)

    def test_sdk_must_match_android_qnn_runtime_line(self):
        self.assertEqual("2.48.0.260626", sdk_version(Path("/sdk/2.48.0.260626")))
        with self.assertRaisesRegex(ValueError, "does not match"):
            sdk_version(Path("/sdk/2.42.0.251225"))
        self.assertEqual("2.48.0", ANDROID_QNN_RUNTIME_VERSION)

    def test_converter_builds_fp16_for_one_soc(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sdk = root / "2.48.0.260626"
            tools = sdk / "bin/x86_64-linux-clang"
            tools.mkdir(parents=True)
            for name in ("qnn-onnx-converter", "qnn-model-lib-generator", "qnn-context-binary-generator"):
                (tools / name).touch()
            commands = build_commands(
                sdk,
                root / "graph.onnx",
                root / "work",
                TARGETS["snapdragon_8_elite"],
                [("audio", (1, 32000))],
            )
            converter, _, context, expected = commands
            config_argument = next(
                value for value in context if value.startswith("--config_file=")
            )
            main_config = Path(config_argument.split("=", 1)[1])
            main_document = json.loads(main_config.read_text(encoding="utf-8"))
            backend_config = Path(main_document["backend_extensions"]["config_file_path"])
            backend_document = json.loads(backend_config.read_text(encoding="utf-8"))
        self.assertIn("--float_bitwidth", converter)
        self.assertEqual("16", converter[converter.index("--float_bitwidth") + 1])
        self.assertIn("--float_bias_bitwidth", converter)
        self.assertEqual("16", converter[converter.index("--float_bias_bitwidth") + 1])
        self.assertNotIn("--preserve_io", converter)
        self.assertIn("--input_dim", converter)
        self.assertNotIn("--htp_socs=sm8750", context)
        self.assertEqual(
            [{"soc_model": 69, "dsp_arch": "v79"}], backend_document["devices"]
        )
        self.assertEqual(2, backend_document["graphs"][0]["O"])
        self.assertTrue(str(expected).endswith("snapdragon_8_elite_fp16.SM8750.bin"))

    def test_converter_can_preserve_layout_without_preserving_fp32_dtype(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sdk = root / "2.48.0.260626"
            tools = sdk / "bin/x86_64-linux-clang"
            tools.mkdir(parents=True)
            for name in (
                "qnn-onnx-converter",
                "qnn-model-lib-generator",
                "qnn-context-binary-generator",
            ):
                (tools / name).touch()
            converter, _, _, _ = build_commands(
                sdk,
                root / "graph.onnx",
                root / "work",
                TARGETS["snapdragon_8_elite"],
                [],
                preserve_io_layout=True,
            )
        self.assertIn("--float_bitwidth", converter)
        self.assertEqual("16", converter[converter.index("--float_bitwidth") + 1])
        self.assertEqual(
            ["--preserve_io", "layout"],
            converter[converter.index("--preserve_io") : converter.index("--preserve_io") + 2],
        )
        self.assertNotIn("datatype", converter)

    def test_htp_optimization_level_is_explicit_and_validated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sdk = root / "2.48.0.260626"
            (sdk / "lib/x86_64-linux-clang").mkdir(parents=True)
            main_config = write_htp_backend_configs(
                sdk,
                root / "work",
                TARGETS["snapdragon_8_elite"],
                optimization_level=1,
            )
            main_document = json.loads(main_config.read_text(encoding="utf-8"))
            backend_config = Path(main_document["backend_extensions"]["config_file_path"])
            backend_document = json.loads(backend_config.read_text(encoding="utf-8"))
            self.assertEqual(1, backend_document["graphs"][0]["O"])
            o0_config = write_htp_backend_configs(
                sdk,
                root / "o0",
                TARGETS["snapdragon_8_elite"],
                optimization_level=0,
            )
            o0_main = json.loads(o0_config.read_text(encoding="utf-8"))
            o0_backend = Path(o0_main["backend_extensions"]["config_file_path"])
            self.assertEqual(0, json.loads(o0_backend.read_text(encoding="utf-8"))["graphs"][0]["O"])
            with self.assertRaisesRegex(ValueError, "between 0 and 3"):
                write_htp_backend_configs(
                    sdk,
                    root / "invalid",
                    TARGETS["snapdragon_8_elite"],
                    optimization_level=4,
                )

    def test_context_metadata_must_match_exact_soc_and_arch(self):
        document = {
            "info": {
                "buildId": "v2.48.0.260626120635",
                "numGraphs": 1,
                "socModel": 69,
                "contextMetadata": {"info": {"dspArch": 79}},
            }
        }
        validate_context_metadata(
            document, TARGETS["snapdragon_8_elite"], "2.48.0.260626"
        )
        document["info"]["socModel"] = 0
        document["info"]["contextMetadata"]["info"]["dspArch"] = 68
        with self.assertRaisesRegex(RuntimeError, "wrong HTP target"):
            validate_context_metadata(
                document, TARGETS["snapdragon_8_elite"], "2.48.0.260626"
            )

    def test_shape_override_parser_is_strict(self):
        self.assertEqual({"audio": (1, 32000)}, parse_dimensions(["audio=1,32000"]))
        with self.assertRaises(ValueError):
            parse_dimensions(["audio=1,-1"])
        with self.assertRaises(ValueError):
            parse_dimensions(["audio=1", "audio=2"])

    def test_preserved_layout_validation_rejects_axis_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.onnx"
            graph = helper.make_graph(
                [helper.make_node("Identity", ["/input"], ["/output"])],
                "source",
                [helper.make_tensor_value_info("/input", TensorProto.FLOAT, [1, 3, 8])],
                [helper.make_tensor_value_info("/output", TensorProto.FLOAT, [1, 3, 8])],
            )
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
            model.ir_version = 9
            onnx.save(model, source)
            info = root / "context-info.json"

            def document(shape):
                return {
                    "info": {
                        "graphs": [
                            {
                                "info": {
                                    "graphInputs": [
                                        {"info": {"name": "_input", "dimensions": shape}}
                                    ],
                                    "graphOutputs": [
                                        {"info": {"name": "_output", "dimensions": shape}}
                                    ],
                                }
                            }
                        ]
                    }
                }

            info.write_text(json.dumps(document([1, 3, 8])), encoding="utf-8")
            validate_preserved_io_layout(source, info)
            info.write_text(json.dumps(document([1, 8, 3])), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "did not preserve"):
                validate_preserved_io_layout(source, info)

    def test_gradle_runtime_pin_matches_builder(self):
        gradle = (ROOT / "app/build.gradle.kts").read_text(encoding="utf-8")
        self.assertIn(f'val qnnRuntimeVersion = "{ANDROID_QNN_RUNTIME_VERSION}"', gradle)
        self.assertIn('implementation("com.qualcomm.qti:qnn-runtime:$qnnRuntimeVersion")', gradle)

    def test_apk_excludes_unused_qnn_backends_and_unsupported_htp_architectures(self):
        gradle = (ROOT / "app/build.gradle.kts").read_text(encoding="utf-8")
        for library in (
            "libQnnGpu.so",
            "libQnnDsp.so",
            "libQnnDspV66Skel.so",
            "libQnnDspV66Stub.so",
            "libQnnHtpV68Skel.so",
            "libQnnHtpV68Stub.so",
            "libQnnHtpV69Skel.so",
            "libQnnHtpV69Stub.so",
            "libQnnHtpV73Skel.so",
            "libQnnHtpV73Stub.so",
        ):
            self.assertIn(f'"**/{library}"', gradle)
        for architecture in ("V75", "V79", "V81"):
            self.assertNotIn(f'"**/libQnnHtp{architecture}Skel.so"', gradle)
            self.assertNotIn(f'"**/libQnnHtp{architecture}Stub.so"', gradle)

    def test_static_acceptance_disables_cpu_fallback_at_both_package_boundaries(self):
        assembler = (ROOT / "tools/assemble_v2pp_qnn_static_acceptance.py").read_text(
            encoding="utf-8"
        )
        android_probe = (
            ROOT / "app/src/main/java/ai/gsv/mobile/QnnV2ppRuntime.kt"
        ).read_text(encoding="utf-8")
        self.assertEqual(2, assembler.count('"cpu_neural_fallback": False'))
        self.assertEqual(2, android_probe.count('getBoolean("cpu_neural_fallback")'))

    def test_qnn_profiling_is_debug_only(self):
        runtime = (ROOT / "app/src/main/java/ai/gsv/mobile/QnnV2ppRuntime.kt").read_text(
            encoding="utf-8"
        )
        g2pw = (ROOT / "app/src/main/java/ai/gsv/mobile/G2pwOnnx.kt").read_text(
            encoding="utf-8"
        )
        self.assertIn("val profilePrefix = if (BuildConfig.DEBUG)", runtime)
        self.assertIn("private val profilingEnabled = qnnEnabled && BuildConfig.DEBUG", g2pw)
        self.assertNotIn("if (qnnEnabled) finishProfiling()", g2pw)

    def test_adb_acceptance_entrypoints_are_debug_only(self):
        activity = (ROOT / "app/src/main/java/ai/gsv/mobile/MainActivity.kt").read_text(
            encoding="utf-8"
        )
        runner = (ROOT / "app/src/main/java/ai/gsv/mobile/DebugAcceptanceRunner.kt").read_text(
            encoding="utf-8"
        )
        self.assertIn("DebugAcceptanceRunner.launch(this, intent)", activity)
        self.assertNotIn("QnnV2ppRuntime", activity)
        self.assertNotIn("QnnProductAcceptance", activity)
        self.assertNotIn("QnnEpContextProbe", activity)
        debug_gate = runner.index("if (!BuildConfig.DEBUG) return")
        for extra in (
            '"qnn_epcontext_probe"',
            '"qnn_v2pp_acceptance"',
            '"qnn_product_model"',
        ):
            self.assertGreater(runner.index(extra), debug_gate)

    def test_qnn_runtime_dispatches_by_artifact_engine_abi(self):
        backend = (ROOT / "app/src/main/java/ai/gsv/mobile/TtsBackend.kt").read_text(
            encoding="utf-8"
        )
        self.assertIn('when (val engine = descriptor.optString("engine"))', backend)
        self.assertNotIn("when (model.version)", backend)


if __name__ == "__main__":
    unittest.main()
