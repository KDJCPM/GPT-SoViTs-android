import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_v2pp_qnn_product as product


class QnnProductBuildTests(unittest.TestCase):
    def write_json(self, path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def component(self, root: Path, source: Path, optimization_level: int = 1) -> product.GraphSpec:
        component = root / "component"
        backend = component / "backend/model.bin"
        backend.parent.mkdir(parents=True)
        backend.write_bytes(b"compiled-context")
        info = component / "context-info.json"
        self.write_json(
            info,
            {
                "info": {
                    "socModel": 69,
                    "buildId": "v2.48.0.260626-test",
                    "numGraphs": 1,
                    "contextMetadata": {"info": {"dspArch": 79}},
                }
            },
        )
        files = []
        for path in (backend, info):
            files.append(
                {
                    "path": path.relative_to(component).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": product.sha256(path),
                }
            )
        self.write_json(
            component / "manifest.json",
            {
                "format": "gsv-qnn-compiled-component",
                "target_soc": "snapdragon_8_elite",
                "target_asic": "SM8750",
                "target_soc_model": 69,
                "htp_arch": "V79",
                "qairt_version": "2.48.0.260626",
                "qnn_runtime_version": "2.48.0",
                "precision": "fp16",
                "quantization": "none",
                "graph_io_dtype": "float16",
                "htp_graph_optimization_level": optimization_level,
                "source_onnx_sha256": product.sha256(source),
                "files": files,
            },
        )
        return product.GraphSpec("test", source, component, optimization_level)

    def test_reuses_only_exact_verified_component(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "model.onnx"
            source.write_bytes(b"onnx-source")
            spec = self.component(root, source)
            reusable, reason = product.reusable_component(
                spec,
                target_soc="snapdragon_8_elite",
                qairt_version="2.48.0.260626",
            )
            self.assertTrue(reusable, reason)

            layout_spec = product.GraphSpec(
                spec.name,
                spec.source,
                spec.component,
                spec.optimization_level,
                preserve_io_layout=True,
            )
            reusable, reason = product.reusable_component(
                layout_spec,
                target_soc="snapdragon_8_elite",
                qairt_version="2.48.0.260626",
            )
            self.assertFalse(reusable)
            self.assertIn("preserve_io_layout", reason)

            source.write_bytes(b"changed-source")
            reusable, reason = product.reusable_component(
                spec,
                target_soc="snapdragon_8_elite",
                qairt_version="2.48.0.260626",
            )
            self.assertFalse(reusable)
            self.assertIn("source_onnx_sha256", reason)

    def test_venv_executable_path_does_not_resolve_its_python_symlink(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            target = root / "python3"
            target.write_bytes(b"python")
            link = root / "python"
            link.symlink_to(target.name)
            actual = product.absolute_executable(link)
            self.assertEqual(Path(os.path.abspath(link)), actual)
            self.assertNotEqual(target.resolve(), actual)

    def test_rejects_tampered_component_payload(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "model.onnx"
            source.write_bytes(b"onnx-source")
            spec = self.component(root, source)
            (spec.component / "backend/model.bin").write_bytes(b"tampered")
            reusable, reason = product.reusable_component(
                spec,
                target_soc="snapdragon_8_elite",
                qairt_version="2.48.0.260626",
            )
            self.assertFalse(reusable)
            self.assertIn("payload size mismatch", reason)

    def test_split_provenance_binds_source_rules_and_output(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source.onnx"
            output = root / "split.onnx"
            source.write_bytes(b"source")
            output.write_bytes(b"split")
            product.atomic_json(
                product.split_descriptor_path(output),
                product.split_document(source, output),
            )
            reusable, reason = product.reusable_split(source, output)
            self.assertTrue(reusable, reason)

            output.write_bytes(b"different")
            reusable, _ = product.reusable_split(source, output)
            self.assertFalse(reusable)

    def test_split_provenance_can_bind_target_specific_rules(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source.onnx"
            output = root / "split.onnx"
            source.write_bytes(b"source")
            output.write_bytes(b"split")
            groups = ((4, r"/decoder/resblocks\\.(6|7|8)/Conv"),)
            product.atomic_json(
                product.split_descriptor_path(output),
                product.split_document(source, output, groups),
            )
            reusable, reason = product.reusable_split(source, output, groups)
            self.assertTrue(reusable, reason)
            reusable, reason = product.reusable_split(source, output)
            self.assertFalse(reusable)
            self.assertIn("provenance", reason)

    def test_reference_validation_binds_every_graph_hash(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            graphs = {}
            for name in (
                "reference_ssl",
                "reference_prompt_semantic",
                "reference_conditioning",
                "t2s_reference_prefill",
                "vits_reference",
            ):
                path = root / f"{name}.onnx"
                path.write_bytes(name.encode())
                graphs[name] = path
            report = root / "validation.json"
            self.write_json(
                report,
                {
                    "format": "gsv-v2pp-qnn-reference-onnx-validation",
                    "absolute_tolerance": 0.0002,
                    "maximum_absolute_error": 0.0001,
                    "graphs": {
                        name: {"sha256": product.sha256(path)}
                        for name, path in graphs.items()
                    },
                },
            )
            reusable, reason = product.reusable_validation(report, graphs)
            self.assertTrue(reusable, reason)
            graphs["reference_ssl"].write_bytes(b"changed")
            reusable, reason = product.reusable_validation(report, graphs)
            self.assertFalse(reusable)
            self.assertIn("reference_ssl", reason)

    def test_preset_validation_binds_all_final_graphs(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            graphs = {}
            for name in ("bert", "t2s_prefill", "t2s_step", "vits"):
                path = root / f"{name}.onnx"
                path.write_bytes(name.encode())
                graphs[name] = path
            report = root / "validation.json"
            self.write_json(
                report,
                {
                    "format": "gsv-v2pp-qnn-onnx-fp32-validation",
                    "graphs": {
                        name: {"sha256": product.sha256(path)}
                        for name, path in graphs.items()
                    },
                    "bert": {},
                    "t2s": {},
                    "vits": {},
                },
            )
            reusable, reason = product.reusable_preset_validation(report, graphs)
            self.assertTrue(reusable, reason)
            graphs["vits"].write_bytes(b"changed")
            reusable, reason = product.reusable_preset_validation(report, graphs)
            self.assertFalse(reusable)
            self.assertIn("vits", reason)

    def test_partition_validation_binds_manifest_and_every_stage(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source.onnx"
            source.write_bytes(b"source")
            parts = []
            for index in range(2):
                path = root / f"part_{index}.onnx"
                path.write_bytes(f"part-{index}".encode())
                parts.append(
                    {
                        "index": index,
                        "path": path.name,
                        "sha256": product.sha256(path),
                    }
                )
            manifest = root / "partitions.json"
            self.write_json(
                manifest,
                {
                    "format": "gsv-onnx-contiguous-partitions",
                    "format_version": 1,
                    "source": {"path": str(source), "sha256": product.sha256(source)},
                    "partitions": parts,
                },
            )
            recorded = {
                "manifest": {"sha256": product.sha256(manifest)},
                "parts": [{"sha256": product.sha256(root / item["path"])} for item in parts],
            }
            reusable, reason = product.validation_covers_partitions(recorded, manifest)
            self.assertTrue(reusable, reason)
            (root / "part_1.onnx").write_bytes(b"changed")
            reusable, reason = product.validation_covers_partitions(recorded, manifest)
            self.assertFalse(reusable)
            self.assertIn("verification failed", reason)


if __name__ == "__main__":
    unittest.main()
