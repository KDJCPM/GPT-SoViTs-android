import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).parents[1]


class CpuPackageTest(unittest.TestCase):
    @staticmethod
    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def build(self, root: Path, validated: bool, version: str = "v2ProPlus", options: bool = False) -> tuple[dict, dict]:
        artifacts = root / "artifacts"
        frontend = root / "frontend"
        artifacts.mkdir()
        frontend.mkdir()
        (artifacts / "bert.pt").write_bytes(b"bert")
        (artifacts / "acoustic.pt").write_bytes(b"acoustic")
        (frontend / "frontend.json").write_text("{}", encoding="utf-8")
        pipeline = root / "pipeline.gsvm"
        model = root / "model.gsvm"
        command = [
            sys.executable,
            str(ROOT / "tools/build_cpu_package.py"),
            "--artifacts", str(artifacts),
            "--pipeline-output", str(pipeline),
            "--model-output", str(model),
            "--name", "test",
            "--version", version,
            "--frontend", str(frontend),
            "--bert-stage", "bert.pt",
            "--acoustic-stage", "acoustic.pt",
            "--frontend-profile", "test-frontend",
        ]
        if validated:
            report = root / "validation.json"
            files = [
                {
                    "path": "runtime/bert.pt",
                    "size": (artifacts / "bert.pt").stat().st_size,
                    "sha256": self.digest(artifacts / "bert.pt"),
                },
                {
                    "path": "runtime/acoustic.pt",
                    "size": (artifacts / "acoustic.pt").stat().st_size,
                    "sha256": self.digest(artifacts / "acoustic.pt"),
                },
                {
                    "path": "runtime/frontend/frontend.json",
                    "size": (frontend / "frontend.json").stat().st_size,
                    "sha256": self.digest(frontend / "frontend.json"),
                },
            ]
            report.write_text(json.dumps({
                "format": "gsv-v2pp-cpu-upstream-validation",
                "format_version": 1,
                "passed": True,
                "model_version": version,
                "frontend_profile": "test-frontend",
                "sources": {
                    "gpt": {"sha256": "1" * 64},
                    "sovits": {"sha256": "2" * 64},
                },
                "files": files,
            }), encoding="utf-8")
            command.extend(("--upstream-equivalent", "--validation-report", str(report)))
        if options:
            command.append("--runtime-options")
        subprocess.run(command, check=True, capture_output=True, text=True)
        with zipfile.ZipFile(pipeline) as archive:
            pipeline_manifest = json.loads(archive.read("manifest.json"))
        with zipfile.ZipFile(model) as archive:
            model_manifest = json.loads(archive.read("manifest.json"))
        return pipeline_manifest, model_manifest

    def test_unvalidated_package_is_not_deployable(self):
        with tempfile.TemporaryDirectory() as directory:
            manifests = self.build(Path(directory), validated=False)
        for manifest in manifests:
            self.assertFalse(manifest["deployable"])
            self.assertFalse(manifest["upstream_equivalent"])

    def test_validated_package_is_deployable(self):
        with tempfile.TemporaryDirectory() as directory:
            manifests = self.build(Path(directory), validated=True)
        for manifest in manifests:
            self.assertTrue(manifest["deployable"])
            self.assertTrue(manifest["upstream_equivalent"])
            self.assertEqual(
                "gsv-v2pp-cpu-upstream-validation",
                manifest["upstream_validation"]["format"],
            )

    def test_naked_upstream_equivalent_claim_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            frontend = root / "frontend"
            artifacts.mkdir(); frontend.mkdir()
            (artifacts / "pipeline.pt").write_bytes(b"model")
            (frontend / "frontend.json").write_text("{}", encoding="utf-8")
            command = [
                sys.executable, str(ROOT / "tools/build_cpu_package.py"),
                "--artifacts", str(artifacts), "--output", str(root / "model.gsvm"),
                "--name", "test", "--version", "v2ProPlus",
                "--frontend", str(frontend), "--upstream-equivalent",
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(0, result.returncode)
            self.assertIn("--validation-report", result.stderr)

    def test_validation_report_is_bound_to_exact_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            frontend = root / "frontend"
            artifacts.mkdir(); frontend.mkdir()
            (artifacts / "bert.pt").write_bytes(b"bert")
            (artifacts / "acoustic.pt").write_bytes(b"acoustic")
            (frontend / "frontend.json").write_text("{}", encoding="utf-8")
            report = root / "validation.json"
            report.write_text(json.dumps({
                "format": "gsv-v2pp-cpu-upstream-validation", "format_version": 1,
                "passed": True, "model_version": "v2ProPlus",
                "frontend_profile": "test-frontend",
                "sources": {"gpt": {"sha256": "1" * 64}, "sovits": {"sha256": "2" * 64}},
                "files": [],
            }), encoding="utf-8")
            result = subprocess.run([
                sys.executable, str(ROOT / "tools/build_cpu_package.py"),
                "--artifacts", str(artifacts), "--model-output", str(root / "model.gsvm"),
                "--name", "test", "--version", "v2ProPlus", "--frontend", str(frontend),
                "--bert-stage", "bert.pt", "--acoustic-stage", "acoustic.pt",
                "--frontend-profile", "test-frontend", "--upstream-equivalent",
                "--validation-report", str(report),
            ], capture_output=True, text=True)
            self.assertNotEqual(0, result.returncode)
            self.assertIn("does not bind the exact deployment files", result.stderr)

    def test_runtime_options_are_model_specific_and_include_seed(self):
        for version, has_sample_steps in (("v2ProPlus", False), ("v4", True)):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as directory:
                manifests = self.build(
                    Path(directory),
                    validated=False,
                    version=version,
                    options=True,
                )
                for manifest in manifests:
                    options = manifest["runtime_options"]
                    self.assertIn("seed", options)
                    self.assertEqual(has_sample_steps, "sample_steps" in options)


if __name__ == "__main__":
    unittest.main()
