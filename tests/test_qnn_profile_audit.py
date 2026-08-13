import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from audit_android_qnn_profiles import audit_profile, collect_profiles


class QnnProfileAuditTest(unittest.TestCase):
    def write_profile(self, root: Path, providers: list[str]) -> Path:
        path = root / "profile.json"
        path.write_text(
            json.dumps([
                {
                    "cat": "Node",
                    "dur": "7",
                    "args": {"provider": provider},
                }
                for provider in providers
            ]),
            encoding="utf-8",
        )
        return path

    def test_accepts_qnn_only_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            result = audit_profile(
                self.write_profile(Path(directory), ["QNNExecutionProvider", "QNN"])
            )
        self.assertEqual(2, result["qnn_nodes"])
        self.assertEqual(0, result["cpu_nodes"])
        self.assertEqual(14, result["node_duration_us"])

    def test_rejects_any_cpu_node(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_profile(
                Path(directory), ["QNNExecutionProvider", "CPUExecutionProvider"]
            )
            with self.assertRaisesRegex(ValueError, "to CPU"):
                audit_profile(path)

    def test_rejects_profile_without_qnn_nodes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_profile(Path(directory), ["UnknownProvider"])
            with self.assertRaisesRegex(ValueError, "no QNN"):
                audit_profile(path)

    def test_collects_profiles_recursively(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "nested"
            nested.mkdir()
            first = self.write_profile(root, ["QNN"])
            second = nested / "other.json"
            second.write_text("[]", encoding="utf-8")
            self.assertEqual(sorted([first.resolve(), second.resolve()]), collect_profiles([root]))


if __name__ == "__main__":
    unittest.main()
