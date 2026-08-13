import hashlib
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from validate_v2pp_cpu_artifacts import deployment_files


class CpuValidationTest(unittest.TestCase):
    def test_output_path_is_fixed_before_upstream_changes_working_directory(self):
        source = (
            Path(__file__).parents[1] / "tools/validate_v2pp_cpu_artifacts.py"
        ).read_text(encoding="utf-8")
        main = source[source.index("def main()") :]
        self.assertLess(
            main.index("output = args.output.resolve()"),
            main.index("source_t2s, source_vits = load_upstream_modules"),
        )

    def test_deployment_file_manifest_matches_packaged_paths_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frontend = root / "frontend"
            frontend.mkdir()
            bert = root / "bert.pt"
            acoustic = root / "acoustic.pt"
            bert.write_bytes(b"bert")
            acoustic.write_bytes(b"acoustic")
            (frontend / "frontend.json").write_bytes(b"frontend")
            nested = frontend / "english"
            nested.mkdir()
            (nested / "lexicon.tsv").write_bytes(b"lexicon")

            values = deployment_files(bert, acoustic, frontend)

            self.assertEqual(
                [
                    "runtime/bert.pt",
                    "runtime/acoustic.pt",
                    "runtime/frontend/english/lexicon.tsv",
                    "runtime/frontend/frontend.json",
                ],
                [item["path"] for item in values],
            )
            for item in values:
                relative = item["path"]
                if relative == "runtime/bert.pt":
                    source = bert
                elif relative == "runtime/acoustic.pt":
                    source = acoustic
                else:
                    source = frontend / relative.removeprefix("runtime/frontend/")
                self.assertEqual(source.stat().st_size, item["size"])
                self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), item["sha256"])


if __name__ == "__main__":
    unittest.main()
