import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from model_profiles import HEADER_PROFILES, detect_sovits
from convert_model import SUPPORTED_DEPLOYMENT_PROFILES


class ModelProfilesTest(unittest.TestCase):
    def test_mobile_scope_is_v2_pro_plus_and_v4_only(self):
        self.assertEqual({"v2ProPlus", "v4"}, SUPPORTED_DEPLOYMENT_PROFILES)

    def test_all_version_headers(self):
        for header, (expected, lora) in HEADER_PROFILES.items():
            with self.subTest(header=header), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "model.pth"
                path.write_bytes(header + b"not-needed-for-header-detection")
                profile, actual_lora, actual_header = detect_sovits(path)
                self.assertEqual(expected, profile.id)
                self.assertEqual(lora, actual_lora)
                self.assertEqual(header, actual_header)

    def test_unknown_header_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pth"
            path.write_bytes(b"XXinvalid")
            with self.assertRaises(ValueError):
                detect_sovits(path)

    def test_android_builder_uses_full_bilingual_staged_artifacts(self):
        builder = (
            Path(__file__).parents[1] / "tools/build_android_cpu_pipeline.py"
        ).read_text(encoding="utf-8")
        self.assertIn("build/g2pw-mobile-v3", builder)
        self.assertIn("full-zh-en-g2pw-v3", builder)
        self.assertIn("'--bert-stage',bert.name,'--acoustic-stage','pipeline_core.pt'", builder)
        self.assertNotIn("here/'build_pipeline.py'", builder)

if __name__ == "__main__":
    unittest.main()
