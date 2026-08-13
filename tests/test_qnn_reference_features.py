import sys
import unittest
from pathlib import Path

import torch
from torchaudio.compliance import kaldi


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from export_v2pp_qnn_reference_onnx import (
    FBANK_FRAMES,
    PCM_16K_SAMPLES,
    ReferenceConditioningEncoder,
    fbank_kernels,
)


class QnnReferenceFeatureTest(unittest.TestCase):
    def test_fixed_fbank_frontend_matches_torchaudio_kaldi(self):
        generator = torch.Generator().manual_seed(7)
        pcm = torch.randn((1, PCM_16K_SAMPLES), generator=generator) * 0.02
        real_weight, imaginary_weight, mel_weight = fbank_kernels()
        signal = pcm.unsqueeze(1)
        real = torch.nn.functional.conv1d(signal, real_weight, stride=160)
        imaginary = torch.nn.functional.conv1d(signal, imaginary_weight, stride=160)
        power = real.square() + imaginary.square()
        actual = torch.nn.functional.conv1d(power, mel_weight).clamp_min(
            torch.finfo(torch.float32).eps
        ).log().transpose(1, 2)
        expected = kaldi.fbank(
            pcm,
            num_mel_bins=80,
            sample_frequency=16000,
            dither=0,
        ).unsqueeze(0)
        self.assertEqual((1, FBANK_FRAMES, 80), tuple(actual.shape))
        self.assertLess(float((actual - expected).abs().max()), 2e-4)


if __name__ == "__main__":
    unittest.main()
