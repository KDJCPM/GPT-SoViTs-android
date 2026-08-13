import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from build_pipeline import Utf8TtsPipeline
from export_v4_cpu_artifacts import V4AcousticPipeline


class FakeBert(torch.nn.Module):
    def forward(
        self,
        token_ids: torch.Tensor,
        mask: torch.Tensor,
        token_types: torch.Tensor,
        word2ph: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ones((word2ph.numel(), 4), dtype=torch.float32)


class FakeV2Acoustic(torch.nn.Module):
    def forward(
        self,
        text_seq: torch.Tensor,
        text_bert: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 10,
        top_p: float = 1.0,
        repetition_penalty: float = 1.35,
        speed_factor: float = 1.0,
        seed: int = -1,
    ) -> tuple[int, torch.Tensor]:
        return 32000, torch.ones(16, dtype=torch.int32)

    @torch.jit.export
    def synthesize_reference_options(
        self,
        text_seq: torch.Tensor,
        text_bert: torch.Tensor,
        reference_pcm_16k: torch.Tensor,
        reference_pcm_32k: torch.Tensor,
        prompt_phone_ids: torch.Tensor,
        prompt_bert: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 10,
        top_p: float = 1.0,
        repetition_penalty: float = 1.35,
        speed_factor: float = 1.0,
        sample_steps: int = 32,
        seed: int = -1,
    ) -> tuple[int, torch.Tensor]:
        return 32000, torch.tensor([top_k, sample_steps, seed], dtype=torch.int32)


class FakeV4T2s(torch.nn.Module):
    def forward(
        self,
        prompt_semantic: torch.Tensor,
        prompt_phone_ids: torch.Tensor,
        text_seq: torch.Tensor,
        prompt_bert: torch.Tensor,
        text_bert: torch.Tensor,
        top_k: torch.Tensor,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
    ) -> torch.Tensor:
        return torch.zeros((1, 1, 8), dtype=torch.long)


class FakeV4Voice(torch.nn.Module):
    def forward(
        self,
        semantic: torch.Tensor,
        text_seq: torch.Tensor,
        speaker_embedding: torch.Tensor,
        speed_factor: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ones((1, 512, 30), dtype=torch.float32)

    @torch.jit.export
    def reference(
        self,
        ssl_content: torch.Tensor,
        pcm_32k: torch.Tensor,
        prompt_phone_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.zeros((1, 8), dtype=torch.long),
            torch.ones((1, 512, 1), dtype=torch.float32),
            torch.ones((1, 512, 4), dtype=torch.float32),
            torch.ones((1, 100, 4), dtype=torch.float32),
        )


class FakeCfm(torch.nn.Module):
    def forward(
        self,
        feature: torch.Tensor,
        lengths: torch.Tensor,
        reference_mel: torch.Tensor,
        sample_steps: torch.Tensor,
    ) -> torch.Tensor:
        return torch.full((1, 100, feature.shape[1]), 0.25, dtype=torch.float32)


class FakeVocoder(torch.nn.Module):
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return torch.full((1, 1, mel.shape[2] * 2), 0.1, dtype=torch.float32)


class ReferenceExportAbiTest(unittest.TestCase):
    def test_v2pp_fused_reference_method_schema_and_order(self):
        trie = {
            "trie_heads": [-1],
            "trie_chars": [],
            "trie_children": [],
            "trie_next": [],
            "trie_phone_offsets": [-1],
            "trie_phone_lengths": [0],
            "trie_count_offsets": [-1],
            "trie_count_lengths": [0],
            "trie_phones": [],
            "trie_counts": [],
        }
        module = torch.jit.script(
            Utf8TtsPipeline(
                torch.jit.script(FakeBert()),
                torch.jit.script(FakeV2Acoustic()),
                {ord("a"): [1]},
                {ord("a"): 1},
                {ord("a"): False},
                trie,
            ).eval()
        )
        self.assertIn("synthesize_reference_preprocessed_options", module._c._method_names())
        schema = str(module._c._get_method("synthesize_reference_preprocessed_options").schema)
        self.assertLess(schema.index("prompt_phone_ids"), schema.index("reference_pcm_16k"))

        phone = torch.tensor([1, 2], dtype=torch.long)
        tokens = torch.tensor([101, 1, 2, 102], dtype=torch.long)
        word2ph = torch.tensor([1, 1], dtype=torch.int32)
        chinese = torch.ones(2, dtype=torch.float32)
        sample_rate, values = module.synthesize_reference_preprocessed_options(
            phone, tokens, word2ph, chinese,
            phone, tokens, word2ph, chinese,
            torch.zeros(16000), torch.zeros(32000),
            123, 0.8, 0.9, 7, 1.2, 1.1, 19,
        )
        self.assertEqual(32000, sample_rate)
        self.assertEqual([7, 19, 123], values.tolist())

    def test_v4_staged_reference_method_is_scriptable_and_callable(self):
        conditioning = {
            "prompt_semantic": torch.zeros((1, 8), dtype=torch.long),
            "prompt_phone_ids": torch.ones((1, 2), dtype=torch.long),
            "prompt_bert": torch.ones((2, 4), dtype=torch.float32),
            "speaker_embedding": torch.ones((1, 512, 1), dtype=torch.float32),
            "reference_feature": torch.ones((1, 512, 4), dtype=torch.float32),
            "reference_mel": torch.ones((1, 100, 4), dtype=torch.float32),
        }
        module = torch.jit.script(
            V4AcousticPipeline(
                torch.jit.script(FakeV4T2s()),
                torch.jit.script(FakeV4Voice()),
                torch.jit.script(FakeCfm()),
                torch.jit.script(FakeVocoder()),
                conditioning,
                32,
            ).eval()
        )
        module = torch.jit.freeze(module, preserved_attrs=["synthesize_reference_options"])
        self.assertIn("synthesize_reference_options", module._c._method_names())
        sample_rate, pcm = module.synthesize_reference_options(
            torch.ones((1, 2), dtype=torch.long),
            torch.ones((2, 4), dtype=torch.float32),
            torch.zeros((1, 16000), dtype=torch.float32),
            torch.zeros((1, 32000), dtype=torch.float32),
            torch.ones((1, 2), dtype=torch.long),
            torch.ones((2, 4), dtype=torch.float32),
            1.0, 10, 1.0, 1.35, 1.0, 17, 9,
        )
        self.assertEqual(48000, sample_rate)
        self.assertGreater(pcm.numel(), 0)
        self.assertEqual(torch.int32, pcm.dtype)


if __name__ == "__main__":
    unittest.main()
