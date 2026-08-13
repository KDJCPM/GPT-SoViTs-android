#!/usr/bin/env python3
"""Fuse V2PP modules, preset conditioning and runtime reference encoding into one CPU graph."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from safetensors.torch import load_file


class AcousticPipeline(torch.nn.Module):
    def __init__(self, t2s, vits, ssl, conditioning: dict[str, torch.Tensor], eos: int = 1024, max_tokens: int = 1500):
        super().__init__()
        self.t2s = t2s
        self.vits = vits
        self.ssl = ssl
        self.eos = eos
        self.max_tokens = max_tokens
        self.register_buffer("prompt_semantic", conditioning["prompt_semantic"])
        self.register_buffer("prompt_phone_ids", conditioning["prompt_phone_ids"])
        self.register_buffer("prompt_bert", conditioning["prompt_bert"])
        self.register_buffer("reference_spectrogram", conditioning["reference_spectrogram"])
        self.register_buffer("speaker_embedding", conditioning["speaker_embedding"])

    def synthesize_conditioned(
        self,
        text_seq: torch.Tensor,
        text_bert: torch.Tensor,
        prompt_semantic: torch.Tensor,
        prompt_phone_ids: torch.Tensor,
        prompt_bert: torch.Tensor,
        reference_spectrogram: torch.Tensor,
        speaker_embedding: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 10,
        top_p: float = 1.0,
        repetition_penalty: float = 1.35,
        speed_factor: float = 1.0,
        seed: int = -1,
    ) -> tuple[int, torch.Tensor]:
        if seed >= 0:
            torch.manual_seed(seed)
        y_len, y, xy_pos, k_cache, v_cache = self.t2s.pre_infer(
            prompt_semantic, prompt_phone_ids, text_seq, prompt_bert, text_bert,
            top_k, temperature, top_p, repetition_penalty
        )
        generated = 1
        for idx in range(1, self.max_tokens + 1):
            y, xy_pos, last_token, k_cache, v_cache = self.t2s(
                idx, top_k, y_len, y, xy_pos, k_cache, v_cache,
                temperature, top_p, repetition_penalty
            )
            generated = idx + 1
            if last_token == self.eos:
                break
        semantic = y[:, -generated:].unsqueeze(0)
        speed = torch.tensor(speed_factor, dtype=torch.float32, device=text_seq.device)
        audio = self.vits(semantic, text_seq, reference_spectrogram, speaker_embedding, speed)
        audio = audio.reshape(-1).float().clamp(-1.0, 1.0)
        pcm = torch.round(audio * 32767.0).to(torch.int32)
        return 32000, pcm

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
        return self.synthesize_conditioned(
            text_seq, text_bert, self.prompt_semantic, self.prompt_phone_ids, self.prompt_bert,
            self.reference_spectrogram, self.speaker_embedding,
            temperature, top_k, top_p, repetition_penalty, speed_factor, seed,
        )

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
        ssl_content = self.ssl(reference_pcm_16k)
        prompt_semantic = self.vits.extract_latent(ssl_content)
        reference_spectrogram, speaker_embedding = self.vits.ref_handle(reference_pcm_32k)
        return self.synthesize_conditioned(
            text_seq, text_bert, prompt_semantic, prompt_phone_ids, prompt_bert,
            reference_spectrogram, speaker_embedding,
            temperature, top_k, top_p, repetition_penalty, speed_factor, seed,
        )


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts',required=True,type=Path); p.add_argument('--output',required=True,type=Path)
    p.add_argument('--ssl',required=True,type=Path)
    a=p.parse_args(); root=a.artifacts.resolve()
    conditioning=load_file(root/'conditioning.safetensors',device='cpu')
    module=AcousticPipeline(
        torch.jit.load(root/'t2s.pt',map_location='cpu'),
        torch.jit.load(root/'vits.pt',map_location='cpu'),
        torch.jit.load(str(a.ssl.resolve()),map_location='cpu'),
        conditioning,
    )
    module.eval(); scripted=torch.jit.script(module)
    scripted=torch.jit.freeze(scripted,preserved_attrs=['synthesize_reference_options'])
    scripted.save(str(a.output))
    print(f'Created {a.output} ({a.output.stat().st_size} bytes)')

if __name__=='__main__': main()
