#!/usr/bin/env python3
"""Export fixed-shape V2 Pro Plus runtime-reference graphs for QNN HTP."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from torchaudio.compliance.kaldi import get_mel_banks
from transformers import HubertModel

from export_v2pp_qnn_onnx import (
    T2SLayer,
    VitsGeneratorPadded,
    export_onnx,
    load_t2s_source,
    load_vits_source,
)


REFERENCE_SECONDS = 5
PCM_16K_SAMPLES = 16_000 * REFERENCE_SECONDS
PCM_32K_SAMPLES = 32_000 * REFERENCE_SECONDS
SPECTROGRAM_FFT = 2048
SPECTROGRAM_HOP = 640
SPECTROGRAM_PAD = (SPECTROGRAM_FFT - SPECTROGRAM_HOP) // 2
SPECTROGRAM_PADDED_SAMPLES = PCM_32K_SAMPLES + 2 * SPECTROGRAM_PAD
SPECTROGRAM_FRAMES = 1 + (SPECTROGRAM_PADDED_SAMPLES - SPECTROGRAM_FFT) // SPECTROGRAM_HOP
FBANK_WINDOW = 400
FBANK_HOP = 160
FBANK_FFT = 512
FBANK_BINS = 80
FBANK_FRAMES = 1 + (PCM_16K_SAMPLES - FBANK_WINDOW) // FBANK_HOP
SSL_FRAMES = 249
PROMPT_SEMANTIC_LENGTH = 124


class ReferenceSslEncoder(torch.nn.Module):
    def __init__(self, model: HubertModel):
        super().__init__()
        self.model = model

    def forward(self, pcm_16k: torch.Tensor) -> torch.Tensor:
        return self.model(pcm_16k)[0].transpose(1, 2).float()


class PromptSemanticEncoder(torch.nn.Module):
    def __init__(self, source: torch.nn.Module):
        super().__init__()
        self.ssl_proj = source.ssl_proj
        self.quantizer = source.quantizer

    def forward(self, ssl_content: torch.Tensor) -> torch.Tensor:
        return self.quantizer(self.ssl_proj(ssl_content))[1].transpose(0, 1)[0]


def dft_kernels(n_fft: int, window: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    sample = torch.arange(window.numel(), dtype=torch.float64)
    frequency = torch.arange(n_fft // 2 + 1, dtype=torch.float64).unsqueeze(1)
    angle = 2.0 * math.pi * frequency * sample.unsqueeze(0) / n_fft
    return (
        (torch.cos(angle) * window.double()).float().unsqueeze(1),
        (-torch.sin(angle) * window.double()).float().unsqueeze(1),
    )


def fbank_kernels() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    identity = torch.eye(FBANK_WINDOW, dtype=torch.float64)
    remove_dc = identity - torch.full_like(identity, 1.0 / FBANK_WINDOW)
    preemphasis = identity.clone()
    preemphasis[0, 0] = 1.0 - 0.97
    indices = torch.arange(1, FBANK_WINDOW)
    preemphasis[indices, indices - 1] = -0.97
    window = torch.hann_window(FBANK_WINDOW, periodic=False, dtype=torch.float64).pow(0.85)
    frame_transform = window.diag() @ preemphasis @ remove_dc
    sample = torch.arange(FBANK_WINDOW, dtype=torch.float64)
    frequency = torch.arange(FBANK_FFT // 2 + 1, dtype=torch.float64).unsqueeze(1)
    angle = 2.0 * math.pi * frequency * sample.unsqueeze(0) / FBANK_FFT
    real = (torch.cos(angle) @ frame_transform).float().unsqueeze(1)
    imaginary = (-torch.sin(angle) @ frame_transform).float().unsqueeze(1)
    mel, _ = get_mel_banks(
        FBANK_BINS,
        FBANK_FFT,
        16_000.0,
        20.0,
        0.0,
        100.0,
        -500.0,
        1.0,
    )
    mel = F.pad(mel.float(), (0, 1)).unsqueeze(2)
    return real, imaginary, mel


class SpeakerEncoder(torch.nn.Module):
    def __init__(self, source: torch.nn.Module):
        super().__init__()
        self.bn1 = source.bn1
        self.conv1 = source.conv1
        self.layer1 = source.layer1
        self.layer2 = source.layer2
        self.layer3 = source.layer3
        self.layer4 = source.layer4
        self.layer3_ds = source.layer3_ds
        self.fuse34 = source.fuse34

    def forward(self, fbank: torch.Tensor) -> torch.Tensor:
        value = fbank.permute(0, 2, 1).unsqueeze(1)
        value = F.relu(self.bn1(self.conv1(value)))
        value1 = self.layer1(value)
        value2 = self.layer2(value1)
        value3 = self.layer3(value2)
        value4 = self.layer4(value3)
        fused = self.fuse34(value4, self.layer3_ds(value3))
        return fused.flatten(start_dim=1, end_dim=2).mean(-1)


class ReferenceConditioningEncoder(torch.nn.Module):
    def __init__(self, speaker: torch.nn.Module):
        super().__init__()
        fbank_real, fbank_imaginary, mel = fbank_kernels()
        spec_window = torch.hann_window(SPECTROGRAM_FFT, periodic=True)
        spec_real, spec_imaginary = dft_kernels(SPECTROGRAM_FFT, spec_window)
        self.register_buffer("fbank_real", fbank_real)
        self.register_buffer("fbank_imaginary", fbank_imaginary)
        self.register_buffer("mel", mel)
        self.register_buffer("spec_real", spec_real)
        self.register_buffer("spec_imaginary", spec_imaginary)
        self.speaker = SpeakerEncoder(speaker)

    def forward(
        self,
        pcm_16k: torch.Tensor,
        reflected_pcm_32k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        signal = pcm_16k.unsqueeze(1)
        real = F.conv1d(signal, self.fbank_real, stride=FBANK_HOP)
        imaginary = F.conv1d(signal, self.fbank_imaginary, stride=FBANK_HOP)
        power = real.square() + imaginary.square()
        fbank = F.conv1d(power, self.mel).clamp_min(torch.finfo(torch.float32).eps).log()
        speaker_embedding = self.speaker(fbank.transpose(1, 2))

        padded = reflected_pcm_32k.unsqueeze(1)
        spec_real = F.conv1d(padded, self.spec_real, stride=SPECTROGRAM_HOP)
        spec_imaginary = F.conv1d(padded, self.spec_imaginary, stride=SPECTROGRAM_HOP)
        spectrogram = torch.sqrt(spec_real.square() + spec_imaginary.square() + 1e-8)
        return spectrogram, speaker_embedding


class T2SReferencePrefill(torch.nn.Module):
    def __init__(self, source: torch.nn.Module):
        super().__init__()
        self.text_embedding = source.ar_text_embedding
        self.text_position = source.ar_text_position
        self.audio_embedding = source.ar_audio_embedding
        self.audio_position = source.ar_audio_position
        self.bert_proj = source.bert_proj
        self.predict = source.ar_predict_layer
        self.layers = torch.nn.ModuleList(
            [T2SLayer(layer, source.num_head) for layer in source.h.layers]
        )
        self.heads = source.num_head

    def forward(
        self,
        text_seq: torch.Tensor,
        text_bert: torch.Tensor,
        text_valid: torch.Tensor,
        prompt_semantic: torch.Tensor,
        prompt_phone_ids: torch.Tensor,
        prompt_bert: torch.Tensor,
        prompt_phone_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phone_ids = torch.cat([prompt_phone_ids, text_seq], dim=1)
        bert = torch.cat([prompt_bert, text_bert], dim=0).unsqueeze(0)
        text = self.text_position(self.text_embedding(phone_ids) + self.bert_proj(bert))
        prompt = self.audio_position(self.audio_embedding(prompt_semantic))
        value = torch.cat([text, prompt], dim=1)
        text_valid_full = torch.cat(
            [prompt_phone_valid > 0.5, text_valid > 0.5], dim=1
        )
        text_length = text.shape[1]
        prompt_length = prompt.shape[1]
        x_blocked = F.pad(
            torch.zeros((text_length, text_length), dtype=torch.bool, device=value.device),
            (0, prompt_length),
            value=True,
        )
        y_blocked = F.pad(
            torch.triu(
                torch.ones((prompt_length, prompt_length), dtype=torch.bool, device=value.device),
                diagonal=1,
            ),
            (text_length, 0),
            value=False,
        )
        blocked = torch.cat([x_blocked, y_blocked], dim=0).reshape(
            1, 1, text_length + prompt_length, text_length + prompt_length
        )
        invalid_keys = torch.cat(
            [
                ~text_valid_full,
                torch.zeros((1, prompt_length), dtype=torch.bool, device=value.device),
            ],
            dim=1,
        ).reshape(1, 1, 1, text_length + prompt_length)
        blocked = (blocked | invalid_keys).expand(1, self.heads, -1, -1)
        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        for layer in self.layers:
            value, key, item = layer.prefill(value, blocked)
            keys.append(key)
            values.append(item)
        return self.predict(value[:, -1, :]), torch.stack(keys), torch.stack(values)


class VitsReferenceDecoderPadded(torch.nn.Module):
    def __init__(self, source: torch.nn.Module):
        super().__init__()
        self.ref_enc = source.ref_enc
        self.sv_emb = source.sv_emb
        self.prelu = source.prelu
        self.quantizer = source.quantizer
        self.ge_to512 = source.ge_to512
        self.enc_p = source.enc_p
        self.flow = source.flow
        self.dec = VitsGeneratorPadded(source.dec)
        self.ssl_dim = source.ssl_dim

    def forward(
        self,
        pred_semantic: torch.Tensor,
        text_seq: torch.Tensor,
        noise: torch.Tensor,
        semantic_valid: torch.Tensor,
        text_valid: torch.Tensor,
        reference_spectrogram: torch.Tensor,
        speaker_embedding: torch.Tensor,
    ) -> torch.Tensor:
        refer_mask = torch.ones_like(reference_spectrogram[:1, :1, :])
        ge = self.ref_enc(reference_spectrogram[:, :704] * refer_mask, refer_mask)
        ge = self.prelu(ge + self.sv_emb(speaker_embedding).unsqueeze(-1))
        quantized = self.quantizer.decode(pred_semantic)
        quantized = torch.cat([quantized, quantized]).permute(1, 2, 0)
        quantized = quantized.contiguous().view(1, self.ssl_dim, -1)
        y_mask = semantic_valid.unsqueeze(1).repeat_interleave(2, dim=2)
        text_mask = text_valid.unsqueeze(1)
        encoder = self.enc_p
        y = encoder.ssl_proj(quantized * y_mask) * y_mask
        y = encoder.encoder_ssl(y * y_mask, y_mask)
        text = encoder.text_embedding(text_seq).transpose(1, 2)
        text = encoder.encoder_text(text * text_mask, text_mask)
        ge_for_text = self.ge_to512(ge.transpose(2, 1)).transpose(2, 1)
        y = encoder.mrte(y, y_mask, text, text_mask, ge_for_text)
        y = encoder.encoder2(y * y_mask, y_mask)
        stats = encoder.proj(y) * y_mask
        mean, log_scale = torch.split(stats, encoder.out_channels, dim=1)
        latent = mean + noise * torch.exp(log_scale) * 0.5
        latent = self.flow(latent, y_mask, g=ge, reverse=True) * y_mask
        return self.dec(latent, ge, y_mask)[0, 0]


def load_speaker_source(upstream: Path) -> torch.nn.Module:
    previous = Path.cwd()
    try:
        os.chdir(upstream)
        sys.path.insert(0, str(upstream / "GPT_SoVITS"))
        from sv import SV

        return SV("cpu", False).embedding_model.eval()
    finally:
        os.chdir(previous)


def export_ssl(model_root: Path, output: Path) -> dict:
    model = HubertModel.from_pretrained(
        model_root,
        local_files_only=True,
        torchscript=True,
    ).float().eval()
    module = ReferenceSslEncoder(model).eval()
    pcm = torch.zeros((1, PCM_16K_SAMPLES), dtype=torch.float32)
    with torch.inference_mode():
        ssl = module(pcm)
    if tuple(ssl.shape) != (1, 768, SSL_FRAMES):
        raise ValueError(f"unexpected fixed HuBERT output shape: {tuple(ssl.shape)}")
    path = output / "reference_ssl_5s.onnx"
    export_onnx(module, (pcm,), path, ["reference_pcm_16k"], ["ssl_content"])
    torch.save({"reference_pcm_16k": pcm, "ssl_content": ssl}, output / "reference_ssl_5s.pt")
    return {"stage": "reference_ssl", "graph": path.name, "pcm_samples": PCM_16K_SAMPLES, "ssl_frames": SSL_FRAMES}


def export_prompt(upstream: Path, checkpoint: Path, output: Path) -> dict:
    source = load_vits_source(upstream, checkpoint)
    module = PromptSemanticEncoder(source).eval()
    ssl = torch.zeros((1, 768, SSL_FRAMES), dtype=torch.float32)
    with torch.inference_mode():
        semantic = module(ssl)
    if tuple(semantic.shape) != (1, PROMPT_SEMANTIC_LENGTH):
        raise ValueError(f"unexpected prompt semantic shape: {tuple(semantic.shape)}")
    path = output / "reference_prompt_semantic_5s.onnx"
    export_onnx(module, (ssl,), path, ["ssl_content"], ["prompt_semantic"])
    torch.save({"ssl_content": ssl, "prompt_semantic": semantic}, output / "reference_prompt_semantic_5s.pt")
    return {"stage": "reference_prompt_semantic", "graph": path.name, "semantic_length": PROMPT_SEMANTIC_LENGTH}


def export_conditioning(upstream: Path, output: Path) -> dict:
    module = ReferenceConditioningEncoder(load_speaker_source(upstream)).eval()
    pcm16 = torch.zeros((1, PCM_16K_SAMPLES), dtype=torch.float32)
    padded32 = torch.zeros((1, SPECTROGRAM_PADDED_SAMPLES), dtype=torch.float32)
    with torch.inference_mode():
        spectrogram, speaker = module(pcm16, padded32)
    expected_spec = (1, SPECTROGRAM_FFT // 2 + 1, SPECTROGRAM_FRAMES)
    if tuple(spectrogram.shape) != expected_spec or tuple(speaker.shape) != (1, 20480):
        raise ValueError(
            f"unexpected reference conditioning shapes: {tuple(spectrogram.shape)}, {tuple(speaker.shape)}"
        )
    path = output / "reference_conditioning_5s.onnx"
    export_onnx(
        module,
        (pcm16, padded32),
        path,
        ["reference_pcm_16k", "reference_pcm_32k_reflected"],
        ["reference_spectrogram", "speaker_embedding"],
    )
    torch.save(
        {
            "reference_pcm_16k": pcm16,
            "reference_pcm_32k_reflected": padded32,
            "reference_spectrogram": spectrogram,
            "speaker_embedding": speaker,
        },
        output / "reference_conditioning_5s.pt",
    )
    return {
        "stage": "reference_conditioning",
        "graph": path.name,
        "pcm_16k_samples": PCM_16K_SAMPLES,
        "pcm_32k_samples": PCM_32K_SAMPLES,
        "spectrogram_reflect_pad": SPECTROGRAM_PAD,
        "spectrogram_bins": int(spectrogram.shape[1]),
        "spectrogram_frames": SPECTROGRAM_FRAMES,
        "fbank_frames": FBANK_FRAMES,
        "speaker_embedding_size": int(speaker.shape[1]),
    }


def export_t2s_reference(
    upstream: Path,
    checkpoint: Path,
    output: Path,
    phone_capacity: int,
    cache_capacity: int,
) -> dict:
    source = load_t2s_source(upstream, checkpoint)
    module = T2SReferencePrefill(source).eval()
    text_seq = torch.zeros((1, phone_capacity), dtype=torch.int64)
    text_bert = torch.zeros((phone_capacity, 1024), dtype=torch.float32)
    text_valid = torch.ones((1, phone_capacity), dtype=torch.float32)
    prompt_semantic = torch.zeros((1, PROMPT_SEMANTIC_LENGTH), dtype=torch.int64)
    prompt_phone_ids = torch.zeros((1, phone_capacity), dtype=torch.int64)
    prompt_bert = torch.zeros((phone_capacity, 1024), dtype=torch.float32)
    prompt_phone_valid = torch.ones((1, phone_capacity), dtype=torch.float32)
    inputs = (
        text_seq,
        text_bert,
        text_valid,
        prompt_semantic,
        prompt_phone_ids,
        prompt_bert,
        prompt_phone_valid,
    )
    with torch.inference_mode():
        logits, keys, values = module(*inputs)
    compact_length = int(keys.shape[2])
    if compact_length + 512 > cache_capacity:
        raise ValueError("runtime-reference prefill does not fit the T2S cache")
    path = output / f"t2s_reference_prefill_pc{phone_capacity}.onnx"
    export_onnx(
        module,
        inputs,
        path,
        [
            "text_seq",
            "text_bert",
            "text_valid",
            "prompt_semantic",
            "prompt_phone_ids",
            "prompt_bert",
            "prompt_phone_valid",
        ],
        ["logits", "k_cache", "v_cache"],
    )
    torch.save(dict(zip(
        ["text_seq", "text_bert", "text_valid", "prompt_semantic", "prompt_phone_ids", "prompt_bert", "prompt_phone_valid"],
        inputs,
    )) | {"logits": logits, "k_cache": keys, "v_cache": values}, output / "t2s_reference_prefill.pt")
    return {
        "stage": "t2s_reference_prefill",
        "graph": path.name,
        "phone_capacity": phone_capacity,
        "prompt_phone_capacity": phone_capacity,
        "prompt_semantic_length": PROMPT_SEMANTIC_LENGTH,
        "prefill_cache_length": compact_length,
        "cache_capacity": cache_capacity,
        "layers": int(keys.shape[0]),
        "hidden_size": int(keys.shape[-1]),
    }


def export_vits_reference(
    upstream: Path,
    checkpoint: Path,
    output: Path,
    phone_capacity: int,
    semantic_capacity: int,
) -> dict:
    source = load_vits_source(upstream, checkpoint)
    module = VitsReferenceDecoderPadded(source).eval()
    pred_semantic = torch.zeros((1, 1, semantic_capacity), dtype=torch.int64)
    text_seq = torch.zeros((1, phone_capacity), dtype=torch.int64)
    semantic_valid = torch.ones((1, semantic_capacity), dtype=torch.float32)
    text_valid = torch.ones((1, phone_capacity), dtype=torch.float32)
    reference_spectrogram = torch.zeros(
        (1, SPECTROGRAM_FFT // 2 + 1, SPECTROGRAM_FRAMES), dtype=torch.float32
    )
    speaker_embedding = torch.zeros((1, 20480), dtype=torch.float32)
    with torch.inference_mode():
        refer_mask = torch.ones_like(reference_spectrogram[:1, :1, :])
        ge = module.ref_enc(reference_spectrogram[:, :704] * refer_mask, refer_mask)
        ge = module.prelu(ge + module.sv_emb(speaker_embedding).unsqueeze(-1))
        quantized = module.quantizer.decode(pred_semantic)
        quantized = torch.cat([quantized, quantized]).permute(1, 2, 0)
        quantized = quantized.contiguous().view(1, module.ssl_dim, -1)
        mask = semantic_valid.unsqueeze(1).repeat_interleave(2, dim=2)
        encoder = module.enc_p
        encoded = encoder.ssl_proj(quantized * mask) * mask
        encoded = encoder.encoder_ssl(encoded, mask)
        text = encoder.text_embedding(text_seq).transpose(1, 2)
        text_mask = text_valid.unsqueeze(1)
        text = encoder.encoder_text(text * text_mask, text_mask)
        encoded = encoder.mrte(
            encoded,
            mask,
            text,
            text_mask,
            module.ge_to512(ge.transpose(2, 1)).transpose(2, 1),
        )
        encoded = encoder.encoder2(encoded * mask, mask)
        mean, _ = torch.split(encoder.proj(encoded) * mask, encoder.out_channels, dim=1)
        noise = torch.zeros_like(mean)
        audio = module(
            pred_semantic,
            text_seq,
            noise,
            semantic_valid,
            text_valid,
            reference_spectrogram,
            speaker_embedding,
        )
    inputs = (
        pred_semantic,
        text_seq,
        noise,
        semantic_valid,
        text_valid,
        reference_spectrogram,
        speaker_embedding,
    )
    path = output / f"vits_reference_pc{phone_capacity}_sc{semantic_capacity}.onnx"
    export_onnx(
        module,
        inputs,
        path,
        [
            "pred_semantic",
            "text_seq",
            "noise",
            "semantic_valid",
            "text_valid",
            "reference_spectrogram",
            "speaker_embedding",
        ],
        ["audio"],
    )
    torch.save({"inputs": inputs, "audio": audio}, output / "vits_reference.pt")
    return {
        "stage": "vits_reference",
        "graph": path.name,
        "phone_capacity": phone_capacity,
        "semantic_capacity": semantic_capacity,
        "reference_spectrogram_frames": SPECTROGRAM_FRAMES,
        "audio_capacity_samples": int(audio.numel()),
        "samples_per_semantic": int(audio.numel()) // semantic_capacity,
        "sample_rate": 32000,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("ssl", "prompt", "conditioning", "t2s", "vits"),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--upstream", type=Path, default=Path(".."))
    parser.add_argument("--gpt", type=Path)
    parser.add_argument("--sovits", type=Path)
    parser.add_argument("--hubert", type=Path)
    parser.add_argument("--phone-capacity", type=int, default=128)
    parser.add_argument("--semantic-capacity", type=int, default=512)
    parser.add_argument("--cache-capacity", type=int, default=1024)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    upstream = args.upstream.resolve()
    if args.stage == "ssl":
        if args.hubert is None:
            raise SystemExit("--hubert is required for --stage ssl")
        metadata = export_ssl(args.hubert.resolve(), output)
    elif args.stage == "prompt":
        if args.sovits is None:
            raise SystemExit("--sovits is required for --stage prompt")
        metadata = export_prompt(upstream, args.sovits.resolve(), output)
    elif args.stage == "conditioning":
        metadata = export_conditioning(upstream, output)
    elif args.stage == "t2s":
        if args.gpt is None:
            raise SystemExit("--gpt is required for --stage t2s")
        metadata = export_t2s_reference(
            upstream,
            args.gpt.resolve(),
            output,
            args.phone_capacity,
            args.cache_capacity,
        )
    else:
        if args.sovits is None:
            raise SystemExit("--sovits is required for --stage vits")
        metadata = export_vits_reference(
            upstream,
            args.sovits.resolve(),
            output,
            args.phone_capacity,
            args.semantic_capacity,
        )
    path = output / f"{metadata['stage']}.json"
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {metadata['stage']}: {metadata}")


if __name__ == "__main__":
    main()
