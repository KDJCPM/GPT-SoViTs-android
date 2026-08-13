#!/usr/bin/env python3
"""Export a voice-specialized, FP32-only V4 acoustic pipeline for Android CPU.

Checkpoint parsing and graph adaptation are performed here. The package keeps preset conditioning
for the no-input path and a converted HuBERT/reference encoder for request-scoped reference audio.
No quantization, truncation or dtype conversion is used.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio


class AttrDict(dict):
    def __init__(self, value):
        super().__init__()
        for key, item in value.items():
            item = AttrDict(item) if isinstance(item, dict) else item
            self[key] = item
            setattr(self, key, item)


class DitBlocks(torch.nn.Module):
    def __init__(self, dit):
        super().__init__()
        self.transformer_blocks = dit.transformer_blocks
        self.norm_out = dit.norm_out
        self.proj_out = dit.proj_out

    def forward(self, x, t, mask, rope):
        for block in self.transformer_blocks:
            x = block(x, t, mask=mask, rope=(rope, 1.0))
        return self.proj_out(self.norm_out(x, t))


class DitEmbed(torch.nn.Module):
    def __init__(self, dit):
        super().__init__()
        self.time_embed = dit.time_embed
        self.d_embed = dit.d_embed
        self.text_embed = dit.text_embed
        self.input_embed = dit.input_embed
        self.rotary_embed = dit.rotary_embed

    def forward(self, x0, cond0, x_lens, time, dt, text0):
        from module import commons
        x = x0.transpose(2, 1)
        cond = cond0.transpose(2, 1)
        text = text0.transpose(2, 1)
        mask = commons.sequence_mask(x_lens, max_length=x.size(1)).to(x.device)
        t = self.time_embed(time) + self.d_embed(dt)
        text_embed = self.text_embed(text, x.shape[1])
        positions = torch.arange(x.shape[1], device=x.device)
        rope, _ = self.rotary_embed(positions)
        return self.input_embed(x, cond, text_embed), t, mask, rope


class ExportDiT(torch.nn.Module):
    def __init__(self, dit):
        super().__init__()
        self.embed = DitEmbed(dit)
        self.blocks = DitBlocks(dit)

    def forward(self, x, prompt_x, x_lens, t, dt, mu):
        x, t, mask, rope = self.embed(x, prompt_x, x_lens, t, dt, mu)
        return self.blocks(x, t, mask, rope)


class V4VoiceModel(torch.nn.Module):
    """Share one V4 weight set between target decoding and runtime reference encoding."""

    def __init__(self, model, ssl, hps):
        super().__init__()
        self.model = model
        self.ssl = ssl
        self.filter_length = int(hps.data.filter_length)
        self.sample_rate = int(hps.data.sampling_rate)
        self.hop_length = int(hps.data.hop_length)
        self.win_length = int(hps.data.win_length)

    def forward(self, codes: torch.Tensor, text: torch.Tensor, ge: torch.Tensor, speed: torch.Tensor):
        return self.model(codes, text, ge, speed=speed)

    def reference(self, pcm_16k: torch.Tensor, pcm_32k: torch.Tensor, prompt_phone_ids: torch.Tensor):
        from module.mel_processing import mel_spectrogram_torch, spectrogram_torch
        tail = torch.zeros((1, 4800), dtype=pcm_16k.dtype, device=pcm_16k.device)
        ssl_content = self.ssl(torch.cat([pcm_16k.reshape(1, -1), tail], 1))["last_hidden_state"]
        ssl_content = ssl_content.transpose(1, 2).float()
        prompt_semantic = self.model.extract_latent(ssl_content)[0, 0].unsqueeze(0)
        reference_spectrogram = spectrogram_torch(
            pcm_32k.float(), self.filter_length, self.sample_rate,
            self.hop_length, self.win_length, center=False,
        ).float()
        speaker_embedding = self.model.create_ge(reference_spectrogram)
        reference_feature = self.model(
            prompt_semantic.unsqueeze(0), prompt_phone_ids, speaker_embedding,
        )
        reference_mel = mel_spectrogram_torch(
            pcm_32k.float(), n_fft=1280, win_size=1280, hop_size=320, num_mels=100,
            sampling_rate=32000, fmin=0, fmax=None, center=False,
        )
        reference_mel = (reference_mel + 12.0) / 14.0 * 2.0 - 1.0
        return prompt_semantic, speaker_embedding, reference_feature, reference_mel


class V4AcousticPipeline(torch.nn.Module):
    def __init__(self, t2s, voice, cfm, vocoder, conditioning: dict[str, torch.Tensor], sample_steps: int):
        super().__init__()
        self.t2s = t2s
        self.voice = voice
        self.cfm = cfm
        self.vocoder = vocoder
        self.sample_steps = sample_steps
        self.register_buffer("prompt_semantic", conditioning["prompt_semantic"])
        self.register_buffer("prompt_phone_ids", conditioning["prompt_phone_ids"])
        self.register_buffer("prompt_bert", conditioning["prompt_bert"])
        self.register_buffer("speaker_embedding", conditioning["speaker_embedding"])
        self.register_buffer("reference_feature", conditioning["reference_feature"])
        self.register_buffer("reference_mel", conditioning["reference_mel"])

    def synthesize_conditioned(
        self,
        text_seq: torch.Tensor,
        text_bert: torch.Tensor,
        prompt_semantic: torch.Tensor,
        prompt_phone_ids: torch.Tensor,
        prompt_bert: torch.Tensor,
        speaker_embedding: torch.Tensor,
        reference_feature: torch.Tensor,
        reference_mel: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 10,
        top_p: float = 1.0,
        repetition_penalty: float = 1.35,
        speed_factor: float = 1.0,
        sample_steps: int = 32,
        seed: int = -1,
    ):
        if seed >= 0:
            torch.manual_seed(seed)
        top_k_tensor = torch.tensor([top_k], dtype=torch.long, device=text_seq.device)
        semantic = self.t2s(
            prompt_semantic, prompt_phone_ids, text_seq,
            prompt_bert, text_bert, top_k_tensor, temperature, top_p, repetition_penalty,
        )
        speed = torch.tensor(speed_factor, dtype=torch.float32, device=text_seq.device)
        feature_todo = self.voice(semantic, text_seq, speaker_embedding, speed)
        # V4 decoder contains ten tail frames which upstream intentionally discards.
        feature_todo = feature_todo[:, :, :-10]
        feature_ref = reference_feature
        mel_ref = reference_mel
        reference_len = feature_ref.shape[2]
        chunk_len = 1000 - reference_len
        chunks = torch.jit.annotate(list[torch.Tensor], [])
        offset = 0
        while offset < feature_todo.shape[2]:
            todo = feature_todo[:, :, offset:offset + chunk_len]
            feature = torch.cat([feature_ref, todo], 2).transpose(2, 1)
            lengths = torch.tensor([feature.shape[1]], dtype=torch.long, device=feature.device)
            result = self.cfm(feature, lengths, mel_ref, torch.tensor([sample_steps], dtype=torch.long, device=feature.device))
            result = result[:, :, mel_ref.shape[2]:]
            mel_ref = result[:, :, -reference_len:]
            feature_ref = todo[:, :, -reference_len:]
            chunks.append(result)
            offset += chunk_len
        mel = torch.cat(chunks, 2)
        mel = (mel + 1.0) * 7.0 - 12.0
        audio = self.vocoder(mel).reshape(-1).float().clamp(-1.0, 1.0)
        return 48000, torch.round(audio * 32767.0).to(torch.int32)

    def forward(
        self,
        text_seq: torch.Tensor,
        text_bert: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 10,
        top_p: float = 1.0,
        repetition_penalty: float = 1.35,
        speed_factor: float = 1.0,
        sample_steps: int = 32,
        seed: int = -1,
    ):
        return self.synthesize_conditioned(
            text_seq, text_bert, self.prompt_semantic, self.prompt_phone_ids, self.prompt_bert,
            self.speaker_embedding, self.reference_feature, self.reference_mel,
            temperature, top_k, top_p, repetition_penalty, speed_factor, sample_steps, seed,
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
    ):
        prompt_semantic, speaker_embedding, reference_feature, reference_mel = self.voice.reference(
            reference_pcm_16k.reshape(1, -1), reference_pcm_32k.reshape(1, -1),
            prompt_phone_ids.reshape(1, -1),
        )
        reference_len = min(reference_feature.shape[2], reference_mel.shape[2], 500)
        reference_feature = reference_feature[:, :, -reference_len:].contiguous()
        reference_mel = reference_mel[:, :, -reference_len:].contiguous()
        return self.synthesize_conditioned(
            text_seq, text_bert, prompt_semantic, prompt_phone_ids, prompt_bert,
            speaker_embedding, reference_feature, reference_mel,
            temperature, top_k, top_p, repetition_penalty, speed_factor, sample_steps, seed,
        )


def load_v4(sovits_path: Path):
    from module.models_onnx import SynthesizerTrnV3
    from process_ckpt import load_sovits_new
    value = load_sovits_new(str(sovits_path))
    hps = AttrDict(value["config"])
    hps.model.semantic_frame_rate = "25hz"
    hps.model.version = "v4"
    model = SynthesizerTrnV3(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    ).float().eval()
    missing, unexpected = model.load_state_dict(value["weight"], strict=False)
    print(f"Loaded V4: missing={len(missing)}, unexpected={len(unexpected)}")
    return hps, model


def load_vocoder(path: Path):
    from module.models_onnx import Generator
    model = Generator(
        initial_channel=100, resblock="1", resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[10, 6, 2, 2, 2], upsample_initial_channel=512,
        upsample_kernel_sizes=[20, 12, 4, 4, 4], gin_channels=0, is_bias=True,
    ).float().eval()
    model.remove_weight_norm()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpt", required=True, type=Path)
    parser.add_argument("--sovits", required=True, type=Path)
    parser.add_argument("--vocoder", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--language", default="all_zh")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--upstream", type=Path, default=Path(".."))
    parser.add_argument("--sample-steps", type=int, default=32)
    parser.add_argument("--trace-text", default="这是用于导出第四版模型的较长测试句子，可以验证动态长度是否正常工作。")
    args = parser.parse_args()
    if args.sample_steps < 1:
        raise SystemExit("--sample-steps must be positive")
    upstream = args.upstream.resolve()
    output = args.output.resolve()
    gpt_path = args.gpt.resolve()
    sovits_path = args.sovits.resolve()
    vocoder_path = args.vocoder.resolve()
    reference_path = args.reference.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(upstream)
    sys.path.insert(0, str(upstream / "GPT_SoVITS"))
    os.environ.update(is_half="False", version="v4")

    from export_torch_script import T2SModel, get_raw_t2s_model
    from inference_webui import get_phones_and_bert, get_spepc, ssl_model
    from module.mel_processing import mel_spectrogram_torch

    hps, vq = load_v4(sovits_path)
    cfm = vq.cfm
    del vq.cfm
    raw_t2s = get_raw_t2s_model(torch.load(gpt_path, map_location="cpu", weights_only=False)).float().eval()
    t2s = torch.jit.script(T2SModel(raw_t2s).eval())

    prompt_phones, prompt_bert, _ = get_phones_and_bert(args.prompt, args.language, "v3")
    trace_phones, trace_bert, _ = get_phones_and_bert(args.trace_text, args.language, "v3")
    prompt_phone_ids = torch.tensor(prompt_phones, dtype=torch.long).unsqueeze(0)
    trace_phone_ids = torch.tensor(trace_phones, dtype=torch.long).unsqueeze(0)
    prompt_bert = prompt_bert.T.contiguous().float().cpu()
    trace_bert = trace_bert.T.contiguous().float().cpu()

    zero = np.zeros(int(16000 * 0.3), dtype=np.float32)
    raw_wav16, _ = librosa.load(reference_path, sr=16000, mono=True)
    raw_wav16 = torch.from_numpy(raw_wav16.astype(np.float32)).unsqueeze(0)
    wav16 = torch.cat([raw_wav16, torch.from_numpy(zero).unsqueeze(0)], 1)
    with torch.inference_mode():
        ssl_model.model = ssl_model.model.float().cpu()
        ssl = ssl_model.model(wav16)["last_hidden_state"].transpose(1, 2).float()
        prompt_semantic = vq.extract_latent(ssl)[0, 0].unsqueeze(0)
        reference_spec = get_spepc(hps, str(reference_path), torch.float32, "cpu")
        if isinstance(reference_spec, tuple):
            reference_spec = reference_spec[0]
        reference_spec = reference_spec.float()
        speaker_embedding = vq.create_ge(reference_spec)
        reference_feature = vq(prompt_semantic.unsqueeze(0), prompt_phone_ids, speaker_embedding)

    ref_audio, ref_sr = torchaudio.load(reference_path)
    ref_audio = ref_audio.float().mean(0, keepdim=True)
    if ref_sr != 32000:
        ref_audio = torchaudio.functional.resample(ref_audio, ref_sr, 32000)
    reference_mel = mel_spectrogram_torch(
        ref_audio, n_fft=1280, win_size=1280, hop_size=320, num_mels=100,
        sampling_rate=32000, fmin=0, fmax=None, center=False,
    )
    reference_mel = (reference_mel + 12.0) / 14.0 * 2.0 - 1.0
    ref_len = min(reference_mel.shape[2], reference_feature.shape[2], 500)
    reference_mel = reference_mel[:, :, -ref_len:].contiguous()
    reference_feature = reference_feature[:, :, -ref_len:].contiguous()

    # One traced module shares the SoVITS weights between target decoding and reference encoding.
    target_inputs = (t2s(prompt_semantic, prompt_phone_ids, trace_phone_ids, prompt_bert, trace_bert, torch.tensor([10])),
                     trace_phone_ids, speaker_embedding, torch.tensor(1.0))
    voice = torch.jit.trace_module(
        V4VoiceModel(vq, ssl_model.model, hps).eval(),
        {
            "forward": target_inputs,
            "reference": (raw_wav16, ref_audio, prompt_phone_ids),
        },
        check_trace=False,
        strict=False,
    )

    cfm.estimator = ExportDiT(cfm.estimator).eval()
    example_len = min(1000, ref_len + 160)
    x = torch.randn(1, 100, example_len)
    prompt_x = torch.zeros_like(x)
    mu = torch.randn(1, 512, example_len)
    lengths = torch.tensor([example_len], dtype=torch.long)
    t = torch.zeros(1)
    dt = torch.full((1,), 1.0 / args.sample_steps)
    cfm.estimator = torch.jit.trace(cfm.estimator, (x, prompt_x, lengths, t, dt, mu), check_trace=False, strict=False)
    cfm_script = torch.jit.script(cfm.eval())

    vocoder = load_vocoder(vocoder_path)
    vocoder_script = torch.jit.trace(vocoder, torch.randn(1, 100, 240), check_trace=False, strict=False)
    conditioning = {
        "prompt_semantic": prompt_semantic,
        "prompt_phone_ids": prompt_phone_ids,
        "prompt_bert": prompt_bert,
        "speaker_embedding": speaker_embedding,
        "reference_feature": reference_feature,
        "reference_mel": reference_mel,
    }
    pipeline = torch.jit.script(
        V4AcousticPipeline(t2s, voice, cfm_script, vocoder_script, conditioning, args.sample_steps).eval()
    )
    pipeline = torch.jit.freeze(pipeline, preserved_attrs=["synthesize_reference_options"])
    pipeline.save(str(output / "pipeline_core.pt"))
    print(f"Created {output / 'pipeline_core.pt'} ({(output / 'pipeline_core.pt').stat().st_size} bytes), FP32, steps={args.sample_steps}")

    # A real inference catches shape-specialization and unsupported scripted control flow now.
    torch.manual_seed(1234)
    sr, pcm = pipeline(trace_phone_ids, trace_bert, 1.0, 10)
    print(f"Smoke inference: sr={sr}, samples={pcm.numel()}, peak={int(pcm.abs().max())}")
    torch.manual_seed(1234)
    reference_sr, reference_pcm = pipeline.synthesize_reference_options(
        trace_phone_ids,
        trace_bert,
        raw_wav16,
        ref_audio,
        prompt_phone_ids,
        prompt_bert,
        1.0,
        10,
        1.0,
        1.35,
        1.0,
        args.sample_steps,
        1234,
    )
    print(
        "Reference smoke inference: "
        f"sr={reference_sr}, samples={reference_pcm.numel()}, peak={int(reference_pcm.abs().max())}"
    )
    del raw_t2s, vq, cfm, vocoder
    gc.collect()


if __name__ == "__main__":
    with torch.inference_mode():
        main()
