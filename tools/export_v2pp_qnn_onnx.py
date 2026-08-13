#!/usr/bin/env python3
"""Export fixed-shape V2 Pro Plus neural stages for QAIRT/QNN HTP.

The exported graphs deliberately exclude text preprocessing, autoregressive sampling and loop
control. Those are scalar orchestration tasks. Every learned operation remains in one of the ONNX
graphs, and each graph has static shapes so it can be compiled into a target-specific HTP context.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from safetensors.torch import load_file


def export_onnx(
    module: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    output: Path,
    input_names: list[str],
    output_names: list[str],
) -> None:
    module.eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            module,
            inputs,
            str(output),
            input_names=input_names,
            output_names=output_names,
            opset_version=17,
            dynamo=False,
            do_constant_folding=True,
        )


def export_bert(artifacts: Path, output: Path, token_length: int) -> dict[str, object]:
    if token_length < 3:
        raise ValueError("--token-length must include CLS, at least one token, and SEP")
    bert = torch.jit.load(str(artifacts / "bert_mobile_fp32_eager.pt"), map_location="cpu")
    # Export the registered ScriptModule itself. Pulling it into a new eager wrapper makes recent
    # PyTorch versions reject it as being outside the active trace.
    module = bert.model
    ids = torch.zeros((1, token_length), dtype=torch.int64)
    ids[:, 0] = 101
    ids[:, -1] = 102
    attention = torch.ones_like(ids)
    token_types = torch.zeros_like(ids)
    graph = output / f"bert_tokens_{token_length}.onnx"
    export_onnx(
        module,
        (ids, token_types, attention),
        graph,
        ["input_ids", "token_type_ids", "attention_mask"],
        ["hidden_features"],
    )
    with torch.inference_mode():
        expected = module(ids, token_types, attention)
    torch.save(
        {
            "input_ids": ids,
            "attention_mask": attention,
            "token_type_ids": token_types,
            "hidden_features": expected,
        },
        output / f"bert_tokens_{token_length}_reference.pt",
    )
    return {
        "stage": "bert_token_features",
        "graph": graph.name,
        "token_length": token_length,
        "inputs": {
            "input_ids": list(ids.shape),
            "attention_mask": list(attention.shape),
            "token_type_ids": list(token_types.shape),
        },
        "outputs": {"hidden_features": list(expected.shape)},
    }


class T2SLayer(torch.nn.Module):
    """One inference transformer layer with explicit, QNN-friendly attention."""

    def __init__(self, source: torch.nn.Module, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = source.self_attn.embed_dim // heads
        self.in_proj_weight = source.self_attn.in_proj_weight
        self.in_proj_bias = source.self_attn.in_proj_bias
        self.out_proj = source.self_attn.out_proj
        self.linear1 = source.linear1
        self.linear2 = source.linear2
        self.norm1 = source.norm1
        self.norm2 = source.norm2

    def project(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return F.linear(value, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)

    def finish(self, value: torch.Tensor, attended: torch.Tensor) -> torch.Tensor:
        value = self.norm1(value + self.out_proj(attended))
        return self.norm2(value + self.linear2(F.relu(self.linear1(value))))

    def prefill(
        self, value: torch.Tensor, blocked: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = self.project(value)
        batch, sequence, _ = q.shape
        qh = q.view(batch, sequence, self.heads, self.head_dim).transpose(1, 2)
        kh = k.view(batch, sequence, self.heads, self.head_dim).transpose(1, 2)
        vh = v.view(batch, sequence, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = torch.where(blocked, torch.full_like(scores, -10000.0), scores)
        attended = torch.matmul(torch.softmax(scores, dim=-1), vh)
        attended = attended.transpose(1, 2).reshape(batch, sequence, -1)
        return self.finish(value, attended), k, v

    def step(
        self,
        value: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        write_mask: torch.Tensor,
        attention_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = self.project(value)
        k_all = k_cache + write_mask * k
        v_all = v_cache + write_mask * v
        batch, sequence, _ = q.shape
        cache_length = k_all.shape[1]
        qh = q.view(batch, sequence, self.heads, self.head_dim).transpose(1, 2)
        kh = k_all.view(batch, cache_length, self.heads, self.head_dim).transpose(1, 2)
        vh = v_all.view(batch, cache_length, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + attention_bias
        attended = torch.matmul(torch.softmax(scores, dim=-1), vh)
        attended = attended.transpose(1, 2).reshape(batch, sequence, -1)
        return self.finish(value, attended), k, v


class T2SPrefill(torch.nn.Module):
    def __init__(self, source: torch.nn.Module, conditioning: dict[str, torch.Tensor]):
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
        self.register_buffer("prompt_semantic", conditioning["prompt_semantic"])
        self.register_buffer("prompt_phone_ids", conditioning["prompt_phone_ids"])
        self.register_buffer("prompt_bert", conditioning["prompt_bert"])

    def forward(
        self, text_seq: torch.Tensor, text_bert: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phone_ids = torch.cat([self.prompt_phone_ids, text_seq], dim=1)
        bert = torch.cat([self.prompt_bert, text_bert], dim=0).unsqueeze(0)
        text = self.text_embedding(phone_ids) + self.bert_proj(bert)
        text = self.text_position(text)
        prompt = self.audio_position(self.audio_embedding(self.prompt_semantic))
        value = torch.cat([text, prompt], dim=1)
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
        blocked = blocked.expand(1, self.heads, -1, -1)
        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        for layer in self.layers:
            value, key, item = layer.prefill(value, blocked)
            keys.append(key)
            values.append(item)
        logits = self.predict(value[:, -1, :])
        return logits, torch.stack(keys), torch.stack(values)


class T2SPrefillPadded(T2SPrefill):
    """Static-capacity prefill whose padding is invisible to every attention query."""

    def forward(
        self,
        text_seq: torch.Tensor,
        text_bert: torch.Tensor,
        text_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_valid = text_valid > 0.5
        phone_ids = torch.cat([self.prompt_phone_ids, text_seq], dim=1)
        bert = torch.cat([self.prompt_bert, text_bert], dim=0).unsqueeze(0)
        text = self.text_embedding(phone_ids) + self.bert_proj(bert)
        text = self.text_position(text)
        prompt = self.audio_position(self.audio_embedding(self.prompt_semantic))
        value = torch.cat([text, prompt], dim=1)
        prompt_phone_valid = torch.ones_like(self.prompt_phone_ids, dtype=torch.bool)
        text_valid_full = torch.cat([prompt_phone_valid, target_valid], dim=1)
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
        logits = self.predict(value[:, -1, :])
        return logits, torch.stack(keys), torch.stack(values)


class T2SStep(torch.nn.Module):
    def __init__(self, source: torch.nn.Module):
        super().__init__()
        self.audio_embedding = source.ar_audio_embedding
        self.audio_position = source.ar_audio_position
        self.predict = source.ar_predict_layer
        self.layers = torch.nn.ModuleList(
            [T2SLayer(layer, source.num_head) for layer in source.h.layers]
        )

    def forward(
        self,
        last_token: torch.Tensor,
        position_embedding: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        write_mask: torch.Tensor,
        attention_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        value = self.audio_embedding(last_token) * self.audio_position.x_scale
        value = value + self.audio_position.alpha * position_embedding
        new_keys: list[torch.Tensor] = []
        new_values: list[torch.Tensor] = []
        for index, layer in enumerate(self.layers):
            value, key, item = layer.step(
                value, k_cache[index], v_cache[index], write_mask, attention_bias
            )
            new_keys.append(key)
            new_values.append(item)
        return self.predict(value[:, -1, :]), torch.stack(new_keys), torch.stack(new_values)


def load_t2s_source(upstream: Path, checkpoint: Path) -> torch.nn.Module:
    previous = Path.cwd()
    try:
        os.chdir(upstream)
        sys.path.insert(0, str(upstream / "GPT_SoVITS"))
        sys.path.insert(0, str(upstream / "tools"))
        from export_torch_script import get_raw_t2s_model

        value = torch.load(checkpoint, map_location="cpu", weights_only=False)
        return get_raw_t2s_model(value).model.eval()
    finally:
        os.chdir(previous)


def export_t2s(
    artifacts: Path,
    output: Path,
    upstream: Path,
    checkpoint: Path,
    phone_length: int,
    cache_length: int,
) -> dict[str, object]:
    conditioning = load_file(artifacts / "conditioning.safetensors", device="cpu")
    source = load_t2s_source(upstream, checkpoint)
    prefill = T2SPrefill(source, conditioning).eval()
    step = T2SStep(source).eval()
    text_seq = torch.zeros((1, phone_length), dtype=torch.int64)
    text_bert = torch.zeros((phone_length, 1024), dtype=torch.float32)
    with torch.inference_mode():
        logits, compact_k, compact_v = prefill(text_seq, text_bert)
    compact_length = compact_k.shape[2]
    if cache_length <= compact_length:
        raise ValueError(
            f"--cache-length {cache_length} must exceed prefill length {compact_length}"
        )
    last_token = torch.argmax(logits[:, :-1], dim=-1, keepdim=True).to(torch.int64)
    position_embedding = source.ar_audio_position.pe[:, conditioning["prompt_semantic"].shape[1]].detach()
    k_cache = torch.zeros(
        (len(step.layers), 1, cache_length, compact_k.shape[-1]), dtype=torch.float32
    )
    v_cache = torch.zeros_like(k_cache)
    k_cache[:, :, :compact_length] = compact_k
    v_cache[:, :, :compact_length] = compact_v
    write_mask = torch.zeros((1, cache_length, 1), dtype=torch.float32)
    write_mask[:, compact_length] = 1.0
    attention_bias = torch.full((1, 1, 1, cache_length), -10000.0, dtype=torch.float32)
    attention_bias[..., : compact_length + 1] = 0.0
    with torch.inference_mode():
        step_outputs = step(
            last_token,
            position_embedding,
            k_cache,
            v_cache,
            write_mask,
            attention_bias,
        )

    prefill_graph = output / f"t2s_prefill_p{phone_length}.onnx"
    step_graph = output / f"t2s_step_c{cache_length}.onnx"
    export_onnx(
        prefill,
        (text_seq, text_bert),
        prefill_graph,
        ["text_seq", "text_bert"],
        ["logits", "k_cache", "v_cache"],
    )
    export_onnx(
        step,
        (last_token, position_embedding, k_cache, v_cache, write_mask, attention_bias),
        step_graph,
        [
            "last_token",
            "position_embedding",
            "k_cache",
            "v_cache",
            "write_mask",
            "attention_bias",
        ],
        ["logits", "new_keys", "new_values"],
    )
    torch.save(
        {
            "text_seq": text_seq,
            "text_bert": text_bert,
            "prefill_logits": logits,
            "prefill_k_cache": compact_k,
            "prefill_v_cache": compact_v,
            "last_token": last_token,
            "position_embedding": position_embedding,
            "k_cache": k_cache,
            "v_cache": v_cache,
            "write_mask": write_mask,
            "attention_bias": attention_bias,
            "step_logits": step_outputs[0],
            "step_new_keys": step_outputs[1],
            "step_new_values": step_outputs[2],
        },
        output / f"t2s_p{phone_length}_c{cache_length}_reference.pt",
    )
    return {
        "stage": "t2s",
        "graphs": [prefill_graph.name, step_graph.name],
        "phone_length": phone_length,
        "prefill_cache_length": compact_length,
        "cache_capacity": cache_length,
        "layers": len(step.layers),
        "hidden_size": compact_k.shape[-1],
    }


def export_t2s_padded(
    artifacts: Path,
    output: Path,
    upstream: Path,
    checkpoint: Path,
    phone_capacity: int,
    cache_length: int,
) -> dict[str, object]:
    conditioning = load_file(artifacts / "conditioning.safetensors", device="cpu")
    source = load_t2s_source(upstream, checkpoint)
    prefill = T2SPrefillPadded(source, conditioning).eval()
    step = T2SStep(source).eval()
    text_seq = torch.zeros((1, phone_capacity), dtype=torch.int64)
    text_bert = torch.zeros((phone_capacity, 1024), dtype=torch.float32)
    text_valid = torch.ones((1, phone_capacity), dtype=torch.float32)
    with torch.inference_mode():
        logits, compact_k, compact_v = prefill(text_seq, text_bert, text_valid)
    compact_length = compact_k.shape[2]
    if cache_length <= compact_length:
        raise ValueError(
            f"--cache-length {cache_length} must exceed padded prefill length {compact_length}"
        )
    last_token = torch.argmax(logits[:, :-1], dim=-1, keepdim=True).to(torch.int64)
    position_embedding = source.ar_audio_position.pe[:, conditioning["prompt_semantic"].shape[1]].detach()
    k_cache = torch.zeros(
        (len(step.layers), 1, cache_length, compact_k.shape[-1]), dtype=torch.float32
    )
    v_cache = torch.zeros_like(k_cache)
    k_cache[:, :, :compact_length] = compact_k
    v_cache[:, :, :compact_length] = compact_v
    write_mask = torch.zeros((1, cache_length, 1), dtype=torch.float32)
    write_mask[:, compact_length] = 1.0
    attention_bias = torch.full((1, 1, 1, cache_length), -10000.0, dtype=torch.float32)
    attention_bias[..., : compact_length + 1] = 0.0
    with torch.inference_mode():
        step_outputs = step(
            last_token,
            position_embedding,
            k_cache,
            v_cache,
            write_mask,
            attention_bias,
        )
    prefill_graph = output / f"t2s_prefill_pc{phone_capacity}.onnx"
    step_graph = output / f"t2s_step_c{cache_length}.onnx"
    export_onnx(
        prefill,
        (text_seq, text_bert, text_valid),
        prefill_graph,
        ["text_seq", "text_bert", "text_valid"],
        ["logits", "k_cache", "v_cache"],
    )
    export_onnx(
        step,
        (last_token, position_embedding, k_cache, v_cache, write_mask, attention_bias),
        step_graph,
        [
            "last_token",
            "position_embedding",
            "k_cache",
            "v_cache",
            "write_mask",
            "attention_bias",
        ],
        ["logits", "new_keys", "new_values"],
    )
    torch.save(
        {
            "text_seq": text_seq,
            "text_bert": text_bert,
            "text_valid": text_valid,
            "prefill_logits": logits,
            "prefill_k_cache": compact_k,
            "prefill_v_cache": compact_v,
            "last_token": last_token,
            "position_embedding": position_embedding,
            "k_cache": k_cache,
            "v_cache": v_cache,
            "write_mask": write_mask,
            "attention_bias": attention_bias,
            "step_logits": step_outputs[0],
            "step_new_keys": step_outputs[1],
            "step_new_values": step_outputs[2],
        },
        output / f"t2s_pc{phone_capacity}_c{cache_length}_reference.pt",
    )
    return {
        "stage": "t2s_padded",
        "graphs": [prefill_graph.name, step_graph.name],
        "phone_capacity": phone_capacity,
        "prompt_phone_length": int(conditioning["prompt_phone_ids"].shape[1]),
        "prompt_semantic_length": int(conditioning["prompt_semantic"].shape[1]),
        "prefill_cache_length": int(compact_length),
        "cache_capacity": cache_length,
        "layers": len(step.layers),
        "hidden_size": int(compact_k.shape[-1]),
        "padding_mask_input": True,
    }


class VitsDecoder(torch.nn.Module):
    """SoVITS decoder with explicit noise input and conversion-time reference conditioning."""

    def __init__(self, source: torch.nn.Module, conditioning: dict[str, torch.Tensor]):
        super().__init__()
        self.ref_enc = source.ref_enc
        self.sv_emb = source.sv_emb
        self.prelu = source.prelu
        self.quantizer = source.quantizer
        self.ge_to512 = source.ge_to512
        self.enc_p = source.enc_p
        self.flow = source.flow
        self.dec = source.dec
        self.ssl_dim = source.ssl_dim
        self.register_buffer("reference_spectrogram", conditioning["reference_spectrogram"])
        self.register_buffer("speaker_embedding", conditioning["speaker_embedding"])

    def statistics(
        self, pred_semantic: torch.Tensor, text_seq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        refer_mask = torch.ones_like(self.reference_spectrogram[:1, :1, :])
        ge = self.ref_enc(self.reference_spectrogram[:, :704] * refer_mask, refer_mask)
        ge = self.prelu(ge + self.sv_emb(self.speaker_embedding).unsqueeze(-1))
        quantized = self.quantizer.decode(pred_semantic)
        quantized = torch.cat([quantized, quantized]).permute(1, 2, 0)
        quantized = quantized.contiguous().view(1, self.ssl_dim, -1)
        ge_for_text = self.ge_to512(ge.transpose(2, 1)).transpose(2, 1)
        _, mean, log_scale, mask = self.enc_p(quantized, text_seq, ge_for_text, 1.0)
        return mean, log_scale, mask, ge

    def forward(
        self, pred_semantic: torch.Tensor, text_seq: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        mean, log_scale, mask, ge = self.statistics(pred_semantic, text_seq)
        latent = mean + noise * torch.exp(log_scale) * 0.5
        latent = self.flow(latent, mask, g=ge, reverse=True) * mask
        return self.dec(latent, g=ge)[0, 0]


class VitsGeneratorPadded(torch.nn.Module):
    """HiFi-GAN generator that keeps padded time positions identically zero."""

    def __init__(self, source: torch.nn.Module):
        super().__init__()
        self.conv_pre = source.conv_pre
        self.cond = source.cond
        self.ups = source.ups
        self.resblocks = source.resblocks
        self.conv_post = source.conv_post
        self.num_kernels = source.num_kernels
        self.num_upsamples = source.num_upsamples
        self.upsample_rates = [int(layer.stride[0]) for layer in source.ups]

    def forward(
        self, x: torch.Tensor, g: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        x = self.conv_pre(x * mask)
        x = (x + self.cond(g)) * mask
        for index in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1) * mask
            x = self.ups[index](x)
            mask = torch.repeat_interleave(mask, self.upsample_rates[index], dim=2)
            x = x * mask
            combined = self.resblocks[index * self.num_kernels](x, mask)
            for kernel in range(1, self.num_kernels):
                combined = combined + self.resblocks[
                    index * self.num_kernels + kernel
                ](x, mask)
            x = combined / self.num_kernels
            x = x * mask
        x = F.leaky_relu(x) * mask
        return torch.tanh(self.conv_post(x)) * mask


class VitsDecoderPadded(VitsDecoder):
    """Static-capacity decoder with exact masks for semantic and phone padding."""

    def __init__(self, source: torch.nn.Module, conditioning: dict[str, torch.Tensor]):
        super().__init__(source, conditioning)
        self.dec = VitsGeneratorPadded(self.dec)

    def statistics_padded(
        self,
        pred_semantic: torch.Tensor,
        text_seq: torch.Tensor,
        semantic_valid: torch.Tensor,
        text_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        refer_mask = torch.ones_like(self.reference_spectrogram[:1, :1, :])
        ge = self.ref_enc(self.reference_spectrogram[:, :704] * refer_mask, refer_mask)
        ge = self.prelu(ge + self.sv_emb(self.speaker_embedding).unsqueeze(-1))
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
        y = encoder.mrte(y, y_mask, text, text_mask, self.ge_to512(ge.transpose(2, 1)).transpose(2, 1))
        y = encoder.encoder2(y * y_mask, y_mask)
        stats = encoder.proj(y) * y_mask
        mean, log_scale = torch.split(stats, encoder.out_channels, dim=1)
        return mean, log_scale, y_mask, ge

    def forward(
        self,
        pred_semantic: torch.Tensor,
        text_seq: torch.Tensor,
        noise: torch.Tensor,
        semantic_valid: torch.Tensor,
        text_valid: torch.Tensor,
    ) -> torch.Tensor:
        mean, log_scale, mask, ge = self.statistics_padded(
            pred_semantic, text_seq, semantic_valid, text_valid
        )
        latent = mean + noise * torch.exp(log_scale) * 0.5
        latent = self.flow(latent, mask, g=ge, reverse=True) * mask
        return self.dec(latent, ge, mask)[0, 0]


def load_vits_source(upstream: Path, checkpoint: Path) -> torch.nn.Module:
    previous = Path.cwd()
    try:
        os.chdir(upstream)
        sys.path.insert(0, str(upstream / "GPT_SoVITS"))
        sys.path.insert(0, str(upstream / "tools"))
        from export_torch_script import VitsModel

        return VitsModel(
            checkpoint, version="v2ProPlus", is_half=False, device="cpu"
        ).vq_model.eval()
    finally:
        os.chdir(previous)


def export_vits(
    artifacts: Path,
    output: Path,
    upstream: Path,
    checkpoint: Path,
    phone_length: int,
    semantic_length: int,
) -> dict[str, object]:
    conditioning = load_file(artifacts / "conditioning.safetensors", device="cpu")
    source = load_vits_source(upstream, checkpoint)
    module = VitsDecoder(source, conditioning).eval()
    pred_semantic = torch.zeros((1, 1, semantic_length), dtype=torch.int64)
    text_seq = torch.zeros((1, phone_length), dtype=torch.int64)
    with torch.inference_mode():
        mean, _, _, _ = module.statistics(pred_semantic, text_seq)
    generator = torch.Generator(device="cpu").manual_seed(1234)
    noise = torch.randn(mean.shape, generator=generator, dtype=mean.dtype)
    with torch.inference_mode():
        audio = module(pred_semantic, text_seq, noise)
    graph = output / f"vits_p{phone_length}_s{semantic_length}.onnx"
    export_onnx(
        module,
        (pred_semantic, text_seq, noise),
        graph,
        ["pred_semantic", "text_seq", "noise"],
        ["audio"],
    )
    torch.save(
        {
            "pred_semantic": pred_semantic,
            "text_seq": text_seq,
            "noise": noise,
            "audio": audio,
        },
        output / f"vits_p{phone_length}_s{semantic_length}_reference.pt",
    )
    return {
        "stage": "vits",
        "graph": graph.name,
        "phone_length": phone_length,
        "semantic_length": semantic_length,
        "noise_shape": list(noise.shape),
        "audio_samples": audio.numel(),
        "sample_rate": 32000,
    }


def export_vits_padded(
    artifacts: Path,
    output: Path,
    upstream: Path,
    checkpoint: Path,
    phone_capacity: int,
    semantic_capacity: int,
) -> dict[str, object]:
    conditioning = load_file(artifacts / "conditioning.safetensors", device="cpu")
    source = load_vits_source(upstream, checkpoint)
    module = VitsDecoderPadded(source, conditioning).eval()
    pred_semantic = torch.zeros((1, 1, semantic_capacity), dtype=torch.int64)
    text_seq = torch.zeros((1, phone_capacity), dtype=torch.int64)
    semantic_valid = torch.ones((1, semantic_capacity), dtype=torch.float32)
    text_valid = torch.ones((1, phone_capacity), dtype=torch.float32)
    with torch.inference_mode():
        mean, _, _, _ = module.statistics_padded(
            pred_semantic, text_seq, semantic_valid, text_valid
        )
    generator = torch.Generator(device="cpu").manual_seed(1234)
    noise = torch.randn(mean.shape, generator=generator, dtype=mean.dtype)
    with torch.inference_mode():
        audio = module(pred_semantic, text_seq, noise, semantic_valid, text_valid)
    graph = output / f"vits_pc{phone_capacity}_sc{semantic_capacity}.onnx"
    export_onnx(
        module,
        (pred_semantic, text_seq, noise, semantic_valid, text_valid),
        graph,
        ["pred_semantic", "text_seq", "noise", "semantic_valid", "text_valid"],
        ["audio"],
    )
    torch.save(
        {
            "pred_semantic": pred_semantic,
            "text_seq": text_seq,
            "noise": noise,
            "semantic_valid": semantic_valid,
            "text_valid": text_valid,
            "audio": audio,
        },
        output / f"vits_pc{phone_capacity}_sc{semantic_capacity}_reference.pt",
    )
    return {
        "stage": "vits_padded",
        "graph": graph.name,
        "phone_capacity": phone_capacity,
        "semantic_capacity": semantic_capacity,
        "noise_shape": list(noise.shape),
        "audio_capacity_samples": audio.numel(),
        "samples_per_semantic": audio.numel() // semantic_capacity,
        "sample_rate": 32000,
        "padding_mask_inputs": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stage", choices=["bert", "t2s", "vits"], default="bert")
    parser.add_argument("--token-length", type=int, default=8)
    parser.add_argument("--phone-length", type=int, default=8)
    parser.add_argument("--cache-length", type=int, default=512)
    parser.add_argument("--semantic-length", type=int, default=64)
    parser.add_argument("--padded", action="store_true")
    parser.add_argument("--upstream", type=Path, default=Path(".."))
    parser.add_argument("--gpt", type=Path)
    parser.add_argument("--sovits", type=Path)
    args = parser.parse_args()

    artifacts = args.artifacts.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not (artifacts / "conditioning.safetensors").is_file():
        raise SystemExit(f"missing V2PP artifact directory: {artifacts}")
    # Load once here as an early integrity check; later stages consume the same fixed conditioning.
    conditioning = load_file(artifacts / "conditioning.safetensors", device="cpu")
    required = {
        "prompt_phone_ids",
        "prompt_semantic",
        "prompt_bert",
        "reference_spectrogram",
        "speaker_embedding",
    }
    if set(conditioning) != required:
        raise SystemExit(f"unexpected conditioning tensors: {sorted(conditioning)}")

    if args.stage == "bert":
        manifest = export_bert(artifacts, output, args.token_length)
        manifest_path = output / f"bert_tokens_{args.token_length}.json"
    elif args.stage == "t2s":
        if args.gpt is None:
            raise SystemExit("--gpt is required for --stage t2s")
        if args.padded:
            manifest = export_t2s_padded(
                artifacts,
                output,
                args.upstream.resolve(),
                args.gpt.resolve(),
                args.phone_length,
                args.cache_length,
            )
            manifest_path = output / f"t2s_pc{args.phone_length}_c{args.cache_length}.json"
        else:
            manifest = export_t2s(
                artifacts,
                output,
                args.upstream.resolve(),
                args.gpt.resolve(),
                args.phone_length,
                args.cache_length,
            )
            manifest_path = output / f"t2s_p{args.phone_length}_c{args.cache_length}.json"
    else:
        if args.sovits is None:
            raise SystemExit("--sovits is required for --stage vits")
        if args.padded:
            manifest = export_vits_padded(
                artifacts,
                output,
                args.upstream.resolve(),
                args.sovits.resolve(),
                args.phone_length,
                args.semantic_length,
            )
            manifest_path = output / f"vits_pc{args.phone_length}_sc{args.semantic_length}.json"
        else:
            manifest = export_vits(
                artifacts,
                output,
                args.upstream.resolve(),
                args.sovits.resolve(),
                args.phone_length,
                args.semantic_length,
            )
            manifest_path = output / f"vits_p{args.phone_length}_s{args.semantic_length}.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Exported {args.stage}: {manifest}")


if __name__ == "__main__":
    main()
