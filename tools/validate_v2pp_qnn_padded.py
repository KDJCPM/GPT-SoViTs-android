#!/usr/bin/env python3
"""Validate static-capacity V2PP QNN exports against exact-shape FP32 modules.

This runs the real GPT and SoVITS weights. It proves that padding is invisible to valid T2S
positions, the first autoregressive step sees the same cache, and the valid VITS PCM prefix is
unchanged. It does not validate FP16 QAIRT output; that remains a separate conversion gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from export_v2pp_qnn_onnx import (
    T2SPrefill,
    T2SPrefillPadded,
    T2SStep,
    VitsDecoder,
    VitsDecoderPadded,
    load_t2s_source,
    load_vits_source,
)


def difference(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    if actual.shape != expected.shape:
        raise AssertionError(f"shape mismatch: {tuple(actual.shape)} != {tuple(expected.shape)}")
    delta = (actual.float() - expected.float()).abs()
    denominator = expected.float().abs().clamp_min(1.0e-8)
    return {
        "max_abs": float(delta.max()) if delta.numel() else 0.0,
        "max_rel": float((delta / denominator).max()) if delta.numel() else 0.0,
        "mean_abs": float(delta.mean()) if delta.numel() else 0.0,
    }


def require_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, float]:
    stats = difference(actual, expected)
    if not torch.allclose(
        actual.float(),
        expected.float(),
        atol=absolute_tolerance,
        rtol=relative_tolerance,
    ):
        raise AssertionError(
            f"{name} differs: max_abs={stats['max_abs']:.9g} "
            f"max_rel={stats['max_rel']:.9g} mean_abs={stats['mean_abs']:.9g}"
        )
    return stats


def deterministic_ids(length: int, vocabulary: int, offset: int) -> torch.Tensor:
    if vocabulary <= 1:
        raise ValueError(f"invalid vocabulary size: {vocabulary}")
    values = (torch.arange(length, dtype=torch.int64) * 37 + offset) % (vocabulary - 1)
    return (values + 1).reshape(1, length)


def embedding_vocabulary(module: torch.nn.Module) -> int:
    direct = getattr(module, "num_embeddings", None)
    if direct is not None:
        return int(direct)
    wrapped = getattr(module, "word_embeddings", None)
    if wrapped is not None and getattr(wrapped, "num_embeddings", None) is not None:
        return int(wrapped.num_embeddings)
    raise ValueError(f"cannot determine vocabulary size for {type(module).__name__}")


def validate_t2s(
    artifacts: Path,
    upstream: Path,
    checkpoint: Path,
    phone_capacity: int,
    valid_lengths: list[int],
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, object]:
    conditioning = load_file(artifacts / "conditioning.safetensors", device="cpu")
    source = load_t2s_source(upstream, checkpoint)
    exact = T2SPrefill(source, conditioning).eval()
    padded = T2SPrefillPadded(source, conditioning).eval()
    step = T2SStep(source).eval()
    vocabulary = embedding_vocabulary(source.ar_text_embedding)
    prompt_phone_length = int(conditioning["prompt_phone_ids"].shape[1])
    prompt_semantic_length = int(conditioning["prompt_semantic"].shape[1])
    padded_prefill_length = prompt_phone_length + phone_capacity + prompt_semantic_length
    cache_capacity = padded_prefill_length + 2
    position = source.ar_audio_position.pe[:, prompt_semantic_length].detach()
    results: list[dict[str, object]] = []

    with torch.inference_mode():
        for valid_length in valid_lengths:
            if valid_length < 1 or valid_length > phone_capacity:
                raise ValueError(
                    f"T2S valid length {valid_length} is outside 1..{phone_capacity}"
                )
            phone_ids = deterministic_ids(valid_length, vocabulary, 11)
            generator = torch.Generator(device="cpu").manual_seed(1000 + valid_length)
            text_bert = torch.randn(
                (valid_length, 1024), generator=generator, dtype=torch.float32
            )
            padded_ids = torch.zeros((1, phone_capacity), dtype=torch.int64)
            padded_ids[:, :valid_length] = phone_ids
            padded_bert = torch.zeros((phone_capacity, 1024), dtype=torch.float32)
            padded_bert[:valid_length] = text_bert
            text_valid = torch.zeros((1, phone_capacity), dtype=torch.float32)
            text_valid[:, :valid_length] = 1.0

            exact_logits, exact_keys, exact_values = exact(phone_ids, text_bert)
            padded_logits, padded_keys, padded_values = padded(
                padded_ids, padded_bert, text_valid
            )
            prompt_start = prompt_phone_length + phone_capacity
            valid_indices = torch.cat(
                [
                    torch.arange(prompt_phone_length + valid_length),
                    torch.arange(prompt_start, prompt_start + prompt_semantic_length),
                ]
            )
            gathered_keys = padded_keys.index_select(2, valid_indices)
            gathered_values = padded_values.index_select(2, valid_indices)

            item: dict[str, object] = {
                "valid_phones": valid_length,
                "prefill_logits": require_close(
                    "T2S prefill logits",
                    padded_logits,
                    exact_logits,
                    absolute_tolerance,
                    relative_tolerance,
                ),
                "prefill_keys": require_close(
                    "T2S prefill keys",
                    gathered_keys,
                    exact_keys,
                    absolute_tolerance,
                    relative_tolerance,
                ),
                "prefill_values": require_close(
                    "T2S prefill values",
                    gathered_values,
                    exact_values,
                    absolute_tolerance,
                    relative_tolerance,
                ),
            }

            last_token = torch.argmax(exact_logits[:, :-1], dim=-1, keepdim=True).to(torch.int64)
            exact_length = int(exact_keys.shape[2])
            exact_cache_keys = torch.zeros(
                (exact_keys.shape[0], 1, cache_capacity, exact_keys.shape[-1]),
                dtype=torch.float32,
            )
            exact_cache_values = torch.zeros_like(exact_cache_keys)
            exact_cache_keys[:, :, :exact_length] = exact_keys
            exact_cache_values[:, :, :exact_length] = exact_values
            exact_write = torch.zeros((1, cache_capacity, 1), dtype=torch.float32)
            exact_write[:, exact_length] = 1.0
            exact_attention = torch.full(
                (1, 1, 1, cache_capacity), -10000.0, dtype=torch.float32
            )
            exact_attention[..., : exact_length + 1] = 0.0

            padded_cache_keys = torch.zeros_like(exact_cache_keys)
            padded_cache_values = torch.zeros_like(exact_cache_values)
            padded_cache_keys[:, :, :padded_prefill_length] = padded_keys
            padded_cache_values[:, :, :padded_prefill_length] = padded_values
            padded_write = torch.zeros_like(exact_write)
            padded_write[:, padded_prefill_length] = 1.0
            padded_attention = torch.full_like(exact_attention, -10000.0)
            padded_attention[..., valid_indices] = 0.0
            padded_attention[..., padded_prefill_length] = 0.0

            exact_step = step(
                last_token,
                position,
                exact_cache_keys,
                exact_cache_values,
                exact_write,
                exact_attention,
            )
            padded_step = step(
                last_token,
                position,
                padded_cache_keys,
                padded_cache_values,
                padded_write,
                padded_attention,
            )
            item["step_logits"] = require_close(
                "T2S step logits",
                padded_step[0],
                exact_step[0],
                absolute_tolerance,
                relative_tolerance,
            )
            item["step_keys"] = require_close(
                "T2S step keys",
                padded_step[1],
                exact_step[1],
                absolute_tolerance,
                relative_tolerance,
            )
            item["step_values"] = require_close(
                "T2S step values",
                padded_step[2],
                exact_step[2],
                absolute_tolerance,
                relative_tolerance,
            )
            results.append(item)

    return {
        "phone_capacity": phone_capacity,
        "prompt_phone_length": prompt_phone_length,
        "prompt_semantic_length": prompt_semantic_length,
        "cases": results,
    }


def validate_vits(
    artifacts: Path,
    upstream: Path,
    checkpoint: Path,
    phone_capacity: int,
    semantic_capacity: int,
    cases: list[tuple[int, int]],
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, object]:
    conditioning = load_file(artifacts / "conditioning.safetensors", device="cpu")
    source = load_vits_source(upstream, checkpoint)
    exact = VitsDecoder(source, conditioning).eval()
    padded = VitsDecoderPadded(source, conditioning).eval()
    phone_vocabulary = embedding_vocabulary(source.enc_p.text_embedding)
    semantic_vocabulary = int(source.quantizer.vq.layers[0]._codebook.codebook_size)
    results: list[dict[str, object]] = []

    with torch.inference_mode():
        for phone_length, semantic_length in cases:
            if phone_length < 1 or phone_length > phone_capacity:
                raise ValueError(
                    f"VITS phone length {phone_length} is outside 1..{phone_capacity}"
                )
            if semantic_length < 1 or semantic_length > semantic_capacity:
                raise ValueError(
                    f"VITS semantic length {semantic_length} is outside 1..{semantic_capacity}"
                )
            phone_ids = deterministic_ids(phone_length, phone_vocabulary, 19)
            semantics = deterministic_ids(semantic_length, semantic_vocabulary, 23).reshape(
                1, 1, semantic_length
            )
            padded_phones = torch.zeros((1, phone_capacity), dtype=torch.int64)
            padded_phones[:, :phone_length] = phone_ids
            padded_semantics = torch.zeros((1, 1, semantic_capacity), dtype=torch.int64)
            padded_semantics[:, :, :semantic_length] = semantics
            phone_valid = torch.zeros((1, phone_capacity), dtype=torch.float32)
            phone_valid[:, :phone_length] = 1.0
            semantic_valid = torch.zeros((1, semantic_capacity), dtype=torch.float32)
            semantic_valid[:, :semantic_length] = 1.0

            exact_mean, exact_log_scale, exact_mask, _ = exact.statistics(
                semantics, phone_ids
            )
            padded_mean, padded_log_scale, padded_mask, _ = padded.statistics_padded(
                padded_semantics,
                padded_phones,
                semantic_valid,
                phone_valid,
            )
            valid_frames = int(exact_mean.shape[-1])
            generator = torch.Generator(device="cpu").manual_seed(
                2000 + phone_length * 100 + semantic_length
            )
            exact_noise = torch.randn(
                exact_mean.shape, generator=generator, dtype=exact_mean.dtype
            )
            padded_noise = torch.zeros_like(padded_mean)
            padded_noise[..., :valid_frames] = exact_noise
            exact_audio = exact(semantics, phone_ids, exact_noise)
            padded_audio = padded(
                padded_semantics,
                padded_phones,
                padded_noise,
                semantic_valid,
                phone_valid,
            )
            valid_samples = int(exact_audio.numel())
            results.append(
                {
                    "valid_phones": phone_length,
                    "valid_semantics": semantic_length,
                    "mean": require_close(
                        "VITS mean",
                        padded_mean[..., :valid_frames],
                        exact_mean,
                        absolute_tolerance,
                        relative_tolerance,
                    ),
                    "log_scale": require_close(
                        "VITS log scale",
                        padded_log_scale[..., :valid_frames],
                        exact_log_scale,
                        absolute_tolerance,
                        relative_tolerance,
                    ),
                    "mask": require_close(
                        "VITS mask",
                        padded_mask[..., :valid_frames],
                        exact_mask,
                        0.0,
                        0.0,
                    ),
                    "audio": require_close(
                        "VITS PCM prefix",
                        padded_audio[:valid_samples],
                        exact_audio,
                        absolute_tolerance,
                        relative_tolerance,
                    ),
                    "valid_samples": valid_samples,
                    "padded_samples": int(padded_audio.numel()),
                }
            )

    return {
        "phone_capacity": phone_capacity,
        "semantic_capacity": semantic_capacity,
        "cases": results,
    }


def parse_lengths(raw: str) -> list[int]:
    values = [int(item) for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one length is required")
    return values


def parse_vits_cases(raw: str) -> list[tuple[int, int]]:
    values: list[tuple[int, int]] = []
    for item in raw.split(","):
        if not item.strip():
            continue
        try:
            phones, semantics = item.lower().split("x", 1)
            values.append((int(phones), int(semantics)))
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                "VITS cases must use PHONE_LENGTHxSEMANTIC_LENGTH"
            ) from error
    if not values:
        raise argparse.ArgumentTypeError("at least one VITS case is required")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--gpt", required=True, type=Path)
    parser.add_argument("--sovits", required=True, type=Path)
    parser.add_argument("--phone-capacity", type=int, default=16)
    parser.add_argument("--semantic-capacity", type=int, default=32)
    parser.add_argument("--t2s-valid-lengths", type=parse_lengths, default=[1, 8, 16])
    parser.add_argument("--vits-cases", type=parse_vits_cases, default=[(1, 1), (8, 16), (16, 32)])
    parser.add_argument("--absolute-tolerance", type=float, default=2.0e-4)
    parser.add_argument("--relative-tolerance", type=float, default=2.0e-4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    artifacts = args.artifacts.resolve()
    upstream = args.upstream.resolve()
    result = {
        "format": "gsv-v2pp-qnn-padded-fp32-validation",
        "format_version": 1,
        "absolute_tolerance": args.absolute_tolerance,
        "relative_tolerance": args.relative_tolerance,
        "t2s": validate_t2s(
            artifacts,
            upstream,
            args.gpt.resolve(),
            args.phone_capacity,
            args.t2s_valid_lengths,
            args.absolute_tolerance,
            args.relative_tolerance,
        ),
        "vits": validate_vits(
            artifacts,
            upstream,
            args.sovits.resolve(),
            args.phone_capacity,
            args.semantic_capacity,
            args.vits_cases,
            args.absolute_tolerance,
            args.relative_tolerance,
        ),
    }
    encoded = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
