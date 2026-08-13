#!/usr/bin/env python3
"""Validate the exact staged V2 Pro Plus CPU deployment against upstream FP32 modules.

The report binds every packaged runtime/frontend file and both source checkpoints. Packaging may
claim deployable/upstream_equivalent only when this program completes all neural and end-to-end
checks. Run it with the GPT-SoVITS Python environment, not the QAIRT conversion environment.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def file_entry(path: Path, runtime_path: str) -> dict[str, object]:
    return {"path": runtime_path, "size": path.stat().st_size, "sha256": digest(path)}


def deployment_files(
    bert: Path,
    acoustic: Path,
    frontend: Path,
) -> list[dict[str, object]]:
    values = [
        file_entry(bert, "runtime/bert.pt"),
        file_entry(acoustic, "runtime/acoustic.pt"),
    ]
    values.extend(
        file_entry(path, f"runtime/frontend/{path.relative_to(frontend)}")
        for path in sorted(frontend.rglob("*"))
        if path.is_file()
    )
    return values


def difference(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, object]:
    if actual.shape != expected.shape:
        raise AssertionError(f"shape mismatch: {tuple(actual.shape)} != {tuple(expected.shape)}")
    delta = (actual.detach().float() - expected.detach().float()).abs()
    return {
        "shape": list(actual.shape),
        "max_abs": float(delta.max()) if delta.numel() else 0.0,
        "mean_abs": float(delta.mean()) if delta.numel() else 0.0,
    }


def require_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, object]:
    stats = difference(actual, expected)
    if not torch.allclose(
        actual.detach().float(),
        expected.detach().float(),
        atol=absolute_tolerance,
        rtol=relative_tolerance,
    ):
        raise AssertionError(
            f"{name} differs: max_abs={stats['max_abs']:.9g} mean_abs={stats['mean_abs']:.9g}"
        )
    return stats


def compare_tree(
    name: str,
    actual: Any,
    expected: Any,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, object]:
    results: dict[str, object] = {}

    def walk(label: str, left: Any, right: Any) -> None:
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            results[label] = require_close(
                label, left, right, absolute_tolerance, relative_tolerance
            )
            return
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            if len(left) != len(right):
                raise AssertionError(f"{label} length mismatch: {len(left)} != {len(right)}")
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                walk(f"{label}.{index}", left_item, right_item)
            return
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if left != right:
                raise AssertionError(f"{label} differs: {left} != {right}")
            results[label] = {"value": left}
            return
        raise AssertionError(f"{label} has unsupported values {type(left)} and {type(right)}")

    walk(name, actual, expected)
    return results


def clone_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, list):
        return [clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_tree(item) for item in value)
    return value


def validate_bert(
    model_root: Path,
    exported_path: Path,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, object]:
    from transformers import AutoModel
    from export_bert_mobile import LeanBert

    source_model = AutoModel.from_pretrained(
        model_root,
        local_files_only=True,
        attn_implementation="eager",
        add_pooling_layer=False,
    ).float().eval()
    source = LeanBert(source_model).eval()
    exported = torch.jit.load(str(exported_path), map_location="cpu").eval()
    input_ids = torch.tensor([[101, 872, 1962, 8024, 6821, 3221, 102]], dtype=torch.int64)
    attention = torch.ones_like(input_ids)
    token_types = torch.zeros_like(input_ids)
    word2ph = torch.tensor([2, 2, 1, 2, 1], dtype=torch.int32)
    with torch.inference_mode():
        expected = source(input_ids, attention, token_types, word2ph)
        actual = exported(input_ids, attention, token_types, word2ph)
    result = require_close("bert", actual, expected, absolute_tolerance, relative_tolerance)
    del source, source_model, exported
    gc.collect()
    return result


def validate_ssl(
    model_root: Path,
    exported_path: Path,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, object]:
    from transformers import HubertModel

    source = HubertModel.from_pretrained(
        model_root,
        local_files_only=True,
        torchscript=True,
    ).float().eval()
    exported = torch.jit.load(str(exported_path), map_location="cpu").eval()
    time = torch.arange(16_000, dtype=torch.float32) / 16_000.0
    pcm = (0.1 * torch.sin(2.0 * torch.pi * 220.0 * time)).reshape(1, -1)
    with torch.inference_mode():
        expected = source(pcm)[0].transpose(1, 2).float()
        actual = exported(pcm)
    result = require_close("ssl", actual, expected, absolute_tolerance, relative_tolerance)
    del source, exported
    gc.collect()
    return result


def load_upstream_modules(
    upstream: Path,
    gpt: Path,
    sovits: Path,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    for path in (upstream, upstream / "GPT_SoVITS", upstream / "tools"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    import export_torch_script
    from export_torch_script import T2SModel, VitsModel, get_raw_t2s_model
    from stream_v2pro import ExportERes2NetV2, StepVitsModel, StreamT2SModel, init_sv_cn

    checkpoint = torch.load(gpt, map_location="cpu", weights_only=False)
    source_t2s = StreamT2SModel(T2SModel(get_raw_t2s_model(checkpoint).float().eval())).eval()
    if export_torch_script.sv_cn_model is None:
        init_sv_cn("cpu", False)
    source_vits = StepVitsModel(
        VitsModel(sovits, "v2ProPlus", is_half=False, device="cpu").eval(),
        ExportERes2NetV2(export_torch_script.sv_cn_model).eval(),
    ).eval()
    return source_t2s, source_vits


def validate_t2s(
    source: torch.nn.Module,
    exported_path: Path,
    conditioning: dict[str, torch.Tensor],
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, object]:
    exported = torch.jit.load(str(exported_path), map_location="cpu").eval()
    prompts = conditioning["prompt_semantic"]
    reference_phones = conditioning["prompt_phone_ids"]
    reference_bert = conditioning["prompt_bert"]
    text_phones = reference_phones[:, :8].clone()
    text_bert = reference_bert[:8].clone()
    arguments = (prompts, reference_phones, text_phones, reference_bert, text_bert, 10)
    with torch.inference_mode():
        torch.manual_seed(1234)
        expected_prefill = source.pre_infer(*clone_tree(arguments))
        torch.manual_seed(1234)
        actual_prefill = exported.pre_infer(*clone_tree(arguments))
        prefill = compare_tree(
            "t2s.prefill",
            actual_prefill,
            expected_prefill,
            absolute_tolerance,
            relative_tolerance,
        )
        expected_step_args = (1, 10, *clone_tree(expected_prefill))
        actual_step_args = (1, 10, *clone_tree(actual_prefill))
        torch.manual_seed(4321)
        expected_step = source(*expected_step_args)
        torch.manual_seed(4321)
        actual_step = exported(*actual_step_args)
        step = compare_tree(
            "t2s.step",
            actual_step,
            expected_step,
            absolute_tolerance,
            relative_tolerance,
        )
    del exported
    gc.collect()
    return {"prefill": prefill, "step": step}


def deterministic_reference(seconds: int = 1) -> torch.Tensor:
    time = torch.arange(32_000 * seconds, dtype=torch.float32) / 32_000.0
    value = 0.08 * torch.sin(2.0 * torch.pi * 196.0 * time)
    value += 0.03 * torch.sin(2.0 * torch.pi * 311.0 * time)
    return value.reshape(1, -1)


def validate_vits(
    source: torch.nn.Module,
    exported_path: Path,
    conditioning: dict[str, torch.Tensor],
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[dict[str, object], torch.jit.ScriptModule]:
    exported = torch.jit.load(str(exported_path), map_location="cpu").eval()
    phones = conditioning["prompt_phone_ids"][:, :8].clone()
    semantics = conditioning["prompt_semantic"][:, :16].unsqueeze(0).clone()
    reference_spectrogram = conditioning["reference_spectrogram"]
    speaker_embedding = conditioning["speaker_embedding"]
    with torch.inference_mode():
        torch.manual_seed(2468)
        expected_audio = source(semantics, phones, reference_spectrogram, speaker_embedding, 1.0)
        torch.manual_seed(2468)
        actual_audio = exported(
            semantics, phones, reference_spectrogram, speaker_embedding, torch.tensor(1.0)
        )
        audio = require_close(
            "vits.audio",
            actual_audio,
            expected_audio,
            absolute_tolerance,
            relative_tolerance,
        )
        ssl_content = torch.randn((1, 768, 50), generator=torch.Generator().manual_seed(99))
        latent = require_close(
            "vits.extract_latent",
            exported.extract_latent(ssl_content),
            source.extract_latent(ssl_content),
            absolute_tolerance,
            relative_tolerance,
        )
        pcm32 = deterministic_reference()
        actual_reference = exported.ref_handle(pcm32)
        expected_reference = source.ref_handle(pcm32)
        reference = compare_tree(
            "vits.reference",
            actual_reference,
            expected_reference,
            absolute_tolerance,
            relative_tolerance,
        )
    return {"audio": audio, "extract_latent": latent, "reference": reference}, exported


def synthesize_upstream(
    t2s: torch.nn.Module,
    vits: torch.nn.Module,
    text_phones: torch.Tensor,
    text_bert: torch.Tensor,
    prompt_semantic: torch.Tensor,
    prompt_phone_ids: torch.Tensor,
    prompt_bert: torch.Tensor,
    reference_spectrogram: torch.Tensor,
    speaker_embedding: torch.Tensor,
    seed: int,
) -> tuple[int, torch.Tensor]:
    torch.manual_seed(seed)
    y_len, y, xy_pos, k_cache, v_cache = t2s.pre_infer(
        prompt_semantic,
        prompt_phone_ids,
        text_phones,
        prompt_bert,
        text_bert,
        10,
        1.0,
        1.0,
        1.35,
    )
    generated = 1
    for index in range(1, 1501):
        y, xy_pos, last_token, k_cache, v_cache = t2s(
            index, 10, y_len, y, xy_pos, k_cache, v_cache, 1.0, 1.0, 1.35
        )
        generated = index + 1
        if last_token == 1024:
            break
    semantic = y[:, -generated:].unsqueeze(0)
    audio = vits(
        semantic,
        text_phones,
        reference_spectrogram,
        speaker_embedding,
        1.0,
    ).reshape(-1).float().clamp(-1.0, 1.0)
    return 32_000, torch.round(audio * 32767.0).to(torch.int32)


def validate_fused_preset(
    source_t2s: torch.nn.Module,
    source_vits: torch.nn.Module,
    acoustic_path: Path,
    conditioning: dict[str, torch.Tensor],
) -> dict[str, object]:
    acoustic = torch.jit.load(str(acoustic_path), map_location="cpu").eval()
    phones = conditioning["prompt_phone_ids"][:, :8].clone()
    bert = conditioning["prompt_bert"][:8].clone()
    with torch.inference_mode():
        expected_rate, expected_pcm = synthesize_upstream(
            source_t2s,
            source_vits,
            phones,
            bert,
            conditioning["prompt_semantic"],
            conditioning["prompt_phone_ids"],
            conditioning["prompt_bert"],
            conditioning["reference_spectrogram"],
            conditioning["speaker_embedding"],
            13579,
        )
        actual_rate, actual_pcm = acoustic(phones, bert, 1.0, 10, 1.0, 1.35, 1.0, 13579)
    if int(actual_rate) != expected_rate:
        raise AssertionError(f"fused sample rate {actual_rate} != {expected_rate}")
    if actual_pcm.shape != expected_pcm.shape:
        raise AssertionError(
            f"fused PCM shape {tuple(actual_pcm.shape)} != {tuple(expected_pcm.shape)}"
        )
    delta = (actual_pcm.to(torch.int64) - expected_pcm.to(torch.int64)).abs()
    max_pcm_delta = int(delta.max()) if delta.numel() else 0
    if max_pcm_delta > 2:
        raise AssertionError(f"fused PCM differs from upstream by {max_pcm_delta} integer levels")
    peak = int(actual_pcm.abs().max()) if actual_pcm.numel() else 0
    if actual_pcm.numel() == 0 or peak <= 100:
        raise AssertionError("fused upstream validation returned empty or silent PCM")
    del acoustic
    gc.collect()
    return {
        "sample_rate": int(actual_rate),
        "samples": int(actual_pcm.numel()),
        "peak": peak,
        "max_pcm_delta": max_pcm_delta,
    }


def validate_fused_reference(
    ssl_model_root: Path,
    source_t2s: torch.nn.Module,
    source_vits: torch.nn.Module,
    acoustic_path: Path,
    conditioning: dict[str, torch.Tensor],
) -> dict[str, object]:
    from transformers import HubertModel

    source_ssl = HubertModel.from_pretrained(
        ssl_model_root,
        local_files_only=True,
        torchscript=True,
    ).float().eval()
    acoustic = torch.jit.load(str(acoustic_path), map_location="cpu").eval()
    text_phones = conditioning["prompt_phone_ids"][:, :8].clone()
    text_bert = conditioning["prompt_bert"][:8].clone()
    prompt_phone_ids = conditioning["prompt_phone_ids"][:, :8].clone()
    prompt_bert = conditioning["prompt_bert"][:8].clone()
    pcm32 = deterministic_reference()
    time16 = torch.arange(16_000, dtype=torch.float32) / 16_000.0
    pcm16 = (
        0.08 * torch.sin(2.0 * torch.pi * 196.0 * time16)
        + 0.03 * torch.sin(2.0 * torch.pi * 311.0 * time16)
    ).reshape(1, -1)
    with torch.inference_mode():
        ssl_content = source_ssl(pcm16)[0].transpose(1, 2).float()
        prompt_semantic = source_vits.extract_latent(ssl_content)
        reference_spectrogram, speaker_embedding = source_vits.ref_handle(pcm32)
        expected_rate, expected_pcm = synthesize_upstream(
            source_t2s,
            source_vits,
            text_phones,
            text_bert,
            prompt_semantic,
            prompt_phone_ids,
            prompt_bert,
            reference_spectrogram,
            speaker_embedding,
            97531,
        )
        actual_rate, actual_pcm = acoustic.synthesize_reference_options(
            text_phones,
            text_bert,
            pcm16,
            pcm32,
            prompt_phone_ids,
            prompt_bert,
            1.0,
            10,
            1.0,
            1.35,
            1.0,
            32,
            97531,
        )
    if int(actual_rate) != expected_rate:
        raise AssertionError(f"fused reference sample rate {actual_rate} != {expected_rate}")
    if actual_pcm.shape != expected_pcm.shape:
        raise AssertionError(
            "fused reference PCM shape "
            f"{tuple(actual_pcm.shape)} != {tuple(expected_pcm.shape)}"
        )
    delta = (actual_pcm.to(torch.int64) - expected_pcm.to(torch.int64)).abs()
    max_pcm_delta = int(delta.max()) if delta.numel() else 0
    if max_pcm_delta > 2:
        raise AssertionError(
            f"fused temporary-reference PCM differs from upstream by {max_pcm_delta} integer levels"
        )
    peak = int(actual_pcm.abs().max()) if actual_pcm.numel() else 0
    if actual_pcm.numel() == 0 or peak <= 100:
        raise AssertionError("fused temporary-reference validation returned empty or silent PCM")
    del source_ssl, acoustic
    gc.collect()
    return {
        "sample_rate": int(actual_rate),
        "samples": int(actual_pcm.numel()),
        "peak": peak,
        "max_pcm_delta": max_pcm_delta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--frontend", required=True, type=Path)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--gpt", required=True, type=Path)
    parser.add_argument("--sovits", required=True, type=Path)
    parser.add_argument("--bert", default="bert_mobile_fp32_eager.pt")
    parser.add_argument("--ssl", default="ssl_mobile_fp32.pt")
    parser.add_argument("--acoustic", default="pipeline_core.pt")
    parser.add_argument("--t2s", default="t2s.pt")
    parser.add_argument("--vits", default="vits.pt")
    parser.add_argument("--frontend-profile", default="full-zh-en-g2pw-v3")
    parser.add_argument("--absolute-tolerance", type=float, default=2.0e-4)
    parser.add_argument("--relative-tolerance", type=float, default=2.0e-4)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    original_cwd = Path.cwd()
    artifacts = args.artifacts.resolve()
    frontend = args.frontend.resolve()
    upstream = args.upstream.resolve()
    gpt = args.gpt.resolve()
    sovits = args.sovits.resolve()
    output = args.output.resolve()
    paths = {
        "bert": artifacts / args.bert,
        "ssl": artifacts / args.ssl,
        "acoustic": artifacts / args.acoustic,
        "t2s": artifacts / args.t2s,
        "vits": artifacts / args.vits,
        "conditioning": artifacts / "conditioning.safetensors",
    }
    missing = [str(path) for path in (*paths.values(), frontend, gpt, sovits) if not path.exists()]
    if missing:
        raise SystemExit(f"CPU validation inputs are missing: {', '.join(missing)}")

    try:
        # Upstream inference_webui creates a transient weight.json in cwd; never write into
        # the read-only upstream checkout during validation.
        os.chdir(artifacts)
        checks: dict[str, object] = {}
        bert_source = upstream / "pretrained_models/chinese-roberta-wwm-ext-large"
        if not bert_source.exists():
            bert_source = upstream / "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
        ssl_source = upstream / "pretrained_models/chinese-hubert-base"
        if not ssl_source.exists():
            ssl_source = upstream / "GPT_SoVITS/pretrained_models/chinese-hubert-base"
        checks["bert"] = validate_bert(
            bert_source,
            paths["bert"],
            args.absolute_tolerance,
            args.relative_tolerance,
        )
        checks["ssl"] = validate_ssl(
            ssl_source,
            paths["ssl"],
            args.absolute_tolerance,
            args.relative_tolerance,
        )
        source_t2s, source_vits = load_upstream_modules(upstream, gpt, sovits)
        conditioning = load_file(paths["conditioning"], device="cpu")
        checks["t2s"] = validate_t2s(
            source_t2s,
            paths["t2s"],
            conditioning,
            args.absolute_tolerance,
            args.relative_tolerance,
        )
        checks["vits"], exported_vits = validate_vits(
            source_vits,
            paths["vits"],
            conditioning,
            args.absolute_tolerance,
            args.relative_tolerance,
        )
        del exported_vits
        gc.collect()
        checks["fused_preset"] = validate_fused_preset(
            source_t2s, source_vits, paths["acoustic"], conditioning
        )
        checks["fused_reference"] = validate_fused_reference(
            ssl_source,
            source_t2s,
            source_vits,
            paths["acoustic"],
            conditioning,
        )
        report = {
            "format": "gsv-v2pp-cpu-upstream-validation",
            "format_version": 1,
            "passed": True,
            "model_version": "v2ProPlus",
            "frontend_profile": args.frontend_profile,
            "precision": "fp32",
            "quantization": "none",
            "sources": {
                "gpt": {"sha256": digest(gpt)},
                "sovits": {"sha256": digest(sovits)},
            },
            "files": deployment_files(paths["bert"], paths["acoustic"], frontend),
            "supporting_artifacts": {
                name: {"size": path.stat().st_size, "sha256": digest(path)}
                for name, path in paths.items()
            },
            "absolute_tolerance": args.absolute_tolerance,
            "relative_tolerance": args.relative_tolerance,
            "checks": checks,
        }
        encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
        print(encoded, end="")
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()
