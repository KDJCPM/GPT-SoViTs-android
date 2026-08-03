#!/usr/bin/env python3
"""Reference CPU executor for the canonical GSV ABI.

It proves that a canonical package runs without reading the source .pth/.ckpt. This transitional
adapter materializes upstream structures; native backends consume safetensors directly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import shutil
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file


def legacy_sovits_bytes(value: dict, header: bytes) -> bytes:
    stream = BytesIO()
    torch.save(value, stream)
    data = stream.getvalue()
    return data if header == b"PK" else header + data[2:]


def run(args: argparse.Namespace) -> None:
    model_path = args.model.resolve()
    reference_path = args.reference.resolve()
    output_path = args.output.resolve()
    upstream = args.upstream.resolve()
    os.chdir(upstream)
    sys.path.insert(0, str(upstream / "GPT_SoVITS"))
    with tempfile.TemporaryDirectory(prefix="gsvm-reference-") as temp_name:
        temp = Path(temp_name)
        with zipfile.ZipFile(model_path) as archive:
            archive.extractall(temp)
        manifest = json.loads((temp / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("model_abi") not in {"gsv-v2-pro-plus@1", "gsv-synthesis@1"}:
            raise ValueError(f"Unsupported ABI: {manifest.get('model_abi')}")
        gpt_meta = json.loads((temp / "config/gpt.json").read_text(encoding="utf-8"))
        sovits_meta = json.loads((temp / "config/sovits.json").read_text(encoding="utf-8"))
        gpt_value = {"weight": load_file(temp / "weights/gpt.safetensors"), **gpt_meta}
        sovits_value = {"weight": load_file(temp / "weights/sovits.safetensors"), **sovits_meta}
        gpt_path, sovits_path = temp / "runtime.ckpt", temp / "runtime.pth"
        torch.save(gpt_value, gpt_path)
        source_header = manifest.get("source_header", "PK").encode("ascii")
        sovits_path.write_bytes(legacy_sovits_bytes(sovits_value, source_header))

        # Upstream identifies headerless pretrained V4 by the source file's first-block MD5. A
        # lossless tensor reserialization necessarily changes that container MD5, so use the
        # conversion manifest instead of incorrectly falling back to the V3 size heuristic.
        import importlib
        tts_module=importlib.import_module("TTS_infer_pack.TTS")
        bert_asset=temp/"runtime_assets/bert"
        hubert_asset=temp/"runtime_assets/hubert"
        if not bert_asset.is_dir() or not hubert_asset.is_dir():
            raise RuntimeError("Complete package is missing BERT/HuBERT runtime assets")
        import sv as sv_module
        speaker_asset=temp/"runtime_assets/speaker_encoder/pretrained_eres2netv2w24s4ep4.ckpt"
        if speaker_asset.is_file(): sv_module.sv_path=str(speaker_asset)
        if manifest["model_version"] == "v4":
            original_detect=tts_module.get_sovits_version_from_path_fast
            def detect_converted(path: str):
                return ("v2", "v4", False) if Path(path).resolve()==sovits_path.resolve() else original_detect(path)
            tts_module.get_sovits_version_from_path_fast=detect_converted
            packaged_vocoder=temp/"runtime_assets/v4/vocoder.pth"
            expected_vocoder=temp/"GPT_SoVITS/pretrained_models/gsv-v4-pretrained/vocoder.pth"
            expected_vocoder.parent.mkdir(parents=True,exist_ok=True)
            try: os.link(packaged_vocoder,expected_vocoder)
            except OSError: shutil.copyfile(packaged_vocoder,expected_vocoder)
            tts_module.now_dir=str(temp)
        TTS, TTS_Config=tts_module.TTS,tts_module.TTS_Config

        config = TTS_Config({"custom": {
            "device": "cpu", "is_half": False, "version": manifest["model_version"],
            "t2s_weights_path": str(gpt_path), "vits_weights_path": str(sovits_path),
            "cnhuhbert_base_path": str(hubert_asset),
            "bert_base_path": str(bert_asset),
        }})
        tts = TTS(config)
        request = {
            "text": args.text, "text_lang": args.language,
            "ref_audio_path": str(reference_path),
            "prompt_text": args.prompt, "prompt_lang": args.prompt_language,
            "top_k": args.top_k, "top_p": args.top_p, "temperature": args.temperature,
            "text_split_method": "cut5", "batch_size": 1, "speed_factor": 1.0,
            "split_bucket": False, "return_fragment": False, "seed": args.seed,
            "parallel_infer": False, "repetition_penalty": 1.35, "sample_steps": args.sample_steps,
        }
        results = list(tts.run(request))
        if len(results) != 1:
            raise RuntimeError(f"Expected one output, got {len(results)}")
        sample_rate, pcm = results[0]
        if sample_rate != int(manifest["sample_rate"]):
            raise RuntimeError(f"Wrong model path selected: expected {manifest['sample_rate']} Hz, got {sample_rate} Hz")
        if not isinstance(pcm, np.ndarray) or pcm.dtype != np.int16:
            raise RuntimeError("Reference backend returned invalid PCM")
        sf.write(output_path, pcm, sample_rate, subtype="PCM_16")
        print(f"Wrote {output_path}: {sample_rate} Hz, {pcm.size} samples")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--upstream", type=Path, default=Path(".."))
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--prompt-language", default="zh")
    parser.add_argument("--text", required=True)
    parser.add_argument("--language", default="zh")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--sample-steps", type=int, default=32)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
