#!/usr/bin/env python3
"""Assemble deployable V2 Pro Plus QNN pipeline and voice attachments."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import zipfile
from pathlib import Path

import onnx
from safetensors.torch import load_file

from audit_v2pp_qnn_product import audit_product
from build_qnn_attachment import QNN_SUFFIX, build_attachment, digest
from partition_onnx_contiguous import read_partition_manifest, tensor_description
from wrap_qnn_ep_context import match_tensor_names, wrap_context


FRONTEND_FILES = (
    "frontend.json",
    "jieba-dict.txt",
    "jieba-pos-hmm.bin",
    "polyphonic.rep",
    "polyphonic-fix.rep",
    "english.json",
    "english-lexicon.tsv",
    "english-names.tsv",
    "english-homographs.tsv",
    "english-tagger.bin",
    "english-g2p.bin",
    "english-unigrams.tsv",
    "english-bigrams.tsv",
)


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"required metadata does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_base_manifest(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"base model package does not exist: {path}")
    with zipfile.ZipFile(path) as archive:
        if not archive.namelist() or archive.namelist()[0] != "manifest.json":
            raise ValueError("base model package must place manifest.json first")
        document = json.loads(archive.read("manifest.json"))
    if document.get("artifact_role") != "model" or document.get("model_version") != "v2ProPlus":
        raise ValueError("base model must be a V2 Pro Plus model package")
    if not document.get("deployable"):
        raise ValueError("base model package is not deployable")
    if document.get("executor") not in ("torchscript-cpu-single", "torchscript-cpu-staged"):
        raise ValueError("base model package is not the CPU correctness artifact")
    if document.get("reference_input_version") != 1:
        raise ValueError(
            "base model package must expose reference_input_version=1 before QNN binding"
        )
    profile = document.get("frontend_profile")
    if not isinstance(profile, str) or not profile:
        raise ValueError("base model package is missing frontend_profile")
    return document


def component_backend(component: Path) -> tuple[Path, Path]:
    document = read_json(component / "manifest.json")
    backend = component / document.get("backend_artifact", "")
    info = component / "context-info.json"
    if not backend.is_file() or not info.is_file():
        raise ValueError(f"compiled component is incomplete: {component}")
    return backend, info


def wrap_component(
    *,
    name: str,
    source: Path,
    component: Path,
    output: Path,
) -> tuple[Path, Path]:
    manifest = read_json(component / "manifest.json")
    expected_hash = manifest.get("source_onnx_sha256")
    if not isinstance(expected_hash, str) or digest(source.resolve()) != expected_hash:
        raise ValueError(f"source ONNX does not match compiled component {name}")
    backend, info = component_backend(component)
    return wrap_context(source, backend, info, output / f"{name}.onnx")


def partition_graph_specs(
    manifest: Path,
    components: list[Path],
    prefix: str,
) -> list[dict[str, object]]:
    document, paths = read_partition_manifest(manifest)
    if len(paths) != len(components):
        raise ValueError(
            f"{prefix} has {len(paths)} ONNX partitions but {len(components)} components"
        )
    result: list[dict[str, object]] = []
    for index, (item, source, component) in enumerate(
        zip(document["partitions"], paths, components)
    ):
        name = f"{prefix}_{index:02d}"
        result.append(
            {
                "name": name,
                "source": source,
                "component": component,
                "path": f"runtime/qnn/{name}.onnx",
                "inputs": item["inputs"],
                "outputs": item["outputs"],
            }
        )
    return result


def runtime_partition_contract(
    stage: dict[str, object],
    wrapped_graph: Path,
) -> dict[str, object]:
    model = onnx.load(wrapped_graph, load_external_data=False)
    input_names = match_tensor_names(
        [item["name"] for item in stage["inputs"]],
        [value.name for value in model.graph.input],
        f"{stage['name']} inputs",
    )
    output_names = match_tensor_names(
        [item["name"] for item in stage["outputs"]],
        [value.name for value in model.graph.output],
        f"{stage['name']} outputs",
    )

    def contract(value: onnx.ValueInfoProto, names: dict[str, str]) -> dict[str, object]:
        result = tensor_description(value)
        result["logical_name"] = names[value.name]
        return result

    return {
        "name": stage["name"],
        "path": stage["path"],
        "inputs": [contract(value, input_names) for value in model.graph.input],
        "outputs": [contract(value, output_names) for value in model.graph.output],
    }


def build_executor(
    *,
    conditioning: Path,
    bert_metadata: Path,
    t2s_metadata: Path,
    vits_metadata: Path,
    reference_ssl_metadata: Path,
    reference_prompt_metadata: Path,
    reference_conditioning_metadata: Path,
    reference_t2s_metadata: Path,
    reference_vits_metadata: Path,
    g2pw_sequence_length: int,
    vits_partitions: list[dict[str, object]],
    reference_vits_partitions: list[dict[str, object]],
) -> dict:
    bert = read_json(bert_metadata)
    t2s = read_json(t2s_metadata)
    vits = read_json(vits_metadata)
    reference_ssl = read_json(reference_ssl_metadata)
    reference_prompt = read_json(reference_prompt_metadata)
    reference_conditioning = read_json(reference_conditioning_metadata)
    reference_t2s = read_json(reference_t2s_metadata)
    reference_vits = read_json(reference_vits_metadata)
    preset = load_file(conditioning, device="cpu")
    prompt_semantic = preset["prompt_semantic"].flatten().tolist()
    token_capacity = int(bert["token_length"])
    phone_capacity = int(t2s["phone_capacity"])
    semantic_capacity = int(vits["semantic_capacity"])
    if int(vits["phone_capacity"]) != phone_capacity:
        raise ValueError("T2S and VITS phone capacities do not match")
    if int(t2s["prompt_semantic_length"]) != len(prompt_semantic):
        raise ValueError("T2S metadata and preset prompt semantic lengths do not match")
    compact_length = int(t2s["prefill_cache_length"])
    cache_capacity = int(t2s["cache_capacity"])
    if compact_length + semantic_capacity > cache_capacity:
        raise ValueError("T2S cache cannot hold the configured semantic capacity")
    if g2pw_sequence_length != token_capacity:
        raise ValueError("G2PW sequence length and BERT token capacity must match")
    reference_phone_capacity = int(reference_t2s["phone_capacity"])
    prompt_phone_capacity = int(reference_t2s["prompt_phone_capacity"])
    reference_semantic_length = int(reference_prompt["semantic_length"])
    if reference_phone_capacity != phone_capacity or prompt_phone_capacity != phone_capacity:
        raise ValueError("preset and runtime-reference phone capacities do not match")
    if int(reference_t2s["prompt_semantic_length"]) != reference_semantic_length:
        raise ValueError("runtime-reference prompt semantic metadata does not match")
    reference_compact_length = int(reference_t2s["prefill_cache_length"])
    if reference_compact_length + semantic_capacity > cache_capacity:
        raise ValueError("runtime-reference T2S cache does not fit the semantic capacity")
    if int(reference_t2s["cache_capacity"]) != cache_capacity:
        raise ValueError("preset and runtime-reference T2S cache capacities do not match")
    if int(reference_vits["phone_capacity"]) != phone_capacity:
        raise ValueError("runtime-reference VITS phone capacity does not match T2S")
    if int(reference_vits["semantic_capacity"]) != semantic_capacity:
        raise ValueError("runtime-reference VITS semantic capacity does not match preset VITS")
    if int(reference_vits["sample_rate"]) != int(vits["sample_rate"]):
        raise ValueError("runtime-reference VITS sample rate does not match preset VITS")
    if int(reference_ssl["pcm_samples"]) != int(reference_conditioning["pcm_16k_samples"]):
        raise ValueError("runtime-reference 16 kHz PCM capacities do not match")
    if len(vits_partitions) < 2 or len(reference_vits_partitions) < 2:
        raise ValueError("production QNN VITS requires ordered contiguous partitions")
    return {
        "format": "gsv-qnn-executor",
        "format_version": 1,
        "operation": "synthesize_utf8_to_pcm16",
        "runtime_abi_version": 1,
        "complete": True,
        "utf8_text_input": True,
        "pcm16_output": True,
        "cpu_neural_fallback": False,
        "languages": ["auto", "zh", "en"],
        "engine": "gpt-sovits-v2pp-qnn-buckets",
        "engine_version": 2,
        # Sampling controls remain disabled until the complete options ABI is implemented.
        "runtime_options_version": 0,
        "reference_input_version": 1,
        "sample_rate": int(vits["sample_rate"]),
        "frontend": {
            "root": "runtime/frontend",
            "g2pw_model": "runtime/qnn/g2pw.onnx",
            "g2pw_sequence_length": g2pw_sequence_length,
        },
        "graphs": {
            "bert": "runtime/qnn/bert.onnx",
            "t2s_prefill": "runtime/qnn/t2s_prefill.onnx",
            "t2s_step": "runtime/qnn/t2s_step.onnx",
            "vits": vits_partitions,
            "reference_ssl": "runtime/qnn/reference_ssl.onnx",
            "reference_prompt_semantic": "runtime/qnn/reference_prompt_semantic.onnx",
            "reference_conditioning": "runtime/qnn/reference_conditioning.onnx",
            "t2s_reference_prefill": "runtime/qnn/t2s_reference_prefill.onnx",
            "vits_reference": reference_vits_partitions,
        },
        "shapes": {
            "token_capacity": token_capacity,
            "phone_capacity": phone_capacity,
            "semantic_capacity": semantic_capacity,
            "preset_prompt_phone_length": int(t2s["prompt_phone_length"]),
            "prefill_cache_length": compact_length,
            "cache_capacity": cache_capacity,
            "layers": int(t2s["layers"]),
            "hidden_size": int(t2s["hidden_size"]),
            "samples_per_semantic": int(vits["samples_per_semantic"]),
            "eos_token": 1024,
            "padding_mask_inputs": True,
        },
        "reference": {
            "duration_policy": "exact_samples",
            "pcm_16k_samples": int(reference_conditioning["pcm_16k_samples"]),
            "pcm_32k_samples": int(reference_conditioning["pcm_32k_samples"]),
            "spectrogram_reflect_pad": int(reference_conditioning["spectrogram_reflect_pad"]),
            "ssl_frames": int(reference_ssl["ssl_frames"]),
            "prompt_semantic_length": reference_semantic_length,
            "prompt_phone_capacity": prompt_phone_capacity,
            "prefill_cache_length": reference_compact_length,
            "reference_spectrogram_bins": int(
                reference_conditioning.get("spectrogram_bins", 1025)
            ),
            "reference_spectrogram_frames": int(reference_vits["reference_spectrogram_frames"]),
            "speaker_embedding_size": int(
                reference_conditioning.get("speaker_embedding_size", 20480)
            ),
        },
        "preset": {"prompt_semantic": prompt_semantic},
        "max_text_codepoints": 4000,
        "inter_segment_silence_ms": 150,
    }


def assemble(args: argparse.Namespace) -> tuple[dict, dict]:
    base_manifest = read_base_manifest(args.base_model)
    frontend_profile = base_manifest["frontend_profile"]
    if not args.pipeline_output.name.endswith(QNN_SUFFIX):
        raise ValueError(f"pipeline output must end with {QNN_SUFFIX}")
    if not args.model_output.name.endswith(QNN_SUFFIX):
        raise ValueError(f"model output must end with {QNN_SUFFIX}")
    args.pipeline_output.parent.mkdir(parents=True, exist_ok=True)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gsv-qnn-product-", dir=args.model_output.parent) as raw:
        work = Path(raw)
        wrapped = work / "wrapped"
        vits_stages = partition_graph_specs(
            args.vits_partitions_manifest,
            args.vits_partition_component,
            "vits",
        )
        reference_vits_stages = partition_graph_specs(
            args.vits_reference_partitions_manifest,
            args.vits_reference_partition_component,
            "vits_reference",
        )
        graph_specs = {
            "bert": (args.bert_onnx, args.bert_component),
            "g2pw": (args.g2pw_onnx, args.g2pw_component),
            "t2s_prefill": (args.t2s_prefill_onnx, args.t2s_prefill_component),
            "t2s_step": (args.t2s_step_onnx, args.t2s_step_component),
            "reference_ssl": (args.reference_ssl_onnx, args.reference_ssl_component),
            "reference_prompt_semantic": (
                args.reference_prompt_semantic_onnx,
                args.reference_prompt_semantic_component,
            ),
            "reference_conditioning": (
                args.reference_conditioning_onnx,
                args.reference_conditioning_component,
            ),
            "t2s_reference_prefill": (
                args.t2s_reference_prefill_onnx,
                args.t2s_reference_prefill_component,
            ),
        }
        for stage in [*vits_stages, *reference_vits_stages]:
            graph_specs[stage["name"]] = (stage["source"], stage["component"])
        wrapped_graphs = {
            name: wrap_component(name=name, source=source, component=component, output=wrapped)
            for name, (source, component) in graph_specs.items()
        }
        executor = build_executor(
            conditioning=args.conditioning,
            bert_metadata=args.bert_metadata,
            t2s_metadata=args.t2s_metadata,
            vits_metadata=args.vits_metadata,
            reference_ssl_metadata=args.reference_ssl_metadata,
            reference_prompt_metadata=args.reference_prompt_semantic_metadata,
            reference_conditioning_metadata=args.reference_conditioning_metadata,
            reference_t2s_metadata=args.t2s_reference_prefill_metadata,
            reference_vits_metadata=args.vits_reference_metadata,
            g2pw_sequence_length=args.g2pw_sequence_length,
            vits_partitions=[
                runtime_partition_contract(stage, wrapped_graphs[stage["name"]][0])
                for stage in vits_stages
            ],
            reference_vits_partitions=[
                runtime_partition_contract(stage, wrapped_graphs[stage["name"]][0])
                for stage in reference_vits_stages
            ],
        )
        executor_path = work / "executor.json"
        executor_path.write_text(
            json.dumps(executor, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        frontend_payloads = []
        for name in FRONTEND_FILES:
            source = args.frontend / name
            if not source.is_file():
                raise ValueError(f"frontend asset is missing: {source}")
            frontend_payloads.append((source, f"runtime/frontend/{name}"))
        pipeline_payloads = list(frontend_payloads)
        for name in ("bert", "g2pw", "reference_ssl", "reference_conditioning"):
            onnx_path, bin_path = wrapped_graphs[name]
            pipeline_payloads.extend(
                ((onnx_path, f"runtime/qnn/{name}.onnx"), (bin_path, f"runtime/qnn/{name}.bin"))
            )
        model_payloads = []
        for name in (
            "t2s_prefill",
            "t2s_step",
            "reference_prompt_semantic",
            "t2s_reference_prefill",
            *[stage["name"] for stage in vits_stages],
            *[stage["name"] for stage in reference_vits_stages],
        ):
            onnx_path, bin_path = wrapped_graphs[name]
            model_payloads.extend(
                ((onnx_path, f"runtime/qnn/{name}.onnx"), (bin_path, f"runtime/qnn/{name}.bin"))
            )
        temporary_pipeline = work / f"pipeline{QNN_SUFFIX}"
        temporary_model = work / f"model{QNN_SUFFIX}"
        pipeline_manifest = build_attachment(
            output=temporary_pipeline,
            role="pipeline",
            name=f"V2 Pro Plus {args.target_soc} QNN pipeline attachment",
            version="v2ProPlus",
            frontend_profile=frontend_profile,
            target_soc=args.target_soc,
            components={
                "bert": args.bert_component,
                "g2pw": args.g2pw_component,
                "reference_ssl": args.reference_ssl_component,
                "reference_conditioning": args.reference_conditioning_component,
            },
            payload_mappings=pipeline_payloads,
            executor_descriptor=None,
            base_model=None,
            deployable=True,
        )
        model_manifest = build_attachment(
            output=temporary_model,
            role="model",
            name=args.name,
            version="v2ProPlus",
            frontend_profile=frontend_profile,
            target_soc=args.target_soc,
            components={
                "t2s_prefill": args.t2s_prefill_component,
                "t2s_step": args.t2s_step_component,
                "reference_prompt_semantic": args.reference_prompt_semantic_component,
                "t2s_reference_prefill": args.t2s_reference_prefill_component,
                **{stage["name"]: stage["component"] for stage in vits_stages},
                **{stage["name"]: stage["component"] for stage in reference_vits_stages},
            },
            payload_mappings=model_payloads,
            executor_descriptor=executor_path,
            base_model=args.base_model,
            deployable=True,
        )
        os.replace(temporary_pipeline, args.pipeline_output)
        os.replace(temporary_model, args.model_output)
        audit_product(args.pipeline_output, args.model_output, args.base_model)
        return pipeline_manifest, model_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--target-soc", required=True)
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--frontend", required=True, type=Path)
    parser.add_argument("--conditioning", required=True, type=Path)
    parser.add_argument("--bert-onnx", required=True, type=Path)
    parser.add_argument("--bert-component", required=True, type=Path)
    parser.add_argument("--bert-metadata", required=True, type=Path)
    parser.add_argument("--g2pw-onnx", required=True, type=Path)
    parser.add_argument("--g2pw-component", required=True, type=Path)
    parser.add_argument("--g2pw-sequence-length", type=int, default=130)
    parser.add_argument("--t2s-prefill-onnx", required=True, type=Path)
    parser.add_argument("--t2s-prefill-component", required=True, type=Path)
    parser.add_argument("--t2s-step-onnx", required=True, type=Path)
    parser.add_argument("--t2s-step-component", required=True, type=Path)
    parser.add_argument("--t2s-metadata", required=True, type=Path)
    parser.add_argument("--vits-partitions-manifest", required=True, type=Path)
    parser.add_argument(
        "--vits-partition-component",
        required=True,
        action="append",
        type=Path,
    )
    parser.add_argument("--vits-metadata", required=True, type=Path)
    parser.add_argument("--reference-ssl-onnx", required=True, type=Path)
    parser.add_argument("--reference-ssl-component", required=True, type=Path)
    parser.add_argument("--reference-ssl-metadata", required=True, type=Path)
    parser.add_argument("--reference-prompt-semantic-onnx", required=True, type=Path)
    parser.add_argument("--reference-prompt-semantic-component", required=True, type=Path)
    parser.add_argument("--reference-prompt-semantic-metadata", required=True, type=Path)
    parser.add_argument("--reference-conditioning-onnx", required=True, type=Path)
    parser.add_argument("--reference-conditioning-component", required=True, type=Path)
    parser.add_argument("--reference-conditioning-metadata", required=True, type=Path)
    parser.add_argument("--t2s-reference-prefill-onnx", required=True, type=Path)
    parser.add_argument("--t2s-reference-prefill-component", required=True, type=Path)
    parser.add_argument("--t2s-reference-prefill-metadata", required=True, type=Path)
    parser.add_argument("--vits-reference-partitions-manifest", required=True, type=Path)
    parser.add_argument(
        "--vits-reference-partition-component",
        required=True,
        action="append",
        type=Path,
    )
    parser.add_argument("--vits-reference-metadata", required=True, type=Path)
    parser.add_argument("--pipeline-output", required=True, type=Path)
    parser.add_argument("--model-output", required=True, type=Path)
    args = parser.parse_args()
    pipeline, model = assemble(args)
    print(
        f"Created {args.pipeline_output.resolve()} and {args.model_output.resolve()} "
        f"target={model['target_asic']} profile={model['frontend_profile']} "
        f"deployable={pipeline['deployable'] and model['deployable']}"
    )


if __name__ == "__main__":
    main()
