#!/usr/bin/env python3
"""Audit a final paired V2 Pro Plus QNN product and its bound CPU model."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import onnx

from build_qnn_attachment import QNN_SUFFIX, safe_destination
from build_qnn_htp_context import ANDROID_QNN_RUNTIME_VERSION, TARGETS, TARGET_SOC_FAMILY


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
PIPELINE_GRAPHS = ("bert", "g2pw", "reference_ssl", "reference_conditioning")
MODEL_GRAPHS = (
    "t2s_prefill",
    "t2s_step",
    "reference_prompt_semantic",
    "t2s_reference_prefill",
    *(f"vits_{index:02d}" for index in range(6)),
    *(f"vits_reference_{index:02d}" for index in range(6)),
)
PIPELINE_GRAPH_PATHS = {
    name: f"runtime/qnn/{name}.onnx" for name in PIPELINE_GRAPHS
}
MODEL_GRAPH_PATHS = {name: f"runtime/qnn/{name}.onnx" for name in MODEL_GRAPHS}
EXECUTOR_GRAPH_PATHS = {
    "bert": PIPELINE_GRAPH_PATHS["bert"],
    "t2s_prefill": MODEL_GRAPH_PATHS["t2s_prefill"],
    "t2s_step": MODEL_GRAPH_PATHS["t2s_step"],
    "reference_ssl": PIPELINE_GRAPH_PATHS["reference_ssl"],
    "reference_prompt_semantic": MODEL_GRAPH_PATHS["reference_prompt_semantic"],
    "reference_conditioning": PIPELINE_GRAPH_PATHS["reference_conditioning"],
    "t2s_reference_prefill": MODEL_GRAPH_PATHS["t2s_reference_prefill"],
}


@dataclass(frozen=True)
class AuditedArchive:
    path: Path
    manifest: dict
    names: frozenset[str]
    size: int
    sha256: str


def digest_stream(source) -> str:
    value = hashlib.sha256()
    for block in iter(lambda: source.read(1024 * 1024), b""):
        value.update(block)
    return value.hexdigest()


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return digest_stream(source)


def require_equal(document: dict, name: str, expected: object, label: str) -> None:
    actual = document.get(name)
    if actual != expected:
        raise ValueError(f"{label} has {name}={actual!r}, expected {expected!r}")


def require_positive_int(document: dict, name: str, label: str) -> int:
    value = document.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} has invalid {name}={value!r}")
    return value


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def audit_archive(path: Path) -> AuditedArchive:
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"package does not exist: {path}")
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if not infos or infos[0].filename != "manifest.json":
            raise ValueError(f"{path.name} must place manifest.json first")
        names = [item.filename for item in infos]
        if len(names) != len(set(names)):
            raise ValueError(f"{path.name} contains duplicate ZIP entries")
        if any(item.is_dir() for item in infos):
            raise ValueError(f"{path.name} contains undeclared directory entries")
        try:
            manifest = json.loads(archive.read("manifest.json"))
            declared_items = manifest["files"]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"{path.name} has an invalid manifest: {error}") from error
        if not isinstance(declared_items, list):
            raise ValueError(f"{path.name} manifest files must be an array")
        declared: dict[str, dict] = {}
        for item in declared_items:
            if not isinstance(item, dict):
                raise ValueError(f"{path.name} has a non-object file declaration")
            raw_member = item.get("path")
            if not isinstance(raw_member, str):
                raise ValueError(f"{path.name} has a non-string payload path")
            member = safe_destination(raw_member)
            if member == "manifest.json" or member in declared:
                raise ValueError(f"{path.name} has a duplicate or invalid declaration: {member}")
            declared[member] = item
        actual = set(names[1:])
        if actual != set(declared):
            missing = sorted(set(declared) - actual)
            extra = sorted(actual - set(declared))
            raise ValueError(f"{path.name} payload set differs: missing={missing} extra={extra}")
        for info in infos[1:]:
            item = declared[info.filename]
            expected_size = item.get("size")
            expected_hash = item.get("sha256")
            if (
                not isinstance(expected_size, int)
                or isinstance(expected_size, bool)
                or expected_size < 0
            ):
                raise ValueError(f"{path.name} has invalid size for {info.filename}")
            if info.file_size != expected_size:
                raise ValueError(f"{path.name} payload size mismatch: {info.filename}")
            if not is_sha256(expected_hash):
                raise ValueError(f"{path.name} has invalid SHA-256 for {info.filename}")
            with archive.open(info) as source:
                if digest_stream(source) != expected_hash:
                    raise ValueError(f"{path.name} payload SHA-256 mismatch: {info.filename}")
    return AuditedArchive(path, manifest, frozenset(names), path.stat().st_size, digest(path))


def audit_ep_contexts(
    package: AuditedArchive,
    expected_paths: set[str],
) -> dict[str, onnx.ModelProto]:
    actual_paths = {value for value in package.names if value.endswith(".onnx")}
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)
        extra = sorted(actual_paths - expected_paths)
        raise ValueError(
            f"{package.path.name} ONNX graph set differs: missing={missing} extra={extra}"
        )
    models: dict[str, onnx.ModelProto] = {}
    with zipfile.ZipFile(package.path) as archive:
        for name in sorted(actual_paths):
            model = onnx.load_model_from_string(archive.read(name))
            nodes = list(model.graph.node)
            if len(nodes) != 1 or nodes[0].domain != "com.microsoft" or nodes[0].op_type != "EPContext":
                raise ValueError(f"{package.path.name} {name} is not one prepared EPContext graph")
            if list(nodes[0].input) != [value.name for value in model.graph.input]:
                raise ValueError(f"{package.path.name} {name} EPContext inputs do not match its graph")
            if list(nodes[0].output) != [value.name for value in model.graph.output]:
                raise ValueError(f"{package.path.name} {name} EPContext outputs do not match its graph")
            attributes = {
                value.name: onnx.helper.get_attribute_value(value) for value in nodes[0].attribute
            }
            context_name = attributes.get("ep_cache_context")
            if isinstance(context_name, bytes):
                context_name = context_name.decode("utf-8")
            expected_bin = str(PurePosixPath(name).with_suffix(".bin"))
            if context_name != PurePosixPath(expected_bin).name or expected_bin not in package.names:
                raise ValueError(f"{package.path.name} {name} has no matching external context")
            if attributes.get("embed_mode") != 0 or attributes.get("main_context") != 1:
                raise ValueError(f"{package.path.name} {name} has an unsupported EPContext mode")
            if attributes.get("source") not in (b"QNNExecutionProvider", "QNNExecutionProvider"):
                raise ValueError(f"{package.path.name} {name} targets another execution provider")
            models[name] = model
    return models


def read_json_member(package: AuditedArchive, name: str) -> dict:
    if name not in package.names:
        raise ValueError(f"{package.path.name} is missing {name}")
    with zipfile.ZipFile(package.path) as archive:
        try:
            value = json.loads(archive.read(name))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"{package.path.name} has invalid JSON in {name}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{package.path.name} {name} must contain a JSON object")
    return value


def audit_attachment_components(
    document: dict,
    expected_components: set[str],
    label: str,
) -> None:
    components = document.get("components")
    if not isinstance(components, dict) or set(components) != expected_components:
        raise ValueError(f"{label} attachment has an unexpected component set")
    for name, component in components.items():
        if not isinstance(component, dict):
            raise ValueError(f"{label} attachment component {name} must be an object")
        if not is_sha256(component.get("source_onnx_sha256")):
            raise ValueError(f"{label} attachment component {name} has an invalid source hash")
        static_inputs = component.get("static_inputs")
        if not isinstance(static_inputs, dict) or not static_inputs:
            raise ValueError(f"{label} attachment component {name} has invalid static inputs")
        for input_name, shape in static_inputs.items():
            if (
                not isinstance(input_name, str)
                or not input_name
                or not isinstance(shape, list)
                or not shape
                or any(
                    not isinstance(value, int) or isinstance(value, bool) or value <= 0
                    for value in shape
                )
            ):
                raise ValueError(
                    f"{label} attachment component {name} has an invalid static input"
                )


def tensor_contract(value: onnx.ValueInfoProto) -> dict[str, object]:
    tensor = value.type.tensor_type
    if not tensor.elem_type or not tensor.HasField("shape"):
        raise ValueError(f"tensor {value.name!r} has no static tensor type")
    shape: list[int] = []
    for dimension in tensor.shape.dim:
        if not dimension.HasField("dim_value") or dimension.dim_value <= 0:
            raise ValueError(f"tensor {value.name!r} has a dynamic or invalid shape")
        shape.append(dimension.dim_value)
    return {
        "name": value.name,
        "data_type": int(tensor.elem_type),
        "data_type_name": onnx.TensorProto.DataType.Name(tensor.elem_type),
        "shape": shape,
    }


def audit_partition_contract(
    stage: dict,
    model: onnx.ModelProto,
    label: str,
) -> tuple[set[str], set[str]]:
    logical_names: dict[str, set[str]] = {}
    for direction, actual_values in (
        ("inputs", list(model.graph.input)),
        ("outputs", list(model.graph.output)),
    ):
        declared_values = stage.get(direction)
        if not isinstance(declared_values, list) or len(declared_values) != len(actual_values):
            raise ValueError(f"{label} {direction} do not match its EPContext graph")
        logical: set[str] = set()
        for index, (declared, actual) in enumerate(zip(declared_values, actual_values)):
            if not isinstance(declared, dict):
                raise ValueError(f"{label} {direction}[{index}] must be an object")
            expected = tensor_contract(actual)
            for name, value in expected.items():
                require_equal(declared, name, value, f"{label} {direction}[{index}]")
            logical_name = declared.get("logical_name")
            if not isinstance(logical_name, str) or not logical_name or logical_name in logical:
                raise ValueError(f"{label} has invalid or duplicate logical tensor names")
            logical.add(logical_name)
        logical_names[direction] = logical
    return logical_names["inputs"], logical_names["outputs"]


def audit_product(pipeline_path: Path, model_path: Path, base_model_path: Path) -> dict:
    if not pipeline_path.name.endswith(QNN_SUFFIX) or not model_path.name.endswith(QNN_SUFFIX):
        raise ValueError(f"QNN product files must end with {QNN_SUFFIX}")
    pipeline = audit_archive(pipeline_path)
    model = audit_archive(model_path)
    base_model = audit_archive(base_model_path)
    pipeline_manifest = pipeline.manifest
    model_manifest = model.manifest
    base_manifest = base_model.manifest

    for name, expected in {
        "format": "gsvm-deploy",
        "format_version": 1,
        "artifact_role": "model",
        "model_version": "v2ProPlus",
        "sample_rate": 32000,
        "entrypoint": "synthesize_utf8_to_pcm16",
        "api_version": 1,
        "deployable": True,
        "reference_input_version": 1,
    }.items():
        require_equal(base_manifest, name, expected, "base model")
    if base_manifest.get("executor") not in ("torchscript-cpu-single", "torchscript-cpu-staged"):
        raise ValueError("base model is not a CPU correctness package")
    frontend_profile = base_manifest.get("frontend_profile")
    if not isinstance(frontend_profile, str) or not frontend_profile:
        raise ValueError("base model is missing frontend_profile")

    target_soc = model_manifest.get("target_soc")
    if target_soc not in TARGETS:
        raise ValueError(f"model package has unsupported target_soc={target_soc!r}")
    target = TARGETS[target_soc]
    qairt_version = model_manifest.get("qairt_version")
    if not isinstance(qairt_version, str) or not qairt_version:
        raise ValueError("model package is missing qairt_version")
    qairt_parts = qairt_version.split(".")
    if (
        len(qairt_parts) != 4
        or any(not value.isdigit() for value in qairt_parts)
        or ".".join(qairt_parts[:3]) != ANDROID_QNN_RUNTIME_VERSION
    ):
        raise ValueError(
            f"model package QAIRT version {qairt_version!r} does not match the Android runtime"
        )
    attachment_for = f"gsvm:v2ProPlus:{frontend_profile}:api1"
    bundle_id = f"{attachment_for}:qnn-htp:{target.asic}:qairt-{qairt_version}"
    common = {
        "format": "gsvm-deploy",
        "format_version": 1,
        "model_version": "v2ProPlus",
        "sample_rate": 32000,
        "executor": "qnn-htp",
        "entrypoint": "synthesize_utf8_to_pcm16",
        "api_version": 1,
        "deployable": True,
        "frontend_profile": frontend_profile,
        "target_soc": target_soc,
        "target_soc_family": TARGET_SOC_FAMILY,
        "target_asic": target.asic,
        "target_soc_model": target.soc_model,
        "supported_target_socs": [target_soc],
        "htp_arch": target.htp_arch,
        "qairt_version": qairt_version,
        "qnn_runtime_version": ANDROID_QNN_RUNTIME_VERSION,
        "precision": "fp16",
        "quantization": "none",
        "cpu_neural_fallback": False,
        "attachment_for": attachment_for,
        "bundle_id": bundle_id,
    }
    for label, document in (("pipeline", pipeline_manifest), ("model", model_manifest)):
        for name, expected in common.items():
            require_equal(document, name, expected, label)
    require_equal(pipeline_manifest, "artifact_role", "qnn-pipeline-attachment", "pipeline")
    require_equal(pipeline_manifest, "requires_role", "qnn-model-attachment", "pipeline")
    require_equal(pipeline_manifest, "backend_artifact", "runtime/qnn/attachment.json", "pipeline")
    require_equal(model_manifest, "artifact_role", "qnn-model-attachment", "model")
    require_equal(model_manifest, "requires_role", "qnn-pipeline-attachment", "model")
    require_equal(model_manifest, "backend_artifact", "runtime/qnn/executor.json", "model")
    require_equal(model_manifest, "base_model_sha256", base_model.sha256, "model")
    require_equal(model_manifest, "runtime_options_version", 0, "model")
    require_equal(model_manifest, "reference_input_version", 1, "model")
    if "base_model_sha256" in pipeline_manifest:
        raise ValueError("pipeline attachment must not bind one voice model")
    for name in ("runtime_options_version", "reference_input_version", "reference_input"):
        if name in pipeline_manifest:
            raise ValueError(f"pipeline attachment must not own voice field {name}")

    expected_pipeline_names = {
        "manifest.json",
        "runtime/qnn/attachment.json",
        *(f"runtime/frontend/{name}" for name in FRONTEND_FILES),
        *PIPELINE_GRAPH_PATHS.values(),
        *(str(PurePosixPath(path).with_suffix(".bin")) for path in PIPELINE_GRAPH_PATHS.values()),
    }
    expected_model_names = {
        "manifest.json",
        "runtime/qnn/attachment.json",
        "runtime/qnn/executor.json",
        *MODEL_GRAPH_PATHS.values(),
        *(str(PurePosixPath(path).with_suffix(".bin")) for path in MODEL_GRAPH_PATHS.values()),
    }
    for label, package, expected_names in (
        ("pipeline", pipeline, expected_pipeline_names),
        ("model", model, expected_model_names),
    ):
        if package.names != expected_names:
            missing = sorted(expected_names - package.names)
            extra = sorted(package.names - expected_names)
            raise ValueError(f"{label} package file set differs: missing={missing} extra={extra}")

    reference = model_manifest.get("reference_input")
    if not isinstance(reference, dict):
        raise ValueError("model package is missing reference_input")
    require_equal(reference, "preset_when_omitted", True, "reference input")
    require_equal(reference, "duration_policy", "exact_samples", "reference input")
    pcm = reference.get("pcm")
    expected_pcm = [
        (16000, 1, "float32", 80000),
        (32000, 1, "float32", 160000),
    ]
    if (
        not isinstance(pcm, list)
        or any(not isinstance(value, dict) for value in pcm)
        or [
            (
                value.get("sample_rate"),
                value.get("channels"),
                value.get("dtype"),
                value.get("samples"),
            )
            for value in pcm
        ]
        != expected_pcm
    ):
        raise ValueError("model package has unexpected runtime reference PCM capacities")

    pipeline_attachment = read_json_member(pipeline, "runtime/qnn/attachment.json")
    model_attachment = read_json_member(model, "runtime/qnn/attachment.json")
    for label, document, role, expected_components in (
        ("pipeline", pipeline_attachment, "pipeline", set(PIPELINE_GRAPHS)),
        ("model", model_attachment, "model", set(MODEL_GRAPHS)),
    ):
        require_equal(document, "format", "gsv-qnn-attachment", f"{label} attachment")
        require_equal(document, "format_version", 1, f"{label} attachment")
        require_equal(document, "role", role, f"{label} attachment")
        audit_attachment_components(document, expected_components, label)

    pipeline_models = audit_ep_contexts(pipeline, set(PIPELINE_GRAPH_PATHS.values()))
    model_models = audit_ep_contexts(model, set(MODEL_GRAPH_PATHS.values()))

    executor = read_json_member(model, "runtime/qnn/executor.json")
    executor_expected = {
        "format": "gsv-qnn-executor",
        "format_version": 1,
        "operation": "synthesize_utf8_to_pcm16",
        "runtime_abi_version": 1,
        "complete": True,
        "utf8_text_input": True,
        "pcm16_output": True,
        "cpu_neural_fallback": False,
        "runtime_options_version": 0,
        "reference_input_version": 1,
        "sample_rate": 32000,
        "languages": ["auto", "zh", "en"],
        "engine": "gpt-sovits-v2pp-qnn-buckets",
        "engine_version": 2,
    }
    for name, expected in executor_expected.items():
        require_equal(executor, name, expected, "executor")
    executor_reference = executor.get("reference")
    if not isinstance(executor_reference, dict):
        raise ValueError("executor is missing runtime reference configuration")
    for name, expected in (
        ("duration_policy", "exact_samples"),
        ("pcm_16k_samples", 80000),
        ("pcm_32k_samples", 160000),
    ):
        require_equal(executor_reference, name, expected, "executor reference")
    for name in (
        "spectrogram_reflect_pad",
        "ssl_frames",
        "prompt_semantic_length",
        "prompt_phone_capacity",
        "prefill_cache_length",
        "reference_spectrogram_bins",
        "reference_spectrogram_frames",
        "speaker_embedding_size",
    ):
        require_positive_int(executor_reference, name, "executor reference")

    shapes = executor.get("shapes")
    if not isinstance(shapes, dict):
        raise ValueError("executor is missing its prepared shape contract")
    shape_values = {
        name: require_positive_int(shapes, name, "executor shapes")
        for name in (
            "token_capacity",
            "phone_capacity",
            "semantic_capacity",
            "preset_prompt_phone_length",
            "prefill_cache_length",
            "cache_capacity",
            "layers",
            "hidden_size",
            "samples_per_semantic",
            "eos_token",
        )
    }
    require_equal(shapes, "padding_mask_inputs", True, "executor shapes")
    preset = executor.get("preset")
    prompt_semantic = preset.get("prompt_semantic") if isinstance(preset, dict) else None
    if (
        not isinstance(prompt_semantic, list)
        or not prompt_semantic
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            or value > shape_values["eos_token"]
            for value in prompt_semantic
        )
    ):
        raise ValueError("executor has an invalid preset prompt semantic sequence")
    if (
        shape_values["preset_prompt_phone_length"]
        + shape_values["phone_capacity"]
        + len(prompt_semantic)
        != shape_values["prefill_cache_length"]
    ):
        raise ValueError("executor preset prefill layout does not match its cache length")
    if (
        shape_values["prefill_cache_length"] + shape_values["semantic_capacity"]
        > shape_values["cache_capacity"]
    ):
        raise ValueError("executor preset path exceeds its T2S cache capacity")
    if executor_reference["prompt_phone_capacity"] != shape_values["phone_capacity"]:
        raise ValueError("executor reference phone capacity differs from the synthesis capacity")
    if (
        executor_reference["prefill_cache_length"] + shape_values["semantic_capacity"]
        > shape_values["cache_capacity"]
    ):
        raise ValueError("executor runtime-reference path exceeds its T2S cache capacity")
    max_text_codepoints = require_positive_int(executor, "max_text_codepoints", "executor")
    if max_text_codepoints > 1_000_000:
        raise ValueError("executor max_text_codepoints is unreasonably large")
    silence = executor.get("inter_segment_silence_ms")
    if not isinstance(silence, int) or isinstance(silence, bool) or not 0 <= silence <= 2000:
        raise ValueError("executor has invalid inter_segment_silence_ms")

    graphs = executor.get("graphs")
    if not isinstance(graphs, dict):
        raise ValueError("executor graphs must be an object")
    if set(graphs) != set(EXECUTOR_GRAPH_PATHS) | {"vits", "vits_reference"}:
        raise ValueError("executor has an unexpected graph set")
    graph_count = 0
    for name, expected_path in EXECUTOR_GRAPH_PATHS.items():
        require_equal(graphs, name, expected_path, "executor graphs")
        owner = pipeline if expected_path in pipeline_models else model
        if expected_path not in owner.names:
            raise ValueError(f"executor graph is missing from its attachment: {expected_path}")
        graph_count += 1
    for name, expected_prefix in (("vits", "vits"), ("vits_reference", "vits_reference")):
        stages = graphs.get(name)
        if not isinstance(stages, list) or len(stages) != 6:
            raise ValueError(f"executor has an invalid graph list {name}")
        available: set[str] = set()
        final_outputs: set[str] = set()
        for index, stage in enumerate(stages):
            if not isinstance(stage, dict):
                raise ValueError(f"executor {name} stage is not an object")
            stage_name = f"{expected_prefix}_{index:02d}"
            graph_path = MODEL_GRAPH_PATHS[stage_name]
            require_equal(stage, "name", stage_name, f"executor {name}")
            require_equal(stage, "path", graph_path, f"executor {name} {stage_name}")
            inputs, outputs = audit_partition_contract(
                stage,
                model_models[graph_path],
                f"executor {name} {stage_name}",
            )
            if index and not inputs.issubset(available):
                raise ValueError(
                    f"executor {name} {stage_name} consumes unavailable logical tensors: "
                    f"{sorted(inputs - available)}"
                )
            if outputs.intersection(available):
                raise ValueError(
                    f"executor {name} {stage_name} overwrites logical tensors: "
                    f"{sorted(outputs.intersection(available))}"
                )
            available.update(outputs)
            final_outputs = outputs
            graph_count += 1
        if "audio" not in final_outputs:
            raise ValueError(f"executor {name} does not produce the required audio tensor")
    frontend = executor.get("frontend")
    if not isinstance(frontend, dict):
        raise ValueError("executor is missing frontend configuration")
    require_equal(frontend, "root", "runtime/frontend", "executor frontend")
    require_equal(
        frontend,
        "g2pw_sequence_length",
        shape_values["token_capacity"],
        "executor frontend",
    )
    g2pw_path = frontend.get("g2pw_model")
    if g2pw_path != PIPELINE_GRAPH_PATHS["g2pw"]:
        raise ValueError("executor G2PW graph is missing from the paired product")
    for name in FRONTEND_FILES:
        if f"runtime/frontend/{name}" not in pipeline.names:
            raise ValueError(f"pipeline package is missing frontend asset {name}")

    pipeline_contexts = len(pipeline_models)
    model_contexts = len(model_models)
    return {
        "format": "gsv-v2pp-qnn-product-audit",
        "format_version": 1,
        "verified": True,
        "pipeline": {
            "path": str(pipeline.path),
            "size": pipeline.size,
            "sha256": pipeline.sha256,
            "files": len(pipeline.manifest["files"]),
            "ep_contexts": pipeline_contexts,
        },
        "model": {
            "path": str(model.path),
            "size": model.size,
            "sha256": model.sha256,
            "files": len(model.manifest["files"]),
            "ep_contexts": model_contexts,
            "base_model_sha256": base_model.sha256,
        },
        "bundle_id": bundle_id,
        "target_soc": target_soc,
        "target_asic": target.asic,
        "target_soc_model": target.soc_model,
        "htp_arch": target.htp_arch,
        "qairt_version": qairt_version,
        "qnn_runtime_version": ANDROID_QNN_RUNTIME_VERSION,
        "precision": "fp16",
        "quantization": "none",
        "graph_references": graph_count,
        "reference_pcm_samples": {"16000": 80000, "32000": 160000},
    }


def atomic_json(path: Path, document: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f"{path.name}.pending")
    pending.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(pending, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_product(args.pipeline, args.model, args.base_model)
    if args.output is not None:
        atomic_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
