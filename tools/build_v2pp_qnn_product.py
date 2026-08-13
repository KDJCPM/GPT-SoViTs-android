#!/usr/bin/env python3
"""Build one resumable V2 Pro Plus QNN product for an exact Snapdragon target.

This is the product conversion entrypoint. It validates the final ONNX files, compiles every
required HTP context sequentially, verifies reusable components, and assembles the paired pipeline
and voice attachments. Android never needs to know how these component graphs were prepared.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import onnx

from build_qnn_htp_context import (
    TARGETS,
    sdk_version,
    sha256,
    validate_context_metadata,
    validate_preserved_io_layout,
)
from normalize_qnn_g2pw_descriptor import normalize_descriptor
from partition_onnx_contiguous import read_partition_manifest


TOOLS = Path(__file__).resolve().parent
REFERENCE_SPLIT_GROUPS = (
    (2, r"/dec/resblocks\.(3|4|5)/convs1\.[012]/Conv"),
    (4, r"/dec/resblocks\.(6|7|8)/convs1\.[012]/Conv"),
    (8, r"/dec/resblocks\.(9|10|11)/convs1\.[012]/Conv"),
    (16, r"/dec/resblocks\.(12|13|14)/convs1\.[012]/Conv"),
    (16, r"/dec/conv_post/Conv"),
)
# V75 and V81 have the same 8 MiB HTP VTCM limit for the long decoder stages. Keep the
# conversion-time rewrite target-specific; the existing SM8750 package remains byte-for-byte
# compatible with its already validated V79 graph set.
PRESET_VTCM_SPLIT_TARGETS = {
    "snapdragon_8_gen_3",
    "snapdragon_8_elite_gen_5",
}
VITS_BOUNDARIES = (
    "/dec/conv_pre/Conv",
    "/dec/ups.1/ConvTranspose",
    "/dec/ups.2/ConvTranspose",
    "/dec/ups.3/ConvTranspose",
    "/dec/ups.4/ConvTranspose",
)
VITS_FIXED_DIMENSIONS = {"unk__671": 1}
G2PW_ALIGNMENT_FORMAT = "gsv-qnn-g2pw-alignment"


@dataclass(frozen=True)
class GraphSpec:
    name: str
    source: Path
    component: Path
    optimization_level: int
    input_dimensions: tuple[str, ...] = ()
    preserve_io_layout: bool = False


def run(command: list[object], *, malloc_arena_max: int | None = None) -> None:
    rendered = [str(value) for value in command]
    environment = dict(os.environ)
    if malloc_arena_max is not None:
        environment["MALLOC_ARENA_MAX"] = str(malloc_arena_max)
    print("Running:", " ".join(rendered), flush=True)
    subprocess.run(rendered, check=True, env=environment)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f"{path.name}.pending")
    pending.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(pending, path)


def absolute_executable(path: Path) -> Path:
    # Resolving a venv's python symlink escapes the environment and drops its site-packages.
    return Path(os.path.abspath(path))


def g2pw_alignment_descriptor(source: Path, output: Path, summary: dict) -> dict:
    return {
        "format": G2PW_ALIGNMENT_FORMAT,
        "format_version": 1,
        "source_sha256": sha256(source),
        "output_sha256": sha256(output),
        "normalization": summary,
    }


def reusable_g2pw_alignment(source: Path, output: Path, descriptor: Path) -> tuple[bool, str]:
    if not output.is_file() or not descriptor.is_file():
        return False, "aligned graph or provenance descriptor is missing"
    try:
        document = read_json(descriptor)
        if document.get("format") != G2PW_ALIGNMENT_FORMAT:
            return False, "G2PW alignment provenance format is incorrect"
        if document.get("format_version") != 1:
            return False, "G2PW alignment provenance version is unsupported"
        if document.get("source_sha256") != sha256(source):
            return False, "G2PW alignment source hash does not match"
        if document.get("output_sha256") != sha256(output):
            return False, "G2PW aligned graph hash does not match"
        normalization = document.get("normalization", {})
        if int(normalization.get("padded_rows", 0)) < int(
            normalization.get("used_rows", 0)
        ):
            return False, "G2PW alignment provenance has invalid row counts"
        return True, "verified"
    except (OSError, TypeError, ValueError) as error:
        return False, str(error)


def ensure_g2pw_alignment(args: argparse.Namespace) -> Path:
    source = args.g2pw_onnx
    if not source.is_file():
        raise ValueError(f"G2PW ONNX source is missing: {source}")
    source_hash = sha256(source)
    shared_root = args.components_root.parent / "qnn-shared-inputs"
    output = shared_root / f"{source.stem}-{source_hash[:16]}-qnn-aligned.onnx"
    descriptor = output.with_name(f"{output.name}.build.json")
    reusable, reason = reusable_g2pw_alignment(source, output, descriptor)
    if reusable:
        print(f"Reusing validated G2PW alignment {output}", flush=True)
        return output

    model = onnx.load(str(source), load_external_data=True)
    summary = normalize_descriptor(model)
    if int(summary["added_rows"]) == 0:
        print(f"G2PW graph is already QAIRT-aligned: {source}", flush=True)
        return source
    if output.exists() or descriptor.exists():
        raise ValueError(f"existing G2PW alignment cannot be reused: {reason}")
    shared_root.mkdir(parents=True, exist_ok=True)
    pending = output.with_name(f"{output.name}.pending")
    try:
        onnx.save(model, str(pending))
        os.replace(pending, output)
    finally:
        if pending.exists():
            pending.unlink()
    atomic_json(descriptor, g2pw_alignment_descriptor(source, output, summary))
    print(f"Prepared QAIRT-aligned G2PW graph {output}: {summary}", flush=True)
    return output


def verify_declared_files(root: Path, manifest: dict) -> tuple[bool, str]:
    try:
        canonical_root = root.resolve(strict=True)
        for item in manifest["files"]:
            path = (canonical_root / item["path"]).resolve(strict=True)
            path.relative_to(canonical_root)
            if not path.is_file() or path.stat().st_size != int(item["size"]):
                return False, f"payload size mismatch: {item['path']}"
            if sha256(path) != item["sha256"]:
                return False, f"payload SHA-256 mismatch: {item['path']}"
    except (KeyError, OSError, ValueError, TypeError) as error:
        return False, f"invalid component payload manifest: {error}"
    return True, "verified"


def reusable_component(
    spec: GraphSpec,
    *,
    target_soc: str,
    qairt_version: str,
) -> tuple[bool, str]:
    manifest_path = spec.component / "manifest.json"
    info_path = spec.component / "context-info.json"
    if not manifest_path.is_file() or not info_path.is_file():
        return False, "component metadata is missing"
    try:
        manifest = read_json(manifest_path)
        expected_target = TARGETS[target_soc]
        required = {
            "format": "gsv-qnn-compiled-component",
            "target_soc": target_soc,
            "target_asic": expected_target.asic,
            "target_soc_model": expected_target.soc_model,
            "htp_arch": expected_target.htp_arch,
            "qairt_version": qairt_version,
            "qnn_runtime_version": "2.48.0",
            "precision": "fp16",
            "quantization": "none",
            "graph_io_dtype": "float16",
            "preserve_io_layout": spec.preserve_io_layout,
            "htp_graph_optimization_level": spec.optimization_level,
            "source_onnx_sha256": sha256(spec.source),
        }
        for name, expected in required.items():
            actual = manifest.get(name, False) if name == "preserve_io_layout" else manifest.get(name)
            if actual != expected:
                return False, f"{name}={actual!r}, expected {expected!r}"
        validate_context_metadata(read_json(info_path), expected_target, qairt_version)
        if spec.preserve_io_layout:
            validate_preserved_io_layout(spec.source, info_path)
        return verify_declared_files(spec.component, manifest)
    except (KeyError, OSError, ValueError, RuntimeError, TypeError) as error:
        return False, str(error)


def split_descriptor_path(output: Path) -> Path:
    return output.with_name(f"{output.name}.build.json")


def split_document(
    source: Path,
    output: Path,
    groups: tuple[tuple[int, str], ...] = REFERENCE_SPLIT_GROUPS,
) -> dict:
    return {
        "format": "gsv-qnn-vits-vtcm-split",
        "format_version": 1,
        "source": str(source.resolve()),
        "source_sha256": sha256(source),
        "output": str(output.resolve()),
        "output_sha256": sha256(output),
        "groups": [
            {"chunks": chunks, "node_regex": expression}
            for chunks, expression in groups
        ],
    }


def reusable_split(
    source: Path,
    output: Path,
    groups: tuple[tuple[int, str], ...] = REFERENCE_SPLIT_GROUPS,
) -> tuple[bool, str]:
    descriptor_path = split_descriptor_path(output)
    if not output.is_file() or not descriptor_path.is_file():
        return False, "split graph or provenance descriptor is missing"
    try:
        actual = read_json(descriptor_path)
        expected = split_document(source, output, groups)
        if actual != expected:
            return False, "split graph provenance does not match its source and rules"
        return True, "verified"
    except (OSError, ValueError, TypeError) as error:
        return False, str(error)


def ensure_reference_vits_split(args: argparse.Namespace) -> Path:
    source = args.reference_root / "vits_reference_pc128_sc512.onnx"
    output = args.reference_vits_onnx
    if not source.is_file():
        raise ValueError(f"reference VITS source is missing: {source}")
    reusable, reason = reusable_split(source, output, REFERENCE_SPLIT_GROUPS)
    if reusable:
        print(f"Reusing validated split graph {output}", flush=True)
        return output
    if output.exists() or split_descriptor_path(output).exists():
        raise ValueError(f"existing reference VITS split cannot be reused: {reason}")
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_name(f"{output.name}.pending")
    command: list[object] = [
        args.qairt_python,
        TOOLS / "split_qnn_vits_vtcm_conv.py",
        "--source",
        source,
        "--output",
        pending,
    ]
    for chunks, expression in REFERENCE_SPLIT_GROUPS:
        command.extend(["--split-group", f"{chunks}:{expression}"])
    try:
        run(command)
        os.replace(pending, output)
    finally:
        if pending.exists():
            pending.unlink()
    atomic_json(
        split_descriptor_path(output),
        split_document(source, output, REFERENCE_SPLIT_GROUPS),
    )
    print(f"Prepared reference VITS VTCM graph {output}", flush=True)
    return output


def ensure_preset_vits_split(args: argparse.Namespace) -> Path:
    """Prepare a target-specific preset VITS graph when its decoder exceeds HTP VTCM."""
    source = args.vits_onnx
    if args.target_soc not in PRESET_VTCM_SPLIT_TARGETS:
        return source
    target = TARGETS[args.target_soc]
    output_root = args.components_root.parent / "qnn-target-graphs" / args.target_soc
    output = output_root / (
        f"{source.stem}_htp_{target.htp_arch.lower()}_vtcm{source.suffix}"
    )
    groups = REFERENCE_SPLIT_GROUPS
    reusable, reason = reusable_split(source, output, groups)
    if reusable:
        print(f"Reusing validated preset VITS VTCM graph {output}", flush=True)
        return output
    if output.exists() or split_descriptor_path(output).exists():
        raise ValueError(f"existing preset VITS split cannot be reused: {reason}")
    output_root.mkdir(parents=True, exist_ok=True)
    pending = output.with_name(f"{output.name}.pending")
    command: list[object] = [
        args.qairt_python,
        TOOLS / "split_qnn_vits_vtcm_conv.py",
        "--source",
        source,
        "--output",
        pending,
    ]
    for chunks, expression in groups:
        command.extend(["--split-group", f"{chunks}:{expression}"])
    try:
        run(command)
        os.replace(pending, output)
    finally:
        if pending.exists():
            pending.unlink()
    atomic_json(split_descriptor_path(output), split_document(source, output, groups))
    print(f"Prepared preset VITS VTCM graph {output}", flush=True)
    return output


def reusable_partitions(source: Path, root: Path) -> tuple[bool, str]:
    manifest = root / "partitions.json"
    if not manifest.is_file():
        return False, "partition manifest is missing"
    try:
        document, _ = read_partition_manifest(manifest)
        if document["source"]["sha256"] != sha256(source):
            return False, "partition source hash does not match"
        if document.get("boundaries_before") != list(VITS_BOUNDARIES):
            return False, "partition boundaries do not match the product rules"
        if document.get("fixed_dimensions") != VITS_FIXED_DIMENSIONS:
            return False, "partition static dimension bindings do not match"
        return True, "verified"
    except (KeyError, OSError, TypeError, ValueError) as error:
        return False, str(error)


def ensure_vits_partitions(source: Path, root: Path, prefix: str, args: argparse.Namespace) -> Path:
    reusable, reason = reusable_partitions(source, root)
    if reusable:
        print(f"Reusing validated contiguous partitions {root}", flush=True)
        return root / "partitions.json"
    if root.exists():
        raise ValueError(f"existing VITS partitions cannot be reused: {reason}")
    command: list[object] = [
        args.qairt_python,
        TOOLS / "partition_onnx_contiguous.py",
        "--source",
        source,
        "--output-dir",
        root,
        "--prefix",
        prefix,
    ]
    for name, value in VITS_FIXED_DIMENSIONS.items():
        command.extend(["--dim-param", f"{name}={value}"])
    for boundary in VITS_BOUNDARIES:
        command.extend(["--boundary-before", boundary])
    run(command)
    reusable, reason = reusable_partitions(source, root)
    if not reusable:
        raise RuntimeError(f"new VITS partitions failed verification: {reason}")
    return root / "partitions.json"


def validation_covers_partitions(recorded: dict, manifest: Path) -> tuple[bool, str]:
    try:
        _, paths = read_partition_manifest(manifest)
        if recorded.get("manifest", {}).get("sha256") != sha256(manifest):
            return False, "partition validation manifest hash mismatch"
        actual_parts = recorded.get("parts", [])
        if len(actual_parts) != len(paths):
            return False, "partition validation part count mismatch"
        for index, path in enumerate(paths):
            if actual_parts[index].get("sha256") != sha256(path):
                return False, f"partition validation hash mismatch at stage {index}"
        return True, "verified"
    except (AttributeError, KeyError, OSError, TypeError, ValueError) as error:
        return False, str(error)


def expected_validation_graphs(args: argparse.Namespace, reference_vits: Path) -> dict[str, Path]:
    return {
        "reference_ssl": args.reference_root / "reference_ssl_5s.onnx",
        "reference_prompt_semantic": args.reference_root / "reference_prompt_semantic_5s.onnx",
        "reference_conditioning": args.reference_root / "reference_conditioning_5s.onnx",
        "t2s_reference_prefill": args.reference_root / "t2s_reference_prefill_pc128.onnx",
        "vits_reference": reference_vits,
    }


def reusable_validation(
    report: Path,
    graphs: dict[str, Path],
    partitions: Path | None = None,
) -> tuple[bool, str]:
    if not report.is_file():
        return False, "validation report is missing"
    try:
        document = read_json(report)
        if document.get("format") != "gsv-v2pp-qnn-reference-onnx-validation":
            return False, "validation format is incorrect"
        if float(document.get("maximum_absolute_error", float("inf"))) > float(
            document.get("absolute_tolerance", 0.0)
        ):
            return False, "validation error exceeds tolerance"
        recorded = document["graphs"]
        for name, path in graphs.items():
            if recorded.get(name, {}).get("sha256") != sha256(path):
                return False, f"validation graph hash mismatch: {name}"
        if partitions is not None:
            reusable, reason = validation_covers_partitions(
                recorded.get("vits_reference_partitions", {}),
                partitions,
            )
            if not reusable:
                return False, reason
            if "vits_reference_partition_equivalence" not in document:
                return False, "reference partition equivalence measurement is missing"
        return True, "verified"
    except (KeyError, OSError, TypeError, ValueError) as error:
        return False, str(error)


def ensure_reference_validation(
    args: argparse.Namespace,
    reference_vits: Path,
    partitions: Path,
) -> Path:
    graphs = expected_validation_graphs(args, reference_vits)
    missing = [str(path) for path in graphs.values() if not path.is_file()]
    if missing:
        raise ValueError(f"reference graphs are missing: {', '.join(missing)}")
    output = args.reference_validation
    reusable, _ = reusable_validation(output, graphs, partitions)
    if reusable:
        print(f"Reusing reference FP32 validation {output}", flush=True)
        return output
    pending = output.with_name(f"{output.name}.pending")
    if pending.exists():
        raise ValueError(f"stale validation output exists: {pending}")
    run(
        [
            args.validation_python,
            TOOLS / "validate_v2pp_qnn_reference.py",
            "--root",
            args.reference_root,
            "--vits-onnx",
            reference_vits,
            "--vits-partitions",
            partitions,
            "--output",
            pending,
        ]
    )
    os.replace(pending, output)
    reusable, reason = reusable_validation(output, graphs, partitions)
    if not reusable:
        raise RuntimeError(f"new reference validation report is unusable: {reason}")
    return output


def compile_component(
    args: argparse.Namespace,
    spec: GraphSpec,
    qairt_version: str,
) -> None:
    if not spec.source.is_file():
        raise ValueError(f"QNN graph source is missing: {spec.source}")
    reusable, reason = reusable_component(
        spec,
        target_soc=args.target_soc,
        qairt_version=qairt_version,
    )
    if reusable:
        print(f"Reusing component {spec.name}: {spec.component}", flush=True)
        return
    if spec.component.exists():
        raise ValueError(f"existing component {spec.name} cannot be reused: {reason}")
    command: list[object] = [
        args.qairt_python,
        TOOLS / "build_qnn_htp_context.py",
        "--onnx",
        spec.source,
        "--output",
        spec.component,
        "--qairt-sdk",
        args.qairt_sdk,
        "--target-soc",
        args.target_soc,
        "--optimization-level",
        spec.optimization_level,
    ]
    if spec.preserve_io_layout:
        command.append("--preserve-io-layout")
    for value in spec.input_dimensions:
        command.extend(["--input-dim", value])
    for value in args.extra_ld_library_path:
        command.extend(["--extra-ld-library-path", value])
    for value in args.extra_path:
        command.extend(["--extra-path", value])
    run(command, malloc_arena_max=2)
    reusable, reason = reusable_component(
        spec,
        target_soc=args.target_soc,
        qairt_version=qairt_version,
    )
    if not reusable:
        raise RuntimeError(f"compiled component {spec.name} failed verification: {reason}")


def graph_specs(
    args: argparse.Namespace,
    vits_partitions: Path,
    reference_vits_partitions: Path,
) -> list[GraphSpec]:
    components = args.components_root
    specs = [
        GraphSpec("bert", args.bert_onnx, components / "bert_tokens_130", 3),
        GraphSpec(
            "g2pw",
            args.g2pw_onnx,
            components / "g2pw_s130",
            1,
            (
                "input_ids=1,130",
                "token_type_ids=1,130",
                "attention_mask=1,130",
                "phoneme_mask=1,1305",
                "char_ids=1",
                "position_ids=1",
            ),
        ),
        GraphSpec("t2s_prefill", args.t2s_prefill_onnx, components / "t2s_prefill_pc128_o1", 1),
        GraphSpec("t2s_step", args.t2s_step_onnx, components / "t2s_step_c1024", 2),
        GraphSpec(
            "reference_ssl",
            args.reference_root / "reference_ssl_5s.onnx",
            components / "reference_ssl",
            1,
        ),
        GraphSpec(
            "reference_prompt_semantic",
            args.reference_root / "reference_prompt_semantic_5s.onnx",
            components / "reference_prompt_semantic",
            1,
        ),
        GraphSpec(
            "reference_conditioning",
            args.reference_root / "reference_conditioning_5s.onnx",
            components / "reference_conditioning",
            1,
        ),
        GraphSpec(
            "t2s_reference_prefill",
            args.reference_root / "t2s_reference_prefill_pc128.onnx",
            components / "t2s_reference_prefill",
            1,
        ),
    ]
    for prefix, manifest, component_prefix in (
        ("vits", vits_partitions, args.vits_component_name),
        ("vits_reference", reference_vits_partitions, args.reference_vits_component_name),
    ):
        _, paths = read_partition_manifest(manifest)
        specs.extend(
            GraphSpec(
                f"{prefix}_{index:02d}",
                path,
                components / f"{component_prefix}_p{index:02d}_o0",
                0,
                preserve_io_layout=True,
            )
            for index, path in enumerate(paths)
        )
    return specs


def expected_preset_graphs(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "bert": args.bert_onnx,
        "t2s_prefill": args.t2s_prefill_onnx,
        "t2s_step": args.t2s_step_onnx,
        "vits": args.vits_onnx,
    }


def reusable_preset_validation(
    report: Path,
    expected: dict[str, Path],
    partitions: Path | None = None,
) -> tuple[bool, str]:
    if not report.is_file():
        return False, "validation report is missing"
    try:
        document = read_json(report)
        if document.get("format") != "gsv-v2pp-qnn-onnx-fp32-validation":
            return False, "preset ONNX validation report has the wrong format"
        if not all(name in document for name in ("bert", "t2s", "vits")):
            return False, "preset ONNX validation report has incomplete measurements"
        graphs = document.get("graphs", {})
        for name, path in expected.items():
            if graphs.get(name, {}).get("sha256") != sha256(path):
                return False, f"preset validation does not cover the final {name} graph"
        if partitions is not None:
            reusable, reason = validation_covers_partitions(
                graphs.get("vits_partitions", {}),
                partitions,
            )
            if not reusable:
                return False, reason
            if "partition_equivalence" not in document["vits"]:
                return False, "preset partition equivalence measurement is missing"
        return True, "verified"
    except (OSError, TypeError, ValueError) as error:
        return False, str(error)


def ensure_preset_validation(args: argparse.Namespace, partitions: Path) -> Path:
    expected = expected_preset_graphs(args)
    reusable, _ = reusable_preset_validation(args.preset_validation, expected, partitions)
    if reusable:
        print(f"Reusing preset FP32 validation {args.preset_validation}", flush=True)
        return args.preset_validation
    pending = args.preset_validation.with_name(f"{args.preset_validation.name}.pending")
    if pending.exists():
        raise ValueError(f"stale validation output exists: {pending}")
    run(
        [
            args.validation_python,
            TOOLS / "validate_v2pp_qnn_onnx.py",
            "--root",
            args.bert_onnx.parent,
            "--vits-graph",
            args.vits_onnx,
            "--vits-partitions",
            partitions,
            "--output",
            pending,
        ]
    )
    os.replace(pending, args.preset_validation)
    reusable, reason = reusable_preset_validation(args.preset_validation, expected, partitions)
    if not reusable:
        raise RuntimeError(f"new preset validation report is unusable: {reason}")
    return args.preset_validation


def assemble(
    args: argparse.Namespace,
    specs: list[GraphSpec],
    vits_partitions: Path,
    reference_vits_partitions: Path,
) -> None:
    by_name = {spec.name: spec for spec in specs}
    command: list[object] = [
        args.validation_python,
        TOOLS / "assemble_v2pp_qnn_attachments.py",
        "--name",
        args.name,
        "--target-soc",
        args.target_soc,
        "--base-model",
        args.base_model,
        "--frontend",
        args.frontend,
        "--conditioning",
        args.conditioning,
        "--g2pw-sequence-length",
        130,
        "--pipeline-output",
        args.pipeline_output,
        "--model-output",
        args.model_output,
        "--bert-metadata",
        args.bert_metadata,
        "--t2s-metadata",
        args.t2s_metadata,
        "--vits-metadata",
        args.vits_metadata,
        "--reference-ssl-metadata",
        args.reference_root / "reference_ssl.json",
        "--reference-prompt-semantic-metadata",
        args.reference_root / "reference_prompt_semantic.json",
        "--reference-conditioning-metadata",
        args.reference_root / "reference_conditioning.json",
        "--t2s-reference-prefill-metadata",
        args.reference_root / "t2s_reference_prefill.json",
        "--vits-reference-metadata",
        args.reference_root / "vits_reference.json",
        "--vits-partitions-manifest",
        vits_partitions,
        "--vits-reference-partitions-manifest",
        reference_vits_partitions,
    ]
    graph_arguments = {
        "bert": "bert",
        "g2pw": "g2pw",
        "t2s_prefill": "t2s-prefill",
        "t2s_step": "t2s-step",
        "reference_ssl": "reference-ssl",
        "reference_prompt_semantic": "reference-prompt-semantic",
        "reference_conditioning": "reference-conditioning",
        "t2s_reference_prefill": "t2s-reference-prefill",
    }
    for name, argument in graph_arguments.items():
        spec = by_name[name]
        command.extend([f"--{argument}-onnx", spec.source, f"--{argument}-component", spec.component])
    for spec in specs:
        if spec.name.startswith("vits_reference_"):
            command.extend(["--vits-reference-partition-component", spec.component])
        elif spec.name.startswith("vits_"):
            command.extend(["--vits-partition-component", spec.component])
    run(command)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--target-soc", required=True, choices=sorted(TARGETS))
    parser.add_argument("--qairt-sdk", required=True, type=Path)
    parser.add_argument("--qairt-python", required=True, type=Path)
    parser.add_argument("--validation-python", required=True, type=Path)
    parser.add_argument("--components-root", required=True, type=Path)
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--frontend", type=Path)
    parser.add_argument("--conditioning", required=True, type=Path)
    parser.add_argument("--bert-onnx", required=True, type=Path)
    parser.add_argument("--bert-metadata", required=True, type=Path)
    parser.add_argument("--g2pw-onnx", required=True, type=Path)
    parser.add_argument("--t2s-prefill-onnx", required=True, type=Path)
    parser.add_argument("--t2s-step-onnx", required=True, type=Path)
    parser.add_argument("--t2s-metadata", required=True, type=Path)
    parser.add_argument("--vits-onnx", required=True, type=Path)
    parser.add_argument("--vits-partitions-root", required=True, type=Path)
    parser.add_argument("--vits-metadata", required=True, type=Path)
    parser.add_argument("--preset-validation", required=True, type=Path)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--reference-vits-onnx", required=True, type=Path)
    parser.add_argument("--reference-vits-partitions-root", required=True, type=Path)
    parser.add_argument("--reference-validation", required=True, type=Path)
    parser.add_argument("--pipeline-output", type=Path)
    parser.add_argument("--model-output", type=Path)
    parser.add_argument("--contexts-only", action="store_true")
    parser.add_argument(
        "--vits-component-name",
        default="vits_pc128_sc512_htp_v79_vtcm_v2_pio_layout_o0",
    )
    parser.add_argument(
        "--reference-vits-component-name",
        default="vits_reference_pc128_sc512_htp_v79_vtcm_pio_layout_o0",
    )
    parser.add_argument("--extra-ld-library-path", action="append", default=[], type=Path)
    parser.add_argument("--extra-path", action="append", default=[], type=Path)
    args = parser.parse_args()
    path_fields = (
        "qairt_sdk",
        "components_root",
        "conditioning",
        "bert_onnx",
        "bert_metadata",
        "g2pw_onnx",
        "t2s_prefill_onnx",
        "t2s_step_onnx",
        "t2s_metadata",
        "vits_onnx",
        "vits_partitions_root",
        "vits_metadata",
        "preset_validation",
        "reference_root",
        "reference_vits_onnx",
        "reference_vits_partitions_root",
        "reference_validation",
    )
    for name in path_fields:
        setattr(args, name, getattr(args, name).resolve())
    args.qairt_python = absolute_executable(args.qairt_python)
    args.validation_python = absolute_executable(args.validation_python)
    args.extra_ld_library_path = [value.resolve() for value in args.extra_ld_library_path]
    args.extra_path = [value.resolve() for value in args.extra_path]
    if args.contexts_only:
        if args.pipeline_output is not None or args.model_output is not None:
            parser.error("--contexts-only cannot be combined with attachment outputs")
    elif None in (args.base_model, args.frontend, args.pipeline_output, args.model_output):
        parser.error(
            "--base-model, --frontend, --pipeline-output and --model-output are required for assembly"
        )
    else:
        args.base_model = args.base_model.resolve()
        args.frontend = args.frontend.resolve()
        args.pipeline_output = args.pipeline_output.resolve()
        args.model_output = args.model_output.resolve()
    return args


def main() -> None:
    args = parse_args()
    version = sdk_version(args.qairt_sdk)
    args.g2pw_onnx = ensure_g2pw_alignment(args)
    args.vits_onnx = ensure_preset_vits_split(args)
    reference_vits = ensure_reference_vits_split(args)
    vits_partitions = ensure_vits_partitions(
        args.vits_onnx,
        args.vits_partitions_root,
        "vits",
        args,
    )
    reference_vits_partitions = ensure_vits_partitions(
        reference_vits,
        args.reference_vits_partitions_root,
        "vits_reference",
        args,
    )
    ensure_reference_validation(args, reference_vits, reference_vits_partitions)
    ensure_preset_validation(args, vits_partitions)
    specs = graph_specs(args, vits_partitions, reference_vits_partitions)
    for spec in specs:
        compile_component(args, spec, version)
    if not args.contexts_only:
        assemble(args, specs, vits_partitions, reference_vits_partitions)
    print(
        f"Completed V2 Pro Plus QNN product build for {args.target_soc}; "
        f"contexts={len(specs)} attachments={'no' if args.contexts_only else 'yes'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
