#!/usr/bin/env python3
"""Compile one static ONNX graph into one SoC-specific QAIRT FP16 HTP context.

This tool produces a compiled graph component, not a deployable GPT-SoVITS package. A complete NPU
package still needs every required acoustic graph plus an executor that exposes the stable GSVM
text-to-PCM operation. Keeping this boundary explicit prevents a frontend-only QNN graph from being
mistaken for complete Android TTS support.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


ANDROID_QNN_RUNTIME_VERSION = "2.48.0"
TARGET_SOC_FAMILY = "qualcomm_snapdragon_8"


@dataclass(frozen=True)
class QnnTarget:
    target_soc: str
    asic: str
    htp_arch: str
    soc_model: int


TARGETS = {
    "snapdragon_8_gen_3": QnnTarget("snapdragon_8_gen_3", "SM8650", "V75", 57),
    "snapdragon_8_elite": QnnTarget("snapdragon_8_elite", "SM8750", "V79", 69),
    "snapdragon_8_elite_gen_5": QnnTarget("snapdragon_8_elite_gen_5", "SM8850", "V81", 87),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_dimensions(values: list[str]) -> dict[str, tuple[int, ...]]:
    result: dict[str, tuple[int, ...]] = {}
    for value in values:
        name, separator, raw_dimensions = value.partition("=")
        if not separator or not name.strip():
            raise ValueError(f"invalid --input-dim {value!r}; expected NAME=1,2,3")
        dimensions = tuple(int(item) for item in raw_dimensions.split(","))
        if not dimensions or any(item <= 0 for item in dimensions):
            raise ValueError(f"all dimensions must be positive in --input-dim {value!r}")
        if name in result:
            raise ValueError(f"duplicate --input-dim for {name}")
        result[name] = dimensions
    return result


def validate_static_onnx(path: Path, overrides: dict[str, tuple[int, ...]]) -> list[tuple[str, tuple[int, ...]]]:
    try:
        import onnx
    except ImportError as error:
        raise RuntimeError("the conversion environment must provide the onnx Python package") from error

    model = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(model)
    initializers = {value.name for value in model.graph.initializer}
    inputs: list[tuple[str, tuple[int, ...]]] = []
    input_names = {value.name for value in model.graph.input if value.name not in initializers}
    unknown = set(overrides).difference(input_names)
    if unknown:
        raise ValueError(f"--input-dim names are not graph inputs: {', '.join(sorted(unknown))}")
    for value in model.graph.input:
        if value.name in initializers:
            continue
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            raise ValueError(f"input {value.name} has no tensor shape")
        dimensions = []
        dynamic = False
        for dimension in tensor_type.shape.dim:
            if dimension.HasField("dim_value") and dimension.dim_value > 0:
                dimensions.append(int(dimension.dim_value))
            else:
                dimensions.append(-1)
                dynamic = True
        selected = overrides.get(value.name)
        if selected is not None:
            if len(selected) != len(dimensions):
                raise ValueError(
                    f"input {value.name} has rank {len(dimensions)}, override has rank {len(selected)}"
                )
            dimensions = list(selected)
            dynamic = False
        if dynamic:
            raise ValueError(
                f"input {value.name} is dynamic; provide --input-dim {value.name}=... for one prepared artifact"
            )
        inputs.append((value.name, tuple(dimensions)))
    if not inputs:
        raise ValueError("ONNX graph has no runtime inputs")
    return inputs


def validate_preserved_io_layout(
    source: Path,
    context_info: Path,
    resolved_inputs: dict[str, tuple[int, ...]] | None = None,
) -> None:
    try:
        import onnx
    except ImportError as error:
        raise RuntimeError("the conversion environment must provide the onnx Python package") from error
    from wrap_qnn_ep_context import load_graph_io, match_tensor_names

    model = onnx.load(str(source), load_external_data=False)
    initializers = {value.name for value in model.graph.initializer}
    source_inputs = [value for value in model.graph.input if value.name not in initializers]
    context_inputs, context_outputs = load_graph_io(context_info)

    def source_shapes(values: list[onnx.ValueInfoProto], overrides: dict[str, tuple[int, ...]]) -> dict[str, list[int]]:
        result: dict[str, list[int]] = {}
        for value in values:
            if value.name in overrides:
                result[value.name] = list(overrides[value.name])
                continue
            dimensions = [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]
            if not dimensions or any(dimension <= 0 for dimension in dimensions):
                raise ValueError(
                    f"cannot verify preserved layout for dynamic tensor {value.name!r}"
                )
            result[value.name] = dimensions
        return result

    def compare(
        source_values: list[onnx.ValueInfoProto],
        context_values: list[dict],
        label: str,
        overrides: dict[str, tuple[int, ...]],
    ) -> None:
        expected = source_shapes(source_values, overrides)
        mapping = match_tensor_names(
            list(expected),
            [value.get("name", "") for value in context_values],
            label,
        )
        for value in context_values:
            actual = value.get("dimensions")
            source_name = mapping[value["name"]]
            if actual != expected[source_name]:
                raise ValueError(
                    f"context did not preserve {label} layout for {source_name!r}: "
                    f"source={expected[source_name]} context={actual}"
                )

    input_overrides = resolved_inputs or {}
    compare(source_inputs, context_inputs, "inputs", input_overrides)
    compare(list(model.graph.output), context_outputs, "outputs", {})


def sdk_version(sdk: Path) -> str:
    version = sdk.name
    if not re.fullmatch(r"\d+\.\d+\.\d+\.\d+", version):
        raise ValueError(
            "QAIRT SDK directory must retain its exact version name, for example 2.48.0.260626"
        )
    runtime_line = ".".join(version.split(".")[:3])
    if runtime_line != ANDROID_QNN_RUNTIME_VERSION:
        raise ValueError(
            f"QAIRT {version} does not match Android qnn-runtime {ANDROID_QNN_RUNTIME_VERSION}"
        )
    return version


def tool_path(sdk: Path, name: str) -> Path:
    path = sdk / "bin/x86_64-linux-clang" / name
    if not path.is_file():
        raise FileNotFoundError(f"QAIRT tool is missing: {path}")
    return path


def write_htp_backend_configs(
    sdk: Path,
    work: Path,
    target: QnnTarget,
    optimization_level: int,
) -> Path:
    """Pin the actual cache target; --htp_socs alone can silently emit generic V68 metadata."""
    if optimization_level not in (0, 1, 2, 3):
        raise ValueError("HTP optimization level must be between 0 and 3")
    work.mkdir(parents=True, exist_ok=True)
    backend_config = work / "htp-backend.json"
    backend_config.write_text(
        json.dumps(
            {
                "graphs": [
                    {
                        "graph_names": ["model"],
                        "vtcm_mb": 0,
                        "O": optimization_level,
                    }
                ],
                "devices": [
                    {
                        "soc_model": target.soc_model,
                        "dsp_arch": target.htp_arch.lower(),
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    main_config = work / "htp-main.json"
    main_config.write_text(
        json.dumps(
            {
                "backend_extensions": {
                    "shared_library_path": str(
                        sdk / "lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so"
                    ),
                    "config_file_path": str(backend_config),
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return main_config


def build_commands(
    sdk: Path,
    source: Path,
    work: Path,
    target: QnnTarget,
    shape_overrides: list[tuple[str, tuple[int, ...]]],
    optimization_level: int = 2,
    preserve_io_layout: bool = False,
) -> tuple[list[str], list[str], list[str], Path]:
    model_cpp = work / "model.cpp"
    model_bin = work / "model.bin"
    library_name = f"gsv_{target.target_soc}"
    library_root = work / "model-lib"
    context_root = work / "context"
    context_name = f"{library_name}_fp16"
    main_config = write_htp_backend_configs(sdk, work, target, optimization_level)

    converter = [
        str(tool_path(sdk, "qnn-onnx-converter")),
        "--input_network", str(source),
        "--output_path", str(model_cpp),
        "--float_bitwidth", "16",
        # QAIRT 2.48 documents an HTP context-build failure when an otherwise FP16
        # graph retains FP32 bias tensors. Keep the complete floating-point graph in
        # FP16 so model.cpp and model.bin agree on every static tensor payload.
        "--float_bias_bitwidth", "16",
    ]
    if preserve_io_layout:
        converter.extend(["--preserve_io", "layout"])
    for name, dimensions in shape_overrides:
        converter.extend(["--input_dim", name, ",".join(map(str, dimensions))])
    model_library = [
        str(tool_path(sdk, "qnn-model-lib-generator")),
        "--cpp", str(model_cpp),
        "--bin", str(model_bin),
        "--lib_targets", "x86_64-linux-clang",
        "--lib_name", library_name,
        "--output_dir", str(library_root),
    ]
    host_library = library_root / "x86_64-linux-clang" / f"lib{library_name}.so"
    context = [
        str(tool_path(sdk, "qnn-context-binary-generator")),
        f"--model={host_library}",
        f"--backend={sdk / 'lib/x86_64-linux-clang/libQnnHtp.so'}",
        f"--binary_file={context_name}.{target.asic}",
        f"--output_dir={context_root}",
        f"--config_file={main_config}",
    ]
    return converter, model_library, context, context_root / f"{context_name}.{target.asic}.bin"


def qairt_environment(
    sdk: Path,
    extra_library_paths: list[Path],
    extra_binary_paths: list[Path],
) -> dict[str, str]:
    environment = dict(os.environ)
    python_paths = [sdk / "lib/python", sdk / "benchmarks/QNN"]
    library_paths = [sdk / "lib/x86_64-linux-clang", *extra_library_paths]
    environment.update(
        QAIRT_SDK_ROOT=str(sdk),
        QNN_SDK_ROOT=str(sdk),
        PYTHONPATH=os.pathsep.join(map(str, python_paths))
        + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""),
        LD_LIBRARY_PATH=os.pathsep.join(map(str, library_paths))
        + (os.pathsep + environment["LD_LIBRARY_PATH"] if environment.get("LD_LIBRARY_PATH") else ""),
        PATH=os.pathsep.join(
            [
                str(Path(sys.executable).parent),
                str(sdk / "bin/x86_64-linux-clang"),
                *map(str, extra_binary_paths),
            ]
        )
        + os.pathsep
        + environment.get("PATH", ""),
    )
    return environment


def run(command: list[str], environment: dict[str, str]) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=environment)


def validate_context_metadata(document: dict, target: QnnTarget, version: str) -> None:
    info = document.get("info", {})
    actual_soc_model = info.get("socModel")
    actual_arch = info.get("contextMetadata", {}).get("info", {}).get("dspArch")
    build_id = info.get("buildId", "")
    if actual_soc_model != target.soc_model or actual_arch != int(target.htp_arch[1:]):
        raise RuntimeError(
            "QAIRT emitted a context for the wrong HTP target: "
            f"expected {target.asic}/{target.htp_arch} "
            f"(socModel={target.soc_model}), got socModel={actual_soc_model}, "
            f"dspArch={actual_arch}"
        )
    if not isinstance(build_id, str) or not build_id.startswith(f"v{version}"):
        raise RuntimeError(
            f"context buildId {build_id!r} does not match QAIRT SDK {version}"
        )
    if info.get("numGraphs") != 1:
        raise RuntimeError(f"expected one compiled graph, found {info.get('numGraphs')!r}")


def inspect_context(
    sdk: Path,
    context: Path,
    output: Path,
    target: QnnTarget,
    version: str,
    environment: dict[str, str],
) -> Path:
    command = [
        str(tool_path(sdk, "qnn-context-binary-utility")),
        f"--context_binary={context}",
        f"--json_file={output}",
    ]
    run(command, environment)
    document = json.loads(output.read_text(encoding="utf-8"))
    validate_context_metadata(document, target, version)
    return output


def write_artifact(
    output: Path,
    source: Path,
    context: Path,
    context_info: Path,
    target: QnnTarget,
    version: str,
    inputs: list[tuple[str, tuple[int, ...]]],
    optimization_level: int,
    preserve_io_layout: bool,
) -> None:
    backend_directory = output / "backend"
    backend_directory.mkdir(parents=True)
    backend = backend_directory / context.name
    shutil.copy2(context, backend)
    deployed_context_info = output / "context-info.json"
    shutil.copy2(context_info, deployed_context_info)
    manifest = {
        "format": "gsv-qnn-compiled-component",
        "format_version": 1,
        "deployable": False,
        "executor": "qnn-htp",
        "precision": "fp16",
        "quantization": "none",
        "graph_io_dtype": "float16",
        "preserve_io_layout": preserve_io_layout,
        "target_soc": target.target_soc,
        "target_soc_family": TARGET_SOC_FAMILY,
        "target_asic": target.asic,
        "target_soc_model": target.soc_model,
        "supported_target_socs": [target.target_soc],
        "htp_arch": target.htp_arch,
        "qairt_version": version,
        "qnn_runtime_version": ANDROID_QNN_RUNTIME_VERSION,
        "htp_graph_optimization_level": optimization_level,
        "backend_artifact": backend.relative_to(output).as_posix(),
        "source_onnx": source.name,
        "source_onnx_sha256": sha256(source),
        "static_inputs": {name: list(dimensions) for name, dimensions in inputs},
        "files": [
            {
                "path": backend.relative_to(output).as_posix(),
                "size": backend.stat().st_size,
                "sha256": sha256(backend),
            },
            {
                "path": deployed_context_info.relative_to(output).as_posix(),
                "size": deployed_context_info.stat().st_size,
                "sha256": sha256(deployed_context_info),
            },
        ],
        "deployment_blocker": (
            "compiled graph component only; a complete acoustic graph set and text-to-PCM executor are required"
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--qairt-sdk", required=True, type=Path)
    parser.add_argument("--target-soc", required=True, choices=sorted(TARGETS))
    parser.add_argument(
        "--optimization-level",
        type=int,
        choices=(0, 1, 2, 3),
        default=2,
        help=(
            "HTP offline graph optimization level; use 1 or 0 when higher-level graph "
            "finalization exceeds host memory or target VTCM."
        ),
    )
    parser.add_argument(
        "--input-dim",
        action="append",
        default=[],
        metavar="NAME=1,2,3",
        help="Resolve a dynamic input to one static shape; repeat for multiple inputs.",
    )
    parser.add_argument(
        "--preserve-io-layout",
        action="store_true",
        help="Preserve source ONNX I/O layout while converting floating-point I/O to FP16.",
    )
    parser.add_argument(
        "--extra-ld-library-path",
        action="append",
        default=[],
        type=Path,
        help="Host dependency directory needed by QAIRT tools; repeat when necessary.",
    )
    parser.add_argument(
        "--extra-path",
        action="append",
        default=[],
        type=Path,
        help="Host executable directory needed by QAIRT tools; repeat when necessary.",
    )
    parser.add_argument("--print-commands", action="store_true", help="Validate and print without compiling.")
    args = parser.parse_args()

    source = args.onnx.resolve()
    output = args.output.resolve()
    sdk = args.qairt_sdk.resolve()
    if not source.is_file():
        raise SystemExit(f"ONNX source does not exist: {source}")
    if output.exists():
        raise SystemExit(f"output already exists; choose a new directory: {output}")
    version = sdk_version(sdk)
    overrides = parse_dimensions(args.input_dim)
    inputs = validate_static_onnx(source, overrides)
    target = TARGETS[args.target_soc]

    with tempfile.TemporaryDirectory(prefix="gsv-qnn-") as temporary:
        work = Path(temporary)
        selected_overrides = [(name, dimensions) for name, dimensions in inputs if name in overrides]
        commands = build_commands(
            sdk,
            source,
            work,
            target,
            selected_overrides,
            args.optimization_level,
            args.preserve_io_layout,
        )
        if args.print_commands:
            for command in commands[:3]:
                print(" ".join(command))
            return
        environment = qairt_environment(
            sdk,
            [value.resolve() for value in args.extra_ld_library_path],
            [value.resolve() for value in args.extra_path],
        )
        for command in commands[:3]:
            run(command, environment)
        context = commands[3]
        if not context.is_file() or context.stat().st_size == 0:
            raise RuntimeError(f"QAIRT did not produce the expected context binary: {context}")
        context_info = inspect_context(
            sdk,
            context,
            work / "context-info.json",
            target,
            version,
            environment,
        )
        if args.preserve_io_layout:
            validate_preserved_io_layout(source, context_info, dict(inputs))
        output.parent.mkdir(parents=True, exist_ok=True)
        pending = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
        try:
            write_artifact(
                pending,
                source,
                context,
                context_info,
                target,
                version,
                inputs,
                args.optimization_level,
                args.preserve_io_layout,
            )
            pending.rename(output)
        except Exception:
            shutil.rmtree(pending, ignore_errors=True)
            raise
    print(f"Created {output} for {target.target_soc} ({target.htp_arch}, QAIRT {version})")


if __name__ == "__main__":
    main()
