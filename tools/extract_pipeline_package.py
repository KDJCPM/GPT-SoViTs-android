#!/usr/bin/env python3
"""Extract a verified, voice-independent frontend pipeline from a deployable GSVM package."""

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--executor", required=True, choices=("torchscript-cpu-single", "torchscript-cpu-staged"))
    args = parser.parse_args()

    with zipfile.ZipFile(args.source) as source:
        manifest = json.loads(source.read("manifest.json"))
        files = [item for item in manifest["files"] if item["path"].startswith("runtime/frontend/")]
        if not files:
            raise SystemExit("source package has no runtime/frontend assets")
        for item in files:
            digest = hashlib.sha256()
            with source.open(item["path"]) as payload:
                for chunk in iter(lambda: payload.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != item["sha256"]:
                raise SystemExit(f'{item["path"]} failed source hash verification')

        version = manifest["model_version"]
        frontend_profile = manifest.get("frontend_profile", "full-g2pw-v2")
        options_abi = manifest.get("runtime_options_version", 0)
        pipeline_manifest = {
            "format": "gsvm-deploy",
            "format_version": 1,
            "name": f"GPT-SoVITS {version} shared pipeline",
            "model_version": version,
            "sample_rate": manifest["sample_rate"],
            "runtime": args.executor,
            "executor": args.executor,
            "entrypoint": "synthesize_utf8_to_pcm16",
            "api_version": 1,
            "deployable": True,
            "frontend_profile": frontend_profile,
            "runtime_options_version": options_abi,
            "artifact_role": "pipeline",
            "requires_role": "model",
            "bundle_id": f"gsvm:{version}:{frontend_profile}:api1:options{options_abi}",
            "target_soc": "any",
            "target_soc_family": "cpu",
            "backend_artifact": args.executor,
            "files": files,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        partial = args.output.with_suffix(args.output.suffix + ".partial")
        with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as output:
            output.writestr("manifest.json", json.dumps(pipeline_manifest, ensure_ascii=False, indent=2))
            for item in files:
                with source.open(item["path"]) as incoming, output.open(item["path"], "w", force_zip64=True) as outgoing:
                    shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
        partial.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
