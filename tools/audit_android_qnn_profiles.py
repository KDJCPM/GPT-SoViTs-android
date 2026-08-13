#!/usr/bin/env python3
"""Audit Android ONNX Runtime profiles for strict QNN-only neural execution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def provider(event: dict) -> str:
    arguments = event.get("args") if isinstance(event.get("args"), dict) else {}
    for value in (
        arguments.get("provider"),
        arguments.get("execution_provider"),
        event.get("provider"),
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def audit_profile(path: Path) -> dict[str, object]:
    events = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(events, list):
        raise ValueError(f"{path} is not an ONNX Runtime event array")
    providers: set[str] = set()
    qnn_nodes = 0
    cpu_nodes = 0
    duration_us = 0
    for event in events:
        if not isinstance(event, dict) or event.get("cat") != "Node":
            continue
        name = provider(event)
        if not name:
            continue
        providers.add(name)
        duration = event.get("dur", 0)
        try:
            duration_us += int(duration)
        except (TypeError, ValueError):
            pass
        normalized = name.lower()
        if normalized in {"qnn", "qnnexecutionprovider"}:
            qnn_nodes += 1
        elif normalized in {"cpu", "cpuexecutionprovider"}:
            cpu_nodes += 1
    if cpu_nodes:
        raise ValueError(f"{path} assigned {cpu_nodes} neural node event(s) to CPU")
    if qnn_nodes == 0:
        raise ValueError(f"{path} contains no QNN node events")
    return {
        "path": str(path.resolve()),
        "providers": sorted(providers),
        "qnn_nodes": qnn_nodes,
        "cpu_nodes": cpu_nodes,
        "node_duration_us": duration_us,
    }


def collect_profiles(values: list[Path]) -> list[Path]:
    profiles: set[Path] = set()
    for value in values:
        value = value.resolve()
        if value.is_dir():
            profiles.update(path for path in value.rglob("*.json") if path.is_file())
        elif value.is_file():
            profiles.add(value)
        else:
            raise ValueError(f"profile path does not exist: {value}")
    if not profiles:
        raise ValueError("no ONNX Runtime JSON profiles were found")
    return sorted(profiles)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", nargs="+", type=Path, help="Profile JSON file or directory")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = [audit_profile(path) for path in collect_profiles(args.profile)]
    report = {
        "format": "gsv-android-qnn-profile-audit",
        "format_version": 1,
        "passed": True,
        "profiles": results,
        "totals": {
            "profiles": len(results),
            "qnn_nodes": sum(int(item["qnn_nodes"]) for item in results),
            "cpu_nodes": sum(int(item["cpu_nodes"]) for item in results),
            "node_duration_us": sum(int(item["node_duration_us"]) for item in results),
        },
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.output.resolve().write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
