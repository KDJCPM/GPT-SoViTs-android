from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib

@dataclass(frozen=True)
class ModelProfile:
    id: str
    checkpoint_family: str
    sample_rate: int
    cpu_exporter: str
    vocoder: str
    supports_lora: bool = False

PROFILES = {
    "v1": ModelProfile("v1", "v1", 32000, "torchscript_legacy", "sovits"),
    "v2": ModelProfile("v2", "v2", 32000, "torchscript_legacy", "sovits"),
    "v2Pro": ModelProfile("v2Pro", "v2", 32000, "torchscript_stream_pro", "sovits"),
    "v2ProPlus": ModelProfile("v2ProPlus", "v2", 32000, "torchscript_stream_pro", "sovits"),
    "v3": ModelProfile("v3", "v2", 24000, "torchscript_cfm", "bigvgan", True),
    "v4": ModelProfile("v4", "v2", 48000, "torchscript_cfm", "hifigan", True),
}

HEADER_PROFILES = {
    b"00": ("v1", False), b"01": ("v2", False), b"02": ("v3", False),
    b"03": ("v3", True), b"04": ("v4", True), b"05": ("v2Pro", False),
    b"06": ("v2ProPlus", False),
}

PRETRAINED_MD5 = {
    "dc3c97e17592963677a4a1681f30c653": ("v1", False),
    "6642b37f3dbb1f76882b69937c95a5f3": ("v2", False),
    "43797be674a37c1c83ee81081941ed0f": ("v3", False),
    "4f26b9476d0c5033e04162c486074374": ("v4", False),
    "c7e9fce2223f3db685cdfa1e6368728a": ("v2Pro", False),
    "66b313e39455b57ab1b0bc0b239c9d0a": ("v2ProPlus", False),
}

def first_block_md5(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.md5(stream.read(8192)).hexdigest()

def detect_sovits(path: Path) -> tuple[ModelProfile, bool, bytes]:
    known = PRETRAINED_MD5.get(first_block_md5(path))
    with path.open("rb") as stream:
        header = stream.read(2)
    if known:
        version, lora = known
    elif header in HEADER_PROFILES:
        version, lora = HEADER_PROFILES[header]
    elif header == b"PK":
        size = path.stat().st_size
        if size < 82978 * 1024: version = "v1"
        elif size < 700 * 1024 * 1024: version = "v2"
        else: version = "v3"
        lora = False
    else:
        raise ValueError(f"Unknown SoVITS checkpoint header {header!r}")
    return PROFILES[version], lora, header
