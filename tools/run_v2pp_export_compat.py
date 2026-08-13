#!/usr/bin/env python3
"""Run upstream V2 Pro Plus export with a trace-compatible speed input.

PyTorch tracing does not accept Python floats in example input tuples. The upstream module uses a
float speed argument, so this launcher applies the minimal source-level compatibility adjustment in
memory. The upstream checkout is never modified; the resulting graph still consumes the same value
as a scalar tensor and the final GSVM wrapper retains the stable float option API.
"""

import argparse
import os
import sys
from pathlib import Path


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"Upstream exporter compatibility marker changed: {old!r}")
    return source.replace(old, new)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--upstream-script", required=True, type=Path)
    known, remaining = parser.parse_known_args()
    script = known.upstream_script.resolve()
    source = script.read_text(encoding="utf-8")
    source = replace_once(
        source,
        "def forward(self, pred_semantic, text_seq, refer, sv_emb=None, speed: float = 1.0):",
        "def forward(self, pred_semantic, text_seq, refer, sv_emb=None, speed: Tensor = torch.tensor(1.0)):",
    )
    source = replace_once(
        source,
        '"forward": (y, text_seq, refer, sv_emb, 1.0),',
        '"forward": (y, text_seq, refer, sv_emb, torch.tensor(1.0, device=y.device)),',
    )
    # User checkpoints are frequently saved with CUDA storage tags.  CPU export must be
    # deterministic on hosts without CUDA, so remap both GPT checkpoint loads explicitly.
    source = source.replace("torch.load(gpt_path, weights_only=False)",
                            "torch.load(gpt_path, map_location=\"cpu\", weights_only=False)")
    sys.argv = [str(script), *remaining]
    # The upstream web UI writes a transient weight.json relative to cwd.  Keep the
    # checkout read-only and allow the dispatcher to provide a writable workspace.
    os.chdir(os.environ.get("GSV_EXPORT_CWD", str(script.parents[1])))
    namespace = {"__name__": "__main__", "__file__": str(script)}
    exec(compile(source, str(script), "exec"), namespace)


if __name__ == "__main__":
    main()
