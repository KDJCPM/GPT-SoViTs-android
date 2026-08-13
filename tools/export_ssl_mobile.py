#!/usr/bin/env python3
"""Export the shared FP32 Chinese HuBERT reference encoder for Android TorchScript."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import HubertModel


class ReferenceSslEncoder(torch.nn.Module):
    def __init__(self, model: HubertModel):
        super().__init__()
        self.model = model

    def forward(self, pcm_16k: torch.Tensor) -> torch.Tensor:
        return self.model(pcm_16k)[0].transpose(1, 2).float()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    model = HubertModel.from_pretrained(
        args.model.resolve(),
        local_files_only=True,
        torchscript=True,
    ).float().eval()
    example = torch.zeros(1, 16_000 * 5, dtype=torch.float32)
    with torch.inference_mode():
        exported = torch.jit.trace(ReferenceSslEncoder(model).eval(), example, check_trace=False, strict=False)
        exported = torch.jit.freeze(exported)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        exported.save(str(args.output))
    print(f"Created {args.output} ({args.output.stat().st_size} bytes), FP32")


if __name__ == "__main__":
    main()
