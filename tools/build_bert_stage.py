#!/usr/bin/env python3
"""Wrap the unchanged FP32 BERT graph as a disposable phone-feature stage."""
import argparse
from pathlib import Path
import torch


class BertStage(torch.nn.Module):
    def __init__(self, bert):
        super().__init__(); self.bert=bert

    def forward(self, token_ids: torch.Tensor, word2ph: torch.Tensor, chinese_mask: torch.Tensor):
        ids=token_ids.to(torch.int64).reshape(1,-1)
        features=self.bert(ids,torch.ones_like(ids),torch.zeros_like(ids),word2ph.to(torch.int32).reshape(-1))
        return features*chinese_mask.to(features.dtype).reshape(-1,1)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bert',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    module=torch.jit.script(BertStage(torch.jit.load(str(a.bert.resolve()),map_location='cpu')).eval())
    module=torch.jit.freeze(module);module.save(str(a.output.resolve()))
    print(f'Created {a.output.resolve()} ({a.output.resolve().stat().st_size} bytes)')

if __name__=='__main__':main()
