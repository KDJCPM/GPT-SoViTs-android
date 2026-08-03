#!/usr/bin/env python3
"""Export only the BERT encoder required for phone features, excluding the unused MLM head."""
import argparse
from pathlib import Path
import torch
from transformers import AutoModel, AutoTokenizer

@torch.jit.script
def expand_phone_features(features: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    rows=torch.jit.annotate(list[torch.Tensor],[])
    for i in range(counts.numel()):
        for _ in range(int(counts[i].item())): rows.append(features[i])
    return torch.stack(rows)

class LeanBert(torch.nn.Module):
    def __init__(self, model): super().__init__(); self.model=model
    def forward(self,input_ids,attention_mask,token_type_ids,word2ph):
        outputs=self.model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,output_hidden_states=True,return_dict=False)
        return expand_phone_features(outputs[2][-3][0,1:-1],word2ph).float()

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--half',action='store_true');a=p.parse_args()
    tokenizer=AutoTokenizer.from_pretrained(a.model); text='你好,这是移动端模型导出测试.'; inputs=tokenizer(text,return_tensors='pt'); counts=torch.tensor([2 if '\u4e00'<=c<='\u9fff' else 1 for c in text],dtype=torch.int32)
    # PyTorch Android 1.13 has no aten::scaled_dot_product_attention; force the mathematically
    # equivalent eager matmul/softmax path during conversion.
    model=AutoModel.from_pretrained(a.model,attn_implementation='eager',add_pooling_layer=False).eval(); model=model.half() if a.half else model; wrapper=LeanBert(model).eval()
    traced=torch.jit.trace(wrapper,(inputs['input_ids'],inputs['attention_mask'],inputs['token_type_ids'],counts),strict=False)
    traced.save(str(a.output));print(f'Created {a.output} ({a.output.stat().st_size} bytes)')
if __name__=='__main__':main()
