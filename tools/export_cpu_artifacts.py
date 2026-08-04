#!/usr/bin/env python3
"""Dispatch upstream graph exporters by model family.

The output is an intermediate conversion directory, never an Android-facing API. A later fusion
stage must produce one pipeline.pt accepting UTF-8 text.
"""
from __future__ import annotations
import argparse, os, subprocess, sys
from pathlib import Path
from model_profiles import detect_sovits

def run(command: list[str], cwd: Path, env: dict[str,str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=cwd, env=env, check=True)

def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpt',required=True,type=Path); p.add_argument('--sovits',required=True,type=Path)
    p.add_argument('--reference',required=True,type=Path); p.add_argument('--prompt',required=True)
    p.add_argument('--vocoder',type=Path,help='V4 HiFi-GAN checkpoint (defaults to the upstream pretrained vocoder)')
    p.add_argument('--language',default='all_zh'); p.add_argument('--output',required=True,type=Path)
    p.add_argument('--upstream',type=Path,default=Path('..')); a=p.parse_args()
    upstream=a.upstream.resolve(); python=upstream/'gpt/bin/python'; output=a.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    profile,lora,_=detect_sovits(a.sovits.resolve())
    if profile.id not in {'v2ProPlus','v4'}:
        raise SystemExit(f'Android scope supports only v2ProPlus and v4, got {profile.id}')
    env=dict(os.environ, is_half='False', version=profile.id)
    common=[str(python)]
    if profile.cpu_exporter == 'torchscript_stream_pro':
        command=common+[str(Path(__file__).resolve().parent/'run_v2pp_export_compat.py'),
          "--upstream-script",str(upstream/'GPT_SoVITS/stream_v2pro.py'),"--gpt_model",str(a.gpt.resolve()),"--sovits_model",str(a.sovits.resolve()),
          "--ref_audio",str(a.reference.resolve()),"--ref_text",a.prompt,"--output_path",str(output),"--device","cpu",
          "--version",profile.id,"--no-half","--lang",a.language]
    elif profile.cpu_exporter == 'torchscript_legacy':
        command=common+["GPT_SoVITS/export_torch_script.py","--gpt_model",str(a.gpt.resolve()),"--sovits_model",str(a.sovits.resolve()),
          "--ref_audio",str(a.reference.resolve()),"--ref_text",a.prompt,"--output_path",str(output),"--device","cpu",
          "--version",profile.id,"--no-half"]
    elif profile.id == 'v4':
        vocoder=(a.vocoder.resolve() if a.vocoder else upstream/'GPT_SoVITS/pretrained_models/gsv-v4-pretrained/vocoder.pth')
        command=common+[str(Path(__file__).resolve().parent/'export_v4_cpu_artifacts.py'),
          '--gpt',str(a.gpt.resolve()),'--sovits',str(a.sovits.resolve()),'--vocoder',str(vocoder),
          '--reference',str(a.reference.resolve()),'--prompt',a.prompt,'--language',a.language,
          '--output',str(output),'--upstream',str(upstream),'--sample-steps','32']
    else:
        raise SystemExit(f'No CPU exporter for {profile.id}')
    run(command,upstream,env)
    print(f"Exported {profile.id}; lora={lora}; intermediate={output}")

if __name__=='__main__': main()
