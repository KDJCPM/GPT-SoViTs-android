#!/usr/bin/env python3
"""Package fully converted CPU files behind the single Android runtime contract."""
import argparse, hashlib, json, os, zipfile
from pathlib import Path
from model_profiles import PROFILES

def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

p=argparse.ArgumentParser()
p.add_argument('--artifacts',required=True,type=Path)
p.add_argument('--output',type=Path,help='Combined deployment package output')
p.add_argument('--pipeline-output',type=Path,help='Standalone pipeline package output')
p.add_argument('--model-output',type=Path,help='Standalone model package output')
p.add_argument('--name',required=True)
p.add_argument('--version',required=True,choices=sorted(PROFILES))
p.add_argument('--frontend',type=Path,help='Converted frontend bundle; required for deployable=true')
p.add_argument('--pipeline',default='pipeline.pt',help='Pipeline filename inside --artifacts')
p.add_argument('--bert-stage',help='Optional staged FP32 BERT filename inside --artifacts')
p.add_argument('--acoustic-stage',help='Optional staged acoustic filename inside --artifacts')
p.add_argument('--frontend-profile',default='portable-char-v1')
p.add_argument('--upstream-equivalent',action='store_true')
p.add_argument('--minimum-ram-mb',type=int,default=6144)
p.add_argument('--runtime-options',action='store_true',help='Advertise option-aware graph ABI v1; only use with an exporter that consumes all options')
a=p.parse_args()
split_outputs = a.pipeline_output is not None or a.model_output is not None
if a.pipeline_output is not None and a.model_output is None:
    raise SystemExit('--pipeline-output requires --model-output')
if a.output is not None and split_outputs:
    raise SystemExit('--output cannot be combined with standalone outputs')
if not split_outputs and a.output is None:
    raise SystemExit('--output is required unless standalone outputs are supplied')
staged=bool(a.bert_stage or a.acoustic_stage)
if staged and not (a.bert_stage and a.acoustic_stage): raise SystemExit('--bert-stage and --acoustic-stage must be supplied together')
required=[a.bert_stage,a.acoustic_stage] if staged else [a.pipeline]
for name in required:
    if not (a.artifacts/name).is_file(): raise SystemExit(f'missing {name}')
files=[]
for name in required:
    src=a.artifacts/name
    runtime_name=('bert.pt' if name==a.bert_stage else 'acoustic.pt') if staged else 'pipeline.pt'
    files.append({'path':f'runtime/{runtime_name}','source':src,'size':src.stat().st_size,'sha256':digest(src)})
if a.frontend:
    for src in sorted(a.frontend.rglob('*')):
        if src.is_file(): files.append({'path':f'runtime/frontend/{src.relative_to(a.frontend)}','source':src,'size':src.stat().st_size,'sha256':digest(src)})
profile=PROFILES[a.version]
def option_manifest(manifest: dict) -> None:
    if a.runtime_options:
        manifest['runtime_options_version']=1
        manifest['runtime_options']={
            'temperature':{'type':'float','min':0.0,'max':2.0,'default':1.0,'stage':'semantic_sampling'},
            'top_p':{'type':'float','min':0.0,'max':1.0,'default':1.0,'stage':'semantic_sampling'},
            'top_k':{'type':'int','min':1,'max':100,'default':10,'stage':'semantic_sampling'},
            'repetition_penalty':{'type':'float','min':0.0,'max':3.0,'default':1.35,'stage':'semantic_sampling'},
            'speed_factor':{'type':'float','min':0.25,'max':4.0,'default':1.0,'stage':'acoustic_decode'},
            'sample_steps':{'type':'int','min':1,'max':128,'default':32,'stage':'cfm'},
        }

def write_package(output: Path, artifact_role: str|None, package_files: list[dict]) -> None:
    backend_artifact = 'torchscript-cpu-staged' if staged else 'torchscript-cpu-single'
    manifest={'format':'gsvm-deploy','format_version':1,'name':a.name,'model_version':profile.id,'sample_rate':profile.sample_rate,
      'executor':'torchscript-cpu-staged' if staged else 'torchscript-cpu-single','entrypoint':'synthesize_utf8_to_pcm16','api_version':1,
      'deployable':True,'frontend_profile':a.frontend_profile,'upstream_equivalent':a.upstream_equivalent,
      'minimum_ram_mb':a.minimum_ram_mb,
      'target_soc':'any','target_soc_family':'cpu','backend_artifact':backend_artifact,
      'files':[{k:v for k,v in x.items() if k!='source'} for x in package_files]}
    if artifact_role:
        manifest['artifact_role']=artifact_role
        # Compatibility belongs to the prepared runtime ABI, not to a voice/model name. This lets
        # one verified frontend pipeline serve every voice converted for the same product profile.
        manifest['bundle_id']=f'gsvm:{profile.id}:{a.frontend_profile}:api1'
        manifest['requires_role']='model' if artifact_role=='pipeline' else 'pipeline'
    option_manifest(manifest)
    output.parent.mkdir(parents=True,exist_ok=True); partial=output.with_suffix(output.suffix+'.partial')
    with zipfile.ZipFile(partial,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        z.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
        for item in package_files: z.write(item['source'],item['path'])
    os.replace(partial,output)
    print(f'Created {output}; role={artifact_role or "combined"}; deployable={manifest["deployable"]}; files={len(package_files)}')

if a.pipeline_output is not None:
    pipeline_files=[item for item in files if item['path'].startswith('runtime/frontend/')]
    model_files=[item for item in files if not item['path'].startswith('runtime/frontend/')]
    write_package(a.pipeline_output,'pipeline',pipeline_files)
    write_package(a.model_output,'model',model_files)
elif a.model_output is not None:
    model_files=[item for item in files if not item['path'].startswith('runtime/frontend/')]
    write_package(a.model_output,'model',model_files)
else:
    write_package(a.output,None,files)
