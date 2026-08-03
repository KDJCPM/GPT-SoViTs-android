#!/usr/bin/env python3
"""Precompute upstream G2PW/jieba/tone-sandhi results for the mobile phrase trie.

This intentionally makes conversion expensive so the Android runtime only performs a compact
longest-prefix lookup. The output contains final phone IDs, never raw pinyin guesses.
"""
import argparse, ast, hashlib, json, os, sys, time
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream',type=Path,default=Path('..'))
    p.add_argument('--output',type=Path,default=Path('build/frontend-phrases-v2.json'))
    a=p.parse_args(); upstream=a.upstream.resolve(); output=a.output.resolve()
    sources=[upstream/'GPT_SoVITS/text/g2pw/polyphonic.rep',upstream/'GPT_SoVITS/text/g2pw/polyphonic-fix.rep']
    phrases={}
    for source in sources:
        for line in source.read_text(encoding='utf-8').splitlines():
            if line.strip():
                key,value=line.split(':',1); phrases[key.strip()]=ast.literal_eval(value.strip())
    os.chdir(upstream); sys.path.insert(0,str(upstream/'GPT_SoVITS'))
    os.environ['version']='v2'; os.environ['is_half']='False'
    os.environ['bert_path']=str(upstream/'GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large')
    from text.cleaner import clean_text
    from text import cleaned_text_to_sequence
    entries={}; rejected={}; started=time.time()
    for index,phrase in enumerate(phrases,1):
        try:
            phones,word2ph,normalized=clean_text(phrase,'zh','v2')
            if normalized!=phrase or word2ph is None or len(word2ph)!=len(phrase):
                raise ValueError(f'normalization/alignment changed to {normalized!r}')
            ids=cleaned_text_to_sequence(phones,'v2')
            if sum(word2ph)!=len(ids): raise ValueError('phone alignment mismatch')
            entries[phrase]={'phones':ids,'word2ph':word2ph}
        except Exception as error:
            rejected[phrase]=str(error)
        if index%1000==0: print(f'{index}/{len(phrases)} elapsed={time.time()-started:.1f}s',flush=True)
    value={'format':'gsv-mobile-phrase-frontend','version':1,'upstream':{x.name:digest(x) for x in sources},
           'entries':entries,'rejected':rejected}
    output.parent.mkdir(parents=True,exist_ok=True)
    partial=output.with_suffix(output.suffix+'.partial'); partial.write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8'); os.replace(partial,output)
    print(f'Created {output}; entries={len(entries)} rejected={len(rejected)} elapsed={time.time()-started:.1f}s')


if __name__=='__main__': main()
