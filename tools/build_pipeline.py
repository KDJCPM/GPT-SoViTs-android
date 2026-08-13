#!/usr/bin/env python3
"""Build a single UTF-8-bytes-to-PCM TorchScript CPU pipeline.

All model-specific preprocessing is frozen here. Android only converts String to UTF-8 bytes and
calls forward(bytes, language, seed).
"""
import argparse, ast, contextlib, io, json, os, sys
from pathlib import Path
from typing import Dict, List, Tuple
import torch


class Utf8TtsPipeline(torch.nn.Module):
    phone_table: Dict[int, List[int]]
    token_table: Dict[int, int]
    chinese_table: Dict[int, bool]
    punctuation_map: Dict[int, int]

    def __init__(self, bert, acoustic, phone_table, token_table, chinese_table, phrase_trie):
        super().__init__(); self.bert=bert; self.acoustic=acoustic
        self.phone_table=phone_table; self.token_table=token_table; self.chinese_table=chinese_table
        self.punctuation_map={0xFF0C:44,0x3002:46,0xFF01:33,0xFF1F:63,0xFF1A:44,0xFF1B:44,10:46,0x3001:44}
        for name, value in phrase_trie.items():
            self.register_buffer(name, torch.tensor(value, dtype=torch.int32))

    def find_child(self, node: int, cp: int) -> int:
        edge=int(self.trie_heads[node].item())
        while edge>=0:
            if int(self.trie_chars[edge].item())==cp:
                return int(self.trie_children[edge].item())
            edge=int(self.trie_next[edge].item())
        return -1

    def append_character(self, cp: int, phones: List[int], word2ph: List[int], chinese_phone: List[float]) -> None:
        if cp in self.phone_table:
            values=self.phone_table[cp]; phones.extend(values); word2ph.append(len(values))
            is_zh=self.chinese_table.get(cp,False)
            for _ in values: chinese_phone.append(1.0 if is_zh else 0.0)

    def decode_utf8(self, data: torch.Tensor) -> List[int]:
        values=data.to(torch.int64).reshape(-1); result=torch.jit.annotate(List[int],[])
        i=0; n=values.numel()
        while i<n:
            b0=int(values[i].item()) & 255
            if b0<128: result.append(b0); i+=1
            elif b0<224 and i+1<n:
                result.append(((b0&31)<<6)|(int(values[i+1].item())&63)); i+=2
            elif b0<240 and i+2<n:
                result.append(((b0&15)<<12)|((int(values[i+1].item())&63)<<6)|(int(values[i+2].item())&63)); i+=3
            elif i+3<n:
                result.append(((b0&7)<<18)|((int(values[i+1].item())&63)<<12)|((int(values[i+2].item())&63)<<6)|(int(values[i+3].item())&63)); i+=4
            else: i+=1
        return result

    def forward(self, utf8: torch.Tensor, language: str="auto", seed: int=-1,
                temperature: float=1.0, top_p: float=1.0, top_k: int=10,
                repetition_penalty: float=1.35, speed_factor: float=1.0) -> Tuple[int,torch.Tensor]:
        codepoints=self.decode_utf8(utf8)
        phones=torch.jit.annotate(List[int],[]); word2ph=torch.jit.annotate(List[int],[])
        token_ids=torch.jit.annotate(List[int],[101]); chinese_phone=torch.jit.annotate(List[float],[])
        normalized=torch.jit.annotate(List[int],[])
        for raw in codepoints: normalized.append(self.punctuation_map.get(raw,raw))
        pos=0
        while pos<len(normalized):
            node=0; cursor=pos; best_node=-1; best_end=pos
            while cursor<len(normalized):
                node=self.find_child(node,normalized[cursor])
                if node<0: break
                cursor+=1
                if int(self.trie_phone_offsets[node].item())>=0:
                    best_node=node; best_end=cursor
            if best_node>=0:
                phone_start=int(self.trie_phone_offsets[best_node].item()); phone_len=int(self.trie_phone_lengths[best_node].item())
                count_start=int(self.trie_count_offsets[best_node].item()); count_len=int(self.trie_count_lengths[best_node].item())
                for j in range(phone_len): phones.append(int(self.trie_phones[phone_start+j].item()))
                for j in range(count_len):
                    count=int(self.trie_counts[count_start+j].item()); word2ph.append(count)
                    for _ in range(count): chinese_phone.append(1.0)
                for j in range(pos,best_end): token_ids.append(self.token_table.get(normalized[j],100))
                pos=best_end
            else:
                cp=normalized[pos]
                if cp in self.phone_table:
                    self.append_character(cp,phones,word2ph,chinese_phone)
                    token_ids.append(self.token_table.get(cp,100))
                pos+=1
        if len(phones)==0:
            raise RuntimeError("text has no supported symbols")
        token_ids.append(102)
        ids=torch.tensor(token_ids,dtype=torch.int64).unsqueeze(0)
        mask=torch.ones_like(ids); types=torch.zeros_like(ids); counts=torch.tensor(word2ph,dtype=torch.int32)
        features=self.bert(ids,mask,types,counts)
        feature_mask=torch.tensor(chinese_phone,dtype=features.dtype).unsqueeze(1)
        features=features*feature_mask
        phone_tensor=torch.tensor(phones,dtype=torch.int64).unsqueeze(0)
        if seed>=0: torch.manual_seed(seed)
        return self.acoustic(phone_tensor,features,temperature,top_k,top_p,repetition_penalty,speed_factor,seed)

    @torch.jit.export
    def synthesize_preprocessed(self, phone_ids: torch.Tensor, token_ids: torch.Tensor,
                                word2ph: torch.Tensor, chinese_mask: torch.Tensor,
                                seed: int=-1) -> Tuple[int,torch.Tensor]:
        return self.synthesize_preprocessed_options(
            phone_ids, token_ids, word2ph, chinese_mask, seed,
            1.0, 1.0, 10, 1.35, 1.0, 32,
        )

    @torch.jit.export
    def synthesize_preprocessed_options(self, phone_ids: torch.Tensor, token_ids: torch.Tensor,
                                        word2ph: torch.Tensor, chinese_mask: torch.Tensor,
                                        seed: int=-1, temperature: float=1.0, top_p: float=1.0,
                                        top_k: int=10, repetition_penalty: float=1.35,
                                        speed_factor: float=1.0, sample_steps: int=32) -> Tuple[int,torch.Tensor]:
        ids=token_ids.to(torch.int64).reshape(1,-1);mask=torch.ones_like(ids);types=torch.zeros_like(ids)
        features=self.bert(ids,mask,types,word2ph.to(torch.int32).reshape(-1))
        features=features*chinese_mask.to(features.dtype).reshape(-1,1)
        if seed>=0:torch.manual_seed(seed)
        return self.acoustic(phone_ids.to(torch.int64).reshape(1,-1),features,temperature,top_k,top_p,repetition_penalty,speed_factor,seed)

    @torch.jit.export
    def synthesize_reference_preprocessed_options(
        self,
        phone_ids: torch.Tensor,
        token_ids: torch.Tensor,
        word2ph: torch.Tensor,
        chinese_mask: torch.Tensor,
        prompt_phone_ids: torch.Tensor,
        prompt_token_ids: torch.Tensor,
        prompt_word2ph: torch.Tensor,
        prompt_chinese_mask: torch.Tensor,
        reference_pcm_16k: torch.Tensor,
        reference_pcm_32k: torch.Tensor,
        seed: int=-1,
        temperature: float=1.0,
        top_p: float=1.0,
        top_k: int=10,
        repetition_penalty: float=1.35,
        speed_factor: float=1.0,
        sample_steps: int=32,
    ) -> Tuple[int,torch.Tensor]:
        ids=token_ids.to(torch.int64).reshape(1,-1)
        features=self.bert(ids,torch.ones_like(ids),torch.zeros_like(ids),word2ph.to(torch.int32).reshape(-1))
        features=features*chinese_mask.to(features.dtype).reshape(-1,1)
        prompt_ids=prompt_token_ids.to(torch.int64).reshape(1,-1)
        prompt_features=self.bert(
            prompt_ids,torch.ones_like(prompt_ids),torch.zeros_like(prompt_ids),
            prompt_word2ph.to(torch.int32).reshape(-1),
        )
        prompt_features=prompt_features*prompt_chinese_mask.to(prompt_features.dtype).reshape(-1,1)
        if seed>=0:torch.manual_seed(seed)
        return self.acoustic.synthesize_reference_options(
            phone_ids.to(torch.int64).reshape(1,-1),features,
            reference_pcm_16k.float().reshape(1,-1),reference_pcm_32k.float().reshape(1,-1),
            prompt_phone_ids.to(torch.int64).reshape(1,-1),prompt_features,
            temperature,top_k,top_p,repetition_penalty,speed_factor,sample_steps,seed,
        )


def make_tables(upstream: Path, cache: Path) -> tuple[dict[int,list[int]],dict[int,int],dict[int,bool]]:
    if cache.is_file():
        value=json.loads(cache.read_text()); return ({int(k):v for k,v in value['phones'].items()},{int(k):v for k,v in value['tokens'].items()},{int(k):v for k,v in value['chinese'].items()})
    sys.path.insert(0,str(upstream/'GPT_SoVITS')); os.environ['is_g2pw']='False'; os.environ['version']='v2'
    os.chdir(upstream)
    from transformers import AutoTokenizer
    import text.chinese2 as chinese2; chinese2.is_g2pw=False
    from text.cleaner import clean_text
    from text import cleaned_text_to_sequence
    tokenizer=AutoTokenizer.from_pretrained(upstream/'GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large')
    phones={}; tokens={}; chinese={}
    cps=list(range(0x4E00,0x9FA6))+list(range(32,127))+[0xFF0C,0x3002,0xFF01,0xFF1F,0xFF1A,0xFF1B,0x3001]
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        for cp in cps:
            char=chr(cp); lang='zh' if cp>=0x4E00 or cp>127 else 'en'
            try:
                ph,_,_=clean_text(char,lang,'v2'); ids=cleaned_text_to_sequence(ph,'v2')
                if ids: phones[cp]=ids; tokens[cp]=int(tokenizer.convert_tokens_to_ids(char)); chinese[cp]=(lang=='zh')
            except Exception: pass
    cache.parent.mkdir(parents=True,exist_ok=True); cache.write_text(json.dumps({'phones':phones,'tokens':tokens,'chinese':chinese},ensure_ascii=False))
    return phones,tokens,chinese


def pinyin_to_phone_ids(pinyin: str, symbol_ids: Dict[str, int], pinyin_map: Dict[str, str]) -> List[int]:
    """Apply the same pinyin post-processing as chinese2._g2p without importing G2PW."""
    from pypinyin.contrib.tone_convert import to_finals_tone3, to_initials
    initial=to_initials(pinyin); final=to_finals_tone3(pinyin, neutral_tone_with_five=True)
    if not final or final[-1] not in "12345": raise ValueError(f"invalid toned pinyin: {pinyin}")
    bare=final[:-1]; tone=final[-1]
    if initial:
        key=initial+{"uei":"ui","iou":"iu","uen":"un"}.get(bare,bare)
    else:
        key={"ing":"ying","i":"yi","in":"yin","u":"wu"}.get(bare,bare)
        if key==bare and bare and bare[0] in {"v":"yu","e":"e","i":"y","u":"w"}:
            key={"v":"yu","e":"e","i":"y","u":"w"}[bare[0]]+bare[1:]
    first, second=pinyin_map[key].split()
    return [symbol_ids[first],symbol_ids[second+tone]]


def apply_tone_sandhi(phrase: str, readings: List[str], modifier, psg) -> List[str]:
    """Freeze upstream word segmentation and tone changes into a phrase pronunciation."""
    from pypinyin.contrib.tone_convert import to_finals_tone3, to_initials
    segments=modifier.pre_merge_for_modify(psg.lcut(phrase)); result: List[str]=[]; offset=0
    for word,pos in segments:
        part=readings[offset:offset+len(word)]
        if len(part)!=len(word): return readings
        initials=[to_initials(value) for value in part]
        finals=[to_finals_tone3(value,neutral_tone_with_five=True) for value in part]
        finals=modifier.modified_tone(word,pos,finals)
        result.extend(initial+final for initial,final in zip(initials,finals)); offset+=len(word)
    return result if offset==len(readings) else readings


def make_phrase_trie(upstream: Path, exact_cache: Path|None=None) -> Dict[str, List[int]]:
    """Compile upstream pronunciation overrides into a compact longest-match trie."""
    sys.path.insert(0,str(upstream/'GPT_SoVITS'))
    from text.symbols2 import symbols
    from text.tone_sandhi import ToneSandhi
    import jieba_fast.posseg as psg
    symbol_ids={value:index for index,value in enumerate(symbols)}
    modifier=ToneSandhi()
    pinyin_map={line.split('\t')[0]:line.rstrip().split('\t')[1] for line in (upstream/'GPT_SoVITS/text/opencpop-strict.txt').read_text().splitlines()}
    phrases: Dict[str, List[str]]={}
    for filename in ('polyphonic.rep','polyphonic-fix.rep'):
        for line_number,line in enumerate((upstream/'GPT_SoVITS/text/g2pw'/filename).read_text(encoding='utf-8').splitlines(),1):
            if not line.strip(): continue
            key,value=line.split(':',1); readings=ast.literal_eval(value.strip())
            if len(key.strip())!=len(readings): raise ValueError(f'{filename}:{line_number}: character/reading count mismatch')
            phrases[key.strip()]=readings

    # Node zero is the root. Children are stored as linked edge lists so lookup stays scriptable.
    children: List[Dict[int,int]]=[{}]; terminals: List[Tuple[List[int],List[int]]|None]=[None]
    rejected=0
    exact={}
    if exact_cache is not None and exact_cache.is_file():
        exact=json.loads(exact_cache.read_text(encoding='utf-8')).get('entries',{})
    for phrase,readings in phrases.items():
        node=0; phrase_phones: List[int]=[]; counts: List[int]=[]
        try:
            if phrase in exact:
                phrase_phones=[int(x) for x in exact[phrase]['phones']]; counts=[int(x) for x in exact[phrase]['word2ph']]
            else:
                for reading in apply_tone_sandhi(phrase,readings,modifier,psg):
                    ids=pinyin_to_phone_ids(reading,symbol_ids,pinyin_map); phrase_phones.extend(ids); counts.append(len(ids))
        except (KeyError,ValueError):
            rejected+=1; continue
        for cp in map(ord,phrase):
            child=children[node].get(cp)
            if child is None:
                child=len(children); children[node][cp]=child; children.append({}); terminals.append(None)
            node=child
        terminals[node]=(phrase_phones,counts)
    heads=[-1]*len(children); chars=[]; child_nodes=[]; next_edges=[]
    for node,entries in enumerate(children):
        for cp,child in entries.items():
            edge=len(chars); chars.append(cp); child_nodes.append(child); next_edges.append(heads[node]); heads[node]=edge
    phone_offsets=[]; phone_lengths=[]; count_offsets=[]; count_lengths=[]; flat_phones=[]; flat_counts=[]
    for terminal in terminals:
        if terminal is None:
            phone_offsets.append(-1); phone_lengths.append(0); count_offsets.append(-1); count_lengths.append(0)
        else:
            values,counts=terminal; phone_offsets.append(len(flat_phones)); phone_lengths.append(len(values)); flat_phones.extend(values)
            count_offsets.append(len(flat_counts)); count_lengths.append(len(counts)); flat_counts.extend(counts)
    print(f'Compiled pronunciation trie: phrases={len(phrases)-rejected}, exact={len(exact)}, rejected={rejected}, nodes={len(children)}, edges={len(chars)}')
    return {'trie_heads':heads,'trie_chars':chars,'trie_children':child_nodes,'trie_next':next_edges,
            'trie_phone_offsets':phone_offsets,'trie_phone_lengths':phone_lengths,'trie_count_offsets':count_offsets,
            'trie_count_lengths':count_lengths,'trie_phones':flat_phones,'trie_counts':flat_counts}


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--artifacts',required=True,type=Path); p.add_argument('--upstream',type=Path,default=Path('..')); p.add_argument('--output',required=True,type=Path); p.add_argument('--bert',type=Path); p.add_argument('--acoustic',type=Path); p.add_argument('--tables-cache',type=Path,default=Path('build/frontend-v2.json')); p.add_argument('--phrases-cache',type=Path,default=Path('build/frontend-phrases-v2.json')); a=p.parse_args()
    artifacts=a.artifacts.resolve(); output=a.output.resolve(); cache=a.tables_cache.resolve(); upstream=a.upstream.resolve()
    phones,tokens,chinese=make_tables(upstream,cache); phrase_trie=make_phrase_trie(upstream,a.phrases_cache.resolve())
    bert_path=a.bert.resolve() if a.bert else artifacts/'bert_model.pt'
    acoustic_path=a.acoustic.resolve() if a.acoustic else artifacts/'pipeline_core.pt'
    bert=torch.jit.load(str(bert_path),map_location='cpu'); acoustic=torch.jit.load(str(acoustic_path),map_location='cpu')
    module=torch.jit.script(Utf8TtsPipeline(bert,acoustic,phones,tokens,chinese,phrase_trie).eval())
    module=torch.jit.freeze(module,preserved_attrs=[
        'synthesize_preprocessed', 'synthesize_preprocessed_options',
        'synthesize_reference_preprocessed_options',
    ]); module.save(str(output))
    print(f'Created {output} ({output.stat().st_size} bytes); symbols={len(phones)}')

if __name__=='__main__': main()
