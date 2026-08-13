#!/usr/bin/env python3
"""Build the exact G2PW Android runtime bundle without changing model precision."""
import argparse, json, os, shutil, struct, sys
from pathlib import Path


def _write_string(handle, value: str) -> None:
    encoded = value.encode("utf-8")
    if len(encoded) > 65535:
        raise ValueError("frontend string exceeds the binary ABI limit")
    handle.write(struct.pack(">H", len(encoded)))
    handle.write(encoded)


def _write_english_assets(root: Path, output: Path, symbol_ids: dict[str, int]) -> None:
    """Freeze the upstream English frontend data into Android-friendly files."""
    from text import english
    import g2p_en
    import nltk
    import numpy as np
    import wordsegment

    # Use the initialized upstream frontend, not get_dict(): en_G2p removes six known-bad
    # abbreviation pronunciations during initialization. Keep the title-case name dictionary
    # separate because upstream only consults it for an original token where str.istitle() is true.
    lexicon = dict(english._g2p.cmu)
    names = dict(english._g2p.namedict)
    with (output / "english-lexicon.tsv").open("w", encoding="utf-8", newline="\n") as handle:
        for word in sorted(lexicon):
            pronunciations = lexicon[word]
            if pronunciations:
                handle.write(f"{word}\t{' '.join(pronunciations[0])}\n")
    with (output / "english-names.tsv").open("w", encoding="utf-8", newline="\n") as handle:
        for word in sorted(names):
            pronunciations = names[word]
            if pronunciations:
                handle.write(f"{word}\t{' '.join(pronunciations[0])}\n")

    homographs = dict(english._g2p.homograph2features)
    with (output / "english-homographs.tsv").open("w", encoding="utf-8", newline="\n") as handle:
        for word in sorted(homographs):
            first, second, tag = homographs[word]
            handle.write(f"{word}\t{' '.join(first)}\t{' '.join(second)}\t{tag}\n")

    tagger_root = Path(str(nltk.data.find("taggers/averaged_perceptron_tagger_eng")))
    weights = json.loads(
        (tagger_root / "averaged_perceptron_tagger_eng.weights.json").read_text(encoding="utf-8")
    )
    tagdict = json.loads(
        (tagger_root / "averaged_perceptron_tagger_eng.tagdict.json").read_text(encoding="utf-8")
    )
    classes = sorted(
        json.loads(
            (tagger_root / "averaged_perceptron_tagger_eng.classes.json").read_text(
                encoding="utf-8"
            )
        )
    )
    class_ids = {value: index for index, value in enumerate(classes)}
    with (output / "english-tagger.bin").open("wb") as handle:
        handle.write(b"EPT1")
        handle.write(struct.pack(">H", len(classes)))
        for value in classes:
            _write_string(handle, value)
        handle.write(struct.pack(">I", len(tagdict)))
        for word, tag in sorted(tagdict.items()):
            _write_string(handle, word)
            handle.write(struct.pack(">H", class_ids[tag]))
        handle.write(struct.pack(">I", len(weights)))
        for feature, values in sorted(weights.items()):
            _write_string(handle, feature)
            handle.write(struct.pack(">H", len(values)))
            for tag, weight in sorted(values.items()):
                handle.write(struct.pack(">Hf", class_ids[tag], float(weight)))

    model_root = Path(g2p_en.__file__).resolve().parent
    variables = np.load(model_root / "checkpoint20.npz")
    array_names = (
        "enc_emb",
        "enc_w_ih",
        "enc_w_hh",
        "enc_b_ih",
        "enc_b_hh",
        "dec_emb",
        "dec_w_ih",
        "dec_w_hh",
        "dec_b_ih",
        "dec_b_hh",
        "fc_w",
        "fc_b",
    )
    with (output / "english-g2p.bin").open("wb") as handle:
        handle.write(b"EGP1")
        handle.write(struct.pack(">H", len(array_names)))
        for name in array_names:
            array = np.asarray(variables[name], dtype=">f4", order="C")
            _write_string(handle, name)
            handle.write(struct.pack(">B", array.ndim))
            for size in array.shape:
                handle.write(struct.pack(">I", size))
            handle.write(array.tobytes(order="C"))

    segment_root = Path(wordsegment.__file__).resolve().parent
    shutil.copyfile(segment_root / "unigrams.txt", output / "english-unigrams.tsv")
    shutil.copyfile(segment_root / "bigrams.txt", output / "english-bigrams.tsv")

    english_config = {
        "format": "gsv-english-mobile",
        "version": 1,
        "symbol_ids": symbol_ids,
        "punctuation": ["!", "?", "...", ",", ".", "-"],
        "wordsegment_total": 1024908267229.0,
        "wordsegment_limit": 24,
    }
    (output / "english.json").write_text(
        json.dumps(english_config, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--upstream',type=Path,default=Path('..'));p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    root=a.upstream.resolve();output=a.output.resolve();os.chdir(root);sys.path.insert(0,str(root/'GPT_SoVITS'))
    from text.g2pw.onnx_api import G2PWOnnxConverter
    from text.g2pw.g2pw import read_dict
    from text.tone_sandhi import ToneSandhi
    from text.chinese2 import must_erhua,not_erhua
    from text.zh_normalization.char_convert import t2s_dict
    from text.zh_normalization.num import COM_QUANTIFIERS
    from pypinyin import lazy_pinyin,Style
    from text.symbols2 import symbols
    from pypinyin.contrib.tone_convert import to_finals_tone3,to_initials
    c=G2PWOnnxConverter('GPT_SoVITS/text/G2PWModel',style='pinyin',model_source='GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large')
    tokenizer=json.loads((root/'GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/tokenizer.json').read_text())
    all_chars=sorted(set(c.char_bopomofo_dict)|set(c.monophonic_chars_dict)|set(c.chars));default_pinyin={x:lazy_pinyin(x,neutral_tone_with_five=True,style=Style.TONE3)[0] for x in all_chars}
    tone=ToneSandhi()
    symbol_ids={value:index for index,value in enumerate(symbols)};pinyin_map={line.split('\t')[0]:line.rstrip().split('\t')[1] for line in (root/'GPT_SoVITS/text/opencpop-strict.txt').read_text().splitlines()}
    def phone_ids(pinyin):
        initial=to_initials(pinyin);final=to_finals_tone3(pinyin,neutral_tone_with_five=True);bare,tone=final[:-1],final[-1]
        if initial:key=initial+{'uei':'ui','iou':'iu','uen':'un'}.get(bare,bare)
        else:
            key={'ing':'ying','i':'yi','in':'yin','u':'wu'}.get(bare,bare)
            if key==bare and bare and bare[0] in {'v','e','i','u'}:key={'v':'yu','e':'e','i':'y','u':'w'}[bare[0]]+bare[1:]
        if key not in pinyin_map:return [symbol_ids['UNK']]
        first,second=pinyin_map[key].split();return [symbol_ids[first],symbol_ids[second+tone]]
    pronunciations=set(default_pinyin.values())
    for values in read_dict().values():pronunciations.update(values)
    for label in c.labels:
        base=c.bopomofo_convert_dict.get(label[:-1]);
        if base:pronunciations.add(base+label[-1])
    pronunciations.update({value[:-1]+tone for value in list(pronunciations) if value and value[-1:] in '12345' for tone in '12345'})
    pinyin_phone_ids={}
    for value in sorted(pronunciations):
        if value and value[-1:] in '12345':
            try:pinyin_phone_ids[value]=phone_ids(value)
            except (IndexError,KeyError):pinyin_phone_ids[value]=[symbol_ids['UNK']]
    value={'format':'gsv-g2pw-mobile','version':1,'labels':c.labels,'chars':c.chars,'char2phonemes':c.char2phonemes,
           'polyphonic_chars':sorted(c.polyphonic_chars_new),'monophonic':c.monophonic_chars_dict,
           'char_bopomofo':c.char_bopomofo_dict,'bopomofo_to_pinyin':c.bopomofo_convert_dict,
           'default_pinyin':default_pinyin,'pronunciation_overrides':read_dict(),
           'traditional_to_simplified':{k:v for k,v in t2s_dict.items() if k!=v},
           'common_quantifiers_pattern':COM_QUANTIFIERS,
           'pinyin_phone_ids':pinyin_phone_ids,'punctuation_phone_ids':{x:symbol_ids[x] for x in ',.!?;:' if x in symbol_ids},
           'must_neutral_tone_words':sorted(tone.must_neural_tone_words),'must_not_neutral_tone_words':sorted(tone.must_not_neural_tone_words),
           'must_erhua':sorted(must_erhua),'not_erhua':sorted(not_erhua),
           'vocab':tokenizer['model']['vocab'],'unk_token':tokenizer['model']['unk_token'],'continuing_prefix':tokenizer['model']['continuing_subword_prefix'],
           'max_input_chars_per_word':tokenizer['model']['max_input_chars_per_word'],'use_mask':bool(c.config.use_mask)}
    output.mkdir(parents=True,exist_ok=True);partial=output/'frontend.json.partial';partial.write_text(json.dumps(value,ensure_ascii=False,separators=(',',':')),encoding='utf-8');os.replace(partial,output/'frontend.json')
    shutil.copyfile(root/'GPT_SoVITS/text/G2PWModel/g2pW.onnx',output/'g2pW.onnx')
    for name in ('polyphonic.rep','polyphonic-fix.rep'):
        shutil.copyfile(root/'GPT_SoVITS/text/g2pw'/name,output/name)
    import jieba_fast
    jieba_root=Path(jieba_fast.__file__).resolve().parent
    shutil.copyfile(jieba_root/'dict.txt',output/'jieba-dict.txt')
    from jieba_fast.posseg import char_state_tab_P
    from jieba_fast.posseg import start_P as pos_start,trans_P as pos_trans,emit_P as pos_emit
    (output/'jieba-hmm.tsv').unlink(missing_ok=True)
    states=sorted(pos_start);state_ids={state:i for i,state in enumerate(states)}
    all_chars=sorted(set(char_state_tab_P)|set().union(*(values.keys() for values in pos_emit.values())))
    with (output/'jieba-pos-hmm.bin').open('wb') as handle:
        handle.write(b'JPH1');handle.write(struct.pack('>H',len(states)))
        for state in states:
            boundary,pos=state;encoded=pos.encode();handle.write(boundary.encode());handle.write(struct.pack('>B',len(encoded)));handle.write(encoded);handle.write(struct.pack('>d',pos_start[state]))
        for state in states:
            edges=pos_trans[state];handle.write(struct.pack('>H',len(edges)))
            for target,score in edges.items():handle.write(struct.pack('>Hd',state_ids[target],score))
        handle.write(struct.pack('>I',len(all_chars)))
        for char in all_chars:
            allowed=char_state_tab_P.get(char,states);handle.write(struct.pack('>I',ord(char)));handle.write(struct.pack('>H',len(allowed)))
            for state in allowed:handle.write(struct.pack('>Hd',state_ids[state],pos_emit[state].get(char,-3.14e100)))
    _write_english_assets(root, output, symbol_ids)
    print(f'Created {output}; config={(output/"frontend.json").stat().st_size}; model={(output/"g2pW.onnx").stat().st_size}')

if __name__=='__main__':main()
