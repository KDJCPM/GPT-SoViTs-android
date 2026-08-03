package ai.gsv.mobile

import org.json.JSONObject
import java.io.Closeable
import java.io.File

/** Full contextual G2PW classifier input preparation matching GPT-SoVITS text.g2pw.dataset. */
class FullZhFrontend(
    frontendDir: File,
    qnnTarget: QualcommTargetSoc? = null,
    modelFile: File = File(frontendDir, "g2pW.onnx"),
    staticRows: Int? = null,
    staticSequenceLength: Int? = null,
    htpFp16Precision: Boolean = false,
    htpGraphOptimizationMode: String? = null,
    debugLogProbabilities: Boolean = false,
    rawLogits: Boolean = false,
) : Closeable {
    private val config = JSONObject(File(frontendDir, "frontend.json").readText())
    private val labels = config.getJSONArray("labels").let { a -> List(a.length()) { a.getString(it) } }
    private val chars = config.getJSONArray("chars").let { a -> List(a.length()) { a.getString(it) } }
    private val charIndex = chars.withIndex().associate { it.value to it.index }
    private val polyphonic = config.getJSONArray("polyphonic_chars").let { a -> HashSet<String>(a.length()).apply { repeat(a.length()) { add(a.getString(it)) } } }
    private val char2phonemes = config.getJSONObject("char2phonemes")
    private val bopomofoToPinyin = config.getJSONObject("bopomofo_to_pinyin")
    private val defaults = config.getJSONObject("default_pinyin")
    private val overrides = config.getJSONObject("pronunciation_overrides")
    private val mustNeutral = config.getJSONArray("must_neutral_tone_words").let { a -> HashSet<String>().apply { repeat(a.length()){add(a.getString(it))} } }
    private val mustNotNeutral = config.getJSONArray("must_not_neutral_tone_words").let { a -> HashSet<String>().apply { repeat(a.length()){add(a.getString(it))} } }
    private val mustErhua = config.getJSONArray("must_erhua").let { a -> HashSet<String>().apply { repeat(a.length()){add(a.getString(it))} } }
    private val notErhua = config.getJSONArray("not_erhua").let { a -> HashSet<String>().apply { repeat(a.length()){add(a.getString(it))} } }
    private val vocab = config.getJSONObject("vocab").let { source -> HashMap<String, Int>(source.length()).apply {
        source.keys().forEach { put(it, source.getInt(it)) }
    } }
    private val unknownId = requireNotNull(vocab[config.getString("unk_token")])
    private val prefix = config.getString("continuing_prefix")
    private val maxWordLength = config.getInt("max_input_chars_per_word")
    private val pinyinPhones=config.getJSONObject("pinyin_phone_ids")
    private val punctuationPhones=config.getJSONObject("punctuation_phone_ids")
    private val normalizer=ZhTextNormalizer(config)
    private val model = G2pwOnnx(
        modelFile,
        qnnTarget,
        staticRows = staticRows,
        staticSequenceLength = staticSequenceLength,
        htpFp16Precision = htpFp16Precision,
        htpGraphOptimizationMode = htpGraphOptimizationMode,
        debugLogProbabilities = debugLogProbabilities,
        rawLogits = rawLogits,
    )
    private val jieba = JiebaExact(File(frontendDir,"jieba-dict.txt"),File(frontendDir,"jieba-pos-hmm.bin"))

    data class Prediction(val codePointIndex: Int, val pinyin: String, val confidence: Float)
    data class Prepared(val normalized:String,val phoneIds:LongArray,val tokenIds:LongArray,val word2ph:IntArray,val chineseMask:FloatArray)
    private data class Tokens(val values: List<String>, val textToToken: IntArray)

    val qnnExecutionStats: QnnExecutionStats?
        get() = model.executionStats

    fun finalizeQnnProfiling() = model.finalizeProfiling()

    fun predict(text: String): List<Prediction> {
        val codepoints=text.codePoints().toArray();val tokenized=tokenizeAndMap(codepoints)
        if(tokenized.values.size+2>512){val output=ArrayList<Prediction>();var start=0
            while(start<codepoints.size){var end=minOf(start+480,codepoints.size);if(end<codepoints.size){for(i in end-1 downTo start+1)if(String(Character.toChars(codepoints[i])) in ",.!?;:…"){end=i+1;break}}
                val chunk=String(codepoints,start,end-start);predict(chunk).forEach{output.add(it.copy(codePointIndex=it.codePointIndex+start))};start=end};return output}
        val query=codepoints.indices.filter { polyphonic.contains(String(Character.toChars(codepoints[it]))) }
        if (query.isEmpty()) return emptyList()
        val sequence=tokenized.values.size+2; val rows=query.size
        val ids=LongArray(rows*sequence); val types=LongArray(rows*sequence); val attention=LongArray(rows*sequence){1}
        val mask=FloatArray(rows*labels.size); val charIds=LongArray(rows); val positions=LongArray(rows)
        val oneIds=longArrayOf(requireNotNull(vocab["[CLS]"]).toLong(),*tokenized.values.map { (vocab[it]?:unknownId).toLong() }.toLongArray(),requireNotNull(vocab["[SEP]"]).toLong())
        for ((row,index) in query.withIndex()) {
            oneIds.copyInto(ids,row*sequence); val char=String(Character.toChars(codepoints[index]));charIds[row]=requireNotNull(charIndex[char]).toLong();positions[row]=(tokenized.textToToken[index]+1).toLong()
            val allowed=char2phonemes.getJSONArray(char);repeat(allowed.length()){mask[row*labels.size+allowed.getInt(it)]=1f}
        }
        val probabilities=model.probabilities(G2pwInputs(rows,sequence,labels.size,ids,types,attention,mask,charIds,positions))
        return query.indices.map { row ->
            var best=0;for(i in 1 until labels.size)if(probabilities[row][i]>probabilities[row][best])best=i
            val bopomofo=labels[best];val tone=bopomofo.last();val pinyin=bopomofoToPinyin.optString(bopomofo.dropLast(1),"")+tone
            Prediction(query[row],pinyin,probabilities[row][best])
        }
    }

    /** Contextual pinyin after upstream-style word correction and tone sandhi. */
    fun finalPinyin(text:String):List<String>{
        val cps=text.codePoints().toArray();val values=MutableList(cps.size){index->defaults.optString(String(Character.toChars(cps[index])),String(Character.toChars(cps[index])))}
        TimingContext.measure("frontend.g2pw.predict") {
            predict(text)
        }.forEach{values[it.codePointIndex]=it.pinyin}
        val words=TimingContext.measure("frontend.segment") { preMerge(jieba.cut(text)) }
        return TimingContext.measure("frontend.tone_correction") {
            val result=ArrayList<String>();var offset=0
            for(word in words){
                val length=word.text.codePointCount(0,word.text.length);val raw=values.subList(offset,(offset+length).coerceAtMost(values.size)).toMutableList();val fixed=overrides.optJSONArray(word.text)
                if(fixed!=null&&fixed.length()==length){raw.clear();repeat(fixed.length()){raw.add(fixed.getString(it))}}
                applyTone(word.text,word.pos,raw);result.addAll(raw);offset+=length
            }
            result
        }
    }

    fun prepare(original:String):Prepared{
        val normalized=TimingContext.measure("frontend.normalize") { normalizer.normalize(original) }
        val pinyin=TimingContext.measure("frontend.final_pinyin") { finalPinyin(normalized) }
        val cps=normalized.codePoints().toArray();require(pinyin.size==cps.size){"frontend alignment mismatch"}
        return TimingContext.measure("frontend.phone_pack") {
            val phones=ArrayList<Long>();val counts=IntArray(cps.size);val mask=ArrayList<Float>();val tokens=LongArray(cps.size+2);tokens[0]=requireNotNull(vocab["[CLS]"]).toLong();tokens[tokens.lastIndex]=requireNotNull(vocab["[SEP]"]).toLong()
            for(i in cps.indices){val char=String(Character.toChars(cps[i]));tokens[i+1]=(vocab[char]?:unknownId).toLong()
                if(punctuationPhones.has(char)){counts[i]=1;phones.add(punctuationPhones.getLong(char));mask.add(1f)}else{val source=pinyinPhones.optJSONArray(pinyin[i])?:error("unsupported pinyin ${pinyin[i]}");counts[i]=source.length();repeat(source.length()){phones.add(source.getLong(it));mask.add(1f)}}
            }
            Prepared(normalized,phones.toLongArray(),tokens,counts,mask.toFloatArray())
        }
    }

    fun debugSegments(text:String):List<ZhWord> = preMerge(jieba.cut(text))

    private fun preMerge(input:MutableList<ZhWord>):MutableList<ZhWord>{
        var seg=mutableListOf<ZhWord>();var pending=false
        for(item in input){if(pending){seg.add(ZhWord("不"+item.text,item.pos));pending=false}else if(item.text=="不")pending=true else seg.add(item)};if(pending)seg.add(ZhWord("不","d"))
        val yi=mutableListOf<ZhWord>();var i=0
        while(i<seg.size){if(seg[i].text=="一"&&yi.isNotEmpty()&&i+1<seg.size&&yi.last().text==seg[i+1].text&&yi.last().pos=="v"&&seg[i+1].pos=="v"){yi.last().text += "一"+seg[i+1].text;i+=2}else{if(yi.isNotEmpty()&&yi.last().text=="一")yi.last().text+=seg[i].text else yi.add(seg[i]);i++}}
        seg=mutableListOf();for(item in yi){if(seg.isNotEmpty()&&seg.last().text==item.text)seg.last().text+=item.text else seg.add(item)}
        seg=mergeThree(seg,false);seg=mergeThree(seg,true)
        val er=mutableListOf<ZhWord>();for(item in seg){if(item.text=="儿"&&er.isNotEmpty()&&er.last().text!="#")er.last().text+="儿" else er.add(item)}
        return er
    }

    private fun mergeThree(input:MutableList<ZhWord>,boundaryOnly:Boolean):MutableList<ZhWord>{
        val output=mutableListOf<ZhWord>();var previousMerged=false
        for((index,item) in input.withIndex()){
            val current=defaultFinals(item.text);val previous=if(index>0)defaultFinals(input[index-1].text) else emptyList();val match=if(boundaryOnly)previous.isNotEmpty()&&current.isNotEmpty()&&previous.last().endsWith('3')&&current.first().endsWith('3') else previous.isNotEmpty()&&previous.all{it.endsWith('3')}&&current.all{it.endsWith('3')}
            val redup=input.getOrNull(index-1)?.text?.let{it.length==2&&it[0]==it[1]}?:false
            if(index>0&&match&&!previousMerged&&!redup&&input[index-1].text.length+item.text.length<=3){output.last().text+=item.text;previousMerged=true}else{output.add(item);previousMerged=false}
        };return output
    }

    private fun defaultFinals(word:String)=word.codePoints().toArray().map{defaults.optString(String(Character.toChars(it)),"")}
    private fun tone(value:String,tone:Char)=if(value.isNotEmpty()&&value.last().isDigit())value.dropLast(1)+tone else value
    private fun applyTone(word:String,pos:String,values:MutableList<String>){
        // The preposition 为 is canonically wei2. This also removes a low-margin ONNX/PyTorch
        // classifier divergence observed in long numeric sentences while preserving noun/verb 为.
        if(word=="为"&&pos=="p"&&values.isNotEmpty())values[0]="wei2"
        if(word.length==3&&word[1]=='不')values[1]=tone(values[1],'5') else for(i in word.indices)if(word[i]=='不'&&i+1<values.size&&values[i+1].endsWith('4'))values[i]=tone(values[i],'2')
        if(word.contains('一')&&!word.filter{it!='一'}.all{it.isDigit()}){if(word.length==3&&word[1]=='一'&&word[0]==word[2])values[1]=tone(values[1],'5') else if(word.startsWith("第一")&&values.size>1)values[1]=tone(values[1],'1') else for(i in word.indices)if(word[i]=='一'&&i+1<values.size)values[i]=tone(values[i],if(values[i+1].endsWith('4'))'2' else '4')}
        for(i in 1 until word.length)if(word[i]==word[i-1]&&pos.firstOrNull() in setOf('n','v','a')&&!mustNotNeutral.contains(word))values[i]=tone(values[i],'5')
        if(word.isNotEmpty()&&word.last() in "吧呢哈啊呐噻嘛吖嗨哦哒额滴哩哟喽啰耶喔诶的地得")values[values.lastIndex]=tone(values.last(),'5')
        else if(word.length==1&&word in "了着过"&&pos in setOf("ul","uz","ug"))values[0]=tone(values[0],'5')
        else if(word.length>1&&word.last() in "们子"&&pos in setOf("r","n")&&!mustNotNeutral.contains(word))values[values.lastIndex]=tone(values.last(),'5')
        else if(word.length>1&&word.last() in "上下里"&&pos in setOf("s","l","f"))values[values.lastIndex]=tone(values.last(),'5')
        else if(word.length>1&&word.last() in "来去"&&word[word.lastIndex-1] in "上下进出回过起开")values[values.lastIndex]=tone(values.last(),'5')
        else {
            val ge=word.indexOf('个');if(ge>=1&&(word[ge-1].isDigit()||word[ge-1] in "零一二三四五六七八九十几有两半多各整每做是"))values[ge]=tone(values[ge],'5')
            else if(mustNeutral.contains(word)||mustNeutral.contains(word.takeLast(2)))values[values.lastIndex]=tone(values.last(),'5')
            else if(word.endsWith('儿')&&mustNeutral.contains(word.dropLast(1))&&values.size>1)values[values.lastIndex-1]=tone(values[values.lastIndex-1],'5')
        }
        if(values.size==2&&values.all{it.endsWith('3')})values[0]=tone(values[0],'2')
        else if(values.size==3){val split=jieba.shortestSearchPart(word).length.coerceIn(1,2);if(values.all{it.endsWith('3')}){if(split==2){values[0]=tone(values[0],'2');values[1]=tone(values[1],'2')}else values[1]=tone(values[1],'2')}else{if(split==2&&values[0].endsWith('3')&&values[1].endsWith('3'))values[0]=tone(values[0],'2');if(split==1&&values[0].endsWith('3')&&values[1].endsWith('3'))values[0]=tone(values[0],'2')}}
        else if(values.size==4){if(values[0].endsWith('3')&&values[1].endsWith('3'))values[0]=tone(values[0],'2');if(values[2].endsWith('3')&&values[3].endsWith('3'))values[2]=tone(values[2],'2')}
        if(word.endsWith('儿')&&values.size>1){if(values.last().startsWith("er1"))values[values.lastIndex]=tone(values.last(),'2');if(word !in notErhua&&(mustErhua.contains(word)||pos !in setOf("a","j","nr")))values[values.lastIndex]=tone(values.last(),values[values.lastIndex-1].last())}
    }

    private fun tokenizeAndMap(codepoints: IntArray): Tokens {
        val tokens=ArrayList<String>();val mapping=IntArray(codepoints.size){-1};var pos=0
        while(pos<codepoints.size){
            val cp=codepoints[pos]
            if(Character.isWhitespace(cp)){pos++;continue}
            if(cp<128 && Character.isLetterOrDigit(cp)){
                val start=pos;while(pos<codepoints.size&&codepoints[pos]<128&&Character.isLetterOrDigit(codepoints[pos]))pos++
                val pieces=wordPiece(String(codepoints.sliceArray(start until pos),0,pos-start).lowercase());var cursor=start
                for(piece in pieces){val length=piece.removePrefix(prefix).codePointCount(0,piece.removePrefix(prefix).length).coerceAtLeast(1);val tokenIndex=tokens.size;tokens.add(piece);repeat(length){if(cursor<pos)mapping[cursor++]=tokenIndex}}
                while(cursor<pos){mapping[cursor]=tokens.lastIndex;cursor++}
            }else{
                val value=String(Character.toChars(cp)).lowercase();mapping[pos]=tokens.size;tokens.add(if(vocab.containsKey(value))value else "[UNK]");pos++
            }
        }
        return Tokens(tokens,mapping)
    }

    private fun wordPiece(word: String): List<String> {
        if(word.codePointCount(0,word.length)>maxWordLength)return listOf("[UNK]")
        val result=ArrayList<String>();var start=0
        while(start<word.length){var end=word.length;var found:String?=null
            while(start<end){val raw=word.substring(start,end);val candidate=if(start==0)raw else prefix+raw;if(vocab.containsKey(candidate)){found=candidate;break};end--}
            if(found==null)return listOf("[UNK]");result.add(found);start=end
        }
        return result
    }

    override fun close()=model.close()
}
