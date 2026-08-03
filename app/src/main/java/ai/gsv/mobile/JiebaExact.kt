package ai.gsv.mobile

import java.io.File
import java.io.BufferedInputStream
import java.io.DataInputStream
import kotlin.math.ln

data class ZhWord(var text: String, var pos: String)

/** Jieba dictionary DAG route used by the upstream frontend. HMM fallback is handled separately. */
class JiebaExact(dictionary: File, hmmFile: File?=null) {
    private data class Entry(val frequency: Int, val pos: String)
    private val entries=HashMap<String,Entry>();private val prefixes=HashSet<String>();private var total=0L
    private val hmm=hmmFile?.takeIf(File::isFile)?.let(::JiebaPosHmm)
    init {
        dictionary.forEachLine { line ->
            val parts=line.trim().split(' ');if(parts.size>=2){val frequency=parts[1].toIntOrNull()?:return@forEachLine;val word=parts[0];entries[word]=Entry(frequency,parts.getOrElse(2){"x"});total+=frequency
                for(i in 1..word.length)prefixes.add(word.substring(0,i))
            }
        }
    }

    fun cut(text:String):MutableList<ZhWord>{
        val n=text.length;val routeScore=DoubleArray(n+1);val routeEnd=IntArray(n);routeScore[n]=0.0;val logTotal=ln(total.toDouble())
        for(i in n-1 downTo 0){var best=Double.NEGATIVE_INFINITY;var bestEnd=i;var j=i
            while(j<n){val word=text.substring(i,j+1);if(!prefixes.contains(word))break;val entry=entries[word];if(entry!=null){val score=ln(entry.frequency.toDouble())-logTotal+routeScore[j+1];if(score>best){best=score;bestEnd=j}};j++}
            if(best==Double.NEGATIVE_INFINITY){best=-logTotal+routeScore[i+1];bestEnd=i}
            routeScore[i]=best;routeEnd[i]=bestEnd
        }
        val result=mutableListOf<ZhWord>();val buffer=StringBuilder()
        fun flush(){if(buffer.isEmpty())return;val value=buffer.toString();if(value.length==1){val e=entries[value];result.add(ZhWord(value,e?.pos?:"x"))}else if(entries[value]==null&&hmm!=null)result.addAll(hmm.cut(value))else value.forEach{ch->val word=ch.toString();result.add(ZhWord(word,entries[word]?.pos?:"x"))};buffer.setLength(0)}
        var i=0
        while(i<n){val end=routeEnd[i]+1;val word=text.substring(i,end);if(word.length==1&&word[0].code in 0x4e00..0x9fd5)buffer.append(word)else{flush();val entry=entries[word];result.add(ZhWord(word,entry?.pos?:"x"))};i=end};flush()
        return result
    }

    fun shortestSearchPart(word:String):String{
        if(word.length<=1)return word
        val candidates=ArrayList<String>();for(token in cut(word)){if(token.text.length>2)for(i in 0 until token.text.length-1){val gram=token.text.substring(i,i+2);if(entries.containsKey(gram))candidates.add(gram)};if(token.text.length>3)for(i in 0 until token.text.length-2){val gram=token.text.substring(i,i+3);if(entries.containsKey(gram))candidates.add(gram)};candidates.add(token.text)}
        val shortest=candidates.withIndex().minWithOrNull(compareBy<IndexedValue<String>>{it.value.length}.thenBy{it.index})?.value?:return word.substring(0,1)
        val at=word.indexOf(shortest);return if(at==0)shortest else word.dropLast(shortest.length)
    }
}

/** Exact 256-state Jieba POS BMES Viterbi for consecutive out-of-vocabulary Han text. */
private class JiebaPosHmm(file:File){
    private data class Edge(val target:Int,val score:Double);private data class Observation(val states:IntArray,val scores:DoubleArray)
    private val boundaries:CharArray;private val positions:Array<String>;private val starts:DoubleArray;private val transitions:Array<Array<Edge>>;private val observations=HashMap<Int,Observation>()
    init{DataInputStream(BufferedInputStream(file.inputStream())).use{input->
        require(ByteArray(4).also(input::readFully).contentEquals("JPH1".toByteArray())){"invalid Jieba POS HMM"};val count=input.readUnsignedShort()
        boundaries=CharArray(count);positions=Array(count){""};starts=DoubleArray(count)
        repeat(count){i->boundaries[i]=input.readUnsignedByte().toChar();val n=input.readUnsignedByte();positions[i]=ByteArray(n).also(input::readFully).toString(Charsets.UTF_8);starts[i]=input.readDouble()}
        transitions=Array(count){val n=input.readUnsignedShort();Array(n){Edge(input.readUnsignedShort(),input.readDouble())}}
        repeat(input.readInt()){val cp=input.readInt();val n=input.readUnsignedShort();val ids=IntArray(n);val scores=DoubleArray(n);repeat(n){i->ids[i]=input.readUnsignedShort();scores[i]=input.readDouble()};observations[cp]=Observation(ids,scores)}
    }}
    fun cut(text:String):List<ZhWord>{
        if(text.length==1)return listOf(ZhWord(text,"x"));val count=starts.size;val back=Array(text.length){IntArray(count){-1}}
        var scores=DoubleArray(count){MIN};var active=BooleanArray(count);val first=observation(text[0])
        for(i in first.states.indices){val state=first.states[i];scores[state]=starts[state]+first.scores[i];active[state]=true}
        for(t in 1 until text.length){val obs=observation(text[t]);val allowed=BooleanArray(count);val emission=DoubleArray(count){MIN};for(i in obs.states.indices){allowed[obs.states[i]]=true;emission[obs.states[i]]=obs.scores[i]}
            val next=DoubleArray(count){MIN};val nextActive=BooleanArray(count)
            for(previous in 0 until count)if(active[previous])for(edge in transitions[previous])if(allowed[edge.target]){val value=scores[previous]+edge.score+emission[edge.target];if(!nextActive[edge.target]||value>next[edge.target]){next[edge.target]=value;nextActive[edge.target]=true;back[t][edge.target]=previous}}
            if(!nextActive.any()){for(previous in 0 until count)if(active[previous])for(edge in transitions[previous]){val value=scores[previous]+edge.score+MIN;if(!nextActive[edge.target]||value>next[edge.target]){next[edge.target]=value;nextActive[edge.target]=true;back[t][edge.target]=previous}}}
            scores=next;active=nextActive
        }
        var state=(0 until count).filter{active[it]}.maxByOrNull{scores[it]}?:0;val path=IntArray(text.length);path[path.lastIndex]=state
        for(t in text.lastIndex downTo 1){state=back[t][state].coerceAtLeast(0);path[t-1]=state}
        val output=ArrayList<ZhWord>();var begin=0;var nextIndex=0
        for(i in text.indices)when(boundaries[path[i]]){'B'->begin=i;'E'->{output.add(ZhWord(text.substring(begin,i+1),positions[path[i]]));nextIndex=i+1};'S'->{output.add(ZhWord(text[i].toString(),positions[path[i]]));nextIndex=i+1}}
        if(nextIndex<text.length)output.add(ZhWord(text.substring(nextIndex),positions[path[nextIndex]]));return output
    }
    private fun observation(char:Char)=observations[char.code]?:Observation(IntArray(starts.size){it},DoubleArray(starts.size){MIN})
    companion object{private const val MIN=-3.14e100}
}
