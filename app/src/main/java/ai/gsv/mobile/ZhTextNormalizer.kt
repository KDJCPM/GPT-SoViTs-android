package ai.gsv.mobile

import org.json.JSONObject

/** Deterministic Android port of GPT-SoVITS zh_normalization's ordered NSW rules. */
class ZhTextNormalizer(config: JSONObject) {
    private val traditional = config.optJSONObject("traditional_to_simplified")?.let { source ->
        HashMap<Char,Char>(source.length()).apply { source.keys().forEach { key -> if(key.length==1)put(key[0],source.getString(key)[0]) } }
    } ?: emptyMap()
    private val digits="零一二三四五六七八九"
    private val positiveQuantifiers=config.optString("common_quantifiers_pattern","").takeIf(String::isNotEmpty)?.let{Regex("(\\d+)([多余几+])?$it")}

    fun normalize(input:String):String {
        var text=input.map { ch -> traditional[ch] ?: when(ch.code){in 0xff10..0xff19,in 0xff21..0xff3a,in 0xff41..0xff5a -> (ch.code-0xfee0).toChar();0x3000 -> ' ';else -> ch} }.joinToString("")
        text=text.replace(Regex("[ ——《》【】<>{}()（）#&@“”^_|\\\\]"),"")
        text=DATE.replace(text){m->digit(m.groupValues[1])+"年"+(m.groupValues[3].takeIf(String::isNotEmpty)?.let{cardinal(it)+"月"}?:"")+(m.groupValues[5].takeIf(String::isNotEmpty)?.let{cardinal(it)+m.groupValues[9]}?:"")}
        text=DATE2.replace(text){m->digit(m.groupValues[1])+"年"+cardinal(m.groupValues[3])+"月"+cardinal(m.groupValues[4])+"日"}
        text=TIME_RANGE.replace(text){m->time(m.groupValues[1],m.groupValues[2],m.groupValues[4])+"至"+time(m.groupValues[6],m.groupValues[7],m.groupValues[9])}
        text=TIME.replace(text){m->time(m.groupValues[1],m.groupValues[2],m.groupValues[4])}
        text=TO_RANGE.replace(text){it.value.replace("~","至")}
        text=TEMPERATURE.replace(text){m->(if(m.groupValues[1].isNotEmpty())"零下" else "")+number(m.groupValues[2])+"度"}
        MEASURES.forEach{(raw,value)->text=text.replace(raw,value)}
        // Upstream's national-number path ultimately keeps the hyphen as spoken subtraction.
        text=NATIONAL.replace(text){m->m.value.split('-').joinToString("减"){digit(it,true)}}
        while(ASMD.containsMatchIn(text))text=ASMD.replace(text){m->m.groupValues[1]+OPERATORS.getValue(m.groupValues[8])+m.groupValues[9]}
        text=POWER.replace(text){m->"的"+m.value.map{POWER_CHARS[it]?:it}.joinToString("")+"次方"}
        text=FRACTION.replace(text){m->(if(m.groupValues[1].isNotEmpty())"负" else "")+number(m.groupValues[3])+"分之"+number(m.groupValues[2])}
        text=PERCENT.replace(text){m->(if(m.groupValues[1].isNotEmpty())"负" else "")+"百分之"+number(m.groupValues[2])}
        text=MOBILE.replace(text){phone(it.value,true)}
        text=TELEPHONE.replace(text){phone(it.value,false)}
        text=RANGE.replace(text){m->number(m.groupValues[1])+"到"+number(m.groupValues[5])}
        text=NEGATIVE.replace(text){"负"+number(it.groupValues[2])}
        text=VERSION.replace(text){m->m.value.map{if(it=='.')"点" else digit(it.toString())}.joinToString("")}
        text=DECIMAL.replace(text){number(it.value)}
        positiveQuantifiers?.let{rule->text=rule.replace(text){m->val spoken=number(m.groupValues[1]).let{if(it=="二")"两" else it};spoken+(if(m.groupValues[2]=="+")"多" else m.groupValues[2])+m.groupValues[3]}}
        text=DEFAULT_NUMBER.replace(text){digit(it.value,true)}
        text=NUMBER.replace(text){number(it.value)}
        text=postReplace(text)
        return replacePunctuation(text)
    }

    private fun time(hour:String,minute:String,second:String):String{
        var result=number(hour)+"点";if(minute.trimStart('0').isNotEmpty())result+=if(minute.toInt()==30)"半" else timeNumber(minute)+"分"
        if(second.isNotEmpty()&&second.trimStart('0').isNotEmpty())result+=timeNumber(second)+"秒";return result
    }
    private fun timeNumber(value:String)=if(value.startsWith('0'))"零"+number(value.trimStart('0')) else number(value)
    private fun phone(value:String,mobile:Boolean):String=(if(mobile)value.trim('+').split(' ') else value.split('-')).filter(String::isNotEmpty).joinToString("，"){digit(it,true)}
    private fun number(raw:String):String{
        val negative=raw.startsWith('-');val value=raw.removePrefix("-");val parts=value.split('.',limit=2);var result=cardinal(parts[0])
        if(parts.size==2){var decimal=parts[1];decimal=if(decimal.endsWith('0'))decimal.trimEnd('0')+"0" else decimal.trimEnd('0');if(decimal.isNotEmpty())result=(result.ifEmpty{"零"})+"点"+digit(decimal)}
        return (if(negative)"负" else "")+result
    }
    private fun cardinal(raw:String):String{
        val value=raw.trimStart('0');if(value.isEmpty())return "零"
        fun part(text:String):String{val out=StringBuilder();var zero=false;for(i in text.indices){val n=text[i]-'0';val power=text.length-i-1;if(n==0){zero=out.isNotEmpty()}else{if(zero){out.append('零');zero=false};out.append(digits[n]);when(power){1->out.append('十');2->out.append('百');3->out.append('千')}}};return out.toString()}
        val groups=value.reversed().chunked(4).map(String::reversed).reversed();val units=listOf("","万","亿","万亿");val out=StringBuilder()
        groups.forEachIndexed{i,g->val spoken=part(g);if(spoken.isNotEmpty()){if(out.isNotEmpty()&&g.length==4&&g[0]=='0'&&!out.endsWith('零'))out.append('零');out.append(spoken).append(units[groups.size-1-i])}}
        var result=out.toString();if(result.startsWith("一十"))result=result.drop(1);return result
    }
    private fun digit(value:String,altOne:Boolean=false)=value.mapNotNull{if(it.isDigit())digits[it-'0'] else null}.joinToString("").let{if(altOne)it.replace('一','幺') else it}
    private fun postReplace(source:String):String{var text=source;POST.forEach{(a,b)->text=text.replace(a,b)};return text.replace(Regex("[-——《》【】<=>{}()（）#&@“”^_|\\\\]"),"")}
    private fun replacePunctuation(source:String):String{
        var text=source.replace("嗯","恩").replace("呣","母");PUNCT.forEach{(a,b)->text=text.replace(a,b)}
        text=text.filter{it.code in 0x4e00..0x9fa5||it in ",.!?;:…-"};return text.replace(Regex("([,.!?;:…-])\\1+"),"$1")
    }

    companion object{
        private val DATE=Regex("(\\d{4}|\\d{2})年((0?[1-9]|1[0-2])月)?(((0?[1-9])|((1|2)[0-9])|30|31)([日号]))?")
        private val DATE2=Regex("(\\d{4})([- /.])(0[1-9]|1[012])\\2(0[1-9]|[12][0-9]|3[01])")
        private val TIME=Regex("([0-1]?[0-9]|2[0-3]):([0-5][0-9])(:([0-5][0-9]))?")
        private val TIME_RANGE=Regex("([0-1]?[0-9]|2[0-3]):([0-5][0-9])(:([0-5][0-9]))?(~|-)([0-1]?[0-9]|2[0-3]):([0-5][0-9])(:([0-5][0-9]))?")
        private val TO_RANGE=Regex("-?\\d+(?:\\.\\d+)?(?:%|°C|℃|度|摄氏度|cm2|cm²|cm3|cm³|cm|db|ds|kg|km|m2|m²|m³|m3|ml|m|mm|s)~-?\\d+(?:\\.\\d+)?(?:%|°C|℃|度|摄氏度|cm2|cm²|cm3|cm³|cm|db|ds|kg|km|m2|m²|m³|m3|ml|m|mm|s)")
        private val TEMPERATURE=Regex("(-?)(\\d+(?:\\.\\d+)?)(°C|℃|度|摄氏度)")
        private val FRACTION=Regex("(-?)(\\d+)/(\\d+)");private val PERCENT=Regex("(-?)(\\d+(?:\\.\\d+)?)%")
        private val MOBILE=Regex("(?<!\\d)(?:\\+?86 ?)?1(?:[38]\\d|5[0-35-9]|7[678]|9[89])\\d{8}(?!\\d)")
        private val TELEPHONE=Regex("(?<!\\d)(?:0(?:10|2[1-3]|[3-9]\\d{2})-?)?[1-9]\\d{6,7}(?!\\d)");private val NATIONAL=Regex("400-?\\d{3}-?\\d{4}")
        private val RANGE=Regex("(?<![\\d+\\-×÷=])((-?)(\\d+)(\\.\\d+)?)[-~]((-?)(\\d+)(\\.\\d+)?)(?![\\d+\\-×÷=])")
        private val NEGATIVE=Regex("(-)(\\d+)");private val VERSION=Regex("\\d+\\.\\d+(?:\\.\\d+){1,}")
        private val DECIMAL=Regex("-?(?:\\d+\\.\\d+|\\.\\d+)");private val DEFAULT_NUMBER=Regex("\\d{4,}");private val NUMBER=Regex("-?(?:\\d+(?:\\.\\d+)?|\\.\\d+)")
        private val POWER=Regex("[⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]+");private val ASMD=Regex("((-?)((\\d+)(\\.\\d+)?[⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*)|(\\.\\d+[⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*)|([A-Za-z][⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*))([+\\-×÷=])((-?)((\\d+)(\\.\\d+)?[⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*)|(\\.\\d+[⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*)|([A-Za-z][⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*))")
        private val OPERATORS=mapOf("+" to "加","-" to "减","×" to "乘","÷" to "除","=" to "等于");private val POWER_CHARS="⁰¹²³⁴⁵⁶⁷⁸⁹".zip("0123456789").toMap()+mapOf('ˣ' to 'x','ʸ' to 'y','ⁿ' to 'n')
        private val MEASURES=linkedMapOf("cm²" to "平方厘米","cm2" to "平方厘米","cm³" to "立方厘米","cm3" to "立方厘米","km" to "千米","kg" to "千克","mm" to "毫米","ml" to "毫升","m²" to "平方米","m2" to "平方米","m³" to "立方米","m3" to "立方米","cm" to "厘米","db" to "分贝","ds" to "毫秒","m" to "米","s" to "秒")
        private val POST=linkedMapOf("/" to "每","①" to "一","②" to "二","③" to "三","④" to "四","⑤" to "五","⑥" to "六","⑦" to "七","⑧" to "八","⑨" to "九","⑩" to "十","α" to "阿尔法","β" to "贝塔","γ" to "伽玛","Γ" to "伽玛","δ" to "德尔塔","Δ" to "德尔塔","ε" to "艾普西龙","ζ" to "捷塔","η" to "依塔","θ" to "西塔","Θ" to "西塔","ι" to "艾欧塔","κ" to "喀帕","λ" to "拉姆达","Λ" to "拉姆达","μ" to "缪","ν" to "拗","ξ" to "克西","Ξ" to "克西","ο" to "欧米克伦","π" to "派","Π" to "派","ρ" to "肉","ς" to "西格玛","Σ" to "西格玛","σ" to "西格玛","τ" to "套","υ" to "宇普西龙","φ" to "服艾","Φ" to "服艾","χ" to "器","ψ" to "普赛","Ψ" to "普赛","ω" to "欧米伽","Ω" to "欧米伽","+" to "加","-" to "减","×" to "乘","÷" to "除","=" to "等")
        private val PUNCT=linkedMapOf("：" to ",","；" to ",","，" to ",","。" to ".","！" to "!","？" to "?","\n" to ".","·" to ",","、" to ",","..." to "…","$" to ".","/" to ",","—" to "-","~" to "…","～" to "…")
    }
}
