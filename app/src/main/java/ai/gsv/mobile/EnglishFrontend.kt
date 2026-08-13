package ai.gsv.mobile

import org.json.JSONObject
import java.io.BufferedInputStream
import java.io.DataInputStream
import java.io.File
import java.text.Normalizer
import java.util.Locale
import kotlin.math.log10
import kotlin.math.tanh

/** English frontend frozen from GPT-SoVITS text.english and g2p_en conversion assets. */
internal class EnglishFrontend(private val root: File) {
    private val config = JSONObject(File(root, "english.json").readText())
    private val symbolIds = config.getJSONObject("symbol_ids").let { source ->
        HashMap<String, Long>(source.length()).apply {
            source.keys().forEach { put(it, source.getLong(it)) }
        }
    }
    private val lexicon: Map<String, List<String>> by lazy { loadLexicon() }
    private val names: Map<String, List<String>> by lazy { loadPronunciations("english-names.tsv") }
    private val homographs: Map<String, Homograph> by lazy { loadHomographs() }
    private val tagger: PerceptronTagger by lazy { PerceptronTagger(File(root, "english-tagger.bin")) }
    private val predictor: G2pPredictor by lazy { G2pPredictor(File(root, "english-g2p.bin")) }
    private val segmenter: WordSegmenter by lazy {
        WordSegmenter(
            File(root, "english-unigrams.tsv"),
            File(root, "english-bigrams.tsv"),
            config.getDouble("wordsegment_total"),
            config.getInt("wordsegment_limit"),
        )
    }

    data class Prepared(val normalized: String, val phoneIds: LongArray)
    private data class Homograph(
        val first: List<String>,
        val second: List<String>,
        val firstPosPrefix: String,
    )

    fun prepare(input: String): Prepared {
        val normalized = EnglishNormalizer.normalize(input)
        val tokens = tokenize(normalized)
        if (tokens.isEmpty()) return Prepared(normalized, LongArray(0))
        val needsTags = tokens.any { homographs.containsKey(it.lowercase(Locale.US)) }
        val tags = if (needsTags) tagger.tag(tokens) else List(tokens.size) { "" }
        val phones = ArrayList<Long>()
        tokens.forEachIndexed { index, token ->
            pronunciation(token, tags[index]).forEach { phone ->
                val normalizedPhone = when (phone) {
                    "<unk>" -> "UNK"
                    "'" -> "-"
                    else -> phone
                }
                if (normalizedPhone !in ignoredPhones) {
                    symbolIds[normalizedPhone]?.let(phones::add)
                }
            }
        }
        return Prepared(normalized, phones.toLongArray())
    }

    private fun pronunciation(original: String, pos: String): List<String> {
        val word = original.lowercase(Locale.US)
        if (word.none(Char::isLetter)) return listOf(collapsePunctuation(word))
        if (word.length == 1) {
            if (original == "A") return listOf("EY1")
            return lexicon[word] ?: listOf("UNK")
        }
        homographs[word]?.let { value ->
            return if (
                pos.startsWith(value.firstPosPrefix) ||
                (pos.length < value.firstPosPrefix.length && value.firstPosPrefix.startsWith(pos))
            ) value.first else value.second
        }
        lexicon[word]?.let { return it }
        if (original.isTitleCased()) names[word]?.let { return it }
        if (word.length <= 3) {
            return word.flatMap { letter ->
                if (letter == 'a') listOf("EY1") else lexicon[letter.toString()] ?: listOf("UNK")
            }
        }
        possessive(word)?.let { return it }
        val components = segmenter.segment(word)
        if (components.size > 1) return components.flatMap { pronunciation(it, "") }
        return predictor.predict(segmenter.clean(word))
    }

    private fun possessive(word: String): List<String>? {
        if (!word.endsWith("'s") || word.length <= 2 || word.dropLast(2).any { !it.isLetter() }) return null
        val base = pronunciation(word.dropLast(2), "").toMutableList()
        if (base.isEmpty()) return null
        when (base.last()) {
            "P", "T", "K", "F", "TH", "HH" -> base += "S"
            "S", "Z", "SH", "ZH", "CH", "JH" -> base += listOf("AH0", "Z")
            else -> base += "Z"
        }
        return base
    }

    private fun tokenize(text: String): List<String> = tokenPattern.findAll(text)
        .map { it.value }
        .filter { it.isNotBlank() }
        .toList()

    private fun collapsePunctuation(value: String): String = when {
        value.contains('.') -> "."
        value.contains('!') -> "!"
        value.contains('?') -> "?"
        value.contains(',') -> ","
        else -> value.take(1)
    }

    private fun loadLexicon(): Map<String, List<String>> = loadPronunciations("english-lexicon.tsv")

    private fun loadPronunciations(name: String): Map<String, List<String>> = HashMap<String, List<String>>().apply {
        File(root, name).useLines { lines ->
            lines.forEach { line ->
                val split = line.indexOf('\t')
                if (split > 0) put(line.substring(0, split), line.substring(split + 1).split(' '))
            }
        }
    }

    private fun loadHomographs(): Map<String, Homograph> = HashMap<String, Homograph>().apply {
        File(root, "english-homographs.tsv").useLines { lines ->
            lines.forEach { line ->
                val values = line.split('\t')
                if (values.size == 4) {
                    put(values[0], Homograph(values[1].split(' '), values[2].split(' '), values[3]))
                }
            }
        }
    }

    private class PerceptronTagger(file: File) {
        private val classes: List<String>
        private val tagDictionary: Map<String, Int>
        private val weights: Map<String, List<Pair<Int, Float>>>

        init {
            DataInputStream(BufferedInputStream(file.inputStream())).use { input ->
                require(input.readAscii(4) == "EPT1") { "invalid English POS tagger" }
                classes = List(input.readUnsignedShort()) { input.readUtf8() }
                tagDictionary = HashMap<String, Int>().apply {
                    repeat(input.readInt()) { put(input.readUtf8(), input.readUnsignedShort()) }
                }
                weights = HashMap<String, List<Pair<Int, Float>>>().apply {
                    repeat(input.readInt()) {
                        val feature = input.readUtf8()
                        put(feature, List(input.readUnsignedShort()) { input.readUnsignedShort() to input.readFloat() })
                    }
                }
                require(input.read() == -1) { "trailing English POS tagger data" }
            }
        }

        fun tag(tokens: List<String>): List<String> {
            var previous = "-START-"
            var previous2 = "-START2-"
            val context = listOf("-START-", "-START2-") + tokens.map(::normalize) + listOf("-END-", "-END2-")
            return tokens.mapIndexed { index, word ->
                val tagIndex = tagDictionary[word] ?: predict(features(index + 2, word, context, previous, previous2))
                val tag = classes[tagIndex]
                previous2 = previous
                previous = tag
                tag
            }
        }

        private fun predict(features: List<String>): Int {
            val scores = FloatArray(classes.size)
            features.forEach { feature ->
                weights[feature]?.forEach { (tag, weight) -> scores[tag] += weight }
            }
            var best = 0
            for (index in 1 until classes.size) {
                if (scores[index] > scores[best] || (scores[index] == scores[best] && classes[index] > classes[best])) {
                    best = index
                }
            }
            return best
        }

        private fun features(index: Int, word: String, context: List<String>, prev: String, prev2: String) = listOf(
            "bias",
            "i suffix ${word.takeLast(3)}",
            "i pref1 ${word.firstOrNull() ?: ""}",
            "i-1 tag $prev",
            "i-2 tag $prev2",
            "i tag+i-2 tag $prev $prev2",
            "i word ${context[index]}",
            "i-1 tag+i word $prev ${context[index]}",
            "i-1 word ${context[index - 1]}",
            "i-1 suffix ${context[index - 1].takeLast(3)}",
            "i-2 word ${context[index - 2]}",
            "i+1 word ${context[index + 1]}",
            "i+1 suffix ${context[index + 1].takeLast(3)}",
            "i+2 word ${context[index + 2]}",
        )

        private fun normalize(word: String): String = when {
            '-' in word && !word.startsWith('-') -> "!HYPHEN"
            word.length == 4 && word.all(Char::isDigit) -> "!YEAR"
            word.firstOrNull()?.isDigit() == true -> "!DIGITS"
            else -> word.lowercase(Locale.US)
        }
    }

    private class WordSegmenter(
        unigramFile: File,
        bigramFile: File,
        private val total: Double,
        private val limit: Int,
    ) {
        private val unigrams = readCounts(unigramFile)
        private val bigrams = readCounts(bigramFile)

        fun clean(value: String): String = value.lowercase(Locale.US).filter { it in 'a'..'z' || it.isDigit() }

        fun segment(value: String): List<String> {
            val text = clean(value)
            if (text.isEmpty()) return emptyList()
            val memo = HashMap<Pair<String, String>, Pair<Double, List<String>>>()
            fun search(remaining: String, previous: String): Pair<Double, List<String>> {
                if (remaining.isEmpty()) return 0.0 to emptyList()
                val key = remaining to previous
                memo[key]?.let { return it }
                var bestScore = Double.NEGATIVE_INFINITY
                var bestWords = emptyList<String>()
                for (length in 1..minOf(remaining.length, limit)) {
                    val prefix = remaining.substring(0, length)
                    val suffix = remaining.substring(length)
                    val (suffixScore, suffixWords) = search(suffix, prefix)
                    val candidate = log10(score(prefix, previous)) + suffixScore
                    if (candidate > bestScore) {
                        bestScore = candidate
                        bestWords = listOf(prefix) + suffixWords
                    }
                }
                return (bestScore to bestWords).also { memo[key] = it }
            }
            return search(text, "<s>").second
        }

        private fun score(word: String, previous: String? = null): Double {
            if (previous == null) return unigrams[word]?.div(total) ?: (10.0 / (total * Math.pow(10.0, word.length.toDouble())))
            val bigram = bigrams["$previous $word"]
            return if (bigram != null && previous in unigrams) bigram / total / score(previous) else score(word)
        }

        companion object {
            private fun readCounts(file: File): Map<String, Double> = HashMap<String, Double>().apply {
                file.useLines { lines ->
                    lines.forEach { line ->
                        val split = line.lastIndexOf('\t')
                        if (split > 0) line.substring(split + 1).toDoubleOrNull()?.let { put(line.substring(0, split), it) }
                    }
                }
            }
        }
    }

    private class G2pPredictor(file: File) {
        private data class ArrayValue(val shape: IntArray, val values: FloatArray)
        private val values: Map<String, ArrayValue>

        init {
            DataInputStream(BufferedInputStream(file.inputStream())).use { input ->
                require(input.readAscii(4) == "EGP1") { "invalid English G2P model" }
                values = HashMap<String, ArrayValue>().apply {
                    repeat(input.readUnsignedShort()) {
                        val name = input.readUtf8()
                        val shape = IntArray(input.readUnsignedByte()) { input.readInt() }
                        val count = shape.fold(1) { product, size -> Math.multiplyExact(product, size) }
                        put(name, ArrayValue(shape, FloatArray(count) { input.readFloat() }))
                    }
                }
                require(input.read() == -1) { "trailing English G2P model data" }
            }
        }

        fun predict(word: String): List<String> {
            val encEmb = array("enc_emb")
            var hidden = FloatArray(128)
            val characters = word.map { graphemeIds[it] ?: 1 } + 2
            characters.forEach { id ->
                hidden = gruCell(
                    row(encEmb, id), hidden,
                    array("enc_w_ih"), array("enc_w_hh"), array("enc_b_ih"), array("enc_b_hh"),
                )
            }
            val decEmb = array("dec_emb")
            var input = row(decEmb, 2)
            val output = ArrayList<String>()
            repeat(20) {
                hidden = gruCell(
                    input, hidden,
                    array("dec_w_ih"), array("dec_w_hh"), array("dec_b_ih"), array("dec_b_hh"),
                )
                val logits = linear(hidden, array("fc_w"), array("fc_b"))
                var best = 0
                for (index in 1 until logits.size) if (logits[index] > logits[best]) best = index
                if (best == 3) return output
                output += phonemes.getOrElse(best) { "<unk>" }
                input = row(decEmb, best)
            }
            return output
        }

        private fun gruCell(
            input: FloatArray,
            hidden: FloatArray,
            inputWeights: ArrayValue,
            hiddenWeights: ArrayValue,
            inputBias: ArrayValue,
            hiddenBias: ArrayValue,
        ): FloatArray {
            val inputProjection = linear(input, inputWeights, inputBias)
            val hiddenProjection = linear(hidden, hiddenWeights, hiddenBias)
            val output = FloatArray(hidden.size)
            for (index in hidden.indices) {
                val reset = sigmoid(inputProjection[index] + hiddenProjection[index])
                val update = sigmoid(inputProjection[index + hidden.size] + hiddenProjection[index + hidden.size])
                val candidate = tanh(
                    (inputProjection[index + hidden.size * 2] + reset * hiddenProjection[index + hidden.size * 2]).toDouble()
                ).toFloat()
                output[index] = (1.0f - update) * candidate + update * hidden[index]
            }
            return output
        }

        private fun linear(input: FloatArray, weights: ArrayValue, bias: ArrayValue): FloatArray {
            require(weights.shape.size == 2 && weights.shape[1] == input.size)
            val rows = weights.shape[0]
            return FloatArray(rows) { row ->
                var value = bias.values[row]
                val offset = row * input.size
                for (column in input.indices) value += input[column] * weights.values[offset + column]
                value
            }
        }

        private fun row(array: ArrayValue, index: Int): FloatArray {
            require(array.shape.size == 2 && index in 0 until array.shape[0])
            val columns = array.shape[1]
            return array.values.copyOfRange(index * columns, (index + 1) * columns)
        }

        private fun array(name: String) = requireNotNull(values[name]) { "missing English G2P tensor $name" }
        private fun sigmoid(value: Float) = (1.0 / (1.0 + kotlin.math.exp(-value.toDouble()))).toFloat()

        companion object {
            private val graphemeIds = ('a'..'z').withIndex().associate { (index, value) ->
                value to index + 3
            }
            private val phonemes = listOf(
                "<pad>", "<unk>", "<s>", "</s>", "AA0", "AA1", "AA2", "AE0", "AE1", "AE2",
                "AH0", "AH1", "AH2", "AO0", "AO1", "AO2", "AW0", "AW1", "AW2", "AY0",
                "AY1", "AY2", "B", "CH", "D", "DH", "EH0", "EH1", "EH2", "ER0", "ER1",
                "ER2", "EY0", "EY1", "EY2", "F", "G", "HH", "IH0", "IH1", "IH2", "IY0",
                "IY1", "IY2", "JH", "K", "L", "M", "N", "NG", "OW0", "OW1", "OW2", "OY0",
                "OY1", "OY2", "P", "R", "S", "SH", "T", "TH", "UH0", "UH1", "UH2", "UW",
                "UW0", "UW1", "UW2", "V", "W", "Y", "Z", "ZH",
            )
        }
    }

    private object EnglishNormalizer {
        private val commaNumber = Regex("([0-9][0-9,]+[0-9])")
        private val time = Regex("\\b([01]?[0-9]|2[0-3]):([0-5][0-9])\\b")
        private val money = Regex("([£$])([0-9.,]*[0-9]+)|([0-9.,]*[0-9]+)([£$])")
        private val decimal = Regex("([0-9]+)\\.\\s*([0-9]+)")
        private val fraction = Regex("([0-9]+)/([0-9]+)")
        private val ordinal = Regex("([0-9]+)(st|nd|rd|th)", RegexOption.IGNORE_CASE)
        private val number = Regex("[0-9]+")
        private val uppercaseInsideWord = Regex("(?<!^)(?<!\\s)([A-Z])")
        private val tokenCleanup = Regex("[^ A-Za-z'.,?!-]")
        private val repeatedPunctuation = Regex("([.,?!-])[.,?!-]+")

        fun normalize(source: String): String {
            var text = source
                .replace('，', ',').replace('；', ',').replace('：', ',')
                .replace('。', '.').replace('！', '!').replace('？', '?').replace('’', '\'')
            text = commaNumber.replace(text) { it.groupValues[1].replace(",", "") }
            text = time.replace(text) {
                var hour = it.groupValues[1].toInt()
                val minute = it.groupValues[2].toInt()
                val period = if (hour < 12) "a.m." else "p.m."
                if (hour > 12) hour -= 12
                val minuteWords = if (minute == 0) "o'clock" else spell(minute.toLong())
                "${spell(hour.toLong())} $minuteWords $period"
            }
            text = money.replace(text) {
                val sign = it.groupValues[1].ifEmpty { it.groupValues[4] }
                val raw = it.groupValues[2].ifEmpty { it.groupValues[3] }.replace(",", "")
                val pieces = raw.split('.', limit = 2)
                val whole = pieces[0].ifEmpty { "0" }.toLongOrNull() ?: 0L
                val minor = pieces.getOrNull(1)?.padEnd(2, '0')?.take(2)?.toLongOrNull() ?: 0L
                val majorName = if (sign == "£") "pound" else "dollar"
                val minorName = if (sign == "£") "penny" else "cent"
                buildList {
                    if (whole > 0) add("${spell(whole)} ${plural(majorName, whole)}")
                    if (minor > 0) add("${spell(minor)} ${plural(minorName, minor)}")
                    if (isEmpty()) add("zero ${majorName}s")
                }.joinToString(" and ")
            }
            text = fraction.replace(text) {
                val numerator = it.groupValues[1].toLong()
                val denominator = it.groupValues[2].toLong()
                val denominatorWord = when (denominator) {
                    1L -> return@replace spell(numerator)
                    2L -> if (numerator == 1L) "half" else "halves"
                    else -> ordinal(spell(denominator)) + if (numerator > 1) "s" else ""
                }
                "${spell(numerator)} $denominatorWord"
            }
            text = decimal.replace(text) {
                "${spell(it.groupValues[1].toLong())} point ${it.groupValues[2].map { digit -> spell((digit - '0').toLong()) }.joinToString(" ")}"
            }
            text = ordinal.replace(text) { ordinal(spell(it.groupValues[1].toLong())) }
            text = number.replace(text) { spell(it.value.toLong()) }
            text = text.replace("%", " percent").replace(Regex("(?i)i\\.e\\."), "that is")
                .replace(Regex("(?i)e\\.g\\."), "for example")
            text = Normalizer.normalize(text, Normalizer.Form.NFD).filter { Character.getType(it) != Character.NON_SPACING_MARK.toInt() }
            text = tokenCleanup.replace(text, "")
            text = uppercaseInsideWord.replace(text, " $1")
            text = repeatedPunctuation.replace(text) { it.groupValues[1] }
            return text.replace(Regex("\\s+"), " ").trim()
        }

        private fun spell(value: Long): String {
            if (value == 0L) return "zero"
            if (value < 0L) return "negative ${spell(-value)}"
            if (value in 1001L..2999L) {
                if (value == 2000L) return "two thousand"
                if (value in 2001L..2009L) return "two thousand ${spell(value % 100)}"
                if (value % 100L == 0L) return "${spell(value / 100)} hundred"
                val first = value / 100
                val second = value % 100
                return "${spell(first)} ${if (second < 10) "oh ${spell(second)}" else spell(second)}"
            }
            val parts = ArrayList<String>()
            var remaining = value
            var scale = 0
            while (remaining > 0) {
                val chunk = (remaining % 1000).toInt()
                if (chunk != 0) {
                    val name = scales.getOrElse(scale) { "" }
                    parts += listOf(spellBelowThousand(chunk), name).filter(String::isNotEmpty).joinToString(" ")
                }
                remaining /= 1000
                scale++
            }
            return parts.asReversed().joinToString(" ")
        }

        private fun spellBelowThousand(value: Int): String {
            require(value in 1..999)
            val parts = ArrayList<String>()
            var remaining = value
            if (remaining >= 100) {
                parts += "${ones[remaining / 100]} hundred"
                remaining %= 100
            }
            if (remaining >= 20) {
                val tensValue = tens[remaining / 10]
                val unit = remaining % 10
                parts += if (unit == 0) tensValue else "$tensValue-${ones[unit]}"
            } else if (remaining >= 10) {
                parts += teens[remaining - 10]
            } else if (remaining > 0) {
                parts += ones[remaining]
            }
            return parts.joinToString(" ")
        }
        private fun plural(value: String, count: Long) = if (count == 1L) value else when (value) {
            "penny" -> "pence"
            else -> "${value}s"
        }

        private fun ordinal(value: String): String {
            val split = value.lastIndexOfAny(charArrayOf(' ', '-'))
            val prefix = if (split >= 0) value.substring(0, split + 1) else ""
            val last = value.substring(split + 1)
            val converted = when (last) {
                "one" -> "first"
                "two" -> "second"
                "three" -> "third"
                "five" -> "fifth"
                "eight" -> "eighth"
                "nine" -> "ninth"
                "twelve" -> "twelfth"
                else -> if (last.endsWith('y')) last.dropLast(1) + "ieth" else last + "th"
            }
            return prefix + converted
        }

        private val ones = listOf("", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
        private val teens = listOf("ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen")
        private val tens = listOf("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
        private val scales = listOf("", "thousand", "million", "billion", "trillion", "quadrillion", "quintillion")
    }

    companion object {
        private val tokenPattern = Regex("[A-Za-z]+(?:'[A-Za-z]+)?(?:-[A-Za-z]+)*|\\.{1,3}|[!?,-]")
        private val ignoredPhones = setOf(" ", "<pad>", "UW", "</s>", "<s>")
    }
}

private fun String.isTitleCased(): Boolean {
    var atWordStart = true
    var hasCasedLetter = false
    for (value in this) {
        if (!value.isLetter()) {
            atWordStart = true
            continue
        }
        hasCasedLetter = true
        if (atWordStart) {
            if (!value.isUpperCase() && !value.isTitleCase()) return false
            atWordStart = false
        } else if (!value.isLowerCase()) {
            return false
        }
    }
    return hasCasedLetter
}

private fun DataInputStream.readUtf8(): String = readAscii(readUnsignedShort())
private fun DataInputStream.readAscii(length: Int): String {
    require(length >= 0)
    val bytes = ByteArray(length)
    readFully(bytes)
    return String(bytes, Charsets.UTF_8)
}
