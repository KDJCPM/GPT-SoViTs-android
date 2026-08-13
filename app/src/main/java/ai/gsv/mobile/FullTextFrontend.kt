package ai.gsv.mobile

import java.io.Closeable
import java.io.File

/** Language-aware V2 frontend. English phones carry zero BERT features, matching upstream. */
internal class FullTextFrontend(
    private val frontendDir: File,
    qnnTarget: QualcommTargetSoc? = null,
    modelFile: File = File(frontendDir, "g2pW.onnx"),
    strictQnn: Boolean = false,
    staticRows: Int? = null,
    staticSequenceLength: Int? = null,
    htpFp16Precision: Boolean = false,
    htpGraphOptimizationMode: String? = null,
) : Closeable {
    data class BertSpan(
        val phoneOffset: Int,
        val tokenIds: LongArray,
        val word2ph: IntArray,
    ) {
        val phoneCount: Int = word2ph.sum()
    }

    data class Prepared(
        val normalized: String,
        val phoneIds: LongArray,
        val bertSpans: List<BertSpan>,
    ) {
        val maxTokenCount: Int = bertSpans.maxOfOrNull { it.tokenIds.size } ?: 0
    }

    private val chinese = FullZhFrontend(
        frontendDir,
        qnnTarget,
        modelFile,
        strictQnn,
        staticRows,
        staticSequenceLength,
        htpFp16Precision,
        htpGraphOptimizationMode,
    )
    private val english: EnglishFrontend by lazy { EnglishFrontend(frontendDir) }

    val qnnExecutionStats: QnnExecutionStats?
        get() = chinese.qnnExecutionStats

    fun finalizeQnnProfiling() = chinese.finalizeQnnProfiling()

    fun prepare(text: String, language: String): Prepared {
        val normalizedLanguage = language.lowercase()
        require(normalizedLanguage in supportedLanguages) {
            "unsupported language=$language; use auto, zh, or en"
        }
        val first = prepareInternal(text, normalizedLanguage)
        return if (first.phoneIds.size >= 6 || text.startsWith('.')) first else {
            prepareInternal(".$text", normalizedLanguage)
        }
    }

    private fun prepareInternal(text: String, language: String): Prepared {
        val runs = if (language == "en") listOf(LanguageRun(false, text)) else splitRuns(text)
        val phones = ArrayList<Long>()
        val spans = ArrayList<BertSpan>()
        val normalized = StringBuilder()
        runs.forEach { run ->
            if (run.chinese) {
                val value = chinese.prepare(run.text)
                if (value.phoneIds.isNotEmpty()) {
                    spans += BertSpan(phones.size, value.tokenIds, value.word2ph)
                    value.phoneIds.forEach(phones::add)
                    normalized.append(value.normalized)
                }
            } else {
                val value = english.prepare(run.text)
                value.phoneIds.forEach(phones::add)
                normalized.append(value.normalized)
            }
        }
        require(phones.isNotEmpty()) { "text has no supported symbols for language=$language" }
        spans.forEach { span ->
            require(span.phoneOffset >= 0 && span.phoneOffset + span.phoneCount <= phones.size) {
                "frontend BERT span is outside the phone sequence"
            }
        }
        return Prepared(normalized.toString(), phones.toLongArray(), spans)
    }

    private fun splitRuns(text: String): List<LanguageRun> {
        val codePoints = text.codePoints().toArray()
        if (codePoints.isEmpty()) return emptyList()
        val runs = ArrayList<LanguageRun>()
        var start = 0
        var currentChinese = classify(codePoints, 0)
        for (index in 1 until codePoints.size) {
            val classification = classify(codePoints, index)
            if (classification != currentChinese) {
                runs += LanguageRun(currentChinese, String(codePoints, start, index - start))
                start = index
                currentChinese = classification
            }
        }
        runs += LanguageRun(currentChinese, String(codePoints, start, codePoints.size - start))
        return runs
    }

    private fun classify(codePoints: IntArray, index: Int): Boolean {
        val value = codePoints[index]
        if (isLatin(value)) return false
        if (isCjk(value)) return true
        if (Character.isDigit(value)) {
            val left = (index - 1 downTo 0).firstOrNull { isLatin(codePoints[it]) || isCjk(codePoints[it]) }
            val right = (index + 1 until codePoints.size).firstOrNull { isLatin(codePoints[it]) || isCjk(codePoints[it]) }
            return when {
                left != null -> isCjk(codePoints[left])
                right != null -> isCjk(codePoints[right])
                else -> false
            }
        }
        for (previous in index - 1 downTo 0) {
            if (isLatin(codePoints[previous])) return false
            if (isCjk(codePoints[previous])) return true
        }
        for (next in index + 1 until codePoints.size) {
            if (isLatin(codePoints[next])) return false
            if (isCjk(codePoints[next])) return true
        }
        return false
    }

    private fun isLatin(codePoint: Int): Boolean = Character.UnicodeScript.of(codePoint) == Character.UnicodeScript.LATIN
    private fun isCjk(codePoint: Int): Boolean = when (Character.UnicodeScript.of(codePoint)) {
        Character.UnicodeScript.HAN,
        Character.UnicodeScript.HIRAGANA,
        Character.UnicodeScript.KATAKANA,
        Character.UnicodeScript.HANGUL -> true
        else -> false
    }

    override fun close() = chinese.close()

    private data class LanguageRun(val chinese: Boolean, val text: String)

    companion object {
        private val supportedLanguages = setOf("auto", "zh", "en")
    }
}
