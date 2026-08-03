package ai.gsv.mobile

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtLoggingLevel
import ai.onnxruntime.OrtSession
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.util.HashMap
import java.io.Closeable
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer

data class G2pwInputs(
    val rows: Int,
    val sequenceLength: Int,
    val labelCount: Int,
    val inputIds: LongArray,
    val tokenTypeIds: LongArray,
    val attentionMask: LongArray,
    val phonemeMask: FloatArray,
    val charIds: LongArray,
    val positionIds: LongArray,
)

data class QnnExecutionStats(
    val qnnNodes: Int,
    /** QNN nodes that carry tensor/weight work, excluding tiny shape/index utility nodes. */
    val qnnComputeNodes: Int,
    val cpuNodes: Int,
    val providers: Set<String>,
    val profilePath: String,
    /** Sum of all ORT Node event durations in the profile, in Chrome trace microseconds. */
    val profileNodeDurationUs: Long = 0L,
    /** Sum of QNN Node event durations in the profile, in Chrome trace microseconds. */
    val qnnNodeDurationUs: Long = 0L,
    /** Slowest QNN kernel events, retained for targeted backend investigation. */
    val slowestQnnNodes: List<QnnProfileNode> = emptyList(),
) {
    val usedQnn: Boolean get() = qnnNodes > 0
    val usedQnnCompute: Boolean get() = qnnComputeNodes > 0
}

data class QnnProfileNode(
    val name: String,
    val durationUs: Long,
    val opName: String = "",
)

/** Exact G2PW neural classifier. Tokenization, correction and tone sandhi stay in FullZhFrontend.
 *
 * A strict session disables CPU partition fallback and fails at session creation when HTP cannot
 * accept the complete graph. A mixed session permits CPU partitions, but reports the actual
 * provider assignment from ORT profiling so it cannot be presented as an NPU execution.
 */
class G2pwOnnx(
    model: File,
    qnnTarget: QualcommTargetSoc? = null,
    private val strictQnn: Boolean = false,
    private val staticRows: Int? = null,
    private val staticSequenceLength: Int? = null,
    private val htpFp16Precision: Boolean = false,
    private val htpGraphOptimizationMode: String? = null,
    private val debugLogProbabilities: Boolean = false,
    private val rawLogits: Boolean = false,
) : Closeable {
    init {
        require((staticRows == null) == (staticSequenceLength == null)) {
            "G2PW static shape 必须同时声明 rows 和 sequence length"
        }
        require(staticRows == null || staticRows == 1) {
            "当前静态 G2PW executor 只支持 rows=1 的转换产物"
        }
    }
    private val environment = OrtEnvironment.getEnvironment()
    private val qnnEnabled = qnnTarget != null
    private var profileEnded = false
    var executionStats: QnnExecutionStats? = null
        private set
    private val session = TimingContext.measure("g2pw.session_create") {
        environment.createSession(model.path, OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(2)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            if (qnnTarget != null) {
                setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING)
                addConfigEntry("session.disable_cpu_ep_fallback", if (strictQnn) "1" else "0")
                enableProfiling(File(model.parentFile, "qnn-profile").path)
                val options = HashMap<String, String>().apply {
                    put("backend_type", "htp")
                    // Keep the quality reference explicit. Qualcomm HTP uses FP16 math for FP
                    // operations on this SoC; the provider log/profile is retained for audit.
                    put("enable_htp_fp16_precision", if (htpFp16Precision) "1" else "0")
                    htpGraphOptimizationMode?.let { put("htp_graph_finalization_optimization_mode", it) }
                }
                addQnn(options)
            }
        })
    }

    fun probabilities(input: G2pwInputs): Array<FloatArray> {
        val staticLength = staticSequenceLength
        if (staticLength != null) {
            require(input.sequenceLength <= staticLength) {
                "G2PW 输入长度 ${input.sequenceLength} 超过静态上限 $staticLength"
            }
            val values = Array(input.rows) { row ->
                val padded = padForStatic(input, row, staticLength)
                maskIfRawLogits(
                    TimingContext.measure("g2pw.inference.row_$row") { execute(padded) },
                    padded,
                )[0]
            }
            logProbabilityMargins(values)
            if (qnnEnabled) finishProfiling()
            return values
        }
        val values = maskIfRawLogits(TimingContext.measure("g2pw.inference") { execute(input) }, input)
        logProbabilityMargins(values)
        if (qnnEnabled) finishProfiling()
        return values
    }

    private fun execute(input: G2pwInputs): Array<FloatArray> {
        val sequenceShape = longArrayOf(input.rows.toLong(), input.sequenceLength.toLong())
        val maskShape = longArrayOf(input.rows.toLong(), input.labelCount.toLong())
        OnnxTensor.createTensor(environment, LongBuffer.wrap(input.inputIds), sequenceShape).use { ids ->
            OnnxTensor.createTensor(environment, LongBuffer.wrap(input.tokenTypeIds), sequenceShape).use { types ->
                OnnxTensor.createTensor(environment, LongBuffer.wrap(input.attentionMask), sequenceShape).use { attention ->
                    OnnxTensor.createTensor(environment, FloatBuffer.wrap(input.phonemeMask), maskShape).use { phonemes ->
                        OnnxTensor.createTensor(environment, LongBuffer.wrap(input.charIds), longArrayOf(input.rows.toLong())).use { chars ->
                            OnnxTensor.createTensor(environment, LongBuffer.wrap(input.positionIds), longArrayOf(input.rows.toLong())).use { positions ->
                                return TimingContext.measure("g2pw.ort_session_run") {
                                    session.run(mapOf("input_ids" to ids, "token_type_ids" to types,
                                        "attention_mask" to attention, "phoneme_mask" to phonemes,
                                        "char_ids" to chars, "position_ids" to positions)).use { result ->
                                        @Suppress("UNCHECKED_CAST")
                                        result[0].value as Array<FloatArray>
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private fun padForStatic(input: G2pwInputs, row: Int, sequenceLength: Int): G2pwInputs {
        fun padded(values: LongArray): LongArray {
            val output = LongArray(sequenceLength)
            values.copyInto(output, 0, row * input.sequenceLength, (row + 1) * input.sequenceLength)
            return output
        }
        val phonemes = FloatArray(input.labelCount)
        input.phonemeMask.copyInto(phonemes, 0, row * input.labelCount, (row + 1) * input.labelCount)
        return G2pwInputs(
            rows = 1,
            sequenceLength = sequenceLength,
            labelCount = input.labelCount,
            inputIds = padded(input.inputIds),
            tokenTypeIds = padded(input.tokenTypeIds),
            attentionMask = padded(input.attentionMask),
            phonemeMask = phonemes,
            charIds = longArrayOf(input.charIds[row]),
            positionIds = longArrayOf(input.positionIds[row]),
        )
    }

    private fun logProbabilityMargins(values: Array<FloatArray>) {
        if (!debugLogProbabilities) return
        values.forEachIndexed { row, probabilities ->
            var best = -1
            var second = -1
            for (index in probabilities.indices) {
                if (best < 0 || probabilities[index] > probabilities[best]) {
                    second = best
                    best = index
                } else if (second < 0 || probabilities[index] > probabilities[second]) {
                    second = index
                }
            }
            val top = probabilities.getOrElse(best) { Float.NaN }
            val next = probabilities.getOrElse(second) { Float.NaN }
            Log.i("GSV_G2PW_PROB", "row=$row best=$best top=$top second=$next margin=${top - next}")
        }
    }

    private fun maskIfRawLogits(values: Array<FloatArray>, input: G2pwInputs): Array<FloatArray> {
        if (!rawLogits) return values
        values.forEachIndexed { row, logits ->
            for (label in logits.indices) {
                if (input.phonemeMask[row * input.labelCount + label] <= 0f) {
                    logits[label] = -Float.MAX_VALUE
                }
            }
        }
        return values
    }

    override fun close() {
        if (qnnEnabled) finishProfiling()
        session.close()
    }

    /** Close the audit profile even when a sentence contains no queried polyphonic characters. */
    fun finalizeProfiling() {
        if (qnnEnabled) finishProfiling()
    }

    private fun finishProfiling() {
        if (profileEnded) return
        runCatching { TimingContext.measure("qnn.profile_end") { session.endProfiling() } }
            .onSuccess { path ->
                executionStats = TimingContext.measure("qnn.profile_parse") { readExecutionStats(path) }
                val stats = executionStats
                Log.i(
                    "GSV_QNN",
                    "G2PW profiling=$path providers=${stats?.providers?.joinToString(",") ?: "none"} " +
                        "qnnNodes=${stats?.qnnNodes ?: 0} qnnComputeNodes=${stats?.qnnComputeNodes ?: 0} " +
                        "cpuNodes=${stats?.cpuNodes ?: 0} profileNodeUs=${stats?.profileNodeDurationUs ?: 0} " +
                        "qnnNodeUs=${stats?.qnnNodeDurationUs ?: 0} " +
                        "slowestQnn=${stats?.slowestQnnNodes?.joinToString(";") { "${it.opName.ifBlank { it.name }}:${it.durationUs}us" } ?: "none"} " +
                        "strict=$strictQnn",
                )
            }
            .onFailure { error -> Log.w("GSV_QNN", "profiling close failed", error) }
        profileEnded = true
    }

    private fun readExecutionStats(path: String): QnnExecutionStats {
        val providers = linkedSetOf<String>()
        var qnnNodes = 0
        var qnnComputeNodes = 0
        var cpuNodes = 0
        var profileNodeDurationUs = 0L
        var qnnNodeDurationUs = 0L
        val qnnProfileNodes = ArrayList<QnnProfileNode>()
        val file = File(path)
        if (file.isFile) {
            runCatching {
                val events = JSONArray(file.readText())
                for (index in 0 until events.length()) {
                    val event = events.optJSONObject(index) ?: continue
                    if (event.optString("cat") != "Node") continue
                    val durationUs = event.optString("dur").toLongOrNull() ?: event.optLong("dur", 0L)
                    profileNodeDurationUs += durationUs
                    val args = event.optJSONObject("args")
                    val provider = listOf(
                        args?.optString("provider"),
                        args?.optString("execution_provider"),
                        event.optString("provider"),
                    ).firstOrNull { !it.isNullOrBlank() } ?: continue
                    providers += provider
                    when {
                        provider.equals("QNNExecutionProvider", ignoreCase = true) ||
                            provider.equals("QNN", ignoreCase = true) -> {
                            qnnNodes++
                            qnnNodeDurationUs += durationUs
                            qnnProfileNodes += QnnProfileNode(
                                name = event.optString("name", "unknown"),
                                durationUs = durationUs,
                                opName = args?.optString("op_name", "") ?: "",
                            )
                            if (isComputeNode(args)) qnnComputeNodes++
                        }
                        provider.equals("CPUExecutionProvider", ignoreCase = true) ||
                            provider.equals("CPU", ignoreCase = true) -> cpuNodes++
                    }
                }
            }.onFailure { error -> Log.w("GSV_QNN", "profiling parse failed: ${error.message}") }
        }
        return QnnExecutionStats(
            qnnNodes = qnnNodes,
            qnnComputeNodes = qnnComputeNodes,
            cpuNodes = cpuNodes,
            providers = providers,
            profilePath = path,
            profileNodeDurationUs = profileNodeDurationUs,
            qnnNodeDurationUs = qnnNodeDurationUs,
            slowestQnnNodes = qnnProfileNodes.sortedByDescending { it.durationUs }.take(10),
        )
    }

    /**
     * ORT's QNN profile also contains Shape/Gather bookkeeping partitions. Count a partition as
     * compute only when it carries a non-trivial tensor/weight payload or a non-index datatype.
     * This keeps the UI from calling a graph NPU-backed when HTP only accepted shape plumbing.
     */
    private fun isComputeNode(args: JSONObject?): Boolean {
        if (args == null) return false
        val parameterBytes = args.optString("parameter_size").toLongOrNull() ?: 0L
        val outputBytes = args.optString("output_size").toLongOrNull() ?: 0L
        if (parameterBytes > 1024L || outputBytes > 1024L) return true
        for (field in listOf("input_type_shape", "output_type_shape")) {
            val values = args.optJSONArray(field) ?: continue
            for (index in 0 until values.length()) {
                val tensor = values.optJSONObject(index) ?: continue
                val types = tensor.keys()
                while (types.hasNext()) {
                    when (types.next().lowercase()) {
                        "float", "float16", "double", "int8", "uint8", "int16", "uint16" ->
                            return true
                    }
                }
            }
        }
        return false
    }
}
