package ai.gsv.mobile

import java.io.Closeable
import java.io.File
import java.io.RandomAccessFile
import android.util.Log
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtProvider
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor

data class SynthesisOptions(
    val temperature: Float = 1.0f,
    val topP: Float = 1.0f,
    val topK: Int = 10,
    val repetitionPenalty: Float = 1.35f,
    val speedFactor: Float = 1.0f,
    val sampleSteps: Int = 32,
) {
    init {
        require(temperature >= 0.0f && temperature <= 2.0f) { "temperature 必须在 0 到 2 之间" }
        require(topP >= 0.0f && topP <= 1.0f) { "top_p 必须在 0 到 1 之间" }
        require(topK in 1..100) { "top_k 必须在 1 到 100 之间" }
        require(repetitionPenalty > 0.0f && repetitionPenalty <= 3.0f) { "重复惩罚必须在 0 到 3 之间" }
        require(speedFactor in 0.25f..4.0f) { "语速必须在 0.25 到 4 之间" }
        require(sampleSteps in 1..128) { "采样步数必须在 1 到 128 之间" }
    }

    fun isDefault() = this == SynthesisOptions()
}

data class SynthesisRequest(
    val text: String,
    val language: String = "auto",
    val seed: Long = -1,
    val options: SynthesisOptions = SynthesisOptions(),
)

interface ExecutionSession : Closeable {
    val displayName: String
    fun synthesize(request: SynthesisRequest, output: File): File
}

interface ExecutionBackend {
    fun supports(model: ModelPackage): Boolean
    fun open(model: ModelPackage): ExecutionSession
}

/**
 * Boundary for the licensed Qualcomm AI Runtime JNI layer.  The repository intentionally does
 * not ship QAIRT headers or proprietary HTP libraries, so the default implementation is a
 * closed/unavailable runtime.  A future JNI module can implement this interface without changing
 * the Android synthesis contract or model conversion format.
 */
interface QnnRuntime {
    val available: Boolean
    fun open(model: ModelPackage, target: QualcommTargetSoc): ExecutionSession
}

private object OrtQnnRuntime : QnnRuntime {
    override val available: Boolean
        get() = runCatching { OrtProvider.QNN in OrtEnvironment.getAvailableProviders() }.getOrDefault(false)

    override fun open(model: ModelPackage, target: QualcommTargetSoc): ExecutionSession {
        require(available) { "ONNX Runtime QNN EP 未加载" }
        require(model.runtimeFile("runtime/frontend/g2pW.onnx").isFile) {
            "当前 QNN 混合包缺少完整 G2PW ONNX 前端"
        }
        return QnnHtpG2pwSession(model, target)
    }
}

class QnnHtpBackend(private val runtime: QnnRuntime = OrtQnnRuntime) : ExecutionBackend {
    override fun supports(model: ModelPackage): Boolean {
        if (!runtime.available) return false
        val status = QualcommTargetPolicy.current()
        if (!status.isProductTarget) return false
        // A target-labelled QNN package is preferred.  During the transition, a complete CPU
        // package can still be opened as a mixed artifact: only its exact G2PW graph is executed
        // by QNN HTP and the frozen acoustic graph remains the CPU reference.
        val targetPackage = QualcommTargetPolicy.isCompatible(model, status)
        val mixedFrontend = model.executor in setOf("torchscript-cpu-single", "torchscript-cpu-staged") &&
            model.runtimeFile("runtime/frontend/g2pW.onnx").isFile
        return targetPackage || mixedFrontend
    }

    override fun open(model: ModelPackage): ExecutionSession {
        val status = QualcommTargetPolicy.current()
        require(status.isProductTarget) {
            "当前设备不在 QNN 产品目标内（仅支持 Snapdragon 8 Gen 3、8 Elite、8 Elite Gen 5）"
        }
        if (model.executor == "qnn-htp") {
            require(QualcommTargetPolicy.isCompatible(model, status)) {
                "QNN HTP artifact 不可用：${QualcommTargetPolicy.explain(model, status)}"
            }
        }
        return runtime.open(model, requireNotNull(status.target))
    }
}

/** Mixed V2PP session: the artifact decides QNN/CPU graph placement; acoustic stays FP32 CPU. */
private class QnnHtpG2pwSession(model: ModelPackage, target: QualcommTargetSoc) : ExecutionSession {
    private val delegate = NativeCpuSession(model, target, strictQnn = model.strictCpuFallback)
    override val displayName: String
        get() = delegate.displayName
    override fun synthesize(request: SynthesisRequest, output: File): File = delegate.synthesize(request, output)
    override fun close() = delegate.close()
}

/**
 * Thin host contract required by AGENTS.md. Topology, conditioning and preprocessing belong to
 * the converted artifact/native runtime, never to this Kotlin layer.
 */
class CpuBackend : ExecutionBackend {
    override fun supports(model: ModelPackage) =
        model.executor in setOf("torchscript-cpu-single", "torchscript-cpu-staged") ||
            (model.executor == "qnn-htp" && !model.strictCpuFallback)
    override fun open(model: ModelPackage): ExecutionSession {
        require(model.entrypoint == "synthesize_utf8_to_pcm16") { "未知入口 ${model.entrypoint}" }
        require(model.deployable) { "CPU 图已生成，但文本前端尚未融合，包未标记为可部署" }
        return if(model.executor=="torchscript-cpu-staged") StagedCpuSession(model) else NativeCpuSession(model)
    }
}

/** Keeps only one large FP32 stage resident at a time. This is numerically identical to the fused
 * graph and is required for quality-first V4 on an 8 GB device. */
private class StagedCpuSession(private val model: ModelPackage) : ExecutionSession {
    override val displayName = "Android CPU + G2PW (FP32 staged)"
    override fun synthesize(request: SynthesisRequest, output: File): File {
        val trace = TimingTrace("synthesis.cpu.staged", output)
        return try {
            val result = TimingContext.with(trace) {
                TimingContext.measure("request.validate") {
                    require(request.text.isNotBlank()) { "文本不能为空" }
                    requireOptions(model, request.options)
                }
                val prepared = TimingContext.measure("frontend.prepare") {
                    FullZhFrontend(model.runtimeFile("runtime/frontend")).use { it.prepare(request.text) }
                }
                val features = TimingContext.measure("bert.module_load") {
                    Module.load(model.runtimeFile("runtime/bert.pt").path)
                }.useModule { bert ->
                    TimingContext.measure("bert.inference") {
                        bert.forward(
                            IValue.from(Tensor.fromBlob(prepared.tokenIds,longArrayOf(prepared.tokenIds.size.toLong()))),
                            IValue.from(Tensor.fromBlob(prepared.word2ph,longArrayOf(prepared.word2ph.size.toLong()))),
                            IValue.from(Tensor.fromBlob(prepared.chineseMask,longArrayOf(prepared.chineseMask.size.toLong())))
                        ).toTensor().let { Tensor.fromBlob(it.dataAsFloatArray,it.shape()) }
                    }
                }
                val acoustic = TimingContext.measure("acoustic.module_load") {
                    Module.load(model.runtimeFile("runtime/acoustic.pt").path)
                }
                val acousticResult = acoustic.useModule { module ->
                    TimingContext.measure("acoustic.inference") {
                        val phone = IValue.from(Tensor.fromBlob(prepared.phoneIds,longArrayOf(1,prepared.phoneIds.size.toLong())))
                        val bert = IValue.from(features)
                        if (model.runtimeOptionsVersion >= 1) module.forward(
                            phone, bert,
                            IValue.from(request.options.temperature.toDouble()),
                            IValue.from(request.options.topK.toLong()),
                            IValue.from(request.options.topP.toDouble()),
                            IValue.from(request.options.repetitionPenalty.toDouble()),
                            IValue.from(request.options.speedFactor.toDouble()),
                            IValue.from(request.options.sampleSteps.toLong()),
                            IValue.from(request.seed),
                        ) else module.forward(phone, bert, IValue.from(10L))
                    }
                }
                TimingContext.measure("pcm.convert_validate_wav_write") { writeResult(acousticResult,output) }
            }
            trace.finish(true)
            result
        } catch (error: Throwable) {
            trace.finish(false, error)
            throw error
        }
    }
    override fun close() = Unit
}

private inline fun <T> Module.useModule(block:(Module)->T):T=try{block(this)}finally{destroy()}

private fun writeResult(result:IValue,output:File):File {
    val tuple=result.toTuple();val sampleRate=tuple[0].toLong().toInt();val pcm32=tuple[1].toTensor().dataAsIntArray
    require(pcm32.isNotEmpty()){ "模型返回空 PCM" };var peak=0;var clipped=0;var sumSquares=0.0
    pcm32.forEach{value->val sample=value.coerceIn(-32768,32767);val magnitude=kotlin.math.abs(sample);peak=maxOf(peak,magnitude);if(magnitude>=32767)clipped++;sumSquares+=sample.toDouble()*sample}
    val rms=kotlin.math.sqrt(sumSquares/pcm32.size);val clippedRatio=clipped.toDouble()/pcm32.size
    require(peak>100&&rms>20.0){"模型返回静音 PCM: peak=$peak rms=$rms"}
    require(rms<20000.0&&clippedRatio<0.01){"模型返回严重削波 PCM: peak=$peak rms=$rms clippedRatio=$clippedRatio"}
    WavWriter.write(output,pcm32.map{it.coerceIn(-32768,32767).toShort()}.toShortArray(),sampleRate);return output
}

private class NativeCpuSession(
    private val model: ModelPackage,
    private val qnnFrontendTarget: QualcommTargetSoc? = null,
    private val strictQnn: Boolean = false,
) : ExecutionSession {
    override val displayName: String
        get() {
            if (qnnFrontendTarget == null) return "Android CPU${if (frontend != null) " + G2PW" else ""}"
            val stats = frontend?.qnnExecutionStats
            return when {
                stats == null -> "QNN HTP pending (${qnnFrontendTarget.displayName}) + CPU acoustic"
                stats.usedQnnCompute -> "QNN HTP G2PW (${qnnFrontendTarget.displayName}) + CPU acoustic"
                stats.usedQnn -> "QNN HTP frontend utility (${stats.qnnNodes} nodes) + CPU G2PW/acoustic"
                strictQnn -> "QNN HTP failed (${qnnFrontendTarget.displayName})"
                else -> "Android CPU fallback (QNN assigned 0 nodes)"
            }
        }
    private val module = TimingContext.measure("acoustic.module_load") {
        Module.load(model.runtimeFile("runtime/pipeline.pt").path)
    }
    private val frontend = model.runtimeFile("runtime/frontend/g2pW.onnx").parentFile.takeIf { File(it,"g2pW.onnx").isFile }
        ?.let { FullZhFrontend(it, qnnFrontendTarget) }
    override fun synthesize(request: SynthesisRequest, output: File): File {
        val trace = TimingTrace(
            if (qnnFrontendTarget == null) "synthesis.cpu" else "synthesis.qnn_mixed",
            output,
        )
        return try {
            val result = TimingContext.with(trace) {
                TimingContext.measure("request.validate") {
                    require(request.text.isNotBlank()) { "文本不能为空" }
                    requireOptions(model, request.options)
                }
                val result = frontend?.let { frontend ->
                    val prepared = TimingContext.measure("frontend.prepare") { frontend.prepare(request.text) }
                    val inputs = TimingContext.measure("acoustic.input_tensor_build") {
                        arrayOf(
                            IValue.from(Tensor.fromBlob(prepared.phoneIds,longArrayOf(prepared.phoneIds.size.toLong()))),
                            IValue.from(Tensor.fromBlob(prepared.tokenIds,longArrayOf(prepared.tokenIds.size.toLong()))),
                            IValue.from(Tensor.fromBlob(prepared.word2ph,longArrayOf(prepared.word2ph.size.toLong()))),
                            IValue.from(Tensor.fromBlob(prepared.chineseMask,longArrayOf(prepared.chineseMask.size.toLong()))),
                        )
                    }
                    TimingContext.measure("acoustic.inference") {
                        val phone = inputs[0]
                        val tokens = inputs[1]
                        val word2ph = inputs[2]
                        val chinese = inputs[3]
                        if (model.runtimeOptionsVersion >= 1) module.runMethod("synthesize_preprocessed_options",
                            phone, tokens, word2ph, chinese, IValue.from(request.seed),
                            IValue.from(request.options.temperature.toDouble()), IValue.from(request.options.topP.toDouble()),
                            IValue.from(request.options.topK.toLong()), IValue.from(request.options.repetitionPenalty.toDouble()),
                            IValue.from(request.options.speedFactor.toDouble()), IValue.from(request.options.sampleSteps.toLong()))
                        else module.runMethod("synthesize_preprocessed", phone, tokens, word2ph, chinese, IValue.from(request.seed))
                    }
                } ?: run {
                    val utf8 = request.text.toByteArray(Charsets.UTF_8)
                    val input = Tensor.fromBlobUnsigned(utf8, longArrayOf(utf8.size.toLong()))
                    TimingContext.measure("acoustic.inference_raw_utf8") {
                        module.forward(IValue.from(input), IValue.from(request.language), IValue.from(request.seed))
                    }
                }
                TimingContext.measure("qnn.profile_finalize") { frontend?.finalizeQnnProfiling() }
                if (qnnFrontendTarget != null) {
                    val stats = frontend?.qnnExecutionStats
                    TimingContext.mark(
                        "qnn.profile_summary",
                        "providers=${stats?.providers?.joinToString(",") ?: "none"} " +
                            "qnnNodes=${stats?.qnnNodes ?: 0} qnnComputeNodes=${stats?.qnnComputeNodes ?: 0} " +
                            "cpuNodes=${stats?.cpuNodes ?: 0} profileNodeUs=${stats?.profileNodeDurationUs ?: 0}",
                    )
                    Log.i("GSV_QNN", "G2PW backend=${displayName} stats=$stats")
                }
                TimingContext.measure("pcm.convert_validate_wav_write") {
                    val tuple = result.toTuple()
                    val sampleRate = tuple[0].toLong().toInt()
                    // PyTorch Android has no Int16 tensor bridge; pipeline returns PCM16 values in Int32.
                    val pcm32 = tuple[1].toTensor().dataAsIntArray
                    require(pcm32.isNotEmpty()){ "模型返回空 PCM" }
                    var peak=0;var clipped=0;var sumSquares=0.0
                    pcm32.forEach { value ->
                        val sample=value.coerceIn(-32768,32767);val magnitude=kotlin.math.abs(sample)
                        peak=maxOf(peak,magnitude);if(magnitude>=32767)clipped++;sumSquares+=sample.toDouble()*sample
                    }
                    val rms=kotlin.math.sqrt(sumSquares/pcm32.size);val clippedRatio=clipped.toDouble()/pcm32.size
                    require(peak>100&&rms>20.0){"模型返回静音 PCM: peak=$peak rms=$rms"}
                    require(rms<20000.0&&clippedRatio<0.01){"模型返回严重削波 PCM: peak=$peak rms=$rms clippedRatio=$clippedRatio"}
                    val pcm = pcm32.map { it.coerceIn(-32768, 32767).toShort() }.toShortArray()
                    WavWriter.write(output, pcm, sampleRate)
                    output
                }
            }
            trace.finish(true)
            result
        } catch (error: Throwable) {
            trace.finish(false, error)
            throw error
        }
    }
    override fun close() { frontend?.close(); module.destroy() }
}

private fun requireOptions(model: ModelPackage, options: SynthesisOptions) {
    if (!options.isDefault()) {
        require(model.runtimeOptionsVersion >= 1) {
            "当前模型包未导出 runtime_options_version=1，请重新转换后再调节采样参数"
        }
    }
}

private object WavWriter {
    fun write(file: File, pcm: ShortArray, sampleRate: Int) {
        RandomAccessFile(file, "rw").use { out ->
            out.setLength(0)
            val dataSize = pcm.size * 2
            out.writeBytes("RIFF"); writeIntLE(out, 36 + dataSize); out.writeBytes("WAVE")
            out.writeBytes("fmt "); writeIntLE(out, 16); writeShortLE(out, 1); writeShortLE(out, 1)
            writeIntLE(out, sampleRate); writeIntLE(out, sampleRate * 2)
            writeShortLE(out, 2); writeShortLE(out, 16); out.writeBytes("data"); writeIntLE(out, dataSize)
            pcm.forEach { writeShortLE(out, it.toInt()) }
        }
    }
    private fun writeIntLE(out: RandomAccessFile, value: Int) {
        out.write(value); out.write(value ushr 8); out.write(value ushr 16); out.write(value ushr 24)
    }
    private fun writeShortLE(out: RandomAccessFile, value: Int) {
        out.write(value); out.write(value ushr 8)
    }
}

class TtsEngine(private val backends: List<ExecutionBackend>) : Closeable {
    @Volatile
    private var session: ExecutionSession? = null
    var lastLoadDiagnostic: String = ""
        private set
    val isLoaded get() = session != null
    val backendName get() = session?.displayName ?: "未加载"
    fun load(model: ModelPackage) {
        val trace = TimingTrace("engine.load")
        try {
            TimingContext.with(trace) {
                TimingContext.measure("previous_session.close") { session?.close() }
                val errors = mutableListOf<String>()
                for (backend in backends) {
                    val supported = TimingContext.measure("backend.supports.${backend.javaClass.simpleName}") {
                        runCatching { backend.supports(model) }.getOrElse { false }
                    }
                    if (!supported) continue
                    try {
                        session = TimingContext.measure("backend.open.${backend.javaClass.simpleName}") { backend.open(model) }
                        lastLoadDiagnostic = if (errors.isEmpty()) "" else errors.joinToString(" | ")
                        TimingContext.mark("backend.selected", session?.displayName)
                        return@with
                    } catch (error: Throwable) {
                        errors += "${backend.javaClass.simpleName}: ${error.message ?: error::class.simpleName}"
                    }
                }
                val suffix = if (errors.isEmpty()) "" else "；尝试记录：${errors.joinToString(" | ")}"
                error("不支持 executor=${model.executor}$suffix")
            }
            trace.finish(true)
        } catch (error: Throwable) {
            trace.finish(false, error)
            throw error
        }
    }
    fun synthesize(request: SynthesisRequest, output: File) =
        requireNotNull(session) { "尚未加载模型" }.synthesize(request, output)
    override fun close() { session?.close(); session=null }
}
