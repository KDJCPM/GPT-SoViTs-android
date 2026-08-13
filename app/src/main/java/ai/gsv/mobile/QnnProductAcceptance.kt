package ai.gsv.mobile

import android.content.Context
import android.net.Uri
import fi.iki.elonen.NanoHTTPD
import org.json.JSONObject
import java.io.File
import java.io.RandomAccessFile
import java.net.HttpURLConnection
import java.net.InetAddress
import java.net.ServerSocket
import java.net.URL
import java.util.Base64
import kotlin.math.abs
import kotlin.math.sqrt

/** ADB-only product acceptance that exercises the same package and engine APIs as the UI. */
object QnnProductAcceptance {
    data class Inputs(
        val cpuPipeline: File,
        val cpuModel: File,
        val qnnPipeline: File,
        val qnnModel: File,
        val text: String,
        val referenceAudio: File? = null,
        val referenceText: String = "",
        val referenceLanguage: String = "auto",
    )

    fun run(context: Context, inputs: Inputs, outputRoot: File): String {
        listOf(inputs.cpuPipeline, inputs.cpuModel, inputs.qnnPipeline, inputs.qnnModel).forEach {
            require(it.isFile) { "acceptance input is missing: $it" }
        }
        require(inputs.text.isNotBlank()) { "acceptance text is empty" }
        inputs.referenceAudio?.let {
            require(it.isFile) { "acceptance reference is missing: $it" }
            require(inputs.referenceText.isNotBlank()) { "acceptance reference transcript is empty" }
        }
        outputRoot.mkdirs()
        val resultFile = File(outputRoot, "result.json")
        val presetOutput = File(outputRoot, "preset.wav")
        val referenceOutput = File(outputRoot, "reference.wav")
        val apiPresetOutput = File(outputRoot, "openai-preset.wav")
        val apiReferenceOutput = File(outputRoot, "openai-reference.wav")
        try {
            ModelPackage.installPipeline(
                context,
                Uri.fromFile(inputs.cpuPipeline),
                "v2ProPlus",
            )
            val qnnPipeline = ModelPackage.installQnnPipelineAttachment(
                context,
                Uri.fromFile(inputs.qnnPipeline),
                "v2ProPlus",
            )
            val model = ModelPackage.importQnnModelWithInstalledPipeline(
                context,
                Uri.fromFile(inputs.cpuModel),
                Uri.fromFile(inputs.qnnModel),
            )
            require(model.executor == "qnn-htp" && model.deployable && model.strictCpuFallback)
            val artifactTarget = QualcommTargetPolicy.requireArtifactIdentity(model)
            val deviceTarget = requireNotNull(QualcommTargetPolicy.current().target) {
                "acceptance device is outside the supported QNN product targets"
            }
            require(artifactTarget == deviceTarget) {
                "QNN artifact targets ${artifactTarget.displayName}, but the device is ${deviceTarget.displayName}"
            }
            val presetStats: WavStats
            val referenceStats: WavStats?
            val backend: String
            synchronized(GsvRuntime.engineLock) {
                GsvRuntime.engine.load(model)
                backend = GsvRuntime.engine.backendName
                require(backend.startsWith("QNN HTP")) { "unexpected acceptance backend: $backend" }
                GsvRuntime.engine.synthesize(
                    SynthesisRequest(text = inputs.text, language = "auto"),
                    presetOutput,
                )
                presetStats = inspectWav(presetOutput, model.sampleRate)
                referenceStats = inputs.referenceAudio?.let { source ->
                    val reference = ReferenceAudioDecoder.decode(
                        source,
                        inputs.referenceText,
                        inputs.referenceLanguage,
                        GsvRuntime.engine.referenceExactPcm16kSamples,
                    )
                    GsvRuntime.engine.synthesize(
                        SynthesisRequest(
                            text = inputs.text,
                            language = "auto",
                            reference = reference,
                        ),
                        referenceOutput,
                    )
                    inspectWav(referenceOutput, model.sampleRate)
                }
            }
            val (apiPresetStats, apiReferenceStats) = runOpenAiAcceptance(
                inputs,
                outputRoot,
                apiPresetOutput,
                apiReferenceOutput,
                model.sampleRate,
            )
            val referenceCoverageComplete = referenceStats != null && apiReferenceStats != null
            val result = JSONObject()
                .put("format", "gsv-qnn-product-device-acceptance")
                .put("format_version", 1)
                // This runner proves the engine/API path only. UI and profile checks are separate
                // acceptance gates and must not be represented as a complete product pass here.
                .put("passed", false)
                .put("runner_passed", true)
                .put("product_acceptance_complete", false)
                .put("reference_coverage_complete", referenceCoverageComplete)
                .put("ui_workflow_validated", false)
                .put("qnn_profile_audit_embedded", false)
                .put("backend", backend)
                .put("target_soc", model.targetSoc)
                .put("target_asic", model.targetAsic)
                .put("htp_arch", model.htpArch)
                .put("qairt_version", model.qairtVersion)
                .put("qnn_runtime_version", model.qnnRuntimeVersion)
                .put("cpu_neural_fallback", false)
                .put("pipeline_bundle_id", qnnPipeline.bundleId)
                .put("model_bundle_id", model.bundleId)
                .put("base_model_sha256", model.baseModelSha256)
                .put("preset", presetStats.toJson(presetOutput))
                .put("openai_preset", apiPresetStats.toJson(apiPresetOutput))
            referenceStats?.let { result.put("reference", it.toJson(referenceOutput)) }
            apiReferenceStats?.let { result.put("openai_reference", it.toJson(apiReferenceOutput)) }
            resultFile.writeText(result.toString(2) + "\n")
            val status = if (referenceCoverageComplete) {
                "PASS_ENGINE_API_ONLY"
            } else {
                "PASS_PARTIAL_REFERENCE_NOT_RUN"
            }
            return "$status backend=$backend preset_samples=${presetStats.samples} " +
                "reference_samples=${referenceStats?.samples ?: 0} " +
                "openai_preset_samples=${apiPresetStats.samples} " +
                "openai_reference_samples=${apiReferenceStats?.samples ?: 0} " +
                "result=${resultFile.path}"
        } catch (error: Throwable) {
            resultFile.writeText(
                JSONObject()
                    .put("format", "gsv-qnn-product-device-acceptance")
                    .put("format_version", 1)
                    .put("passed", false)
                    .put("runner_passed", false)
                    .put("product_acceptance_complete", false)
                    .put("error", error.stackTraceToString())
                    .toString(2) + "\n"
            )
            throw error
        }
    }

    private fun runOpenAiAcceptance(
        inputs: Inputs,
        outputRoot: File,
        presetOutput: File,
        referenceOutput: File,
        expectedSampleRate: Int,
    ): Pair<WavStats, WavStats?> {
        val port = freeLoopbackPort()
        val server = LocalOpenAiServer(
            port = port,
            outputDir = File(outputRoot, "openai-runtime"),
            isReady = { GsvRuntime.engine.isLoaded },
            backendName = { GsvRuntime.engine.backendName },
            referenceExactPcm16kSamples = { GsvRuntime.engine.referenceExactPcm16kSamples },
            synthesize = { request, output ->
                synchronized(GsvRuntime.engineLock) {
                    GsvRuntime.engine.synthesize(request, output)
                }
            },
            requestLock = GsvRuntime.engineLock,
        )
        server.start(NanoHTTPD.SOCKET_READ_TIMEOUT, false)
        return try {
            val presetRequest = JSONObject()
                .put("model", "gpt-sovits-local")
                .put("voice", "loaded-artifact")
                .put("input", inputs.text)
                .put("language", "auto")
                .put("response_format", "wav")
            requestOpenAiSpeech(server.endpoint, presetRequest, presetOutput)
            val presetStats = inspectWav(presetOutput, expectedSampleRate)
            val referenceStats = inputs.referenceAudio?.let { source ->
                val referenceRequest = JSONObject(presetRequest.toString())
                    .put(
                        "reference_audio",
                        Base64.getEncoder().encodeToString(source.readBytes()),
                    )
                    .put("reference_text", inputs.referenceText)
                    .put("reference_language", inputs.referenceLanguage)
                requestOpenAiSpeech(server.endpoint, referenceRequest, referenceOutput)
                inspectWav(referenceOutput, expectedSampleRate)
            }
            presetStats to referenceStats
        } finally {
            server.stop()
        }
    }

    private fun requestOpenAiSpeech(endpoint: String, request: JSONObject, output: File) {
        val connection = URL("$endpoint/audio/speech").openConnection() as HttpURLConnection
        connection.requestMethod = "POST"
        connection.connectTimeout = 10_000
        connection.readTimeout = 10 * 60_000
        connection.doOutput = true
        connection.setRequestProperty("Content-Type", "application/json; charset=utf-8")
        val body = request.toString().toByteArray(Charsets.UTF_8)
        connection.setFixedLengthStreamingMode(body.size)
        try {
            connection.outputStream.use { it.write(body) }
            val status = connection.responseCode
            val response = (if (status in 200..299) connection.inputStream else connection.errorStream)
                ?.use { it.readBytes() }
                ?: ByteArray(0)
            require(status == HttpURLConnection.HTTP_OK) {
                "OpenAI speech request failed with HTTP $status: " +
                    response.toString(Charsets.UTF_8).take(4096)
            }
            require(connection.contentType?.substringBefore(';') == "audio/wav") {
                "OpenAI speech response is not audio/wav: ${connection.contentType}"
            }
            require(connection.getHeaderField("X-GSV-Backend")?.startsWith("QNN HTP") == true) {
                "OpenAI speech response did not report the QNN HTP backend"
            }
            require(response.isNotEmpty()) { "OpenAI speech response is empty" }
            output.writeBytes(response)
        } finally {
            connection.disconnect()
        }
    }

    private fun freeLoopbackPort(): Int = ServerSocket(
        0,
        1,
        InetAddress.getByName("127.0.0.1"),
    ).use { it.localPort }

    private data class WavStats(
        val sampleRate: Int,
        val samples: Int,
        val peak: Int,
        val rms: Double,
        val clippedRatio: Double,
    ) {
        fun toJson(path: File) = JSONObject()
            .put("path", path.path)
            .put("bytes", path.length())
            .put("sample_rate", sampleRate)
            .put("samples", samples)
            .put("peak", peak)
            .put("rms", rms)
            .put("clipped_ratio", clippedRatio)
    }

    private fun inspectWav(path: File, expectedSampleRate: Int): WavStats {
        require(path.isFile && path.length() >= 46) { "acceptance WAV is missing or empty" }
        RandomAccessFile(path, "r").use { input ->
            val header = ByteArray(44)
            input.readFully(header)
            require(String(header, 0, 4, Charsets.US_ASCII) == "RIFF")
            require(String(header, 8, 4, Charsets.US_ASCII) == "WAVE")
            fun intLe(offset: Int): Int =
                (header[offset].toInt() and 0xff) or
                    ((header[offset + 1].toInt() and 0xff) shl 8) or
                    ((header[offset + 2].toInt() and 0xff) shl 16) or
                    ((header[offset + 3].toInt() and 0xff) shl 24)
            val sampleRate = intLe(24)
            val dataBytes = intLe(40)
            require(sampleRate == expectedSampleRate) {
                "acceptance WAV sample rate $sampleRate != $expectedSampleRate"
            }
            require(dataBytes > 0 && dataBytes % 2 == 0 && dataBytes.toLong() <= path.length() - 44)
            val samples = dataBytes / 2
            var peak = 0
            var clipped = 0
            var sumSquares = 0.0
            repeat(samples) {
                val low = input.readUnsignedByte()
                val high = input.readUnsignedByte()
                val value = ((high shl 8) or low).toShort().toInt()
                val magnitude = abs(value)
                peak = maxOf(peak, magnitude)
                if (magnitude >= 32767) clipped++
                sumSquares += value.toDouble() * value
            }
            val rms = sqrt(sumSquares / samples)
            val clippedRatio = clipped.toDouble() / samples
            require(peak > 100 && rms > 20.0) { "acceptance WAV is silent: peak=$peak rms=$rms" }
            require(rms < 20000.0 && clippedRatio < 0.01) {
                "acceptance WAV is clipped: rms=$rms clippedRatio=$clippedRatio"
            }
            return WavStats(sampleRate, samples, peak, rms, clippedRatio)
        }
    }
}
