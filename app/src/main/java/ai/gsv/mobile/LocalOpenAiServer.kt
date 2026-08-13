package ai.gsv.mobile

import fi.iki.elonen.NanoHTTPD
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileInputStream
import java.util.Base64
import java.util.UUID

/** Loopback-only OpenAI-compatible speech endpoint backed by the currently loaded artifact. */
class LocalOpenAiServer(
    port: Int,
    private val outputDir: File,
    private val isReady: () -> Boolean,
    private val backendName: () -> String,
    private val referenceExactPcm16kSamples: () -> Int?,
    private val synthesize: (SynthesisRequest, File) -> File,
    private val requestLock: Any = Any(),
    private val decodeReferenceFile: (File, String, String, Int?) -> ReferenceInput =
        ReferenceAudioDecoder::decode,
) : NanoHTTPD(LOOPBACK, port) {
    val endpoint: String = "http://$LOOPBACK:$port/v1"

    override fun serve(session: IHTTPSession): Response = try {
        when {
            session.method == Method.OPTIONS -> cors(newFixedLengthResponse(Response.Status.NO_CONTENT, MIME_PLAINTEXT, ""))
            session.method == Method.GET && session.uri == "/v1/models" -> modelsResponse()
            session.method == Method.POST && session.uri == "/v1/audio/speech" -> speechResponse(session)
            session.uri.startsWith("/v1/") -> errorResponse(Response.Status.NOT_FOUND, "unknown endpoint ${session.uri}", "invalid_request_error")
            else -> errorResponse(Response.Status.NOT_FOUND, "not found", "invalid_request_error")
        }
    } catch (error: IllegalArgumentException) {
        errorResponse(Response.Status.BAD_REQUEST, error.message ?: "invalid request", "invalid_request_error")
    } catch (error: ModelUnavailableException) {
        errorResponse(Response.Status.SERVICE_UNAVAILABLE, error.message ?: "model unavailable", "server_error")
    } catch (error: Throwable) {
        errorResponse(Response.Status.INTERNAL_ERROR, error.message ?: error::class.java.simpleName, "server_error")
    }

    private fun speechResponse(session: IHTTPSession): Response = synchronized(requestLock) {
        if (!isReady()) throw ModelUnavailableException("model is not loaded")
        val bodyFiles = HashMap<String, String>()
        session.parseBody(bodyFiles)
        val body = bodyFiles["postData"] ?: throw IllegalArgumentException("request body is required")
        val json = runCatching { JSONObject(body) }
            .getOrElse { throw IllegalArgumentException("request body must be valid JSON") }
        val input = json.optString("input").trim()
        require(input.isNotEmpty()) { "input is required" }
        val format = json.optString("response_format", "wav").lowercase()
        require(format == "wav") { "response_format=$format is unsupported; use wav" }

        val request = SynthesisRequest(
            text = input,
            language = json.optString("language", "auto"),
            seed = json.optLong("seed", -1L),
            options = SynthesisOptions(
                temperature = json.number("temperature", 1.0).toFloat(),
                topP = json.number("top_p", 1.0).toFloat(),
                topK = json.integer("top_k", 10),
                repetitionPenalty = json.number("repetition_penalty", 1.35).toFloat(),
                speedFactor = json.number("speed", 1.0).toFloat(),
                sampleSteps = json.integer("sample_steps", 32),
            ),
            reference = decodeReference(json),
        )
        outputDir.mkdirs()
        val output = File(outputDir, "openai-${UUID.randomUUID()}.wav")
        return try {
            val result = synthesize(request, output)
            val timing = File("${result.path}.timing.json")
            val stream = object : FileInputStream(result) {
                override fun close() {
                    try {
                        super.close()
                    } finally {
                        result.delete()
                        timing.delete()
                    }
                }
            }
            cors(newChunkedResponse(Response.Status.OK, "audio/wav", stream)).apply {
                addHeader("Content-Disposition", "inline; filename=tts.wav")
                addHeader("X-GSV-Backend", backendName())
            }
        } catch (error: Throwable) {
            output.delete()
            File("${output.path}.timing.json").delete()
            throw error
        }
    }

    private fun decodeReference(json: JSONObject): ReferenceInput? {
        val encoded = json.optString("reference_audio", "").trim()
        if (encoded.isEmpty()) return null
        val prompt = json.optString("reference_text", "").trim()
        require(prompt.isNotEmpty()) { "reference_text is required when reference_audio is supplied" }
        require(encoded.count { !it.isWhitespace() } <= MAX_REFERENCE_BASE64_CHARS) {
            "reference_audio exceeds the 50 MiB limit"
        }
        val compact = encoded.filterNot(Char::isWhitespace)
        val bytes = runCatching { Base64.getDecoder().decode(compact) }
            .getOrElse { throw IllegalArgumentException("reference_audio must be base64-encoded audio") }
        require(bytes.size <= MAX_REFERENCE_BYTES) { "reference_audio exceeds the 50 MiB limit" }
        outputDir.mkdirs()
        val source = File(outputDir, "reference-${UUID.randomUUID()}.audio")
        return try {
            source.writeBytes(bytes)
            decodeReferenceFile(
                source,
                prompt,
                json.optString("reference_language", "auto"),
                referenceExactPcm16kSamples(),
            )
        } finally {
            source.delete()
        }
    }

    private fun modelsResponse(): Response {
        val model = JSONObject()
            .put("id", "gpt-sovits-local")
            .put("object", "model")
            .put("created", 0)
            .put("owned_by", "local")
            .put("ready", isReady())
            .put("backend", backendName())
        return jsonResponse(Response.Status.OK, JSONObject().put("object", "list").put("data", JSONArray().put(model)))
    }

    private fun errorResponse(status: Response.Status, message: String, type: String): Response = jsonResponse(
        status,
        JSONObject().put(
            "error",
            JSONObject()
                .put("message", message)
                .put("type", type)
                .put("param", JSONObject.NULL)
                .put("code", JSONObject.NULL),
        ),
    )

    private fun jsonResponse(status: Response.Status, value: JSONObject): Response =
        cors(newFixedLengthResponse(status, "application/json; charset=utf-8", value.toString()))

    private fun cors(response: Response): Response = response.apply {
        addHeader("Access-Control-Allow-Origin", "*")
        addHeader("Access-Control-Allow-Headers", "Authorization, Content-Type")
        addHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    }

    private fun JSONObject.number(name: String, default: Double): Double {
        if (!has(name)) return default
        return opt(name)?.toString()?.toDoubleOrNull()
            ?: throw IllegalArgumentException("$name must be a number")
    }

    private fun JSONObject.integer(name: String, default: Int): Int {
        if (!has(name)) return default
        return opt(name)?.toString()?.toIntOrNull()
            ?: throw IllegalArgumentException("$name must be an integer")
    }

    private class ModelUnavailableException(message: String) : IllegalStateException(message)

    companion object {
        private const val LOOPBACK = "127.0.0.1"
        private const val MAX_REFERENCE_BYTES = 50 * 1024 * 1024
        private const val MAX_REFERENCE_BASE64_CHARS = ((MAX_REFERENCE_BYTES + 2) / 3) * 4
    }
}
