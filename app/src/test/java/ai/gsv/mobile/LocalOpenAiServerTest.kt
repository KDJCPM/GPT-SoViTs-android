package ai.gsv.mobile

import fi.iki.elonen.NanoHTTPD
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.net.HttpURLConnection
import java.net.InetAddress
import java.net.ServerSocket
import java.net.URL
import java.nio.file.Files
import java.util.Base64

class LocalOpenAiServerTest {
    private val root = Files.createTempDirectory("gsv-openai-test-").toFile()
    private var server: LocalOpenAiServer? = null

    @After
    fun cleanup() {
        server?.stop()
        root.deleteRecursively()
    }

    @Test
    fun omittedReferenceUsesPresetAndReturnsWav() {
        var captured: SynthesisRequest? = null
        val wav = "RIFF-fake-WAVE".toByteArray()
        startServer(
            isReady = { true },
            synthesize = { request, output ->
                captured = request
                output.writeBytes(wav)
                output
            },
        )

        val response = post(
            JSONObject()
                .put("model", "gpt-sovits-local")
                .put("voice", "loaded-artifact")
                .put("input", "Hello from the local API")
                .put("language", "en")
                .put("response_format", "wav")
                .put("temperature", 0.8)
                .put("top_p", 0.9)
                .put("top_k", 20)
                .put("repetition_penalty", 1.2)
                .put("speed", 1.1)
                .put("sample_steps", 24)
                .put("seed", 1234),
        )

        assertEquals(200, response.status)
        assertEquals("audio/wav", response.contentType.substringBefore(';'))
        assertEquals("QNN HTP (test)", response.backend)
        assertArrayEquals(wav, response.body)
        val request = assertNotNull(captured).let { captured!! }
        assertEquals("Hello from the local API", request.text)
        assertEquals("en", request.language)
        assertEquals(1234L, request.seed)
        assertEquals(0.8f, request.options.temperature)
        assertEquals(0.9f, request.options.topP)
        assertEquals(20, request.options.topK)
        assertEquals(1.2f, request.options.repetitionPenalty)
        assertEquals(1.1f, request.options.speedFactor)
        assertEquals(24, request.options.sampleSteps)
        assertNull(request.reference)
    }

    @Test
    fun temporaryReferenceIsDecodedForOneRequest() {
        val requestLock = Any()
        var decodedSource: File? = null
        var decodedExactSamples: Int? = null
        var captured: SynthesisRequest? = null
        val encoded = Base64.getEncoder().encodeToString("encoded-audio".toByteArray())
            .chunked(4)
            .joinToString("\n")
        val expectedReference = ReferenceInput(
            pcm16k = floatArrayOf(0.1f),
            pcm32k = floatArrayOf(0.1f, 0.2f),
            text = "temporary prompt",
            language = "zh",
        )
        startServer(
            isReady = { true },
            exactSamples = { 80_000 },
            requestLock = requestLock,
            decodeReference = { source, text, language, exact ->
                assertTrue(Thread.holdsLock(requestLock))
                decodedSource = source
                decodedExactSamples = exact
                assertArrayEquals("encoded-audio".toByteArray(), source.readBytes())
                assertEquals("temporary prompt", text)
                assertEquals("zh", language)
                expectedReference
            },
            synthesize = { request, output ->
                assertTrue(Thread.holdsLock(requestLock))
                captured = request
                output.writeBytes("RIFF-reference-WAVE".toByteArray())
                output
            },
        )

        val response = post(
            JSONObject()
                .put("input", "使用临时参考语音")
                .put("response_format", "wav")
                .put("reference_audio", encoded)
                .put("reference_text", "temporary prompt")
                .put("reference_language", "zh"),
        )

        assertEquals(200, response.status)
        assertEquals(80_000, decodedExactSamples)
        assertFalse(requireNotNull(decodedSource).exists())
        assertTrue(captured?.reference === expectedReference)
    }

    @Test
    fun unavailableModelReturnsOpenAiServiceUnavailableError() {
        startServer(
            isReady = { false },
            synthesize = { _, _ -> error("must not synthesize") },
        )

        val response = post(JSONObject().put("input", "not ready"))
        assertEquals(503, response.status)
        val error = JSONObject(response.body.toString(Charsets.UTF_8)).getJSONObject("error")
        assertEquals("server_error", error.getString("type"))
        assertTrue(error.getString("message").contains("not loaded"))
    }

    private fun startServer(
        isReady: () -> Boolean,
        exactSamples: () -> Int? = { null },
        requestLock: Any = Any(),
        synthesize: (SynthesisRequest, File) -> File,
        decodeReference: (File, String, String, Int?) -> ReferenceInput = { _, _, _, _ ->
            error("reference decoder must not be called")
        },
    ) {
        val port = freeLoopbackPort()
        server = LocalOpenAiServer(
            port = port,
            outputDir = File(root, "runtime"),
            isReady = isReady,
            backendName = { "QNN HTP (test)" },
            referenceExactPcm16kSamples = exactSamples,
            synthesize = synthesize,
            requestLock = requestLock,
            decodeReferenceFile = decodeReference,
        ).also { it.start(NanoHTTPD.SOCKET_READ_TIMEOUT, false) }
    }

    private fun post(body: JSONObject): HttpResult {
        val endpoint = requireNotNull(server).endpoint
        val connection = URL("$endpoint/audio/speech").openConnection() as HttpURLConnection
        connection.requestMethod = "POST"
        connection.connectTimeout = 5_000
        connection.readTimeout = 5_000
        connection.doOutput = true
        connection.setRequestProperty("Content-Type", "application/json; charset=utf-8")
        val request = body.toString().toByteArray(Charsets.UTF_8)
        connection.setFixedLengthStreamingMode(request.size)
        return try {
            connection.outputStream.use { it.write(request) }
            val status = connection.responseCode
            val response = (if (status in 200..299) connection.inputStream else connection.errorStream)
                ?.use { it.readBytes() }
                ?: ByteArray(0)
            HttpResult(
                status = status,
                contentType = connection.contentType.orEmpty(),
                backend = connection.getHeaderField("X-GSV-Backend"),
                body = response,
            )
        } finally {
            connection.disconnect()
        }
    }

    private fun freeLoopbackPort(): Int = ServerSocket(
        0,
        1,
        InetAddress.getByName("127.0.0.1"),
    ).use { it.localPort }

    private data class HttpResult(
        val status: Int,
        val contentType: String,
        val backend: String?,
        val body: ByteArray,
    )
}
