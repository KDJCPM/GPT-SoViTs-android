package ai.gsv.mobile

import android.content.Context
import android.media.AudioFormat
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.sin

data class ReferenceInput(
    val pcm16k: FloatArray,
    val pcm32k: FloatArray,
    val text: String,
    val language: String = "auto",
)

class ReferenceDurationMismatch(
    val expectedSeconds: Double,
    val actualSeconds: Double,
) : IllegalArgumentException(
    "reference duration must be exactly %.3f seconds (got %.3f)".format(
        expectedSeconds,
        actualSeconds,
    )
)

object ReferenceAudioDecoder {
    fun decode(
        context: Context,
        uri: Uri,
        text: String,
        language: String,
        exactPcm16kSamples: Int? = null,
    ): ReferenceInput {
        val decoded = decodeMedia { extractor -> extractor.setDataSource(context, uri, null) }
        return decoded.toReference(text, language, exactPcm16kSamples)
    }

    fun decode(
        file: File,
        text: String,
        language: String,
        exactPcm16kSamples: Int? = null,
    ): ReferenceInput {
        val decoded = decodeMedia { extractor -> extractor.setDataSource(file.path) }
        return decoded.toReference(text, language, exactPcm16kSamples)
    }

    private fun DecodedAudio.toReference(
        text: String,
        language: String,
        exactPcm16kSamples: Int?,
    ): ReferenceInput {
        require(text.isNotBlank()) { "reference transcript is required" }
        val normalizedLanguage = language.trim().lowercase()
        require(normalizedLanguage in setOf("auto", "zh", "en")) {
            "reference language must be auto, zh, or en"
        }
        val durationSeconds = samples.size.toDouble() / sampleRate
        require(durationSeconds in MIN_REFERENCE_SECONDS..MAX_REFERENCE_SECONDS) {
            "reference duration must be between 1 and 30 seconds (got %.2f)".format(durationSeconds)
        }
        val pcm16k = resample(samples, sampleRate, 16_000)
        val pcm32k = resample(samples, sampleRate, 32_000)
        if (exactPcm16kSamples != null && pcm16k.size != exactPcm16kSamples) {
            throw ReferenceDurationMismatch(
                exactPcm16kSamples / 16_000.0,
                pcm16k.size / 16_000.0,
            )
        }
        return ReferenceInput(
            pcm16k = pcm16k,
            pcm32k = pcm32k,
            text = text.trim(),
            language = normalizedLanguage,
        )
    }

    private data class DecodedAudio(val samples: FloatArray, val sampleRate: Int)

    private fun decodeMedia(setDataSource: (MediaExtractor) -> Unit): DecodedAudio {
        val extractor = MediaExtractor()
        var codec: MediaCodec? = null
        try {
            setDataSource(extractor)
            val track = (0 until extractor.trackCount).firstOrNull { index ->
                extractor.getTrackFormat(index).getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true
            } ?: error("file has no audio track")
            extractor.selectTrack(track)
            val sourceFormat = extractor.getTrackFormat(track)
            if (sourceFormat.containsKey(MediaFormat.KEY_DURATION)) {
                val durationSeconds = sourceFormat.getLong(MediaFormat.KEY_DURATION) / 1_000_000.0
                require(durationSeconds <= MAX_DECODE_SECONDS) {
                    "reference duration exceeds the 30 second limit"
                }
            }
            val mime = requireNotNull(sourceFormat.getString(MediaFormat.KEY_MIME))
            codec = MediaCodec.createDecoderByType(mime).apply {
                configure(sourceFormat, null, null, 0)
                start()
            }

            val output = ByteArrayOutputStream()
            val info = MediaCodec.BufferInfo()
            var inputEnded = false
            var outputEnded = false
            var outputFormat = sourceFormat
            while (!outputEnded) {
                if (!inputEnded) {
                    val inputIndex = codec.dequeueInputBuffer(TIMEOUT_US)
                    if (inputIndex >= 0) {
                        val buffer = requireNotNull(codec.getInputBuffer(inputIndex)).apply { clear() }
                        val size = extractor.readSampleData(buffer, 0)
                        if (size < 0) {
                            codec.queueInputBuffer(inputIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            inputEnded = true
                        } else {
                            codec.queueInputBuffer(inputIndex, 0, size, extractor.sampleTime, 0)
                            extractor.advance()
                        }
                    }
                }

                when (val outputIndex = codec.dequeueOutputBuffer(info, TIMEOUT_US)) {
                    MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> outputFormat = codec.outputFormat
                    MediaCodec.INFO_TRY_AGAIN_LATER -> Unit
                    else -> if (outputIndex >= 0) {
                        codec.getOutputBuffer(outputIndex)?.let { buffer ->
                            if (info.size > 0) {
                                val nextSize = output.size().toLong() + info.size
                                require(nextSize <= decodedByteLimit(outputFormat)) {
                                    "decoded reference audio exceeds the 30 second limit"
                                }
                                buffer.position(info.offset)
                                buffer.limit(info.offset + info.size)
                                val bytes = ByteArray(info.size)
                                buffer.get(bytes)
                                output.write(bytes)
                            }
                        }
                        outputEnded = info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                        codec.releaseOutputBuffer(outputIndex, false)
                    }
                }
            }

            val sampleRate = outputFormat.getInteger(MediaFormat.KEY_SAMPLE_RATE)
            val channels = outputFormat.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
            val encoding = if (outputFormat.containsKey(MediaFormat.KEY_PCM_ENCODING)) {
                outputFormat.getInteger(MediaFormat.KEY_PCM_ENCODING)
            } else {
                AudioFormat.ENCODING_PCM_16BIT
            }
            return DecodedAudio(toMono(output.toByteArray(), channels, encoding), sampleRate)
        } finally {
            runCatching { codec?.stop() }
            codec?.release()
            extractor.release()
        }
    }

    private fun toMono(bytes: ByteArray, channels: Int, encoding: Int): FloatArray {
        require(channels > 0) { "invalid channel count $channels" }
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val scalarCount = when (encoding) {
            AudioFormat.ENCODING_PCM_FLOAT -> bytes.size / 4
            AudioFormat.ENCODING_PCM_8BIT -> bytes.size
            AudioFormat.ENCODING_PCM_16BIT -> bytes.size / 2
            else -> error("unsupported decoded PCM encoding $encoding")
        }
        val frames = scalarCount / channels
        require(frames > 0) { "decoded audio is empty" }
        return FloatArray(frames) { frame ->
            var sum = 0.0f
            repeat(channels) {
                sum += when (encoding) {
                    AudioFormat.ENCODING_PCM_FLOAT -> buffer.float
                    AudioFormat.ENCODING_PCM_8BIT -> ((buffer.get().toInt() and 0xff) - 128) / 128.0f
                    else -> buffer.short / 32768.0f
                }
            }
            (sum / channels).coerceIn(-1.0f, 1.0f)
        }
    }

    private fun decodedByteLimit(format: MediaFormat): Long {
        val sampleRate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
        val channels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
        require(sampleRate > 0 && channels > 0) { "invalid decoded audio format" }
        val bytesPerScalar = if (format.containsKey(MediaFormat.KEY_PCM_ENCODING)) {
            when (format.getInteger(MediaFormat.KEY_PCM_ENCODING)) {
                AudioFormat.ENCODING_PCM_8BIT -> 1
                AudioFormat.ENCODING_PCM_16BIT -> 2
                AudioFormat.ENCODING_PCM_FLOAT -> 4
                else -> 4
            }
        } else {
            4
        }
        return (sampleRate.toLong() * channels * bytesPerScalar * MAX_DECODE_SECONDS.toLong())
    }

    private fun resample(input: FloatArray, sourceRate: Int, targetRate: Int): FloatArray {
        require(sourceRate > 0 && targetRate > 0)
        if (sourceRate == targetRate) return input.copyOf()
        val outputSize = ((input.size.toLong() * targetRate) / sourceRate).toInt().coerceAtLeast(1)
        val sourcePerOutput = sourceRate.toDouble() / targetRate
        val cutoff = minOf(1.0, targetRate.toDouble() / sourceRate) * 0.94
        return FloatArray(outputSize) { index ->
            val source = index * sourcePerOutput
            val center = floor(source).toInt()
            var weighted = 0.0
            var weightSum = 0.0
            for (sample in center - SINC_RADIUS + 1..center + SINC_RADIUS) {
                if (sample !in input.indices) continue
                val distance = source - sample
                if (abs(distance) >= SINC_RADIUS) continue
                val lowPass = if (abs(distance) < 1e-12) {
                    cutoff
                } else {
                    sin(PI * cutoff * distance) / (PI * distance)
                }
                val window = 0.5 + 0.5 * cos(PI * distance / SINC_RADIUS)
                val weight = lowPass * window
                weighted += input[sample] * weight
                weightSum += weight
            }
            if (abs(weightSum) < 1e-12) 0.0f else (weighted / weightSum).toFloat().coerceIn(-1.0f, 1.0f)
        }
    }

    private const val TIMEOUT_US = 10_000L
    private const val SINC_RADIUS = 24
    private const val MIN_REFERENCE_SECONDS = 1.0
    private const val MAX_REFERENCE_SECONDS = 30.0
    private const val MAX_DECODE_SECONDS = 31.0
}
