package ai.gsv.mobile

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtLoggingLevel
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.ShortBuffer

/** ADB-only acceptance probe for externally compiled QAIRT context binaries. */
object QnnEpContextProbe {
    fun run(model: File, resultFile: File): String {
        require(model.isFile) { "EPContext model does not exist: $model" }
        val environment = OrtEnvironment.getEnvironment()
        val profilePrefix = File(resultFile.parentFile, "qnn-epcontext-profile")
        val options = OrtSession.SessionOptions().apply {
            setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO)
            addConfigEntry("session.disable_cpu_ep_fallback", "1")
            enableProfiling(profilePrefix.path)
            addQnn(
                hashMapOf(
                    "backend_type" to "htp",
                    "enable_htp_fp16_precision" to "1",
                    "htp_performance_mode" to "burst",
                    "htp_graph_finalization_optimization_mode" to "3",
                )
            )
        }
        val summary = options.use { sessionOptions ->
            environment.createSession(model.path, sessionOptions).use { session ->
                val inputName = session.inputNames.single()
                val values = shortArrayOf(
                    floatToHalf(1.0f), floatToHalf(2.0f),
                    floatToHalf(3.0f), floatToHalf(4.0f),
                )
                OnnxTensor.createTensor(
                    environment,
                    ShortBuffer.wrap(values),
                    longArrayOf(1, 4),
                    OnnxJavaType.FLOAT16,
                ).use { input ->
                    session.run(mapOf(inputName to input)).use { outputs ->
                        val output = (outputs[0] as OnnxTensor).shortBuffer
                        val result = FloatArray(output.remaining()) { halfToFloat(output.get()) }
                        val profile = session.endProfiling()
                        "input=$inputName output=${result.joinToString(",")} profile=$profile"
                    }
                }
            }
        }
        resultFile.parentFile?.mkdirs()
        resultFile.writeText(summary)
        return summary
    }

    private fun floatToHalf(value: Float): Short {
        val bits = value.toRawBits()
        val sign = (bits ushr 16) and 0x8000
        var exponent = ((bits ushr 23) and 0xff) - 127 + 15
        var mantissa = bits and 0x7fffff
        if (exponent <= 0) {
            if (exponent < -10) return sign.toShort()
            mantissa = (mantissa or 0x800000) shr (1 - exponent)
            return (sign or ((mantissa + 0x1000) shr 13)).toShort()
        }
        if (exponent >= 31) return (sign or 0x7c00).toShort()
        mantissa += 0x1000
        if ((mantissa and 0x800000) != 0) {
            mantissa = 0
            exponent++
            if (exponent >= 31) return (sign or 0x7c00).toShort()
        }
        return (sign or (exponent shl 10) or (mantissa shr 13)).toShort()
    }

    private fun halfToFloat(value: Short): Float {
        val half = value.toInt() and 0xffff
        val sign = (half and 0x8000) shl 16
        var exponent = (half ushr 10) and 0x1f
        var mantissa = half and 0x3ff
        val bits = when (exponent) {
            0 -> {
                if (mantissa == 0) sign
                else {
                    exponent = 1
                    while ((mantissa and 0x400) == 0) { mantissa = mantissa shl 1; exponent-- }
                    mantissa = mantissa and 0x3ff
                    sign or ((exponent + 112) shl 23) or (mantissa shl 13)
                }
            }
            31 -> sign or 0x7f800000 or (mantissa shl 13)
            else -> sign or ((exponent + 112) shl 23) or (mantissa shl 13)
        }
        return Float.fromBits(bits)
    }
}
