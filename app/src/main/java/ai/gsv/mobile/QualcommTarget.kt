package ai.gsv.mobile

import android.os.Build

/** Product QNN target policy. Unknown and older Qualcomm platforms are deliberately rejected. */
enum class QualcommTargetSoc(
    val id: String,
    val displayName: String,
    private val platformPrefixes: Set<String>,
) {
    SNAPDRAGON_8_GEN_3(
        "snapdragon_8_gen_3",
        "Snapdragon 8 Gen 3",
        setOf("SM8650"),
    ),
    SNAPDRAGON_8_ELITE(
        "snapdragon_8_elite",
        "Snapdragon 8 Elite",
        setOf("SM8750"),
    ),
    SNAPDRAGON_8_ELITE_GEN_5(
        "snapdragon_8_elite_gen_5",
        "Snapdragon 8 Elite Gen 5",
        setOf("SM8850"),
    );

    fun matches(raw: String): Boolean {
        val value = normalize(raw)
        return platformPrefixes.any { value == it || value.startsWith(it) }
    }

    companion object {
        val productTargets: Set<String> = entries.mapTo(linkedSetOf()) { it.id }

        fun fromPlatformStrings(values: Iterable<String>): QualcommTargetSoc? {
            values.firstNotNullOfOrNull { value -> entries.firstOrNull { it.matches(value) } }?.let { return it }
            return null
        }

        /** Read only stable platform identifiers; board/product names are not trusted aliases. */
        fun detect(): QualcommTargetSoc? {
            val values = buildList {
                if (Build.VERSION.SDK_INT >= 31) add(Build.SOC_MODEL)
                add(Build.HARDWARE)
                add(Build.BOARD)
            }
            return fromPlatformStrings(values)
        }

        fun normalize(raw: String): String = raw.uppercase().filter { it in 'A'..'Z' || it in '0'..'9' }
    }
}

data class QualcommTargetStatus(
    val target: QualcommTargetSoc?,
    val observed: List<String>,
) {
    val isProductTarget: Boolean get() = target != null
}

object QualcommTargetPolicy {
    const val FAMILY = "qualcomm_snapdragon_8"

    fun current(): QualcommTargetStatus {
        val observed = buildList {
            if (Build.VERSION.SDK_INT >= 31) add(Build.SOC_MODEL)
            add(Build.HARDWARE)
            add(Build.BOARD)
        }.filter { it.isNotBlank() && !it.equals("unknown", ignoreCase = true) }
        return QualcommTargetStatus(QualcommTargetSoc.fromPlatformStrings(observed), observed)
    }

    fun isCompatible(model: ModelPackage, status: QualcommTargetStatus = current()): Boolean {
        if (model.executor != "qnn-htp" || !status.isProductTarget) return false
        val target = requireNotNull(status.target)
        val declared = model.targetSoc.trim().lowercase()
        if (declared != target.id) return false
        if (model.targetSocFamily != FAMILY) return false
        if (model.supportedTargetSocs.isNotEmpty() && target.id !in model.supportedTargetSocs) return false
        return model.htpArch.isNotBlank() && model.qairtVersion.isNotBlank() && model.backendArtifact.isNotBlank()
    }

    fun explain(model: ModelPackage, status: QualcommTargetStatus = current()): String {
        if (model.executor != "qnn-htp") return "不是 QNN HTP 模型包"
        if (!status.isProductTarget) {
            return "当前设备不在 QNN 产品目标内（仅支持 Snapdragon 8 Gen 3、8 Elite、8 Elite Gen 5）"
        }
        val target = requireNotNull(status.target)
        if (model.targetSoc != target.id) {
            return "QNN 包目标为 ${model.targetSoc}，当前设备为 ${target.displayName}"
        }
        if (model.targetSocFamily != FAMILY) return "QNN 包 target_soc_family 不匹配"
        if (model.supportedTargetSocs.isNotEmpty() && target.id !in model.supportedTargetSocs) {
            return "QNN 包未声明当前 SoC"
        }
        if (model.htpArch.isBlank() || model.qairtVersion.isBlank() || model.backendArtifact.isBlank()) {
            return "QNN 包缺少完整的 HTP/QAIRT/backend artifact 兼容信息"
        }
        return "当前版本未链接 QNN JNI/HTP 运行库"
    }
}
