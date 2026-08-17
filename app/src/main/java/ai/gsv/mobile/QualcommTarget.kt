package ai.gsv.mobile

import android.os.Build

/** Product QNN target policy. Unknown and older Qualcomm platforms are deliberately rejected. */
enum class QualcommTargetSoc(
    val id: String,
    val displayName: String,
    val asic: String,
    val socModel: Int,
    val htpArch: String,
    private val platformPrefixes: Set<String>,
) {
    SNAPDRAGON_8_GEN_3(
        "snapdragon_8_gen_3",
        "Snapdragon 8 Gen 3",
        "SM8650",
        57,
        "V75",
        setOf("SM8650"),
    ),
    SNAPDRAGON_8_ELITE(
        "snapdragon_8_elite",
        "Snapdragon 8 Elite",
        "SM8750",
        69,
        "V79",
        setOf("SM8750"),
    ),
    SNAPDRAGON_8_GEN_5(
        "snapdragon_8_gen_5",
        "Snapdragon 8 Gen 5",
        "SM8850",
        87,
        "V81",
        setOf("SM8845"),
    ),
    SNAPDRAGON_8_ELITE_GEN_5(
        "snapdragon_8_elite_gen_5",
        "Snapdragon 8 Elite Gen 5",
        "SM8850",
        87,
        "V81",
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
    const val RUNTIME_VERSION = "2.48.0"

    fun requireArtifactIdentity(model: ModelPackage): QualcommTargetSoc {
        require(model.executor == "qnn-htp") { "QNN package executor must be qnn-htp" }
        val target = QualcommTargetSoc.entries.firstOrNull { it.id == model.targetSoc }
        require(target != null) { "QNN package target_soc is outside the product allowlist" }
        require(model.targetSocFamily == FAMILY) { "QNN package target_soc_family is unsupported" }
        require(model.targetAsic == target.asic) {
            "QNN package target_asic ${model.targetAsic} does not match ${target.asic}"
        }
        require(model.targetSocModel == target.socModel) {
            "QNN package target_soc_model ${model.targetSocModel} does not match ${target.socModel}"
        }
        require(model.htpArch == target.htpArch) {
            "QNN package htp_arch ${model.htpArch} does not match ${target.htpArch}"
        }
        require(model.supportedTargetSocs == setOf(target.id)) {
            "QNN package supported_target_socs must contain only ${target.id}"
        }
        val qairtParts = model.qairtVersion.split('.')
        require(
            qairtParts.size == 4 && qairtParts.all { part -> part.toIntOrNull() != null } &&
                qairtParts.take(3).joinToString(".") == RUNTIME_VERSION
        ) {
            "QNN package QAIRT ${model.qairtVersion} does not match Android QNN $RUNTIME_VERSION"
        }
        require(model.qnnRuntimeVersion == RUNTIME_VERSION) {
            "QNN package runtime ${model.qnnRuntimeVersion} does not match Android QNN $RUNTIME_VERSION"
        }
        require(model.backendArtifact.isNotBlank()) { "QNN package is missing backend_artifact" }
        return target
    }

    fun current(): QualcommTargetStatus {
        val observed = buildList {
            if (Build.VERSION.SDK_INT >= 31) add(Build.SOC_MODEL)
            add(Build.HARDWARE)
        }.filter { it.isNotBlank() && !it.equals("unknown", ignoreCase = true) }
        return QualcommTargetStatus(QualcommTargetSoc.fromPlatformStrings(observed), observed)
    }

    fun isCompatible(model: ModelPackage, status: QualcommTargetStatus = current()): Boolean {
        if (!status.isProductTarget) return false
        val artifactTarget = runCatching { requireArtifactIdentity(model) }.getOrNull() ?: return false
        return artifactTarget == status.target
    }

    fun explain(model: ModelPackage, status: QualcommTargetStatus = current()): String {
        if (model.executor != "qnn-htp") return "not a QNN HTP model package"
        if (!status.isProductTarget) {
            return "device is outside the QNN product targets (Snapdragon 8 Gen 3, 8 Elite, and 8 Elite Gen 5)"
        }
        val target = requireNotNull(status.target)
        if (model.targetSoc != target.id) {
            return "QNN package targets ${model.targetSoc}, but this device is ${target.displayName}"
        }
        if (model.targetSocFamily != FAMILY) return "QNN package target_soc_family does not match"
        if (!model.targetAsic.equals(target.asic, ignoreCase = true)) {
            return "QNN package ASIC is ${model.targetAsic}, but ${target.displayName} requires ${target.asic}"
        }
        if (model.targetSocModel != target.socModel) {
            return "QNN package socModel=${model.targetSocModel}, but ${target.displayName} requires ${target.socModel}"
        }
        if (model.supportedTargetSocs.isNotEmpty() && target.id !in model.supportedTargetSocs) {
            return "QNN package does not declare this SoC"
        }
        if (!model.htpArch.equals(target.htpArch, ignoreCase = true)) {
            return "QNN package HTP architecture is ${model.htpArch}, but ${target.displayName} requires ${target.htpArch}"
        }
        if (model.qairtVersion.isBlank() || model.backendArtifact.isBlank()) {
            return "QNN package is missing complete HTP, QAIRT, or backend artifact compatibility metadata"
        }
        if (runCatching { requireArtifactIdentity(model) }.isFailure) {
            return "QNN package backend identity is internally inconsistent"
        }
        return "this build does not link the QNN JNI/HTP runtime"
    }
}
