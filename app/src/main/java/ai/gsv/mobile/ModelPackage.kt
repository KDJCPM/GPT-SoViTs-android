package ai.gsv.mobile

import android.content.Context
import android.net.Uri
import org.json.JSONObject
import java.io.File
import java.io.InputStream
import java.security.MessageDigest
import java.util.zip.ZipInputStream

data class ModelPackage(
    val root: File,
    val name: String,
    val version: String,
    val sampleRate: Int,
    val runtime: String,
    val formatVersion: Int,
    val executor: String,
    val entrypoint: String,
    val deployable: Boolean,
    val runtimeOptionsVersion: Int = 0,
    val bundleId: String = "",
    val artifactRole: String = "combined",
    val pipelineRoot: File = root,
    val modelRoot: File = root,
    /** Canonical target id for a backend-specific artifact, or `any` for CPU/cross-device data. */
    val targetSoc: String = "any",
    val targetSocFamily: String = "",
    /** Exact HTP architecture from the QAIRT SDK used during conversion. */
    val htpArch: String = "",
    val qairtVersion: String = "",
    /** Stable backend artifact kind, not an Android-side graph description. */
    val backendArtifact: String = "",
    val supportedTargetSocs: Set<String> = emptySet(),
    /** QNN packages may explicitly allow CPU partition fallback during bring-up. */
    val strictCpuFallback: Boolean = true,
) {
    fun runtimeFile(path: String): File {
        val pipelineFile = File(pipelineRoot, path)
        if (pipelineFile.exists()) return pipelineFile
        return File(modelRoot, path)
    }

    fun isSplit() = pipelineRoot.canonicalFile != modelRoot.canonicalFile

    companion object {
        fun import(context: Context, uri: Uri): ModelPackage {
            context.contentResolver.openInputStream(uri).use { raw ->
                requireNotNull(raw) { "无法打开所选文件" }
                return importStream(context,raw, File(context.filesDir, "models/active"))
            }
        }

        fun importFile(context:Context,file:File):ModelPackage=file.inputStream().use { importStream(context,it, File(context.filesDir, "models/active")) }

        fun hasInstalledPipeline(context: Context): Boolean = installedPipelineVersions(context).isNotEmpty()

        fun restorePipelineArchives(context: Context) {
            val archives = File(context.filesDir, "components").also { it.mkdirs() }
            mapOf("v2ProPlus" to "pipeline-v2ProPlus.gsvm", "v4" to "pipeline-v4.gsvm").forEach { (version, name) ->
                val installed = File(context.filesDir, "models/pipelines/$version/manifest.json")
                val archive = File(archives, name)
                if (!installed.isFile && archive.isFile) {
                    runCatching { installPipeline(context, Uri.fromFile(archive), version) }
                }
            }
        }

        fun installedPipelineVersions(context: Context): Set<String> {
            val root = File(context.filesDir, "models/pipelines")
            val legacy = File(context.filesDir, "models/pipeline")
            if (File(legacy, "manifest.json").isFile) {
                runCatching {
                    val version = openExtracted(legacy).version
                    val destination = File(root, version)
                    if (!destination.exists()) {
                        root.mkdirs()
                        require(legacy.renameTo(destination)) { "无法迁移旧 pipeline 组件" }
                    }
                }
            }
            return root.listFiles().orEmpty().mapNotNull { directory ->
                if (directory.name.endsWith(".pending")) return@mapNotNull null
                runCatching {
                    val manifest = JSONObject(File(directory, "manifest.json").readText())
                    if (manifest.optString("artifact_role") == "pipeline") manifest.getString("model_version") else null
                }.getOrNull()
            }.toSet()
        }

        fun installPipeline(
            context: Context,
            pipeline: Uri,
            expectedVersion: String,
        ): ModelPackage {
            val pipelinesRoot = File(context.filesDir, "models/pipelines")
            val pendingRoot = File(pipelinesRoot, "$expectedVersion.pending")
            val pipelinePackage = context.contentResolver.openInputStream(pipeline).use { raw ->
                requireNotNull(raw) { "无法打开 pipeline 包" }
                importStream(context, raw, pendingRoot)
            }
            require(pipelinePackage.artifactRole == "pipeline") { "所选文件不是 pipeline 包" }
            require(pipelinePackage.version == expectedVersion) {
                "选择的是 $expectedVersion，文件实际为 ${pipelinePackage.version}"
            }
            require(pipelinePackage.executor != "qnn-htp") { "不支持该 pipeline 执行后端" }
            val installedRoot = File(pipelinesRoot, expectedVersion)
            pipelinesRoot.mkdirs()
            installedRoot.deleteRecursively()
            require(pendingRoot.renameTo(installedRoot)) { "无法提交 pipeline 组件" }
            return pipelinePackage.copy(root = installedRoot, pipelineRoot = installedRoot, modelRoot = installedRoot)
        }

        fun importModelWithInstalledPipeline(context: Context, model: Uri): ModelPackage {
            val modelRoot = File(context.filesDir, "models/model")
            val modelPackage = context.contentResolver.openInputStream(model).use { raw ->
                requireNotNull(raw) { "无法打开 model 包" }
                importStream(context, raw, modelRoot)
            }
            val pipelineRoot = File(context.filesDir, "models/pipelines/${modelPackage.version}")
            require(File(pipelineRoot, "manifest.json").isFile) {
                "尚未安装 ${modelPackage.version} pipeline 组件"
            }
            val pipelinePackage = openExtracted(pipelineRoot)
            return combineSplit(pipelinePackage, modelPackage, pipelineRoot, modelRoot)
        }

        /** Import independently packaged pipeline and model artifacts and combine them only after
         * both manifests, hashes and compatibility IDs have been checked. */
        fun importPair(context: Context, pipeline: Uri, model: Uri): ModelPackage {
            val pipelineRoot = File(context.filesDir, "models/pipeline")
            val modelRoot = File(context.filesDir, "models/model")
            val pipelinePackage = context.contentResolver.openInputStream(pipeline).use { raw ->
                requireNotNull(raw) { "无法打开 pipeline 包" }
                importStream(context, raw, pipelineRoot)
            }
            val modelPackage = context.contentResolver.openInputStream(model).use { raw ->
                requireNotNull(raw) { "无法打开 model 包" }
                importStream(context, raw, modelRoot)
            }
            return combineSplit(pipelinePackage, modelPackage, pipelineRoot, modelRoot)
        }

        private fun combineSplit(
            pipelinePackage: ModelPackage,
            modelPackage: ModelPackage,
            pipelineRoot: File,
            modelRoot: File,
        ): ModelPackage {
            require(pipelinePackage.version == modelPackage.version) { "pipeline/model 版本不一致" }
            require(pipelinePackage.sampleRate == modelPackage.sampleRate) { "pipeline/model 采样率不一致" }
            require(pipelinePackage.entrypoint == modelPackage.entrypoint) { "pipeline/model 入口不一致" }
            require(pipelinePackage.artifactRole == "pipeline" && modelPackage.artifactRole == "model") { "pipeline/model 角色声明不正确" }
            fun frontendAbi(value: String) = value.replace(Regex(":options\\d+$"), "")
            require(
                pipelinePackage.bundleId.isNotEmpty() &&
                    frontendAbi(pipelinePackage.bundleId) == frontendAbi(modelPackage.bundleId)
            ) { "pipeline/model 前端 ABI 不一致" }
            require(targetsCompatible(pipelinePackage, modelPackage)) { "pipeline/model 目标后端不一致" }
            // The model package owns the executor/backend identity; pipelineRoot contributes
            // frontend assets while modelRoot contributes the selected backend artifact.
            return modelPackage.copy(
                pipelineRoot = pipelineRoot,
                modelRoot = modelRoot,
                artifactRole = "combined",
            )
        }

        fun openExtracted(root:File):ModelPackage {
            val manifest=JSONObject(File(root,"manifest.json").readText())
            val files=manifest.getJSONArray("files");for(i in 0 until files.length()){val item=files.getJSONObject(i);val file=File(root,item.getString("path"));require(file.isFile&&sha256(file)==item.getString("sha256")){"${item.getString("path")} 校验失败"}}
            return fromManifest(root,manifest)
        }

        private fun fromManifest(root:File,manifest:JSONObject):ModelPackage {
            val supported = manifest.optJSONArray("supported_target_socs")?.let { values ->
                mutableSetOf<String>().apply {
                    for (index in 0 until values.length()) add(values.getString(index))
                }
            } ?: emptySet()
            val parsed = ModelPackage(
                root = root,
                name = manifest.getString("name"),
                version = manifest.getString("model_version"),
                sampleRate = manifest.optInt("sample_rate", 32000),
                runtime = manifest.optString("runtime", manifest.optString("executor", "unknown")),
                formatVersion = manifest.getInt("format_version"),
                executor = manifest.optString("executor", "unknown"),
                entrypoint = manifest.optString("entrypoint", "unknown"),
                deployable = manifest.optBoolean("deployable", false),
                runtimeOptionsVersion = manifest.optInt("runtime_options_version", 0),
                bundleId = manifest.optString("bundle_id", ""),
                artifactRole = manifest.optString("artifact_role", "combined"),
                targetSoc = manifest.optString("target_soc", "any"),
                targetSocFamily = manifest.optString("target_soc_family", ""),
                htpArch = manifest.optString("htp_arch", ""),
                qairtVersion = manifest.optString("qairt_version", ""),
                backendArtifact = manifest.optString("backend_artifact", ""),
                supportedTargetSocs = supported,
                strictCpuFallback = manifest.optJSONObject("npu_plan")
                    ?.optBoolean("strict_cpu_fallback", true) ?: true,
            )
            if (parsed.executor == "qnn-htp") {
                require(parsed.targetSoc.isNotBlank() && parsed.targetSoc != "any") { "QNN 包缺少 target_soc" }
                require(parsed.targetSocFamily == "qualcomm_snapdragon_8") { "QNN 包 target_soc_family 不受支持" }
                require(parsed.htpArch.isNotBlank()) { "QNN 包缺少 htp_arch" }
                require(parsed.qairtVersion.isNotBlank()) { "QNN 包缺少 qairt_version" }
                require(parsed.backendArtifact.isNotBlank()) { "QNN 包缺少 backend_artifact" }
            }
            return parsed
        }

        private fun targetsCompatible(pipeline: ModelPackage, model: ModelPackage): Boolean {
            fun compatible(left: String, right: String): Boolean =
                left.isBlank() || left == "any" || right.isBlank() || right == "any" || left == right
            if (!compatible(pipeline.targetSoc, model.targetSoc)) return false
            if (!compatible(pipeline.targetSocFamily, model.targetSocFamily)) return false
            val pipelineTargets = pipeline.supportedTargetSocs
            val modelTargets = model.supportedTargetSocs
            return pipelineTargets.isEmpty() || modelTargets.isEmpty() || pipelineTargets.intersect(modelTargets).isNotEmpty()
        }

        private fun importStream(context:Context,raw:InputStream,unpack:File):ModelPackage {
            unpack.deleteRecursively(); unpack.mkdirs()
            ZipInputStream(raw.buffered()).use { zip ->
                    val first = zip.nextEntry ?: error("空模型包")
                    require(first.name == "manifest.json") { "manifest.json 必须是首个条目" }
                    val manifestBytes = zip.readBytes()
                    val manifest = JSONObject(manifestBytes.toString(Charsets.UTF_8))
                    // Persist the package identity alongside extracted payloads. Startup discovery
                    // and archive recovery must not depend on the in-memory import result.
                    File(unpack, "manifest.json").writeBytes(manifestBytes)
                    val expected = buildMap<String, String> {
                        val files = manifest.getJSONArray("files")
                        for (i in 0 until files.length()) {
                            val item = files.getJSONObject(i); put(item.getString("path"), item.getString("sha256"))
                        }
                    }.toMutableMap()
                    while (true) {
                        val entry = zip.nextEntry ?: break
                        if (entry.isDirectory) continue
                        val hash = expected.remove(entry.name) ?: error("未声明文件 ${entry.name}")
                        val out = File(unpack, entry.name)
                        require(out.canonicalPath.startsWith(unpack.canonicalPath + File.separator)) { "非法模型路径" }
                        out.parentFile?.mkdirs(); out.outputStream().use(zip::copyTo)
                        require(sha256(out) == hash) { "${entry.name} 校验失败" }
                    }
                    require(expected.isEmpty()) { "模型包缺少 ${expected.keys.joinToString()}" }
                return fromManifest(unpack,manifest)
            }
        }

        private fun sha256(file: File): String {
            val digest = MessageDigest.getInstance("SHA-256")
            file.inputStream().use { input ->
                val buffer = ByteArray(1024 * 1024)
                while (true) {
                    val n = input.read(buffer); if (n < 0) break
                    digest.update(buffer, 0, n)
                }
            }
            return digest.digest().joinToString("") { "%02x".format(it) }
        }
    }
}
