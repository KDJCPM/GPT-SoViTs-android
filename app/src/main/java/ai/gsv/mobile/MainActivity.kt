package ai.gsv.mobile

import android.Manifest
import android.media.MediaPlayer
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.Handler
import android.os.ResultReceiver
import android.provider.OpenableColumns
import android.provider.Settings
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.animation.animateContentSize
import androidx.compose.foundation.clickable
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.OpenInNew
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.ExpandLess
import androidx.compose.material.icons.filled.ExpandMore
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.SaveAlt
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.UploadFile
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.res.stringResource
import androidx.lifecycle.lifecycleScope
import androidx.core.content.ContextCompat
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import org.json.JSONArray
import org.json.JSONObject

@OptIn(ExperimentalMaterial3Api::class)
class MainActivity : ComponentActivity() {
    private val engine: TtsEngine get() = GsvRuntime.engine
    private val engineLock: Any get() = GsvRuntime.engineLock
    private var output: File? = null
    private var player: MediaPlayer? = null
    private var pendingQnnModelRecord: ModelRecord? = null
    private var pendingApiPort: Int? = null

    private var ui by mutableStateOf(UiState())

    private val pickCombined = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri?.let { rememberAndLoadModel(it, split = false) }
    }
    private val pickPipelines = registerForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { uris ->
        if (uris.isNotEmpty()) installPipelines(uris)
    }
    private val pickQnnPipelineAttachment = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri?.let(::installQnnPipelineAttachment)
    }
    private val pickModel = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri?.let { rememberAndLoadModel(it, split = true) }
    }
    private val pickQnnModelAttachment = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        val record = pendingQnnModelRecord
        pendingQnnModelRecord = null
        if (uri != null && record != null) {
            runCatching { contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION) }
            loadQnnModelUri(record, uri, remember = true)
        }
    }
    private val pickReference = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri?.let {
            runCatching { contentResolver.takePersistableUriPermission(it, Intent.FLAG_GRANT_READ_URI_PERMISSION) }
            ui = ui.copy(referenceUri = it, referenceName = displayName(it))
        }
    }
    private val saveOutput = registerForActivityResult(ActivityResultContracts.CreateDocument("audio/wav")) { uri ->
        val source = output
        if (uri != null && source != null) {
            setBusy(getString(R.string.saving_audio))
            lifecycleScope.launch {
                runCatching {
                    withContext(Dispatchers.IO) {
                        require(source.isFile) { getString(R.string.audio_output_missing) }
                        contentResolver.openOutputStream(uri, "w").use { destination ->
                            requireNotNull(destination) { getString(R.string.audio_output_open_failed) }
                            source.inputStream().use { it.copyTo(destination) }
                        }
                    }
                }.onSuccess {
                    ui = ui.copy(busy = false, status = getString(R.string.audio_saved))
                }.onFailure {
                    ui = ui.copy(
                        busy = false,
                        status = getString(R.string.audio_save_failed, it.message.orEmpty()),
                    )
                }
            }
        }
    }
    private val requestExternalRead = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) scanExternalModels(silent = false)
        else ui = ui.copy(status = getString(R.string.external_models_permission_required))
    }
    private val manageExternalStorage = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) {
        if (hasExternalModelAccess()) scanExternalModels(silent = false)
        else ui = ui.copy(status = getString(R.string.external_models_permission_required))
    }
    private val requestNotifications = registerForActivityResult(ActivityResultContracts.RequestPermission()) {
        pendingApiPort?.let(::startApiService)
        pendingApiPort = null
    }

    override fun attachBaseContext(newBase: Context) {
        super.attachBaseContext(AppLocale.wrap(newBase))
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        GsvRuntime.retainActivity()
        val prefs = getSharedPreferences("components", MODE_PRIVATE)
        ModelPackage.restorePipelineArchives(this)
        val loaded = engine.loadedPackage
        val restoredOutput = File(cacheDir, "tts.wav").takeIf {
            savedInstanceState?.getBoolean(STATE_CAN_PLAY) == true && it.isFile
        }
        output = restoredOutput
        ui = restoreSynthesisState(savedInstanceState).copy(
            pipelineInstalled = ModelPackage.hasInstalledPipeline(this),
            installedVersions = ModelPackage.installedPipelineVersions(this),
            qnnPipelines = installedQnnPipelines(),
            componentUrl = prefs.getString("pipeline_url_v2pp", null) ?: ComponentVersion.V2PP.defaultUrl,
            showFirstRun = savedInstanceState?.getBoolean(STATE_SHOW_FIRST_RUN)
                ?: !ModelPackage.hasInstalledPipeline(this),
            port = intent.getIntExtra("api_port", 9880).toString(),
            models = readModelRecords(),
            appLanguage = AppLocale.selected(this),
            modelInfo = loaded?.let { "${it.name} / ${it.version} / ${it.sampleRate} Hz" }
                ?: getString(R.string.model_not_loaded),
            backend = loaded?.let { engine.backendName } ?: getString(R.string.backend_not_loaded),
            runtimeOptions = loaded?.runtimeOptions ?: emptySet(),
            referenceOverrideSupported = loaded?.referenceInputVersion?.let { it >= 1 } ?: false,
            referenceExactPcm16kSamples = loaded?.referencePcm16kSamples?.takeIf {
                loaded.referenceDurationPolicy == "exact_samples" && it > 0
            },
            canPlay = restoredOutput != null,
            serverEnabled = GsvRuntime.apiEndpoint != null,
            serverStatus = GsvRuntime.apiEndpoint?.let { "$it · POST /v1/audio/speech" }
                ?: GsvRuntime.apiError?.let { getString(R.string.api_start_failed, it) }
                ?: getString(R.string.api_stopped),
        )
        setContent { GsvTheme { MainScreen() } }
        DebugAcceptanceRunner.launch(this, intent)
        if (hasExternalModelAccess()) scanExternalModels(silent = true)
        if (intent.getBooleanExtra("start_api", false)) startApiServer()
    }

    private fun requestExternalModelScan() {
        if (hasExternalModelAccess()) {
            scanExternalModels(silent = false)
            return
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            val appSettings = Intent(
                Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,
                Uri.parse("package:$packageName"),
            )
            runCatching { manageExternalStorage.launch(appSettings) }
                .onFailure { manageExternalStorage.launch(Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION)) }
        } else {
            requestExternalRead.launch(Manifest.permission.READ_EXTERNAL_STORAGE)
        }
    }

    private fun hasExternalModelAccess(): Boolean = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
        Environment.isExternalStorageManager()
    } else {
        checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED
    }

    private fun scanExternalModels(silent: Boolean) {
        lifecycleScope.launch {
            runCatching {
                withContext(Dispatchers.IO) {
                    val directory = externalModelDirectory()
                    require(directory.exists() || directory.mkdirs()) { "directory could not be created" }
                    val records = directory.listFiles()
                        .orEmpty()
                        .filter {
                            it.isFile &&
                                it.name.endsWith(".gsvm", ignoreCase = true) &&
                                !it.name.endsWith(QNN_ATTACHMENT_SUFFIX, ignoreCase = true)
                        }
                        .sortedBy { it.name.lowercase() }
                        .mapNotNull(::readExternalModelRecord)
                    directory to records
                }
            }.onSuccess { (directory, discovered) ->
                val directoryUri = Uri.fromFile(directory).toString().trimEnd('/') + "/"
                val existing = ui.models.associateBy(ModelRecord::uri)
                val refreshed = discovered.map { record ->
                    existing[record.uri]?.let { previous ->
                        record.copy(
                            qnnUri = previous.qnnUri,
                            baseModelSha256 = previous.baseModelSha256,
                        )
                    } ?: record
                }
                val retained = ui.models.filterNot { it.uri.startsWith(directoryUri) }
                writeModelRecords((refreshed + retained).distinctBy { it.uri })
                if (!silent || discovered.isNotEmpty()) {
                    ui = ui.copy(status = getString(R.string.external_models_scanned, discovered.size))
                }
            }.onFailure { error ->
                if (!silent) ui = ui.copy(
                    status = getString(R.string.external_models_scan_failed, error.message.orEmpty()),
                )
            }
        }
    }

    private fun readExternalModelRecord(file: File): ModelRecord? = runCatching {
        val manifest = file.inputStream().use(ModelPackage::inspectManifest)
        val role = manifest.optString("artifact_role", "combined")
        if (role == "pipeline" || role.startsWith("qnn-") || manifest.optString("executor") == "qnn-htp") {
            return@runCatching null
        }
        ModelRecord(
            uri = Uri.fromFile(file).toString(),
            name = manifest.getString("name"),
            version = manifest.getString("model_version"),
            split = role == "model",
        )
    }.getOrNull()

    private fun externalModelDirectory(): File = File(Environment.getExternalStorageDirectory(), "models/gs")

    private fun installPipelines(uris: List<Uri>) {
        setBusy(getString(R.string.pipeline_installing))
        lifecycleScope.launch {
            runCatching {
                withContext(Dispatchers.IO) {
                    val selectedIds = ui.selectedVersions.map { it.manifestId }.toSet()
                    uris.map { uri ->
                        val version = inspectPipelineVersion(uri)
                        require(version in selectedIds) {
                            getString(R.string.pipeline_version_not_selected, version)
                        }
                        ModelPackage.installPipeline(this@MainActivity, uri, version)
                    }
                }
            }
                .onSuccess {
                    val installed = ModelPackage.installedPipelineVersions(this@MainActivity)
                    ui = ui.copy(busy = false, pipelineInstalled = installed.isNotEmpty(), installedVersions = installed, showFirstRun = false, status = getString(R.string.pipeline_installed_versions, it.joinToString { item -> item.version }))
                }
                .onFailure { ui = ui.copy(busy = false, status = getString(R.string.pipeline_import_failed, it.message.orEmpty())) }
        }
    }

    private fun installQnnPipelineAttachment(uri: Uri) {
        runCatching { contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION) }
        setBusy(getString(R.string.qnn_pipeline_installing))
        lifecycleScope.launch {
            runCatching {
                withContext(Dispatchers.IO) {
                    val version = inspectPipelineVersion(uri)
                    ModelPackage.installQnnPipelineAttachment(this@MainActivity, uri, version)
                }
            }.onSuccess { attachment ->
                ui = ui.copy(
                    busy = false,
                    qnnPipelines = installedQnnPipelines(),
                    status = getString(
                        R.string.qnn_pipeline_installed,
                        attachment.version,
                        attachment.targetSoc,
                    ),
                )
            }.onFailure { error ->
                ui = ui.copy(
                    busy = false,
                    status = getString(R.string.qnn_pipeline_import_failed, error.message.orEmpty()),
                )
            }
        }
    }

    private fun installedQnnPipelines(): Set<String> = ComponentVersion.entries.flatMapTo(linkedSetOf()) { version ->
        ModelPackage.installedQnnPipelineTargets(this, version.manifestId).map { target ->
            "${version.manifestId}:$target"
        }
    }

    private fun inspectPipelineVersion(uri: Uri): String = contentResolver.openInputStream(uri).use { raw ->
        requireNotNull(raw) { getString(R.string.pipeline_open_failed) }
        ModelPackage.inspectManifest(raw).getString("model_version")
    }

    private fun selectComponentVersion(version: ComponentVersion) {
        val prefs = getSharedPreferences("components", MODE_PRIVATE)
        ui = ui.copy(
            componentVersion = version,
            componentUrl = prefs.getString(version.preferenceKey, null) ?: version.defaultUrl,
        )
    }

    private fun downloadPipelines() {
        val versions = ui.selectedVersions.sortedBy { it.ordinal }
        if (versions.isEmpty()) { ui = ui.copy(status = getString(R.string.pipeline_at_least_one)); return }
        setBusy(getString(R.string.pipeline_preparing_download))
        ui = ui.copy(downloadProgress = 0f)
        lifecycleScope.launch {
            runCatching {
                versions.forEachIndexed { index, version ->
                    withContext(Dispatchers.IO) {
                        val prefs = getSharedPreferences("components", MODE_PRIVATE)
                        val requestedUrl = (prefs.getString(version.preferenceKey, null) ?: version.defaultUrl).trim()
                            .replace("https://huggingface.co/", "https://hf-mirror.com/")
                        require(requestedUrl.startsWith("https://")) {
                            getString(R.string.component_url_https, version.label)
                        }
                        downloadPipeline(version, requestedUrl, index, versions.size)
                    }
                }
            }.onSuccess {
                val installed = ModelPackage.installedPipelineVersions(this@MainActivity)
                ui = ui.copy(busy = false, downloadProgress = null, pipelineInstalled = installed.isNotEmpty(), installedVersions = installed, showFirstRun = false, status = getString(R.string.pipeline_install_complete, versions.joinToString { it.label }))
            }.onFailure {
                val installed = ModelPackage.installedPipelineVersions(this@MainActivity)
                ui = ui.copy(busy = false, downloadProgress = null, pipelineInstalled = installed.isNotEmpty(), installedVersions = installed, status = getString(R.string.pipeline_download_failed, it.message.orEmpty()))
            }
        }
    }

    private suspend fun downloadPipeline(version: ComponentVersion, requestedUrl: String, index: Int, count: Int) {
        val archiveDir = File(filesDir, "components").also { it.mkdirs() }
        val archive = File(archiveDir, "pipeline-${version.manifestId}.gsvm")
        val download = File(archiveDir, "pipeline-${version.manifestId}.gsvm.partial")
        try {
                    if (archive.isFile && fileSha256(archive) == version.sha256) {
                        ModelPackage.installPipeline(this@MainActivity, Uri.fromFile(archive), version.manifestId)
                        withContext(Dispatchers.Main) { ui = ui.copy(status = getString(R.string.pipeline_restored, version.label)) }
                        return
                    }
                    val connection = URL(requestedUrl).openConnection() as HttpURLConnection
                    connection.instanceFollowRedirects = true
                    connection.connectTimeout = 20_000
                    connection.readTimeout = 60_000
                    connection.setRequestProperty("User-Agent", "GSV-Mobile/0.1")
                    try {
                        require(connection.responseCode in 200..299) {
                            getString(R.string.download_http_failed, connection.responseCode)
                        }
                        val total = connection.contentLengthLong
                        connection.inputStream.buffered().use { input ->
                            download.outputStream().buffered().use { output ->
                                val buffer = ByteArray(1024 * 1024)
                                var received = 0L
                                while (true) {
                                    val read = input.read(buffer)
                                    if (read < 0) break
                                    output.write(buffer, 0, read)
                                    received += read
                                    if (total > 0) withContext(Dispatchers.Main) {
                                        val fileProgress = received.toFloat() / total.toFloat()
                                        ui = ui.copy(downloadProgress = (index + fileProgress) / count, status = getString(R.string.pipeline_downloading, version.label, received / 1024 / 1024, total / 1024 / 1024))
                                    }
                                }
                            }
                        }
                    } finally { connection.disconnect() }
                    require(fileSha256(download) == version.sha256) {
                        getString(R.string.download_sha_mismatch)
                    }
                    ModelPackage.installPipeline(
                        this@MainActivity, Uri.fromFile(download), version.manifestId,
                    )
                    if (archive.exists()) archive.delete()
                    require(download.renameTo(archive)) {
                        getString(R.string.download_persist_failed)
                    }
        } finally { if (download.exists()) download.delete() }
    }

    private fun fileSha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buffer = ByteArray(1024 * 1024)
            while (true) { val count = input.read(buffer); if (count < 0) break; digest.update(buffer, 0, count) }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    private fun rememberAndLoadModel(uri: Uri, split: Boolean) {
        runCatching {
            contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        loadModelUri(uri, split, remember = true)
    }

    private fun loadModelUri(uri: Uri, split: Boolean, remember: Boolean) {
        if (!modelUriExists(uri)) {
            removeModelRecord(uri.toString())
            ui = ui.copy(status = getString(R.string.model_missing_removed))
            return
        }
        val message = getString(if (split) R.string.model_loading else R.string.model_importing)
        loadPackage(message, importer = {
            if (split) ModelPackage.importModelWithInstalledPipeline(this, uri) else ModelPackage.import(this, uri)
        }) { model ->
            if (remember) saveModelRecord(ModelRecord(uri.toString(), model.name, model.version, split))
        }
    }

    private fun chooseQnnModelAttachment(record: ModelRecord) {
        pendingQnnModelRecord = record
        pickQnnModelAttachment.launch(qnnPackageTypes)
    }

    private fun loadQnnModelUri(record: ModelRecord, attachment: Uri, remember: Boolean) {
        val base = Uri.parse(record.uri)
        if (remember && !modelUriExists(base)) {
            removeModelRecord(record.uri)
            ui = ui.copy(status = getString(R.string.model_missing_removed))
            return
        }
        loadPackage(getString(R.string.qnn_model_loading), importer = {
            if (remember) {
                ModelPackage.importQnnModelWithInstalledPipeline(this, base, attachment)
            } else {
                // Re-hash the current CPU package before selecting an installed QNN copy. The
                // source file may have been replaced in-place since the model record was saved.
                runCatching {
                    ModelPackage.openInstalledQnnModelWithPipeline(this, base)
                }.getOrElse {
                    require(modelUriExists(base)) { "the installed QNN voice and its base model are unavailable" }
                    ModelPackage.importQnnModelWithInstalledPipeline(this, base, attachment)
                }
            }
        }) { model ->
            if (remember) saveModelRecord(
                record.copy(
                    qnnUri = attachment.toString(),
                    baseModelSha256 = model.baseModelSha256,
                )
            )
        }
    }

    private fun loadModelRecord(record: ModelRecord) {
        val qnn = record.qnnUri
        if (qnn.isNullOrBlank()) {
            loadModelUri(Uri.parse(record.uri), record.split, remember = false)
        } else {
            // The verified attachment is copied into app-private storage at install time. Prefer
            // that durable copy so deleting or moving the original download does not disable NPU
            // loading. loadQnnModelUri falls back to the persisted source URI only when the
            // installed copy is absent or belongs to another base model.
            loadQnnModelUri(record, Uri.parse(qnn), remember = false)
        }
    }

    private fun modelUriExists(uri: Uri): Boolean = runCatching {
        contentResolver.openInputStream(uri).use { requireNotNull(it); it.read() }
        true
    }.getOrDefault(false)

    private fun readModelRecords(): List<ModelRecord> {
        val stateFile = File(filesDir, "state/model-records.json")
        val raw = runCatching { stateFile.readText() }.getOrNull()
            ?: getSharedPreferences("models", MODE_PRIVATE).getString("records", "[]") ?: "[]"
        return runCatching {
            val array = JSONArray(raw)
            List(array.length()) { index ->
                array.getJSONObject(index).let {
                    ModelRecord(
                        it.getString("uri"),
                        it.getString("name"),
                        it.getString("version"),
                        it.optBoolean("split", true),
                        it.optString("qnn_uri").takeIf(String::isNotBlank),
                        it.optString("base_model_sha256").takeIf(String::isNotBlank),
                    )
                }
            }
        }.getOrDefault(emptyList())
    }

    private fun writeModelRecords(records: List<ModelRecord>) {
        val array = JSONArray()
        records.forEach { record ->
            array.put(
                JSONObject()
                    .put("uri", record.uri)
                    .put("name", record.name)
                    .put("version", record.version)
                    .put("split", record.split)
                    .put("qnn_uri", record.qnnUri ?: "")
                    .put("base_model_sha256", record.baseModelSha256 ?: ""),
            )
        }
        getSharedPreferences("models", MODE_PRIVATE).edit().putString("records", array.toString()).apply()
        val stateFile = File(filesDir, "state/model-records.json")
        stateFile.parentFile?.mkdirs()
        val pending = File(stateFile.parentFile, "${stateFile.name}.pending")
        pending.writeText(array.toString())
        if (stateFile.exists()) stateFile.delete()
        require(pending.renameTo(stateFile)) { getString(R.string.model_list_save_failed) }
        ui = ui.copy(models = records)
    }

    private fun saveModelRecord(record: ModelRecord) {
        writeModelRecords(listOf(record) + ui.models.filterNot { it.uri == record.uri })
    }

    private fun removeModelRecord(uri: String) = writeModelRecords(ui.models.filterNot { it.uri == uri })

    private fun loadPackage(message: String, importer: () -> ModelPackage, loaded: (ModelPackage) -> Unit = {}) {
        setBusy(message)
        lifecycleScope.launch {
            runCatching {
                withContext(Dispatchers.IO) {
                    TimingContext.measure("model.import_and_engine_load") {
                        importer().also { synchronized(engineLock) { engine.load(it) } }
                    }
                }
            }.onSuccess {
                loaded(it)
                ui = ui.copy(
                    busy = false,
                    modelInfo = "${it.name} / ${it.version} / ${it.sampleRate} Hz",
                    backend = engine.backendName,
                    runtimeOptions = it.runtimeOptions,
                    referenceOverrideSupported = it.referenceInputVersion >= 1,
                    referenceExactPcm16kSamples = it.referencePcm16kSamples.takeIf { samples ->
                        it.referenceDurationPolicy == "exact_samples" && samples > 0
                    },
                    temperature = if ("temperature" in it.runtimeOptions) ui.temperature else "1.0",
                    topP = if ("top_p" in it.runtimeOptions) ui.topP else "1.0",
                    topK = if ("top_k" in it.runtimeOptions) ui.topK else "10",
                    penalty = if ("repetition_penalty" in it.runtimeOptions) ui.penalty else "1.35",
                    speed = if ("speed_factor" in it.runtimeOptions) ui.speed else "1.0",
                    steps = if ("sample_steps" in it.runtimeOptions) ui.steps else "32",
                    referenceUri = null,
                    referenceName = "",
                    referencePrompt = "",
                    status = getString(R.string.model_loaded),
                )
            }.onFailure { error ->
                ui = ui.copy(
                    busy = false,
                    status = if (error is UnsupportedLegacyModelException) {
                        getString(R.string.legacy_model_unsupported)
                    } else {
                        getString(R.string.import_failed, error.message.orEmpty())
                    },
                )
            }
        }
    }

    private fun synthesize() {
        val state = ui
        if (state.referenceUri != null && state.referencePrompt.isBlank()) {
            ui = ui.copy(status = getString(R.string.reference_prompt_required))
            return
        }
        if (state.referenceUri != null && !state.referenceOverrideSupported) {
            ui = ui.copy(status = getString(R.string.reference_not_supported))
            return
        }
        setBusy(if (state.referenceUri == null) getString(R.string.synthesizing) else getString(R.string.reference_decoding))
        lifecycleScope.launch {
            runCatching {
                val options = SynthesisOptions(state.temperature.toFloat(), state.topP.toFloat(), state.topK.toInt(), state.penalty.toFloat(), state.speed.toFloat(), state.steps.toInt())
                val reference = state.referenceUri?.let { uri ->
                    withContext(Dispatchers.IO) {
                        ReferenceAudioDecoder.decode(
                            this@MainActivity,
                            uri,
                            state.referencePrompt,
                            state.referenceLanguage,
                            state.referenceExactPcm16kSamples,
                        )
                    }
                }
                withContext(Dispatchers.Default) {
                    synchronized(engineLock) {
                        engine.synthesize(
                            SynthesisRequest(
                                state.text,
                                language = state.textLanguage,
                                options = options,
                                reference = reference,
                            ),
                            File(cacheDir, "tts.wav"),
                        )
                    }
                }
            }.onSuccess { output = it; ui = ui.copy(busy = false, canPlay = true, status = getString(R.string.synthesis_complete)) }
                .onFailure { error ->
                    val message = if (error is ReferenceDurationMismatch) {
                        getString(
                            R.string.reference_exact_duration_mismatch,
                            error.expectedSeconds,
                            error.actualSeconds,
                        )
                    } else {
                        getString(R.string.synthesis_failed, error.message.orEmpty())
                    }
                    ui = ui.copy(busy = false, status = message)
                }
        }
    }

    private fun play() {
        output?.let { file -> player?.release(); player = MediaPlayer().apply { setDataSource(file.path); prepare(); start() } }
    }

    private fun save() {
        if (output?.isFile != true) {
            ui = ui.copy(status = getString(R.string.audio_output_missing))
            return
        }
        saveOutput.launch("gsv-${System.currentTimeMillis()}.wav")
    }

    private fun startApiServer() {
        val port = ui.port.toIntOrNull()
        if (port == null || port !in 1024..65535) { ui = ui.copy(serverEnabled = false, serverStatus = getString(R.string.api_port_invalid)); return }
        if (
            Build.VERSION.SDK_INT >= 33 &&
            checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
        ) {
            pendingApiPort = port
            ui = ui.copy(serverEnabled = true, serverStatus = getString(R.string.api_starting))
            requestNotifications.launch(Manifest.permission.POST_NOTIFICATIONS)
            return
        }
        startApiService(port)
    }

    private fun startApiService(port: Int) {
        val receiver = object : ResultReceiver(Handler(mainLooper)) {
            override fun onReceiveResult(resultCode: Int, resultData: Bundle?) {
                when (resultCode) {
                    LocalOpenAiService.RESULT_STARTED -> {
                        val endpoint = resultData?.getString(LocalOpenAiService.EXTRA_ENDPOINT).orEmpty()
                        ui = ui.copy(
                            serverEnabled = true,
                            serverStatus = "$endpoint · POST /v1/audio/speech",
                        )
                    }
                    LocalOpenAiService.RESULT_ERROR -> {
                        val error = resultData?.getString(LocalOpenAiService.EXTRA_ERROR).orEmpty()
                        ui = ui.copy(
                            serverEnabled = false,
                            serverStatus = getString(R.string.api_start_failed, error),
                        )
                    }
                }
            }
        }
        ui = ui.copy(serverEnabled = true, serverStatus = getString(R.string.api_starting))
        runCatching {
            ContextCompat.startForegroundService(
                this,
                LocalOpenAiService.startIntent(this, port, receiver),
            )
        }.onFailure {
            ui = ui.copy(
                serverEnabled = false,
                serverStatus = getString(R.string.api_start_failed, it.message.orEmpty()),
            )
        }
    }

    private fun stopApiServer() {
        stopService(LocalOpenAiService.stopIntent(this))
        GsvRuntime.apiEndpoint = null
        GsvRuntime.apiError = null
        ui = ui.copy(serverEnabled = false, serverStatus = getString(R.string.api_stopped))
    }
    private fun setBusy(status: String) { ui = ui.copy(busy = true, status = status) }
    private fun openWebsite(url: String) = startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(url)))

    private fun displayName(uri: Uri): String {
        val value = contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { cursor ->
            if (cursor.moveToFirst()) cursor.getString(0) else null
        }
        return value ?: uri.lastPathSegment ?: "reference audio"
    }

    private fun selectAppLanguage(language: AppLanguage) {
        if (ui.busy || ui.appLanguage == language) return
        AppLocale.set(this, language)
        recreate()
    }

    private fun restoreSynthesisState(state: Bundle?): UiState {
        if (state == null) return ui
        return ui.copy(
            text = state.getString(STATE_TEXT, ui.text),
            textLanguage = state.getString(STATE_TEXT_LANGUAGE, ui.textLanguage),
            temperature = state.getString(STATE_TEMPERATURE, ui.temperature),
            topP = state.getString(STATE_TOP_P, ui.topP),
            topK = state.getString(STATE_TOP_K, ui.topK),
            penalty = state.getString(STATE_PENALTY, ui.penalty),
            speed = state.getString(STATE_SPEED, ui.speed),
            steps = state.getString(STATE_STEPS, ui.steps),
            referenceUri = state.getString(STATE_REFERENCE_URI)?.takeIf(String::isNotBlank)?.let(Uri::parse),
            referenceName = state.getString(STATE_REFERENCE_NAME, ui.referenceName),
            referencePrompt = state.getString(STATE_REFERENCE_PROMPT, ui.referencePrompt),
            referenceLanguage = state.getString(STATE_REFERENCE_LANGUAGE, ui.referenceLanguage),
        )
    }

    @Composable private fun MainScreen() {
        var tab by rememberSaveable { mutableIntStateOf(0) }
        Scaffold(topBar = { TopAppBar(title = { Text(stringResource(R.string.app_name)) }, actions = { IconButton(onClick = { tab = 2 }) { Icon(Icons.Default.Settings, stringResource(R.string.settings)) } }) }) { padding ->
            Column(Modifier.padding(padding).fillMaxSize()) {
                PrimaryTabRow(tab) {
                    Tab(tab == 0, { tab = 0 }, text = { Text(stringResource(R.string.tab_synthesis)) })
                    Tab(tab == 1, { tab = 1 }, text = { Text(stringResource(R.string.tab_models)) })
                    Tab(tab == 2, { tab = 2 }, text = { Text(stringResource(R.string.tab_settings)) })
                    Tab(tab == 3, { tab = 3 }, text = { Text(stringResource(R.string.tab_about)) })
                }
                when (tab) { 0 -> SynthesisPane(); 1 -> ModelLibraryPane(); 2 -> SettingsPane(); else -> AboutPane() }
            }
        }
        if (ui.showFirstRun) ComponentDialog()
    }

    @Composable private fun ComponentDialog() {
        AlertDialog(
            onDismissRequest = { if (!ui.busy) ui = ui.copy(showFirstRun = false) },
            title = { Text(stringResource(R.string.pipeline_choose_title)) },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Text(stringResource(R.string.pipeline_versions_required), style = MaterialTheme.typography.labelLarge)
                    ComponentVersion.entries.forEach { version ->
                        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                            Checkbox(version in ui.selectedVersions, { checked ->
                                val next = if (checked) ui.selectedVersions + version else ui.selectedVersions - version
                                if (next.isNotEmpty()) ui = ui.copy(selectedVersions = next)
                            })
                            Column { Text(version.label); Text(stringResource(if (version.manifestId in ui.installedVersions) R.string.installed else R.string.not_installed), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant) }
                        }
                    }
                    ui.downloadProgress?.let { LinearProgressIndicator(progress = { it }, modifier = Modifier.fillMaxWidth()) }
                    if (ui.pipelineInstalled) {
                        OutlinedButton(
                            { ui = ui.copy(showFirstRun = false); pickQnnPipelineAttachment.launch(qnnPackageTypes) },
                            enabled = !ui.busy,
                            modifier = Modifier.fillMaxWidth(),
                        ) {
                            Icon(Icons.Default.UploadFile, null)
                            Spacer(Modifier.width(8.dp))
                            Text(stringResource(R.string.import_qnn_pipeline_attachment))
                        }
                    }
                }
            },
            confirmButton = { Button(::downloadPipelines, enabled = !ui.busy && ui.selectedVersions.isNotEmpty()) { Icon(Icons.Default.Download, null); Spacer(Modifier.width(8.dp)); Text(if (ui.selectedVersions.size > 1) stringResource(R.string.download_count, ui.selectedVersions.size) else stringResource(R.string.download_component)) } },
            dismissButton = { Row { TextButton(onClick = { ui = ui.copy(showFirstRun = false); pickPipelines.launch(packageTypes) }, enabled = !ui.busy) { Text(stringResource(R.string.import_manually)) }; TextButton(onClick = { ui = ui.copy(showFirstRun = false) }, enabled = !ui.busy) { Text(stringResource(R.string.later)) } } },
        )
    }

    @Composable private fun SynthesisPane() {
        Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(20.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
            Text(ui.backend, style = MaterialTheme.typography.titleMedium)
            Text(ui.modelInfo, color = MaterialTheme.colorScheme.onSurfaceVariant)
            OutlinedTextField(ui.text, { ui = ui.copy(text = it) }, label = { Text(stringResource(R.string.input_text)) }, minLines = 5, modifier = Modifier.fillMaxWidth())
            Text(stringResource(R.string.text_language), style = MaterialTheme.typography.labelLarge)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                listOf("auto" to R.string.language_auto, "zh" to R.string.language_chinese, "en" to R.string.language_english).forEach { (value, label) ->
                    FilterChip(ui.textLanguage == value, { ui = ui.copy(textLanguage = value) }, { Text(stringResource(label)) })
                }
            }
            Text(stringResource(R.string.reference_voice), style = MaterialTheme.typography.titleMedium)
            Text(
                if (ui.referenceUri == null) stringResource(R.string.reference_model_preset) else stringResource(R.string.reference_temporary, ui.referenceName),
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedButton({ pickReference.launch(arrayOf("audio/*", "audio/wav", "audio/flac", "audio/mpeg")) }, enabled = !ui.busy && ui.referenceOverrideSupported, modifier = Modifier.weight(1f)) {
                    Icon(Icons.Default.UploadFile, null); Spacer(Modifier.width(6.dp)); Text(stringResource(R.string.reference_choose_audio))
                }
                TextButton({ ui = ui.copy(referenceUri = null, referenceName = "", referencePrompt = "") }, enabled = !ui.busy && ui.referenceUri != null) { Text(stringResource(R.string.reference_use_preset)) }
            }
            if (ui.referenceUri != null) {
                OutlinedTextField(ui.referencePrompt, { ui = ui.copy(referencePrompt = it) }, label = { Text(stringResource(R.string.reference_prompt)) }, minLines = 2, modifier = Modifier.fillMaxWidth())
                Text(stringResource(R.string.reference_language), style = MaterialTheme.typography.labelLarge)
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    listOf("auto" to R.string.language_auto, "zh" to R.string.language_chinese, "en" to R.string.language_english).forEach { (value, label) ->
                        FilterChip(ui.referenceLanguage == value, { ui = ui.copy(referenceLanguage = value) }, { Text(stringResource(label)) })
                    }
                }
                if (!ui.referenceOverrideSupported) Text(stringResource(R.string.reference_not_supported), color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
            }
            ui.referenceExactPcm16kSamples?.let { samples ->
                Text(
                    stringResource(R.string.reference_exact_duration_required, samples / 16000.0),
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    style = MaterialTheme.typography.bodySmall,
                )
            }
            Text(stringResource(R.string.generation_options), style = MaterialTheme.typography.titleMedium)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                NumberField(stringResource(R.string.temperature), ui.temperature, { ui = ui.copy(temperature = it) }, Modifier.weight(1f), "temperature" in ui.runtimeOptions)
                NumberField(stringResource(R.string.top_p), ui.topP, { ui = ui.copy(topP = it) }, Modifier.weight(1f), "top_p" in ui.runtimeOptions)
            }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                NumberField(stringResource(R.string.top_k), ui.topK, { ui = ui.copy(topK = it) }, Modifier.weight(1f), "top_k" in ui.runtimeOptions)
                NumberField(stringResource(R.string.repetition_penalty), ui.penalty, { ui = ui.copy(penalty = it) }, Modifier.weight(1f), "repetition_penalty" in ui.runtimeOptions)
            }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                NumberField(stringResource(R.string.speed), ui.speed, { ui = ui.copy(speed = it) }, Modifier.weight(1f), "speed_factor" in ui.runtimeOptions)
                NumberField(stringResource(R.string.cfm_steps), ui.steps, { ui = ui.copy(steps = it) }, Modifier.weight(1f), "sample_steps" in ui.runtimeOptions)
            }
            Button(::synthesize, enabled = !ui.busy && ui.text.isNotBlank(), modifier = Modifier.fillMaxWidth()) { Text(if (ui.busy) stringResource(R.string.processing) else stringResource(R.string.synthesize_speech)) }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedButton(::play, enabled = ui.canPlay && !ui.busy, modifier = Modifier.weight(1f)) { Icon(Icons.Default.PlayArrow, null); Spacer(Modifier.width(8.dp)); Text(stringResource(R.string.play)) }
                OutlinedButton(::save, enabled = ui.canPlay && !ui.busy, modifier = Modifier.weight(1f)) { Icon(Icons.Default.SaveAlt, null); Spacer(Modifier.width(8.dp)); Text(stringResource(R.string.save_audio)) }
            }
            if (ui.busy) LinearProgressIndicator(Modifier.fillMaxWidth())
            if (ui.status.isNotBlank()) Text(ui.status, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }

    @Composable private fun ModelLibraryPane() {
        Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(20.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
            Text(stringResource(R.string.pipeline_components), style = MaterialTheme.typography.titleMedium)
            Text(
                if (ui.installedVersions.isEmpty()) stringResource(R.string.not_installed) else stringResource(R.string.pipeline_installed_versions, ComponentVersion.entries.filter { it.manifestId in ui.installedVersions }.joinToString { it.shortLabel }),
                color = if (ui.pipelineInstalled) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.error,
            )
            Text(
                if (ui.qnnPipelines.isEmpty()) stringResource(R.string.qnn_pipeline_not_installed) else stringResource(R.string.qnn_pipeline_targets, ui.qnnPipelines.sorted().joinToString()),
                style = MaterialTheme.typography.bodySmall,
                color = if (ui.qnnPipelines.isEmpty()) MaterialTheme.colorScheme.onSurfaceVariant else MaterialTheme.colorScheme.primary,
            )
            OutlinedButton(
                { pickQnnPipelineAttachment.launch(qnnPackageTypes) },
                enabled = !ui.busy && ui.pipelineInstalled,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Icon(Icons.Default.UploadFile, null)
                Spacer(Modifier.width(8.dp))
                Text(stringResource(R.string.import_qnn_pipeline_attachment))
            }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button({ if (ui.pipelineInstalled) pickModel.launch(packageTypes) else ui = ui.copy(showFirstRun = true) }, enabled = !ui.busy, modifier = Modifier.weight(1f)) { Icon(Icons.Default.UploadFile, null); Spacer(Modifier.width(6.dp)); Text(stringResource(R.string.add_voice_model)) }
                OutlinedButton({ pickCombined.launch(packageTypes) }, enabled = !ui.busy, modifier = Modifier.weight(1f)) { Text(stringResource(R.string.add_complete_package)) }
            }
            Text(stringResource(R.string.external_model_directory), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            OutlinedButton(::requestExternalModelScan, enabled = !ui.busy, modifier = Modifier.fillMaxWidth()) {
                Icon(Icons.Default.Refresh, null)
                Spacer(Modifier.width(8.dp))
                Text(stringResource(R.string.scan_external_models))
            }
            TextButton({ openWebsite(MODEL_CONVERTER_URL) }) {
                Text(stringResource(R.string.how_get_gsvm))
                Spacer(Modifier.width(6.dp))
                Icon(Icons.AutoMirrored.Filled.OpenInNew, stringResource(R.string.open_in_browser))
            }
            Text(
                stringResource(R.string.legacy_model_unsupported),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            HorizontalDivider()
            Text(stringResource(R.string.model_library), style = MaterialTheme.typography.titleLarge)
            if (ui.models.isEmpty()) {
                Text(stringResource(R.string.no_models), color = MaterialTheme.colorScheme.onSurfaceVariant)
            } else ui.models.forEach { record ->
                val expanded = ui.expandedModelUri == record.uri
                Surface(tonalElevation = 1.dp, shape = MaterialTheme.shapes.small, modifier = Modifier.fillMaxWidth().animateContentSize()) {
                    Column {
                        Row(
                            Modifier.fillMaxWidth().clickable { ui = ui.copy(expandedModelUri = if (expanded) null else record.uri) }.padding(horizontal = 12.dp, vertical = 10.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Text(record.name, maxLines = 1, overflow = TextOverflow.Ellipsis, style = MaterialTheme.typography.bodyLarge, modifier = Modifier.weight(1f))
                            Text(record.version, style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                            Icon(if (expanded) Icons.Default.ExpandLess else Icons.Default.ExpandMore, stringResource(if (expanded) R.string.collapse else R.string.expand))
                        }
                        if (expanded) {
                            HorizontalDivider()
                            Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                                Text(stringResource(if (record.split) R.string.split_voice_model else R.string.complete_package), style = MaterialTheme.typography.bodySmall)
                                Text(Uri.decode(record.uri), style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                                Text(
                                    stringResource(if (record.qnnUri == null) R.string.qnn_attachment_not_selected else R.string.qnn_attachment_selected),
                                    style = MaterialTheme.typography.bodySmall,
                                    color = if (record.qnnUri == null) MaterialTheme.colorScheme.onSurfaceVariant else MaterialTheme.colorScheme.primary,
                                )
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.spacedBy(8.dp, Alignment.End),
                                ) {
                                    OutlinedButton({ chooseQnnModelAttachment(record) }, enabled = !ui.busy) {
                                        Text(stringResource(R.string.choose_qnn_attachment))
                                    }
                                    Button({ loadModelRecord(record) }, enabled = !ui.busy) {
                                        Text(stringResource(if (record.qnnUri == null) R.string.load_model else R.string.load_model_npu))
                                    }
                                }
                                if (record.qnnUri != null) {
                                    TextButton(
                                        { saveModelRecord(record.copy(qnnUri = null, baseModelSha256 = null)) },
                                        enabled = !ui.busy,
                                        modifier = Modifier.align(Alignment.End),
                                    ) { Text(stringResource(R.string.remove_qnn_attachment)) }
                                }
                            }
                        }
                    }
                }
            }
            if (ui.busy) LinearProgressIndicator(Modifier.fillMaxWidth())
            if (ui.status.isNotBlank()) Text(ui.status, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }

    @Composable private fun SettingsPane() {
        Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(20.dp), verticalArrangement = Arrangement.spacedBy(18.dp)) {
            Text(stringResource(R.string.app_language), style = MaterialTheme.typography.titleLarge)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                FilterChip(
                    selected = ui.appLanguage == AppLanguage.CHINESE,
                    onClick = { selectAppLanguage(AppLanguage.CHINESE) },
                    label = { Text(stringResource(R.string.app_language_chinese)) },
                    enabled = !ui.busy,
                )
                FilterChip(
                    selected = ui.appLanguage == AppLanguage.ENGLISH,
                    onClick = { selectAppLanguage(AppLanguage.ENGLISH) },
                    label = { Text(stringResource(R.string.app_language_english)) },
                    enabled = !ui.busy,
                )
            }
            HorizontalDivider()
            Text(stringResource(R.string.pipeline_components), style = MaterialTheme.typography.titleLarge)
            Text(if (ui.installedVersions.isEmpty()) stringResource(R.string.not_installed) else stringResource(R.string.pipeline_installed_versions, ComponentVersion.entries.filter { it.manifestId in ui.installedVersions }.joinToString { it.shortLabel }), color = if (ui.pipelineInstalled) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.error)
            Button({ ui = ui.copy(showFirstRun = true) }, enabled = !ui.busy) { Icon(Icons.Default.UploadFile, null); Spacer(Modifier.width(8.dp)); Text(stringResource(if (ui.pipelineInstalled) R.string.change_components else R.string.choose_import_components)) }
            OutlinedButton({ pickQnnPipelineAttachment.launch(qnnPackageTypes) }, enabled = !ui.busy && ui.pipelineInstalled) { Icon(Icons.Default.UploadFile, null); Spacer(Modifier.width(8.dp)); Text(stringResource(R.string.import_qnn_pipeline_attachment)) }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) { ComponentVersion.entries.forEach { version -> FilterChip(ui.componentVersion == version, { selectComponentVersion(version) }, { Text(version.shortLabel) }) } }
            OutlinedTextField(ui.componentUrl, { value -> ui = ui.copy(componentUrl = value); getSharedPreferences("components", MODE_PRIVATE).edit().putString(ui.componentVersion.preferenceKey, value).apply() }, label = { Text(stringResource(R.string.component_download_url, ui.componentVersion.label)) }, leadingIcon = { Icon(Icons.Default.Download, null) }, modifier = Modifier.fillMaxWidth(), singleLine = true)
            HorizontalDivider()
            Text(stringResource(R.string.openai_local_api), style = MaterialTheme.typography.titleLarge)
            OutlinedTextField(ui.port, { ui = ui.copy(port = it) }, enabled = !ui.serverEnabled, label = { Text(stringResource(R.string.port)) }, keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number), modifier = Modifier.fillMaxWidth())
            Row(verticalAlignment = Alignment.CenterVertically) { Switch(ui.serverEnabled, { if (it) startApiServer() else stopApiServer() }); Spacer(Modifier.width(12.dp)); Text(stringResource(if (ui.serverEnabled) R.string.api_enabled else R.string.enable_api)) }
            Text(ui.serverStatus, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }

    @Composable private fun AboutPane() {
        Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(20.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
            Text(stringResource(R.string.app_name), style = MaterialTheme.typography.headlineSmall)
            Text(stringResource(R.string.about_description), color = MaterialTheme.colorScheme.onSurfaceVariant)
            HorizontalDivider()
            Text(stringResource(R.string.project_links), style = MaterialTheme.typography.titleMedium)
            OutlinedButton({ openWebsite(PROJECT_URL) }, modifier = Modifier.fillMaxWidth()) {
                Text(stringResource(R.string.project_name), modifier = Modifier.weight(1f))
                Icon(Icons.AutoMirrored.Filled.OpenInNew, stringResource(R.string.open_in_browser))
            }
            OutlinedButton({ openWebsite(UPSTREAM_URL) }, modifier = Modifier.fillMaxWidth()) {
                Text(stringResource(R.string.upstream_project), modifier = Modifier.weight(1f))
                Icon(Icons.AutoMirrored.Filled.OpenInNew, stringResource(R.string.open_in_browser))
            }
            Text(stringResource(R.string.license_notice), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }

    @Composable private fun NumberField(label: String, value: String, change: (String) -> Unit, modifier: Modifier, enabled: Boolean) = OutlinedTextField(value, change, modifier, enabled = enabled, label = { Text(label) }, singleLine = true, keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Decimal))
    override fun onDestroy() {
        player?.release()
        GsvRuntime.releaseActivity(isFinishing)
        super.onDestroy()
    }

    override fun onSaveInstanceState(outState: Bundle) {
        outState.putString(STATE_TEXT, ui.text)
        outState.putString(STATE_TEXT_LANGUAGE, ui.textLanguage)
        outState.putString(STATE_TEMPERATURE, ui.temperature)
        outState.putString(STATE_TOP_P, ui.topP)
        outState.putString(STATE_TOP_K, ui.topK)
        outState.putString(STATE_PENALTY, ui.penalty)
        outState.putString(STATE_SPEED, ui.speed)
        outState.putString(STATE_STEPS, ui.steps)
        outState.putString(STATE_REFERENCE_URI, ui.referenceUri?.toString())
        outState.putString(STATE_REFERENCE_NAME, ui.referenceName)
        outState.putString(STATE_REFERENCE_PROMPT, ui.referencePrompt)
        outState.putString(STATE_REFERENCE_LANGUAGE, ui.referenceLanguage)
        outState.putBoolean(STATE_CAN_PLAY, ui.canPlay && output?.isFile == true)
        outState.putBoolean(STATE_SHOW_FIRST_RUN, ui.showFirstRun)
        super.onSaveInstanceState(outState)
    }

    companion object {
        private val packageTypes = arrayOf("application/zip", "application/octet-stream")
        private val qnnPackageTypes = arrayOf("application/zip", "application/octet-stream")
        private const val MODEL_CONVERTER_URL = "https://gs.cutefireflyuwu.sbs"
        private const val PROJECT_URL = "https://github.com/tuxKOH/GPT-SoViTs-android"
        private const val UPSTREAM_URL = "https://github.com/RVC-Boss/GPT-SoVITS"
        private const val STATE_TEXT = "synthesis.text"
        private const val STATE_TEXT_LANGUAGE = "synthesis.text_language"
        private const val STATE_TEMPERATURE = "synthesis.temperature"
        private const val STATE_TOP_P = "synthesis.top_p"
        private const val STATE_TOP_K = "synthesis.top_k"
        private const val STATE_PENALTY = "synthesis.penalty"
        private const val STATE_SPEED = "synthesis.speed"
        private const val STATE_STEPS = "synthesis.steps"
        private const val STATE_REFERENCE_URI = "synthesis.reference_uri"
        private const val STATE_REFERENCE_NAME = "synthesis.reference_name"
        private const val STATE_REFERENCE_PROMPT = "synthesis.reference_prompt"
        private const val STATE_REFERENCE_LANGUAGE = "synthesis.reference_language"
        private const val STATE_CAN_PLAY = "synthesis.can_play"
        private const val STATE_SHOW_FIRST_RUN = "components.show_first_run"
    }
}

private data class UiState(
    val pipelineInstalled: Boolean = false, val showFirstRun: Boolean = false, val componentUrl: String = "",
    val componentVersion: ComponentVersion = ComponentVersion.V2PP, val selectedVersions: Set<ComponentVersion> = setOf(ComponentVersion.V2PP),
    val installedVersions: Set<String> = emptySet(), val qnnPipelines: Set<String> = emptySet(), val downloadProgress: Float? = null,
    val models: List<ModelRecord> = emptyList(),
    val expandedModelUri: String? = null,
    val modelInfo: String = "", val backend: String = "", val status: String = "",
    val text: String = "", val textLanguage: String = "auto", val temperature: String = "1.0", val topP: String = "1.0", val topK: String = "10",
    val penalty: String = "1.35", val speed: String = "1.0", val steps: String = "32", val busy: Boolean = false, val canPlay: Boolean = false,
    val referenceUri: Uri? = null, val referenceName: String = "", val referencePrompt: String = "",
    val referenceLanguage: String = "auto", val runtimeOptions: Set<String> = emptySet(), val referenceOverrideSupported: Boolean = false,
    val referenceExactPcm16kSamples: Int? = null,
    val port: String = "9880", val serverEnabled: Boolean = false, val serverStatus: String = "",
    val appLanguage: AppLanguage = AppLanguage.ENGLISH,
)

private data class ModelRecord(
    val uri: String,
    val name: String,
    val version: String,
    val split: Boolean,
    val qnnUri: String? = null,
    val baseModelSha256: String? = null,
)

private enum class ComponentVersion(
    val label: String,
    val shortLabel: String,
    val manifestId: String,
    val preferenceKey: String,
    val defaultUrl: String,
    val sha256: String,
) {
    // This is the shared, frontend-only V2PP pipeline.  Keep the URL/hash bound to the
    // currently released artifact so a first-run download cannot silently restore the old
    // pre-ABI pipeline-v2pp-cpu-fp32.gsvm package.
    V2PP("V2 Pro Plus", "V2PP", "v2ProPlus", "pipeline_url_v2pp", "https://hf-mirror.com/tuxjhtd/GPT-Sovits_pipeline_Torchscript/resolve/main/v2pp-cpu-pipeline.gsvm?download=true", "7fcbc0bb520fa9c3eedd1ccccace443ced2185b8ca1ab2c915fd764f5e6656d7"),
    V4("V4", "V4", "v4", "pipeline_url_v4", "https://hf-mirror.com/tuxjhtd/GPT-Sovits_pipeline_Torchscript/resolve/main/pipeline-v4-cpu-fp32.gsvm?download=true", "8cfec15b4a4cb0a7b6cac8df52f4f44e3493694d73e0107b363ddd1012d4f3fa"),
}

@Composable private fun GsvTheme(content: @Composable () -> Unit) {
    val colors = if (androidx.compose.foundation.isSystemInDarkTheme()) darkColorScheme(primary = androidx.compose.ui.graphics.Color(0xFF8FD5B0)) else lightColorScheme(primary = androidx.compose.ui.graphics.Color(0xFF246B4B), secondary = androidx.compose.ui.graphics.Color(0xFF5B635D))
    MaterialTheme(colorScheme = colors, typography = Typography(), content = content)
}
