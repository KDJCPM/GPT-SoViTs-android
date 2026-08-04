package ai.gsv.mobile

import android.media.MediaPlayer
import android.content.Intent
import android.net.Uri
import android.os.Bundle
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
import androidx.lifecycle.lifecycleScope
import fi.iki.elonen.NanoHTTPD
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
    private val engine = TtsEngine(listOf(CpuBackend()))
    private val engineLock = Any()
    private var output: File? = null
    private var player: MediaPlayer? = null
    private var apiServer: LocalOpenAiServer? = null

    private var ui by mutableStateOf(UiState())

    private val pickCombined = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri?.let { rememberAndLoadModel(it, split = false) }
    }
    private val pickPipelines = registerForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { uris ->
        if (uris.isNotEmpty()) installPipelines(uris)
    }
    private val pickModel = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri?.let { rememberAndLoadModel(it, split = true) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val prefs = getSharedPreferences("components", MODE_PRIVATE)
        ModelPackage.restorePipelineArchives(this)
        ui = ui.copy(
            pipelineInstalled = ModelPackage.hasInstalledPipeline(this),
            installedVersions = ModelPackage.installedPipelineVersions(this),
            componentUrl = prefs.getString("pipeline_url_v2pp", null) ?: ComponentVersion.V2PP.defaultUrl,
            showFirstRun = !ModelPackage.hasInstalledPipeline(this),
            port = intent.getIntExtra("api_port", 9880).toString(),
            models = readModelRecords(),
        )
        setContent { GsvTheme { MainScreen() } }
        if (intent.getBooleanExtra("start_api", false)) startApiServer()
    }

    private fun installPipelines(uris: List<Uri>) {
        setBusy("正在校验并安装 pipeline 组件…")
        lifecycleScope.launch {
            runCatching {
                withContext(Dispatchers.IO) {
                    val selectedIds = ui.selectedVersions.map { it.manifestId }.toSet()
                    uris.map { uri ->
                        val version = inspectPipelineVersion(uri)
                        require(version in selectedIds) { "文件 $version 未在下载目标中勾选" }
                        ModelPackage.installPipeline(this@MainActivity, uri, version)
                    }
                }
            }
                .onSuccess {
                    val installed = ModelPackage.installedPipelineVersions(this@MainActivity)
                    ui = ui.copy(busy = false, pipelineInstalled = installed.isNotEmpty(), installedVersions = installed, showFirstRun = false, status = "已安装：${it.joinToString { item -> item.version }}")
                }
                .onFailure { ui = ui.copy(busy = false, status = "组件导入失败：${it.message}") }
        }
    }

    private fun inspectPipelineVersion(uri: Uri): String = contentResolver.openInputStream(uri).use { raw ->
        requireNotNull(raw) { "无法打开 pipeline 包" }
        java.util.zip.ZipInputStream(raw.buffered()).use { zip ->
            require(zip.nextEntry?.name == "manifest.json") { "manifest.json 必须是首个条目" }
            org.json.JSONObject(zip.readBytes().toString(Charsets.UTF_8)).getString("model_version")
        }
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
        if (versions.isEmpty()) { ui = ui.copy(status = "V2 Pro Plus 与 V4 至少选择一个"); return }
        setBusy("准备下载组件…")
        ui = ui.copy(downloadProgress = 0f)
        lifecycleScope.launch {
            runCatching {
                versions.forEachIndexed { index, version ->
                    withContext(Dispatchers.IO) {
                        val prefs = getSharedPreferences("components", MODE_PRIVATE)
                        val requestedUrl = (prefs.getString(version.preferenceKey, null) ?: version.defaultUrl).trim()
                            .replace("https://huggingface.co/", "https://hf-mirror.com/")
                        require(requestedUrl.startsWith("https://")) { "${version.label} 组件地址必须使用 HTTPS" }
                        downloadPipeline(version, requestedUrl, index, versions.size)
                    }
                }
            }.onSuccess {
                val installed = ModelPackage.installedPipelineVersions(this@MainActivity)
                ui = ui.copy(busy = false, downloadProgress = null, pipelineInstalled = installed.isNotEmpty(), installedVersions = installed, showFirstRun = false, status = "组件安装完成：${versions.joinToString { it.label }}")
            }.onFailure {
                val installed = ModelPackage.installedPipelineVersions(this@MainActivity)
                ui = ui.copy(busy = false, downloadProgress = null, pipelineInstalled = installed.isNotEmpty(), installedVersions = installed, status = "组件下载失败：${it.message}")
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
                        withContext(Dispatchers.Main) { ui = ui.copy(status = "${version.label} 已从本地组件存储恢复") }
                        return
                    }
                    val connection = URL(requestedUrl).openConnection() as HttpURLConnection
                    connection.instanceFollowRedirects = true
                    connection.connectTimeout = 20_000
                    connection.readTimeout = 60_000
                    connection.setRequestProperty("User-Agent", "GSV-Mobile/0.1")
                    try {
                        require(connection.responseCode in 200..299) { "下载失败：HTTP ${connection.responseCode}" }
                        val total = connection.contentLengthLong
                        connection.inputStream.buffered().use { input ->
                            download.outputStream().buffered().use { output ->
                                val buffer = ByteArray(1024 * 1024)
                                var received = 0L
                                while (true) {
                                    val count = input.read(buffer)
                                    if (count < 0) break
                                    output.write(buffer, 0, count)
                                    received += count
                                    if (total > 0) withContext(Dispatchers.Main) {
                                        val fileProgress = received.toFloat() / total.toFloat()
                                        ui = ui.copy(downloadProgress = (index + fileProgress) / count, status = "${version.label}：${received / 1024 / 1024} / ${total / 1024 / 1024} MB")
                                    }
                                }
                            }
                        }
                    } finally { connection.disconnect() }
                    require(fileSha256(download) == version.sha256) { "下载文件 SHA-256 不匹配" }
                    ModelPackage.installPipeline(
                        this@MainActivity, Uri.fromFile(download), version.manifestId,
                    )
                    if (archive.exists()) archive.delete()
                    require(download.renameTo(archive)) { "无法保存下载组件到持久存储" }
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
            ui = ui.copy(status = "模型已被移动或删除，已从清单移除。")
            return
        }
        val message = if (split) "正在加载声音模型…" else "正在导入完整模型包…"
        loadPackage(message, importer = {
            if (split) ModelPackage.importModelWithInstalledPipeline(this, uri) else ModelPackage.import(this, uri)
        }) { model ->
            if (remember) saveModelRecord(ModelRecord(uri.toString(), model.name, model.version, split))
        }
    }

    private fun loadModelRecord(record: ModelRecord) = loadModelUri(Uri.parse(record.uri), record.split, remember = false)

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
                array.getJSONObject(index).let { ModelRecord(it.getString("uri"), it.getString("name"), it.getString("version"), it.optBoolean("split", true)) }
            }
        }.getOrDefault(emptyList())
    }

    private fun writeModelRecords(records: List<ModelRecord>) {
        val array = JSONArray()
        records.forEach { array.put(JSONObject().put("uri", it.uri).put("name", it.name).put("version", it.version).put("split", it.split)) }
        getSharedPreferences("models", MODE_PRIVATE).edit().putString("records", array.toString()).apply()
        val stateFile = File(filesDir, "state/model-records.json")
        stateFile.parentFile?.mkdirs()
        val pending = File(stateFile.parentFile, "${stateFile.name}.pending")
        pending.writeText(array.toString())
        if (stateFile.exists()) stateFile.delete()
        require(pending.renameTo(stateFile)) { "无法保存模型清单" }
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
                ui = ui.copy(busy = false, modelInfo = "${it.name} / ${it.version} / ${it.sampleRate} Hz", backend = engine.backendName, status = "部署包校验通过。")
            }.onFailure { ui = ui.copy(busy = false, status = "导入失败：${it.message}") }
        }
    }

    private fun synthesize() {
        setBusy("正在合成…")
        lifecycleScope.launch {
            runCatching {
                val options = SynthesisOptions(ui.temperature.toFloat(), ui.topP.toFloat(), ui.topK.toInt(), ui.penalty.toFloat(), ui.speed.toFloat(), ui.steps.toInt())
                withContext(Dispatchers.Default) { synchronized(engineLock) { engine.synthesize(SynthesisRequest(ui.text, options = options), File(cacheDir, "tts.wav")) } }
            }.onSuccess { output = it; ui = ui.copy(busy = false, canPlay = true, status = "合成完成。") }
                .onFailure { ui = ui.copy(busy = false, status = "合成未完成：${it.message}") }
        }
    }

    private fun play() {
        output?.let { file -> player?.release(); player = MediaPlayer().apply { setDataSource(file.path); prepare(); start() } }
    }

    private fun startApiServer() {
        val port = ui.port.toIntOrNull()
        if (port == null || port !in 1024..65535) { ui = ui.copy(serverEnabled = false, serverStatus = "端口必须在 1024 到 65535 之间"); return }
        runCatching {
            LocalOpenAiServer(port, File(cacheDir, "openai-api"), { engine.isLoaded }, { engine.backendName },
                { request, file -> synchronized(engineLock) { engine.synthesize(request, file) } }).also { it.start(NanoHTTPD.SOCKET_READ_TIMEOUT, false); apiServer = it }
        }.onSuccess { ui = ui.copy(serverEnabled = true, serverStatus = "${it.endpoint} · POST /v1/audio/speech") }
            .onFailure { ui = ui.copy(serverEnabled = false, serverStatus = "启动失败：${it.message}") }
    }

    private fun stopApiServer() { apiServer?.stop(); apiServer = null; ui = ui.copy(serverEnabled = false, serverStatus = "已停止") }
    private fun setBusy(status: String) { ui = ui.copy(busy = true, status = status) }
    private fun openWebsite(url: String) = startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(url)))

    @Composable private fun MainScreen() {
        var tab by rememberSaveable { mutableIntStateOf(0) }
        Scaffold(topBar = { TopAppBar(title = { Text("GSV Mobile") }, actions = { IconButton(onClick = { tab = 2 }) { Icon(Icons.Default.Settings, "设置") } }) }) { padding ->
            Column(Modifier.padding(padding).fillMaxSize()) {
                PrimaryTabRow(tab) {
                    Tab(tab == 0, { tab = 0 }, text = { Text("合成") })
                    Tab(tab == 1, { tab = 1 }, text = { Text("模型") })
                    Tab(tab == 2, { tab = 2 }, text = { Text("设置") })
                    Tab(tab == 3, { tab = 3 }, text = { Text("关于") })
                }
                when (tab) { 0 -> SynthesisPane(); 1 -> ModelLibraryPane(); 2 -> SettingsPane(); else -> AboutPane() }
            }
        }
        if (ui.showFirstRun) ComponentDialog()
    }

    @Composable private fun ComponentDialog() {
        AlertDialog(
            onDismissRequest = { if (!ui.busy) ui = ui.copy(showFirstRun = false) },
            title = { Text("选择 Pipeline 组件") },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Text("版本（至少一项）", style = MaterialTheme.typography.labelLarge)
                    ComponentVersion.entries.forEach { version ->
                        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                            Checkbox(version in ui.selectedVersions, { checked ->
                                val next = if (checked) ui.selectedVersions + version else ui.selectedVersions - version
                                if (next.isNotEmpty()) ui = ui.copy(selectedVersions = next)
                            })
                            Column { Text(version.label); Text(if (version.manifestId in ui.installedVersions) "已安装" else "未安装", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant) }
                        }
                    }
                    ui.downloadProgress?.let { LinearProgressIndicator(progress = { it }, modifier = Modifier.fillMaxWidth()) }
                }
            },
            confirmButton = { Button(::downloadPipelines, enabled = !ui.busy && ui.selectedVersions.isNotEmpty()) { Icon(Icons.Default.Download, null); Spacer(Modifier.width(8.dp)); Text(if (ui.selectedVersions.size > 1) "下载 ${ui.selectedVersions.size} 个" else "下载组件") } },
            dismissButton = { Row { TextButton(onClick = { ui = ui.copy(showFirstRun = false); pickPipelines.launch(packageTypes) }, enabled = !ui.busy) { Text("手动导入") }; TextButton(onClick = { ui = ui.copy(showFirstRun = false) }, enabled = !ui.busy) { Text("稍后") } } },
        )
    }

    @Composable private fun SynthesisPane() {
        Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(20.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
            Text(ui.backend, style = MaterialTheme.typography.titleMedium)
            Text(ui.modelInfo, color = MaterialTheme.colorScheme.onSurfaceVariant)
            OutlinedTextField(ui.text, { ui = ui.copy(text = it) }, label = { Text("输入文本") }, minLines = 5, modifier = Modifier.fillMaxWidth())
            Text("生成参数", style = MaterialTheme.typography.titleMedium)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) { NumberField("温度", ui.temperature, { ui = ui.copy(temperature = it) }, Modifier.weight(1f)); NumberField("Top P", ui.topP, { ui = ui.copy(topP = it) }, Modifier.weight(1f)); NumberField("Top K", ui.topK, { ui = ui.copy(topK = it) }, Modifier.weight(1f)) }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) { NumberField("惩罚", ui.penalty, { ui = ui.copy(penalty = it) }, Modifier.weight(1f)); NumberField("语速", ui.speed, { ui = ui.copy(speed = it) }, Modifier.weight(1f)); NumberField("CFM 步数", ui.steps, { ui = ui.copy(steps = it) }, Modifier.weight(1f)) }
            Button(::synthesize, enabled = !ui.busy && ui.text.isNotBlank(), modifier = Modifier.fillMaxWidth()) { Text(if (ui.busy) "处理中" else "合成语音") }
            OutlinedButton(::play, enabled = ui.canPlay && !ui.busy, modifier = Modifier.fillMaxWidth()) { Icon(Icons.Default.PlayArrow, null); Spacer(Modifier.width(8.dp)); Text("播放") }
            if (ui.busy) LinearProgressIndicator(Modifier.fillMaxWidth())
            if (ui.status.isNotBlank()) Text(ui.status, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }

    @Composable private fun ModelLibraryPane() {
        Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(20.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
            Text("Pipeline 组件", style = MaterialTheme.typography.titleMedium)
            Text(
                if (ui.installedVersions.isEmpty()) "未安装" else "已安装：${ComponentVersion.entries.filter { it.manifestId in ui.installedVersions }.joinToString { it.shortLabel }}",
                color = if (ui.pipelineInstalled) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.error,
            )
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button({ if (ui.pipelineInstalled) pickModel.launch(packageTypes) else ui = ui.copy(showFirstRun = true) }, enabled = !ui.busy, modifier = Modifier.weight(1f)) { Icon(Icons.Default.UploadFile, null); Spacer(Modifier.width(6.dp)); Text("添加声音模型") }
                OutlinedButton({ pickCombined.launch(packageTypes) }, enabled = !ui.busy, modifier = Modifier.weight(1f)) { Text("添加完整包") }
            }
            TextButton({ openWebsite(MODEL_CONVERTER_URL) }) {
                Text("如何获取 .gsvm?")
                Spacer(Modifier.width(6.dp))
                Icon(Icons.AutoMirrored.Filled.OpenInNew, "在浏览器中打开")
            }
            HorizontalDivider()
            Text("模型清单", style = MaterialTheme.typography.titleLarge)
            if (ui.models.isEmpty()) {
                Text("暂无模型", color = MaterialTheme.colorScheme.onSurfaceVariant)
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
                            Icon(if (expanded) Icons.Default.ExpandLess else Icons.Default.ExpandMore, if (expanded) "收起" else "展开")
                        }
                        if (expanded) {
                            HorizontalDivider()
                            Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                                Text(if (record.split) "分离声音模型" else "完整部署包", style = MaterialTheme.typography.bodySmall)
                                Text(Uri.decode(record.uri), style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                                Button({ loadModelRecord(record) }, enabled = !ui.busy, modifier = Modifier.align(Alignment.End)) { Text("加载模型") }
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
            Text("Pipeline 组件", style = MaterialTheme.typography.titleLarge)
            Text(if (ui.installedVersions.isEmpty()) "未安装" else "已安装：${ComponentVersion.entries.filter { it.manifestId in ui.installedVersions }.joinToString { it.shortLabel }}", color = if (ui.pipelineInstalled) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.error)
            Button({ ui = ui.copy(showFirstRun = true) }, enabled = !ui.busy) { Icon(Icons.Default.UploadFile, null); Spacer(Modifier.width(8.dp)); Text(if (ui.pipelineInstalled) "更换组件" else "选择并导入组件") }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) { ComponentVersion.entries.forEach { version -> FilterChip(ui.componentVersion == version, { selectComponentVersion(version) }, { Text(version.shortLabel) }) } }
            OutlinedTextField(ui.componentUrl, { value -> ui = ui.copy(componentUrl = value); getSharedPreferences("components", MODE_PRIVATE).edit().putString(ui.componentVersion.preferenceKey, value).apply() }, label = { Text("${ui.componentVersion.label} 下载地址") }, leadingIcon = { Icon(Icons.Default.Download, null) }, modifier = Modifier.fillMaxWidth(), singleLine = true)
            HorizontalDivider()
            Text("OpenAI 本地接口", style = MaterialTheme.typography.titleLarge)
            OutlinedTextField(ui.port, { ui = ui.copy(port = it) }, enabled = !ui.serverEnabled, label = { Text("端口") }, keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number), modifier = Modifier.fillMaxWidth())
            Row(verticalAlignment = Alignment.CenterVertically) { Switch(ui.serverEnabled, { if (it) startApiServer() else stopApiServer() }); Spacer(Modifier.width(12.dp)); Text(if (ui.serverEnabled) "接口已启用" else "启用接口") }
            Text(ui.serverStatus, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }

    @Composable private fun AboutPane() {
        Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(20.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
            Text("GSV Mobile", style = MaterialTheme.typography.headlineSmall)
            Text("面向 Android 的 GPT-SoVITS V2 Pro Plus 与 V4 FP32 部署工具。", color = MaterialTheme.colorScheme.onSurfaceVariant)
            HorizontalDivider()
            Text("项目链接", style = MaterialTheme.typography.titleMedium)
            OutlinedButton({ openWebsite(PROJECT_URL) }, modifier = Modifier.fillMaxWidth()) {
                Text("GPT-SoViTs-android", modifier = Modifier.weight(1f))
                Icon(Icons.AutoMirrored.Filled.OpenInNew, "在浏览器中打开")
            }
            OutlinedButton({ openWebsite(UPSTREAM_URL) }, modifier = Modifier.fillMaxWidth()) {
                Text("GPT-SoVITS 上游项目", modifier = Modifier.weight(1f))
                Icon(Icons.AutoMirrored.Filled.OpenInNew, "在浏览器中打开")
            }
            Text("本项目采用 GPL-3.0；上游项目及第三方组件遵循各自许可证。", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }

    @Composable private fun NumberField(label: String, value: String, change: (String) -> Unit, modifier: Modifier) = OutlinedTextField(value, change, modifier, label = { Text(label) }, singleLine = true, keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Decimal))
    override fun onDestroy() { apiServer?.stop(); player?.release(); synchronized(engineLock) { engine.close() }; super.onDestroy() }

    companion object {
        private val packageTypes = arrayOf("application/zip", "application/octet-stream")
        private const val MODEL_CONVERTER_URL = "https://gs.cutefireflyuwu.sbs"
        private const val PROJECT_URL = "https://github.com/tuxKOH/GPT-SoViTs-android"
        private const val UPSTREAM_URL = "https://github.com/RVC-Boss/GPT-SoVITS"
    }
}

private data class UiState(
    val pipelineInstalled: Boolean = false, val showFirstRun: Boolean = false, val componentUrl: String = "",
    val componentVersion: ComponentVersion = ComponentVersion.V2PP, val selectedVersions: Set<ComponentVersion> = setOf(ComponentVersion.V2PP),
    val installedVersions: Set<String> = emptySet(), val downloadProgress: Float? = null,
    val models: List<ModelRecord> = emptyList(),
    val expandedModelUri: String? = null,
    val modelInfo: String = "未加载模型", val backend: String = "CPU · 未加载", val status: String = "",
    val text: String = "", val temperature: String = "1.0", val topP: String = "1.0", val topK: String = "10",
    val penalty: String = "1.35", val speed: String = "1.0", val steps: String = "32", val busy: Boolean = false, val canPlay: Boolean = false,
    val port: String = "9880", val serverEnabled: Boolean = false, val serverStatus: String = "已停止",
)

private data class ModelRecord(val uri: String, val name: String, val version: String, val split: Boolean)

private enum class ComponentVersion(
    val label: String,
    val shortLabel: String,
    val manifestId: String,
    val preferenceKey: String,
    val defaultUrl: String,
    val sha256: String,
) {
    V2PP("V2 Pro Plus", "V2PP", "v2ProPlus", "pipeline_url_v2pp", "https://hf-mirror.com/tuxjhtd/GPT-Sovits_pipeline_Torchscript/resolve/main/pipeline-v2pp-cpu-fp32.gsvm?download=true", "a33bc3d4e297875457c6fc1042869201cb4b97d25ad8b27267154c552f8a9b35"),
    V4("V4", "V4", "v4", "pipeline_url_v4", "https://hf-mirror.com/tuxjhtd/GPT-Sovits_pipeline_Torchscript/resolve/main/pipeline-v4-cpu-fp32.gsvm?download=true", "8cfec15b4a4cb0a7b6cac8df52f4f44e3493694d73e0107b363ddd1012d4f3fa"),
}

@Composable private fun GsvTheme(content: @Composable () -> Unit) {
    val colors = if (androidx.compose.foundation.isSystemInDarkTheme()) darkColorScheme(primary = androidx.compose.ui.graphics.Color(0xFF8FD5B0)) else lightColorScheme(primary = androidx.compose.ui.graphics.Color(0xFF246B4B), secondary = androidx.compose.ui.graphics.Color(0xFF5B635D))
    MaterialTheme(colorScheme = colors, typography = Typography(), content = content)
}
