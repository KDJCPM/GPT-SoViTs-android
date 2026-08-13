package ai.gsv.mobile

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.ExpandLess
import androidx.compose.material.icons.filled.ExpandMore
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.SaveAlt
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.UploadFile
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.res.stringResource

@OptIn(ExperimentalMaterial3Api::class)
@Composable
internal fun GsvAppUi(
    state: UiState,
    onStateChange: (UiState) -> Unit,
    onSynthesize: () -> Unit,
    onPlay: () -> Unit,
    onSave: () -> Unit,
    onPickReference: () -> Unit,
    onInstallFirefly: () -> Unit,
    onSelectPipeline: (ComponentVersion) -> Unit,
    onInstallPipelines: () -> Unit,
    onPickPipeline: () -> Unit,
    onPickQnnPipeline: () -> Unit,
    onPickModel: () -> Unit,
    onPickCombined: () -> Unit,
    onScan: () -> Unit,
    onChooseQnnModel: (ModelRecord) -> Unit,
    onLoadModel: (ModelRecord) -> Unit,
    onRemoveQnnModel: (ModelRecord) -> Unit,
    onOpenConverter: () -> Unit,
    onOpenProject: () -> Unit,
    onOpenUpstream: () -> Unit,
    onSelectLanguage: (AppLanguage) -> Unit,
    onStartApi: () -> Unit,
    onStopApi: () -> Unit,
    onDownloadPipelines: () -> Unit,
) {
    var tab by rememberSaveable { mutableIntStateOf(0) }
    Scaffold(
        topBar = { TopAppBar(title = { Text(stringResource(R.string.app_name)) }, actions = { IconButton({ tab = 2 }) { Icon(Icons.Default.Settings, stringResource(R.string.settings)) } }) },
        bottomBar = { NavigationBar { listOf(R.string.tab_synthesis, R.string.tab_models, R.string.tab_settings).forEachIndexed { index, label -> NavigationBarItem(selected = tab == index, onClick = { tab = index }, icon = { Icon(if (index == 0) Icons.Default.PlayArrow else if (index == 1) Icons.Default.Download else Icons.Default.Settings, null) }, label = { Text(stringResource(label)) }) } } },
    ) { padding ->
        when (tab) {
            0 -> SynthesisScreen(state, onStateChange, onSynthesize, onPlay, onSave, onPickReference, Modifier.padding(padding))
            1 -> ModelsScreen(state, onInstallFirefly, onSelectPipeline, onInstallPipelines, onPickQnnPipeline, onPickModel, onPickCombined, onScan, onChooseQnnModel, onLoadModel, onRemoveQnnModel, onOpenConverter, Modifier.padding(padding))
            else -> SettingsScreen(state, onStateChange, onInstallPipelines, onPickPipeline, onStartApi, onStopApi, onSelectLanguage, onOpenProject, onOpenUpstream, Modifier.padding(padding))
        }
    }
    if (state.showFirstRun) PipelineDialog(state, onStateChange, onDownloadPipelines, onPickPipeline)
}

@Composable private fun SynthesisScreen(s: UiState, change: (UiState) -> Unit, synthesize: () -> Unit, play: () -> Unit, save: () -> Unit, pickReference: () -> Unit, modifier: Modifier) =
    Column(modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(20.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
        SectionTitle(R.string.synthesis_ready)
        StatusPanel(s.modelInfo, s.backend, if (s.referenceUri == null) stringResource(R.string.reference_model_preset) else stringResource(R.string.reference_temporary, s.referenceName))
        OutlinedTextField(s.text, { change(s.copy(text = it)) }, label = { Text(stringResource(R.string.input_text)) }, minLines = 6, modifier = Modifier.fillMaxWidth())
        ChoiceRow(R.string.text_language, s.textLanguage) { change(s.copy(textLanguage = it)) }
        HorizontalDivider()
        SectionTitle(R.string.reference_voice)
        FilledTonalButton(pickReference, enabled = !s.busy && s.referenceOverrideSupported, modifier = Modifier.fillMaxWidth()) { Icon(Icons.Default.UploadFile, null); Spacer(Modifier.width(8.dp)); Text(stringResource(R.string.reference_choose_audio)) }
        if (s.referenceUri != null) {
            OutlinedTextField(s.referencePrompt, { change(s.copy(referencePrompt = it)) }, label = { Text(stringResource(R.string.reference_prompt)) }, modifier = Modifier.fillMaxWidth(), minLines = 2)
            ChoiceRow(R.string.reference_language, s.referenceLanguage) { change(s.copy(referenceLanguage = it)) }
            TextButton({ change(s.copy(referenceUri = null, referenceName = "", referencePrompt = "")) }) { Text(stringResource(R.string.reference_use_preset)) }
        }
        HorizontalDivider()
        SectionTitle(R.string.generation_options)
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) { Number(s.temperature, R.string.temperature, "temperature" in s.runtimeOptions, Modifier.weight(1f)) { change(s.copy(temperature = it)) }; Number(s.topP, R.string.top_p, "top_p" in s.runtimeOptions, Modifier.weight(1f)) { change(s.copy(topP = it)) } }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) { Number(s.topK, R.string.top_k, "top_k" in s.runtimeOptions, Modifier.weight(1f)) { change(s.copy(topK = it)) }; Number(s.steps, R.string.cfm_steps, "sample_steps" in s.runtimeOptions, Modifier.weight(1f)) { change(s.copy(steps = it)) } }
        Button(synthesize, enabled = !s.busy && s.text.isNotBlank(), modifier = Modifier.fillMaxWidth().height(52.dp)) { Text(stringResource(if (s.busy) R.string.processing else R.string.synthesize_speech)) }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) { OutlinedButton(play, enabled = s.canPlay && !s.busy, modifier = Modifier.weight(1f)) { Icon(Icons.Default.PlayArrow, null); Text(stringResource(R.string.play)) }; OutlinedButton(save, enabled = s.canPlay && !s.busy, modifier = Modifier.weight(1f)) { Icon(Icons.Default.SaveAlt, null); Text(stringResource(R.string.save_audio)) } }
        ProgressAndStatus(s)
    }

@Composable private fun ModelsScreen(s: UiState, install: () -> Unit, select: (ComponentVersion) -> Unit, pipelines: () -> Unit, qnnPipeline: () -> Unit, model: () -> Unit, complete: () -> Unit, scan: () -> Unit, qnnModel: (ModelRecord) -> Unit, load: (ModelRecord) -> Unit, remove: (ModelRecord) -> Unit, converter: () -> Unit, modifier: Modifier) =
    Column(modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(20.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
        SectionTitle(R.string.recommended_setup)
        StatusPanel(stringResource(R.string.firefly_default_name), stringResource(R.string.recommended_setup_detail), stringResource(R.string.external_model_directory))
        Button(install, enabled = !s.busy && QualcommTargetSoc.detect() != null, modifier = Modifier.fillMaxWidth().height(52.dp)) { Icon(Icons.Default.Download, null); Spacer(Modifier.width(8.dp)); Text(stringResource(R.string.install_firefly_npu)) }
        if (QualcommTargetSoc.detect() == null) Text(stringResource(R.string.recommended_qnn_unsupported_device), color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
        HorizontalDivider(); SectionTitle(R.string.pipeline_components)
        ComponentVersion.entries.filter { it.manifestId in s.installedVersions }.forEach { version -> val chosen = version.manifestId == s.selectedPipelineVersion; Surface(shape = MaterialTheme.shapes.small, color = if (chosen) MaterialTheme.colorScheme.secondaryContainer else MaterialTheme.colorScheme.surfaceVariant, modifier = Modifier.fillMaxWidth().clickable { select(version) }) { Row(Modifier.padding(12.dp), verticalAlignment = Alignment.CenterVertically) { RadioButton(chosen, { select(version) }); Column(Modifier.padding(start = 8.dp)) { Text(version.label); Text(stringResource(if (s.qnnPipelines.any { it.startsWith(version.manifestId) }) R.string.npu_pipeline_ready else R.string.cpu_pipeline_ready), style = MaterialTheme.typography.bodySmall) } } } }
        OutlinedButton(pipelines, modifier = Modifier.fillMaxWidth()) { Text(stringResource(R.string.install_pipeline)) }
        if (s.selectedPipelineVersion != null) TextButton(qnnPipeline) { Text(stringResource(R.string.import_qnn_pipeline_attachment)) }
        HorizontalDivider(); SectionTitle(R.string.model_library)
        s.models.forEach { record -> var expanded by remember(record.uri) { mutableStateOf(false) }; Surface(shape = MaterialTheme.shapes.small, tonalElevation = 1.dp, modifier = Modifier.fillMaxWidth()) { Column { Row(Modifier.fillMaxWidth().clickable { expanded = !expanded }.padding(14.dp), verticalAlignment = Alignment.CenterVertically) { Column(Modifier.weight(1f)) { Text(record.name, maxLines = 1, overflow = TextOverflow.Ellipsis); Text(record.version, style = MaterialTheme.typography.bodySmall) }; Icon(if (expanded) Icons.Default.ExpandLess else Icons.Default.ExpandMore, null) }; if (expanded) { HorizontalDivider(); Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) { Text(stringResource(if (record.qnnUri == null) R.string.qnn_attachment_not_selected else R.string.qnn_attachment_selected), style = MaterialTheme.typography.bodySmall); Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) { OutlinedButton({ qnnModel(record) }, enabled = !s.busy, modifier = Modifier.weight(1f)) { Text(stringResource(R.string.choose_qnn_attachment)) }; Button({ load(record) }, enabled = !s.busy, modifier = Modifier.weight(1f)) { Text(stringResource(if (record.qnnUri == null) R.string.load_model else R.string.load_model_npu)) } }; if (record.qnnUri != null) TextButton({ remove(record) }) { Text(stringResource(R.string.remove_qnn_attachment)) } } } } } }
        OutlinedButton(model, enabled = !s.busy, modifier = Modifier.fillMaxWidth()) { Icon(Icons.Default.UploadFile, null); Spacer(Modifier.width(8.dp)); Text(stringResource(R.string.add_voice_model)) }
        TextButton(complete) { Text(stringResource(R.string.add_complete_package)) }; TextButton(scan) { Text(stringResource(R.string.scan_external_models)) }; TextButton(converter) { Text(stringResource(R.string.how_get_gsvm)) }; ProgressAndStatus(s)
    }

@Composable private fun SettingsScreen(s: UiState, change: (UiState) -> Unit, pipelines: () -> Unit, pick: () -> Unit, start: () -> Unit, stop: () -> Unit, language: (AppLanguage) -> Unit, project: () -> Unit, upstream: () -> Unit, modifier: Modifier) = Column(modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(20.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) { SectionTitle(R.string.app_language); Row { FilterChip(s.appLanguage == AppLanguage.CHINESE, { language(AppLanguage.CHINESE) }, { Text(stringResource(R.string.app_language_chinese)) }); Spacer(Modifier.width(8.dp)); FilterChip(s.appLanguage == AppLanguage.ENGLISH, { language(AppLanguage.ENGLISH) }, { Text(stringResource(R.string.app_language_english)) }) }; HorizontalDivider(); SectionTitle(R.string.openai_local_api); OutlinedTextField(s.port, { change(s.copy(port = it)) }, label = { Text(stringResource(R.string.port)) }, enabled = !s.serverEnabled); Switch(s.serverEnabled, { if (it) start() else stop() }); Text(s.serverStatus, style = MaterialTheme.typography.bodySmall); HorizontalDivider(); SectionTitle(R.string.pipeline_components); Button(pipelines) { Text(stringResource(R.string.change_components)) }; TextButton(pick) { Text(stringResource(R.string.import_manually)) }; HorizontalDivider(); TextButton(project) { Text(stringResource(R.string.project_name)) }; TextButton(upstream) { Text(stringResource(R.string.upstream_project)) } }

@Composable private fun PipelineDialog(s: UiState, change: (UiState) -> Unit, download: () -> Unit, import: () -> Unit) = AlertDialog(onDismissRequest = { if (!s.busy) change(s.copy(showFirstRun = false)) }, title = { Text(stringResource(R.string.pipeline_choose_title)) }, text = { Column { ComponentVersion.entries.forEach { v -> Row(verticalAlignment = Alignment.CenterVertically) { Checkbox(v in s.selectedVersions, { checked -> change(s.copy(selectedVersions = if (checked) s.selectedVersions + v else s.selectedVersions - v)) }); Text(v.label) } }; ProgressAndStatus(s) } }, confirmButton = { Button(download, enabled = !s.busy && s.selectedVersions.isNotEmpty()) { Text(stringResource(R.string.download_component)) } }, dismissButton = { TextButton(import) { Text(stringResource(R.string.import_manually)) } })
@Composable private fun SectionTitle(id: Int) = Text(stringResource(id), style = MaterialTheme.typography.titleLarge)
@Composable private fun StatusPanel(a: String, b: String, c: String) = Surface(shape = MaterialTheme.shapes.small, color = MaterialTheme.colorScheme.surfaceVariant, modifier = Modifier.fillMaxWidth()) { Column(Modifier.padding(14.dp)) { Text(a, style = MaterialTheme.typography.titleMedium); Text(b, style = MaterialTheme.typography.bodyMedium); Text(c, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant) } }
@Composable private fun ChoiceRow(label: Int, selected: String, set: (String) -> Unit) { Text(stringResource(label), style = MaterialTheme.typography.labelLarge); Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) { listOf("auto" to R.string.language_auto, "zh" to R.string.language_chinese, "en" to R.string.language_english).forEach { (v, l) -> FilterChip(selected == v, { set(v) }, { Text(stringResource(l)) }) } } }
@Composable private fun Number(value: String, label: Int, enabled: Boolean, modifier: Modifier, set: (String) -> Unit) = OutlinedTextField(value, set, modifier = modifier, enabled = enabled, label = { Text(stringResource(label)) }, singleLine = true)
@Composable private fun ProgressAndStatus(s: UiState) { if (s.busy) LinearProgressIndicator(progress = { s.downloadProgress ?: 0f }, modifier = Modifier.fillMaxWidth()); if (s.status.isNotBlank()) Text(s.status, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant) }
