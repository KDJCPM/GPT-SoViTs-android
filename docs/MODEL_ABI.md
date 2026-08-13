# GSVM deployment ABI

Android accepts ZIP-compatible `.gsvm` deployment packages whose first entry is `manifest.json`.
Every payload file must be declared with its path, byte size and SHA-256 hash.

## Supported models

- `v2ProPlus`, 32 kHz output
- `v4`, 48 kHz output

The supported executor values are `torchscript-cpu-single` and `torchscript-cpu-staged`. Both expose
the high-level entrypoint `synthesize_utf8_to_pcm16`.

`qnn-htp` is reserved for a complete, SoC-specific backend package. A standalone context produced
by `build_qnn_htp_context.py` uses `format=gsv-qnn-compiled-component` and `deployable=false`; it is
not accepted as proof of complete Android TTS.

QNN pipeline and voice attachments use the mandatory `*.qnn.gsvm` suffix. The suffix distinguishes
backend-specific NPU attachments from ordinary CPU `*.gsvm` packages; Android checks it before
extracting either attachment.

## Package roles

A self-contained package omits `artifact_role` and contains both the shared text frontend and voice
runtime files.

A split deployment uses:

- `artifact_role: "pipeline"` for `runtime/frontend/*`
- `artifact_role: "model"` for voice-specific TorchScript files

Both packages identify the same model version, sample rate, entrypoint and frontend ABI. Legacy
bundle IDs ending in `:options0` or `:options1` are normalized before frontend compatibility is
checked because synthesis options belong to the model graph, not the shared text frontend.

The shared pipeline supports only packages produced by the current converter contract. Legacy
models using `frontend_profile=full-g2pw-v2` are rejected and must be regenerated with the web
converter; Android does not rewrite old graphs or bypass frontend ABI checks.

## Runtime options

A model package with `runtime_options_version: 1` also carries a `runtime_options` object. Android
enables and accepts only the explicitly declared keys. Common V2 Pro Plus and V4 keys are:

- `temperature`
- `top_p`
- `top_k`
- `repetition_penalty`
- `speed_factor`
- `seed`

V4 additionally declares `sample_steps` for its CFM stage. V2 Pro Plus does not declare this key;
Android disables the control and rejects a non-default API value instead of silently ignoring it.

Android rejects non-default controls when the selected model does not declare the options ABI. A
control is never accepted and silently ignored.

## Runtime reference input

A package with `reference_input_version: 1` accepts an optional request-scoped reference:

- mono FP32 PCM at 16 kHz for HuBERT;
- the same mono audio resampled to 32 kHz for spectrogram and speaker conditioning;
- UTF-8 transcript plus a fixed language selector.

HuBERT, semantic extraction, spectrogram/mel construction and speaker/reference conditioning are
part of the converted graph. Android only decodes the selected media, resamples it and submits the
fixed structure. When the structure is omitted, the graph uses the conditioning embedded during
conversion. Supplying it to a package without version 1 is an error, never a silent fallback.

The local OpenAI endpoint follows the same rule. Omitted `reference_audio` uses the preset;
`reference_audio` requires `reference_text` and applies only to that request.

A static QNN artifact declares its reference PCM capacity and duration policy. The current V2 Pro
Plus QNN product uses `duration_policy=exact_samples` with 80,000 samples at 16 kHz and 160,000
samples at 32 kHz (exactly five seconds). Android rejects a different duration instead of silently
truncating or padding it. CPU artifacts continue to accept the duration range implemented by their
converted high-level executor.

## QNN backend identity

Each complete QNN package must identify exactly one target with `target_soc`,
`target_soc_family=qualcomm_snapdragon_8`, `target_asic`, `target_soc_model`, `htp_arch`,
`qairt_version` and `backend_artifact`. The accepted identities are exact:

- `snapdragon_8_gen_3`: `SM8650`, SoC model 57, HTP `V75`
- `snapdragon_8_elite`: `SM8750`, SoC model 69, HTP `V79`
- `snapdragon_8_elite_gen_5`: `SM8850`, SoC model 87, HTP `V81`

Context binaries are not reusable across these targets. Unknown targets and the ambiguous
`snapdragon_8_gen_5` name are rejected. A package claiming complete HTP execution must also declare
`cpu_neural_fallback=false`; the runtime disables ONNX Runtime CPU EP fallback for every neural
session and treats an unsupported operator as an execution failure.

The CPU artifact remains FP32. A QAIRT graph compiled with `--float_bitwidth 16` has FP16 graph I/O
and is a separate backend artifact; it is not a single file that can be relabelled for both CPU and
HTP execution.

## Runtime boundary

The Android host verifies and extracts packages, selects an artifact matching the requested backend
and device, opens one prepared backend session and submits UTF-8 text plus synthesis options.
Checkpoint parsing, version migrations, graph construction, frontend generation, backend
partitioning and weight conversion remain conversion-time operations.
