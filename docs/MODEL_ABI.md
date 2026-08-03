# GSVM deployment ABI

Android accepts ZIP-compatible `.gsvm` deployment packages whose first entry is `manifest.json`.
Every payload file must be declared with its path, byte size and SHA-256 hash.

## Supported models

- `v2ProPlus`, 32 kHz output
- `v4`, 48 kHz output

The supported executor values are `torchscript-cpu-single` and `torchscript-cpu-staged`. Both expose
the high-level entrypoint `synthesize_utf8_to_pcm16`.

## Package roles

A self-contained package omits `artifact_role` and contains both the shared text frontend and voice
runtime files.

A split deployment uses:

- `artifact_role: "pipeline"` for `runtime/frontend/*`
- `artifact_role: "model"` for voice-specific TorchScript files

Both packages identify the same model version, sample rate, entrypoint and frontend ABI. Legacy
bundle IDs ending in `:options0` or `:options1` are normalized before frontend compatibility is
checked because synthesis options belong to the model graph, not the shared text frontend.

## Runtime options

A model package with `runtime_options_version: 1` consumes:

- `temperature`
- `top_p`
- `top_k`
- `repetition_penalty`
- `speed_factor`
- `sample_steps`
- `seed`

Android rejects non-default controls when the selected model does not declare the options ABI. A
control is never accepted and silently ignored.

## Runtime boundary

The Android host verifies and extracts packages, selects the matching installed pipeline, opens one
CPU session and submits UTF-8 text plus synthesis options. Checkpoint parsing, version migrations,
graph construction, frontend generation and weight conversion remain conversion-time operations.
