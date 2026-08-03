# Project instructions

## Core deployment philosophy

Conversion is allowed to be complex; execution must be as simple as possible.

- Push graph analysis, graph partitioning, operator replacement, weight normalization, precision
  decisions, text preprocessing, reference conditioning, shape specialization, backend compilation,
  compatibility handling and validation into the conversion/build tools.
- A runtime must not understand GPT-SoVITS checkpoint history, Python classes, component topology,
  tensor-name migrations, graph partitioning or backend-specific conversion rules.
- Each deployable artifact is backend-specific and exposes one stable high-level operation:
  load the artifact, submit UTF-8 text and synthesis options, receive PCM audio.
- Prefer a larger or more expensive conversion process when it removes runtime branches,
  dependencies, preprocessing or orchestration.
- Backend switching means selecting another fully prepared artifact with the same high-level API.
  It must not require changing application inference logic.
- Canonical/checkpoint-preserving data may remain in a development package for rebuilding and
  verification, but production Android packages should contain only the files needed by the chosen
  executor.
- CPU is the correctness reference. A converted artifact must be compared with upstream inference
  before it is marked deployable. Lossless mode must not introduce quantization or dtype changes.
- Never report Android TTS as complete when the app only validates weights, emits placeholder audio,
  calls a desktop/server process, or still requires the original `.pth`/`.ckpt` parser.

## Android runtime contract

The Android layer should remain a thin host:

1. import and verify one deployable package;
2. create one backend session;
3. pass UTF-8 text plus a small fixed option structure;
4. receive mono PCM and play/save it.

All model topology and preprocessing details belong to the converted artifact and its backend
runtime library, not to Kotlin UI code.

## Supported model scope

Only GPT-SoVITS V2 Pro Plus and V4 are product targets. Complete quality baselines must retain the
full upstream frontend and conditioning assets. Other historical versions may remain detectable for
safe error reporting, but new runtime/export work must not spend complexity on them.

## Supported Qualcomm NPU scope

The product QNN/HTP target allowlist is intentionally limited to these Snapdragon platforms:

- `snapdragon_8_gen_3` (`SM8650`)
- `snapdragon_8_elite` (`SM8750`)
- `snapdragon_8_elite_gen_5` (`SM8850`)

Older Snapdragon generations, unlisted Qualcomm platforms and unknown devices are not QNN product
targets. The CPU artifact remains the correctness fallback, but it does not make an unlisted SoC a
supported NPU target. Every QNN artifact must carry the exact `target_soc`, `target_soc_family`,
`htp_arch`, `qairt_version` and `backend_artifact` values produced by the matching QAIRT SDK; the
runtime must never infer an HTP architecture or reuse a context binary across generations.

## Memory and quality target

- Product validation targets flagship Android devices with at least 16 GB of physical RAM.
- Keep peak process memory within an 8--12 GB engineering budget where possible by staging exact
  FP32 components and releasing completed stages. Changing weight values, precision, operators,
  CFM step count or output quality is not an acceptable memory optimization without explicit user
  approval.
- A staged executor may alter only component residency and scheduling. It must preserve the same
  frontend, conditioning, FP32 weights, model operations, sampling options and PCM result expected
  from the unstaged quality baseline.
