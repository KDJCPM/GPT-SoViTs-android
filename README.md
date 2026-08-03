# GSV Mobile

GSV Mobile converts and runs [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
V2 Pro Plus and V4 voices locally on Android. The current runtime uses exact FP32 CPU execution and
keeps the text frontend in a reusable pipeline component.

This project is an independent Android deployment implementation. It is not an official
GPT-SoVITS application and is not affiliated with the upstream maintainers.

## Features

- GPT-SoVITS V2 Pro Plus and V4 model conversion
- FP32 weights without quantization or dtype conversion
- Shared, persistent V2PP/V4 pipeline components
- Persistent model list backed by Android Storage Access Framework URIs
- Compose interface for synthesis, sampling controls and audio playback
- OpenAI-compatible local `POST /v1/audio/speech` endpoint
- Model-only packages by default; optional self-contained packages with pipeline assets

The supported runtime is CPU-only, and CPU output is the correctness reference.

## Requirements

### Android

- Android 8.0 (API 26) or newer
- 64-bit Android device
- 16 GB physical RAM recommended
- 8--12 GB of free process memory for large V4 workloads

### Conversion host

- The original GPT-SoVITS repository checked out next to this directory
- The upstream Python environment and pretrained models
- Enough RAM and disk space for FP32 graph export

The expected source layout is:

```text
GPT-SoVITS/
├── android/       # this project
├── GPT_SoVITS/
├── GPT_weights_v2ProPlus/
└── SoVITS_weights_v2ProPlus/
```

## Build Android

Set the Android SDK path in `local.properties`:

```properties
sdk.dir=/path/to/Android/Sdk
```

Build the debug APK:

```bash
./gradlew :app:assembleDebug
```

The APK is written to `app/build/outputs/apk/debug/app-debug.apk`.

## Convert From The Command Line

Create a model-only package:

```bash
../gpt/bin/python tools/build_android_cpu_pipeline.py \
  --gpt ../GPT_weights_v2ProPlus/voice.ckpt \
  --sovits ../SoVITS_weights_v2ProPlus/voice.pth \
  --reference ../reference.wav \
  --prompt '参考音频对应文本' \
  --name voice \
  --work build/voice \
  --model-output converted_models/voice-model.gsvm \
  --upstream ..
```

Use `--output converted_models/voice.gsvm` instead of `--model-output` to create a self-contained
package.

## Android Usage

1. Install one or both pipeline components on the first run.
2. Open the **模型** tab and add a model-only or self-contained `.gsvm` package.
3. Select the saved model from the collapsible model list.
4. Enter text, adjust synthesis options if needed, synthesize and play the WAV result.

Downloaded pipelines are retained in app-private persistent storage. Model list entries retain
persistable document URIs rather than copying every source package. Missing or moved model files are
removed from the list when loading fails.

## Local OpenAI API

Enable the local API from the Android **设置** tab. It listens on device loopback port `9880` by
default. Forward the port to a connected development host:

```bash
adb forward tcp:9880 tcp:9880
```

Synthesize speech:

```bash
curl http://127.0.0.1:9880/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-sovits-local",
    "voice": "loaded-artifact",
    "input": "这是一段本地语音合成测试。",
    "response_format": "wav",
    "speed": 1.0
  }' \
  --output tts.wav
```

Supported extension fields are `temperature`, `top_p`, `top_k`, `repetition_penalty`,
`sample_steps`, `seed` and `language`. Only WAV output is currently supported.

## Package Contract

Android is a thin executor. Conversion handles checkpoint versions, graph construction, frontend
assets, compatibility metadata and weight verification.

Every deployable package exposes one operation:

```text
UTF-8 text + synthesis options -> mono PCM16 audio
```

Model-only packages declare `artifact_role=model`. Shared pipelines declare
`artifact_role=pipeline`. Android verifies package manifests and SHA-256 hashes before combining
compatible artifacts.

## Quality Policy

- No quantization, FP16 conversion, pruning or weight rewriting
- No reduction of V4 CFM steps for memory or speed
- No operator substitution that changes PCM output
- Memory optimization may change component residency and scheduling only
- CPU inference remains the deployment correctness baseline

Model quality is still determined by the source checkpoints and reference conditioning. Conversion
cannot correct pronunciation or breath artifacts learned by a poorly trained voice model.

## Upstream And Third-Party Work

This project builds on [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS), which is
distributed under the MIT License. GPT-SoVITS architecture, upstream inference code and pretrained
asset handling remain attributable to its authors and contributors.

Model checkpoints, pretrained weights, Android libraries and Python dependencies may have their
own licenses or usage restrictions. Users are responsible for reviewing those terms before
redistributing an APK, converted model or hosted conversion service.

## License

GSV Mobile source code in this directory is licensed under the
[GNU General Public License v3.0](LICENSE).

The GPL-3.0 license applies to this project's source code. It does not relicense GPT-SoVITS,
third-party dependencies, pretrained models or user-provided checkpoints.
