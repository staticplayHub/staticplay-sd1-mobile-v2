# SD1 On-Device Plan (Android / NNAPI)

This is the concrete path to a **fully offline** SD1-style image generator running on a modern Android phone.

## What “offline” means here

- No HTTP calls to your PC
- No cloud inference
- No “download on first run” (unless you opt into offline model packs)
- Model + tokenizer + scheduler files live on-device

## Target constraints (realistic)

- Base resolution: `512x512` (then optional upscale)
- Steps: `4–8` (distilled / LCM)
- Batch size: `1`
- Expect to spend effort on: memory, kernel support, fallback paths

## Architecture (recommended)

**React Native UI (Expo prebuild / native build)**  
→ **Native runtime (recommended)**: a tiny Android module (Kotlin/Java) that uses **ONNX Runtime Android** directly (no JSI auto-install)  
→ **Execution provider**: `NNAPI` first, then `CPU` fallback  
→ **Pipeline components (ONNX)**:
- `text_encoder.onnx`
- `unet.onnx` (largest; primary quantization target)
- `vae_decoder.onnx`

### Note on `onnxruntime-react-native`

On our Samsung S24 test device with Expo SDK 54 / RN 0.81 bridgeless runtime, `onnxruntime-react-native` crashes at startup with:

`TypeError: Cannot read property 'install' of null`

So the plan is to integrate ONNX Runtime via a small native module (or switch to TFLite) rather than relying on a JSI auto-install library.

## Model choice

Pick an SD1-class model that is already optimized for few-step sampling:

- SD1.5 + LCM / Lightning-style distillation (preferred)
- SD1.5 + an LCM LoRA (still “offline”, but you ship the LoRA too)

The goal is not “highest possible quality”; it’s “fast + stable on phone”.

Practical rule: target **4–8 steps**, otherwise “offline on phone” becomes “minutes per image”.

## Export to ONNX (dev machine)

Use a conversion pipeline that produces:

- static shapes where possible (mobile likes static)
- ops that NNAPI supports (or graceful fallback)

Two common approaches:

- `diffusers` export scripts + ORT optimization
- `optimum` export to ONNX

## Quantization strategy

- Quantize **UNet** to INT8 if possible (biggest win)
- Keep **VAE decoder** in FP16/BF16 (INT8 can trash images)
- Consider INT8 for the text encoder if it compiles cleanly on NNAPI

Concrete path (dev machine):

1) Export SD1.5 to ONNX (`tools/sd1/`).
2) Run ORT graph optimizations.
3) Try INT8 quantization on UNet first; benchmark on device; keep a CPU fallback if NNAPI won’t compile.

## Runtime strategy

- Prefer NNAPI EP
- Detect if NNAPI fails to compile; re-create session with CPU EP
- Cache text embeddings when prompt doesn’t change
- Use memory-saving knobs:
  - attention slicing / efficient attention (if available in your graph)
  - VAE tiling (if you implement decode tiling)

## Packaging

- Bundle ONNX files under `assets/models/`
- On first launch, copy them to app storage and load from a file path (ORT needs a file path)

### Model pack (offline import)

For SD1 on-device we’ll use 3–4 ONNX files (exact names are up to us, but the app expects you to import `.onnx` files and keeps them in:

`documentDirectory/models/sd1/`

Recommended file names:

- `text_encoder.onnx`
- `unet.onnx`
- `vae_decoder.onnx`
- (optional) `tokenizer.json` / vocab files (if you do tokenizer outside native)

The app can “inspect” a model on-device to show its declared inputs/outputs. This helps catch mismatched exports early, fully offline.

## Milestones

1) Launch-stable offline app shell (done).
2) Native inference “hello world” (ORT Android or TFLite) + NNAPI/CPU switch.
3) Tokenizer + text encoder ONNX (prompt → embeddings).
4) UNet ONNX + scheduler loop (latents step).
5) VAE decoder ONNX (latents → image).
6) Add LCM/turbo model for 4–8 steps.
7) Quantize UNet and benchmark.
