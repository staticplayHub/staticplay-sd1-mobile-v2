# Staticplay SD1 Mobile (On-Device, Android)

Goal: ship a **real offline** Android app (no servers, no cloud) that runs an SD1-class model **on-device** using **NNAPI** (via a native runtime).

This repo currently includes a working offline SD1.5 ONNX "pack" runner (unzip + inspect + generate).

## Key points (no surprises)

- Use a **prebuild** / **native build** (Expo Go won't cut it for native inference).
- "Offline" means the **model files are on the phone** (bundled in the APK or copied into app storage).
- ONNX Runtime is integrated via a tiny **Android native module** (no JSI auto-install).

## Run (Android)

```bash
cd /home/tiny/projects/staticplay-sd1-mobile
export ANDROID_HOME=/home/tiny/Android/sdk
export PATH="$ANDROID_HOME/platform-tools:$PATH"
npx expo prebuild --clean
npx expo run:android   # debug (uses Metro)
```

For a fully offline build (no Metro), install the release build:

```bash
cd /home/tiny/projects/staticplay-sd1-mobile/android
NODE_ENV=production ./gradlew :app:installRelease
```

Then press "Test native ONNX (NNAPI -> CPU)" in the app.

## Bundle With Expo (Windows, release APK)

Use these commands from this project root.

```powershell
# 1) Install dependencies (once)
npm install

# 2) Regenerate native Android project if config/plugins changed
npx expo prebuild --platform android --clean

# 3) Build release APK
cd android
./gradlew.bat :app:assembleRelease

# 4) Install to connected phone (USB debugging on)
./gradlew.bat :app:installRelease
```

APK output path:

`android/app/build/outputs/apk/release/app-release.apk`

Notes:
- This v3 build uses package id `com.anonymous.staticplaysd1mobile.v3`, so v1/v2 can stay installed.
- Some setup buttons can take longer than expected (especially inspect actions).
- Generation works offline but is still being optimized for speed.
- Website: `https://staticplay.co.uk`

## SD1.5 pack + generation (offline)

1) Put the SD1.5 ONNX pack zip at the app external directory as:

`/storage/emulated/0/Android/data/com.anonymous.staticplaysd1mobile.v3/files/sd15-onnx-pack.zip`

2) In the app:
   - (Optional) `Show app external pack path` (for sanity check)
   - `Unzip SD15 pack from external` (one-time; can take minutes)
   - `Load tokenizer (bundled)` (no external files required)
   - Toggle backend:
     - `Backend: NNAPI (experimental)` = fastest, can be unstable on some devices/drivers
     - `Backend: CPU (safe)` = slower, but avoids NNAPI driver issues
   - `Generate 512x512 (quick 4)` to prove the end-to-end pipeline
   - `Generate 512x512 (quality 20)` for higher quality (slower)

3) Output images save to the same app external folder as `sd_out_*.png`.

## SD1 plan (high level)

1) Pick an SD1 distilled variant (LCM/Turbo-style) to keep steps ~4-8.
2) Export components to ONNX (text encoder, UNet, VAE decoder).
3) Quantize mostly UNet (INT8) + keep VAE in FP16/BF16.
4) Run inference via a native runtime with NNAPI; fall back to CPU if NNAPI can't compile a node.
5) Ship the ONNX files in-app (or as offline downloadable packs).

## V3 notes (2026-02-12)

- This branch installs as a separate app id (`com.anonymous.staticplaysd1mobile.v3`) so v1/v2 stay on device.
- Guided mode is enabled to show one button at a time for easier setup flow.
- Added `Auto-pick fastest backend` which benchmarks both NNAPI and CPU and selects the faster one for this device.
- Some actions can take longer than expected, especially:
  - `Inspect pack UNet`
  - `Inspect pack text_encoder`
  - `Inspect pack vae_decoder`
  - first unzip/generation after launch
- Overall generation is currently functional but still slower than target. Ongoing work is focused on speed improvements.
- Public site: `https://staticplay.co.uk`
