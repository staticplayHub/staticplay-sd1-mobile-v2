# SD1.5 ONNX Pack (Offline Import)

The Android app needs SD1.5 exported to ONNX as a **folder**, not a single `.onnx` file.

Reason: the exported `unet.onnx` uses **external tensor data files** (thousands of `onnx__*` files). ONNX Runtime expects them to sit next to `unet.onnx`.

## What the pack contains

From `tools/sd1/_out/sd15-onnx/`:

- `text_encoder.onnx`
- `vae_decoder.onnx`
- `unet.onnx`
- `onnx__*` (external weights for `unet.onnx`)

`tools/sd1/make_sd15_pack_zip.sh` creates:

- `tools/sd1/_out/sd15-onnx-pack.zip`

## Put the pack on the phone

One option (ADB):

```bash
cd /home/tiny/projects/staticplay-sd1-mobile/tools/sd1
./push_sd15_pack_to_phone.sh
```

This pushes to:

- `/sdcard/Download/sd15-onnx-pack.zip`

Alternative (avoids the system file picker / permissions): push into the app’s external files directory:

```bash
adb shell mkdir -p /sdcard/Android/data/com.anonymous.staticplaysd1mobile/files
adb push /home/tiny/projects/staticplay-sd1-mobile/tools/sd1/_out/sd15-onnx-pack.zip \\
  /sdcard/Android/data/com.anonymous.staticplaysd1mobile/files/sd15-onnx-pack.zip
```

Then in the app:

- Tap `Show app external pack path`
- Tap `Unzip SD15 pack from external`

## Tokenizer (required for generation)

For real text prompts, the app needs the SD1.5 CLIP tokenizer files (`vocab.json`, `merges.txt`).

Push them once (ADB):

```bash
cd /home/tiny/projects/staticplay-sd1-mobile/tools/sd1
./push_tokenizer_to_phone.sh
```

Then in the app tap `Load tokenizer from external`.

## Import + inspect (in the app)

1) Tap `Import SD15 pack (.zip)` and pick `sd15-onnx-pack.zip` from Downloads.
2) After unzip, `Pack dir` should show a local path.
3) Tap `Inspect pack UNet` to confirm ONNX Runtime can open it and to view declared inputs/outputs.

Note: the zip can be very large (~3–4GB). The app unzips directly from the picker URI (no “copy to cache”), to avoid duplicating the file on device storage.

If `Inspect` fails, the error string usually tells you what’s wrong (missing files, unsupported ONNX IR/opset, etc.).
