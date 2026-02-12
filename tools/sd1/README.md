# SD1 (SD 1.5) → ONNX Model Pack (offline)

This folder is for **dev-machine** tooling to prepare an **offline** model pack for the Android app.

Target: `stable-diffusion-v1-5/stable-diffusion-v1-5` (not gated), exported to 3 ONNX components:

- `text_encoder.onnx`
- `unet.onnx`
- `vae_decoder.onnx`

These ONNX files can then be imported on-device using the app’s **Import ONNX files (offline)** button.

## 0) Prereqs

- Hugging Face CLI logged in: `hf auth whoami`
- Python 3.10+ available

## 1) Create venv + install deps

```bash
cd /home/tiny/projects/staticplay-sd1-mobile/tools/sd1
./setup_venv.sh
```

## 2) Download SD1.5 (diffusers format)

```bash
cd /home/tiny/projects/staticplay-sd1-mobile/tools/sd1
./download_sd15.sh
```

This downloads into `tools/sd1/_cache/sd15/`.

## 3) Export components to ONNX

```bash
cd /home/tiny/projects/staticplay-sd1-mobile/tools/sd1
./.venv/bin/python export_sd15_to_onnx.py --in ./_cache/sd15 --out ./_out/sd15-onnx
```

Outputs:

- `tools/sd1/_out/sd15-onnx/text_encoder.onnx`
- `tools/sd1/_out/sd15-onnx/unet.onnx`
- `tools/sd1/_out/sd15-onnx/vae_decoder.onnx`

## Notes

- Export uses **static shapes** for mobile friendliness:
  - batch `1`
  - `77` tokens
  - `512x512` image (latent `64x64`)
- This is a first milestone for **inspect + I/O validation on device**. NNAPI compatibility and full diffusion loop come next.

