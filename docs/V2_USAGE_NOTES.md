# Staticplay SD1 Mobile v2 - Usage Notes

## What this build is

- Local-first Android build focused on offline SD1.5 ONNX generation.
- Installed as a separate package so v1 remains untouched:
  - `com.anonymous.staticplaysd1mobile.v2`

## Basic flow

1. Show app external path.
2. Unzip SD15 pack from external.
3. Load tokenizer.
4. Run quick generate first, then quality generate.

## Known behavior

- Some buttons can look stalled while still working.
- The slowest setup actions are typically:
  - `Inspect pack UNet`
  - `Inspect pack text_encoder`
  - `Inspect pack vae_decoder`
- Generation is working but still longer than target. Speed optimization is in progress.

## Output location

- App-local output: app external files dir as `sd_out_*.png`
- Exported output: gallery/public folder (`Pictures/Staticplay` and newer builds target `DCIM/Staticplay`)

## Project links

- Website: `https://staticplay.co.uk`
