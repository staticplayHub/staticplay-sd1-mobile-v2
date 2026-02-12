import argparse
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline


def export_text_encoder(pipe: StableDiffusionPipeline, out: Path) -> None:
    class TextEncoderWrapper(torch.nn.Module):
        def __init__(self, te: torch.nn.Module):
            super().__init__()
            self.te = te

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            # Return only the embeddings (batch, 77, hidden)
            return self.te(input_ids)[0]

    text_encoder = TextEncoderWrapper(pipe.text_encoder).eval()
    input_ids = torch.zeros((1, 77), dtype=torch.long)
    torch.onnx.export(
        text_encoder,
        (input_ids,),
        out.as_posix(),
        input_names=["input_ids"],
        output_names=["last_hidden_state"],
        opset_version=17,
        do_constant_folding=True,
    )


def export_unet(pipe: StableDiffusionPipeline, out: Path) -> None:
    class UnetWrapper(torch.nn.Module):
        def __init__(self, unet: torch.nn.Module):
            super().__init__()
            self.unet = unet

        def forward(
            self,
            sample: torch.Tensor,
            timestep: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
        ) -> torch.Tensor:
            return self.unet(sample, timestep, encoder_hidden_states).sample

    unet_model = pipe.unet
    unet = UnetWrapper(unet_model).eval()
    sample = torch.randn((1, 4, 64, 64), dtype=torch.float32)
    timestep = torch.tensor(1, dtype=torch.int64)
    encoder_hidden_states = torch.randn((1, 77, unet_model.config.cross_attention_dim), dtype=torch.float32)

    # Export forward(sample, timestep, encoder_hidden_states) -> noise_pred
    torch.onnx.export(
        unet,
        (sample, timestep, encoder_hidden_states),
        out.as_posix(),
        input_names=["sample", "timestep", "encoder_hidden_states"],
        output_names=["out_sample"],
        opset_version=17,
        do_constant_folding=True,
    )


def export_vae_decoder(pipe: StableDiffusionPipeline, out: Path) -> None:
    class VaeDecoderWrapper(torch.nn.Module):
        def __init__(self, vae: torch.nn.Module):
            super().__init__()
            self.vae = vae

        def forward(self, latent_sample: torch.Tensor) -> torch.Tensor:
            return self.vae.decode(latent_sample).sample

    vae = VaeDecoderWrapper(pipe.vae).eval()
    latent = torch.randn((1, 4, 64, 64), dtype=torch.float32)
    torch.onnx.export(
        vae,
        (latent,),
        out.as_posix(),
        input_names=["latent_sample"],
        output_names=["sample"],
        opset_version=17,
        do_constant_folding=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True, help="Local diffusers model folder")
    ap.add_argument("--out", dest="out_dir", required=True, help="Output folder for ONNX files")
    args = ap.parse_args()

    in_dir = Path(args.in_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading pipeline from: {in_dir}")
    pipe = StableDiffusionPipeline.from_pretrained(
        in_dir.as_posix(),
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
        torch_dtype=torch.float32,
    )

    # Keep everything on CPU for deterministic export.
    pipe.to("cpu")

    print("Exporting text encoder…")
    export_text_encoder(pipe, out_dir / "text_encoder.onnx")

    print("Exporting UNet…")
    export_unet(pipe, out_dir / "unet.onnx")

    print("Exporting VAE decoder…")
    export_vae_decoder(pipe, out_dir / "vae_decoder.onnx")

    print(f"Done: {out_dir}")


if __name__ == "__main__":
    main()
