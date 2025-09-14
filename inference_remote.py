# This source code is licensed under the license found in the LICENSE file
# in the root directory of this source tree.
"""Simple inference script for generating images from a trained ZigMa model.

The script loads a checkpoint, samples images and computes common
reconstruction metrics such as PSNR and SSIM. Results are written to the
specified output directory.
"""
import os
from einops import rearrange
import torch
import hydra
from omegaconf import OmegaConf
from tqdm import tqdm
from PIL import Image

from datasets.wds_dataloader import WebDataModuleFromConfig
from utils.train_utils import get_model, requires_grad
from train_acc import has_text, is_video
from transport import create_transport, Sampler
from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionPipeline
from utils.metrics import calculate_psnr, calculate_ssim, calculate_sam


@hydra.main(config_path="config", config_name="default", version_base=None)
def main(args):
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    assert args.ckpt is not None, "Must specify a checkpoint to sample from"
    model, in_channels, input_size = get_model(args, device)
    state_dict = torch.load(args.ckpt, map_location="cpu")
    state_dict = state_dict.get("ema", state_dict)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device).eval()
    requires_grad(model, False)

    datamod = WebDataModuleFromConfig(**args.data)
    loader = datamod.train_dataloader()

    transport = create_transport(
        args.train.path_type,
        args.train.prediction,
        args.train.loss_weight,
        args.train.train_eps,
        args.train.sample_eps,
    )
    sampler = Sampler(transport)
    if args.sample_mode == "ODE":
        sample_fn = sampler.sample_ode(
            sampling_method=args.ode.sampling_method,
            num_steps=args.ode.num_sampling_steps,
            atol=args.ode.atol,
            rtol=args.ode.rtol,
            reverse=args.ode.reverse,
        )
    elif args.sample_mode == "SDE":
        sample_fn = sampler.sample_sde(
            sampling_method=args.sde.sampling_method,
            diffusion_form=args.sde.diffusion_form,
            diffusion_norm=args.sde.diffusion_norm,
            last_step=args.sde.last_step,
            last_step_size=args.sde.last_step_size,
            num_steps=args.num_sampling_steps,
        )
    else:
        raise ValueError(f"Unknown sample_mode: {args.sample_mode}")

    if args.is_latent:
        if has_text(args):
            image_model_id = "runwayml/stable-diffusion-v1-5"
            vae = StableDiffusionPipeline.from_pretrained(
                image_model_id, local_files_only=False
            ).vae.to(device)
        else:
            vae = AutoencoderKL.from_pretrained(
                f"stabilityai/sd-vae-ft-{args.vae}"
            ).to(device)
        vae.eval()
    else:
        vae = None

    os.makedirs(args.out_dir, exist_ok=True)

    psnr_vals, ssim_vals, sam_vals = [], [], []

    to_uint8 = lambda x: torch.clamp(127.5 * x + 128.0, 0, 255).to(torch.uint8)

    for idx, data in enumerate(tqdm(loader, desc="inference")):
        if is_video(args):
            raise NotImplementedError("video inference is not supported")
        if args.use_latent:
            inputs = data["img_feature"].to(device)
        else:
            inputs = data["image"].to(device)
        b = inputs.shape[0]
        z = torch.randn(b, in_channels, input_size, input_size, device=device)

        model_kwargs = {}
        if has_text(args):
            y = data["caption_feature"][:, 0].to(device)
            model_kwargs["y"] = y
        elif args.data.num_classes > 0:
            model_kwargs["y"] = data["label"].to(device)

        with torch.no_grad():
            samples = sample_fn(z, model.forward, **model_kwargs)[-1]
            if vae is not None:
                samples = vae.decode(samples / 0.18215).sample
                targets = vae.decode(inputs).sample
            else:
                targets = inputs

        sample_img = to_uint8(samples)
        target_img = to_uint8(targets)

        sample_norm = sample_img.float() / 255.0
        target_norm = target_img.float() / 255.0
        psnr_vals.append(calculate_psnr(sample_norm, target_norm).item())
        ssim_vals.append(calculate_ssim(sample_norm, target_norm).item())
        sam_vals.append(calculate_sam(sample_norm, target_norm).item())

        for b_idx, img in enumerate(sample_img):
            Image.fromarray(rearrange(img, "c h w -> h w c").cpu().numpy()).save(
                os.path.join(args.out_dir, f"{idx*loader.batch_size + b_idx:06d}.png")
            )

    print(
        {
            "psnr": sum(psnr_vals) / len(psnr_vals),
            "ssim": sum(ssim_vals) / len(ssim_vals),
            "sam": sum(sam_vals) / len(sam_vals),
        }
    )


if __name__ == "__main__":
    main()
