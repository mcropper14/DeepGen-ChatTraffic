"""
ChatTraffic inference script.

Supported samplers
------------------
  ddim   : DDIM stochastic sampler (original, ~200 steps)
  plms   : PLMS sampler            (original, ~200 steps)
  flow   : Flow matching ODE       (new, 8-32 steps typical)

Examples
--------
# Original DDIM (unchanged behaviour)
python scripts/chattraffic.py \
    --ckpt logs/ddpm_run/checkpoints/last.ckpt \
    --sampler ddim --steps 200 \
    --prompt "March 21, 2022, 18:00. Road closure on south second ring road."

# Flow matching with default midpoint integrator
python scripts/chattraffic.py \
    --ckpt logs/fm_run/checkpoints/last.ckpt \
    --sampler flow --steps 16 \
    --prompt "March 21, 2022, 18:00. Road closure on south second ring road."

# Flow matching, Euler integrator, with classifier-free guidance
python scripts/chattraffic.py \
    --ckpt logs/fm_run/checkpoints/last.ckpt \
    --sampler flow --flow_method euler --steps 32 --scale 5.0 \
    --prompt "March 21, 2022, 18:00. Road closure on south second ring road."

# Override config explicitly
python scripts/chattraffic.py \
    --ckpt my_checkpoint.ckpt \
    --config configs/latent-diffusion/traffic_fm.yaml \
    --sampler flow --steps 16 \
    --prompt "..."
"""

import argparse
import os
import sys

import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from tqdm import trange

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.flow_sampler import FlowMatchingSampler


# Config auto-selection: each sampler family has a preferred default config.
_SAMPLER_DEFAULT_CONFIG = {
    "ddim": "configs/latent-diffusion/chattraffic.yaml",
    "plms": "configs/latent-diffusion/chattraffic.yaml",
    "flow": "configs/latent-diffusion/traffic_fm.yaml",
}

# Sensible default step counts per sampler
_SAMPLER_DEFAULT_STEPS = {
    "ddim": 200,
    "plms": 200,
    "flow": 16,
}


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd    = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u  = model.load_state_dict(sd, strict=False)
    if verbose:
        if m:
            print("Missing keys:\n", m)
        if u:
            print("Unexpected keys:\n", u)
    model.cuda()
    model.eval()
    return model


def build_sampler(opt, model):
    """Instantiate and return the requested sampler."""
    if opt.sampler == "ddim":
        return DDIMSampler(model)
    elif opt.sampler == "plms":
        return PLMSSampler(model)
    elif opt.sampler == "flow":
        print(f"FlowMatchingSampler  method={opt.flow_method}  steps={opt.steps}")
        return FlowMatchingSampler(model, method=opt.flow_method)
    else:
        raise ValueError(f"Unknown sampler: {opt.sampler!r}")


def run_sampler(sampler, opt, conditioning, uncond_conditioning):
    """Dispatch to sampler.sample() with the correct kwargs for each type."""
    shape = [3, opt.H, opt.W]

    if opt.sampler in ("ddim", "flow"):
        samples, _ = sampler.sample(
            S                            = opt.steps,
            conditioning                 = conditioning,
            batch_size                   = opt.n_samples,
            shape                        = shape,
            verbose                      = False,
            unconditional_guidance_scale = opt.scale,
            unconditional_conditioning   = uncond_conditioning,
            eta                          = opt.ddim_eta,
        )
    elif opt.sampler == "plms":
        samples, _ = sampler.sample(
            S                            = opt.steps,
            conditioning                 = conditioning,
            batch_size                   = opt.n_samples,
            shape                        = shape,
            verbose                      = False,
            unconditional_guidance_scale = opt.scale,
            unconditional_conditioning   = uncond_conditioning,
        )

    return samples


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ---- Required / core ----
    parser.add_argument(
        "--ckpt", type=str, required=True,
        help="Path to the model checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--prompt", type=str, default="",
        help="Traffic event description to generate from.",
    )

    # ---- Sampler selection ----
    parser.add_argument(
        "--sampler", type=str, default="ddim",
        choices=["ddim", "plms", "flow"],
        help=(
            "Sampler to use. "
            "ddim/plms use the DDPM model (chattraffic.yaml). "
            "flow uses the flow matching model (traffic_fm.yaml). "
            "The config is chosen automatically unless --config is given."
        ),
    )
    parser.add_argument(
        "--flow_method", type=str, default="midpoint",
        choices=["euler", "midpoint"],
        help=(
            "ODE integration method for --sampler flow. "
            "midpoint (RK2) gives better quality; euler is faster. "
            "Ignored for ddim/plms."
        ),
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help=(
            "Number of sampling steps. "
            "Defaults per sampler: ddim/plms=200, flow=16."
        ),
    )

    # ---- Config override ----
    parser.add_argument(
        "--config", type=str, default=None,
        help=(
            "Path to model config YAML. "
            "Auto-selected from --sampler if not set."
        ),
    )

    # ---- Output ----
    parser.add_argument(
        "--outdir", type=str, default="./outputs/chattraffic-samples",
        help="Directory to write .npy and .png outputs.",
    )
    parser.add_argument(
        "--n_samples", type=int, default=1,
        help="Samples to generate per prompt.",
    )
    parser.add_argument(
        "--n_iter", type=int, default=1,
        help="Number of independent sampling iterations.",
    )

    # ---- Spatial dimensions ----
    parser.add_argument("--H", type=int, default=36, help="Grid height.")
    parser.add_argument("--W", type=int, default=36, help="Grid width.")

    # ---- Guidance ----
    parser.add_argument(
        "--scale", type=float, default=1.0,
        help=(
            "Classifier-free guidance scale. "
            "1.0 = no guidance. Try 3-7 to strengthen conditioning."
        ),
    )

    # ---- DDIM-specific ----
    parser.add_argument(
        "--ddim_eta", type=float, default=0.0,
        help="DDIM eta (0.0 = deterministic). Ignored for flow and plms.",
    )

    return parser


if __name__ == "__main__":
    parser = get_parser()
    opt    = parser.parse_args()

    # Resolve config path
    config_path = opt.config or _SAMPLER_DEFAULT_CONFIG[opt.sampler]
    if not os.path.exists(config_path):
        print(f"ERROR: Config not found: {config_path!r}", file=sys.stderr)
        sys.exit(1)
    print(f"Config : {config_path}")

    # Resolve step count
    if opt.steps is None:
        opt.steps = _SAMPLER_DEFAULT_STEPS[opt.sampler]
    print(f"Sampler: {opt.sampler}  steps={opt.steps}"
          + (f"  method={opt.flow_method}" if opt.sampler == "flow" else ""))

    # Load model
    config = OmegaConf.load(config_path)
    model  = load_model_from_config(config, opt.ckpt)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model  = model.to(device)

    # Build sampler
    sampler = build_sampler(opt, model)

    # Prepare output directories
    os.makedirs(opt.outdir, exist_ok=True)
    sample_path = os.path.join(opt.outdir, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    # Inference loop
    with torch.no_grad():
        with model.ema_scope():
            uncond_conditioning = None
            if opt.scale != 1.0:
                uncond_conditioning = model.get_learned_conditioning(
                    opt.n_samples * [""]
                )

            for _ in trange(opt.n_iter, desc="Iterations"):
                conditioning = model.get_learned_conditioning(
                    opt.n_samples * [opt.prompt]
                )

                samples = run_sampler(
                    sampler, opt, conditioning, uncond_conditioning
                )

                # Decode from diffusion latent space to traffic features [0, 1]
                x_samples = model.decode_first_stage(samples)
                x_samples = torch.clamp(x_samples, min=0.0, max=1.0)

                for x_sample in x_samples:
                    x_sample = rearrange(x_sample.cpu().numpy(), "c h w -> h w c")

                    # Save normalized array. To recover physical units:
                    #   speed (km/h)    = x[..., 0] * 5.0
                    #   congestion      = x[..., 1] * 150.0
                    #   passing time (s)= x[..., 2] * 3600.0
                    np.save(os.path.join(sample_path, f"{base_count:04}.npy"), x_sample)
                    Image.fromarray(
                        (x_sample * 255.0).astype(np.uint8)
                    ).save(os.path.join(sample_path, f"{base_count:04}.png"))

                    base_count += 1

    print(f"\nDone. Outputs in: {sample_path}")