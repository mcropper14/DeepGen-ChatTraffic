"""
Flow Matching variant of LatentDiffusion.

Instead of predicting noise epsilon at each diffusion step, the model learns
a velocity field v(x_t, t, c) that moves samples along straight linear
trajectories from data (t=0) to noise (t=1):

    Forward process:   x_t = (1 - t) * x_0  +  t * epsilon
    Velocity target:   v   = epsilon - x_0
    Training loss:     MSE( model(x_t, t, c),  v )
    Sampling (ODE):    x_{t - dt} = x_t  -  dt * v(x_t, t, c)

Only p_losses and sample_log are overridden. Everything else — the UNet/GCN,
the BERT encoder, the training loop, checkpointing, and the autoencoder
encode/decode path — is inherited unchanged from LatentDiffusion.

Usage
-----
Training:
    python main.py --base configs/latent-diffusion/traffic_fm.yaml -t --gpus 0,

Inference (via chattraffic.py):
    python scripts/chattraffic.py --sampler flow --ckpt path/to/ckpt.ckpt \
        --prompt "March 07, 2022, 18:00. Road closure on south second ring road."
"""

import torch
import torch.nn.functional as F

from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import default


class LatentFlowMatching(LatentDiffusion):
    """
    Drop-in replacement for LatentDiffusion that uses a flow matching objective.

    Extra constructor parameter
    ---------------------------
    fm_method : str
        ODE integration method used during sampling. One of:
        - "euler"    : first-order, cheapest, ~16 steps sufficient
        - "midpoint" : second-order (RK2), better quality at same step count
        Default: "midpoint"
    """

    def __init__(self, *args, fm_method: str = "midpoint", **kwargs):
        valid = ("euler", "midpoint")
        if fm_method not in valid:
            raise ValueError(f"fm_method must be one of {valid}, got '{fm_method}'")
        super().__init__(*args, **kwargs)
        self.fm_method = fm_method

    # ------------------------------------------------------------------
    # Training objective  (only override in the whole class)
    # ------------------------------------------------------------------

    def p_losses(self, x_start, cond, t, noise=None):
        """
        Flow matching loss: MSE between predicted and target velocity.

        Parameters
        ----------
        x_start : Tensor (B, C, H, W)
            Clean traffic data x_0 in diffusion latent space.
        cond : Tensor or list
            Encoded text conditioning from get_learned_conditioning.
        t : LongTensor (B,)
            Integer timesteps sampled by the parent's forward(), in
            [0, num_timesteps). We convert to continuous [0, 1] here.
        noise : Tensor or None
            Pre-sampled Gaussian noise; drawn fresh if None.
        """
        noise = default(noise, lambda: torch.randn_like(x_start))

        # Convert integer t to continuous t_cont in [0, 1].
        # The UNet still receives the integer t for its sinusoidal embedding —
        # only the data interpolation uses the continuous version.
        t_cont = t.float() / self.num_timesteps          # (B,)
        t_b    = t_cont[:, None, None, None]              # broadcast to (B, 1, 1, 1)

        # Linearly interpolate between clean data and noise
        x_t = (1.0 - t_b) * x_start + t_b * noise

        # Target velocity: direction from data toward noise
        v_target = noise - x_start

        # UNet predicts the velocity field (receives integer t as usual)
        v_pred = self.apply_model(x_t, t, cond)

        loss = F.mse_loss(v_pred, v_target)

        prefix = "train" if self.training else "val"
        loss_dict = {
            f"{prefix}/loss_simple": loss,
            f"{prefix}/loss":        loss,
        }
        return loss, loss_dict

    # ------------------------------------------------------------------
    # Training-time visualisation
    # ------------------------------------------------------------------

    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        """
        Override sample_log to use FlowMatchingSampler instead of DDIM/PLMS.

        Called by the parent's log_images() during training callbacks.
        The `ddim` and `ddim_steps` arguments are accepted for API compatibility
        but ddim is ignored — we always use the flow ODE.
        """
        # Import here to avoid circular imports at module load time
        from ldm.models.diffusion.flow_sampler import FlowMatchingSampler

        sampler = FlowMatchingSampler(self, method=self.fm_method)
        shape   = (self.channels, self.image_size, self.image_size)

        samples, intermediates = sampler.sample(
            S           = ddim_steps,
            batch_size  = batch_size,
            shape       = shape,
            conditioning= cond,
            verbose     = False,
            **kwargs,
        )
        return samples, intermediates