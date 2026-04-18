"""
ODE sampler for flow matching models.

Integrates the learned velocity field v(x_t, t, c) from t=1 (pure noise)
to t=0 (clean data) using either Euler or midpoint (RK2) integration.

The public interface mirrors DDIMSampler.sample() so that chattraffic.py
can switch between samplers without restructuring its inference loop.

Supported methods
-----------------
euler    : x_{t-dt} = x_t - dt * v(x_t, t, c)
           Cheapest. Works well with ~16-32 steps.

midpoint : Evaluate v at t, take a half-step, evaluate v again at t_mid,
           then use that midpoint velocity for the full step (RK2).
           Better quality at the same step count; roughly 2× the FLOPs of euler.
           Recommended default. Works well with ~8-16 steps.
"""

import torch
import numpy as np
from tqdm import tqdm


class FlowMatchingSampler:
    """
    Drop-in replacement for DDIMSampler with a flow-matching ODE backend.

    Parameters
    ----------
    model   : LatentFlowMatching (or any model with .apply_model and .num_timesteps)
    method  : "euler" | "midpoint"
    """

    METHODS = ("euler", "midpoint")

    def __init__(self, model, method: str = "midpoint"):
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}, got '{method}'")
        self.model  = model
        self.method = method

    # ------------------------------------------------------------------
    # Public API  (mirrors DDIMSampler.sample)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        S,                                  # number of ODE integration steps
        batch_size,
        shape,                              # (C, H, W)
        conditioning         = None,
        verbose              = True,
        unconditional_guidance_scale = 1.0,
        unconditional_conditioning   = None,
        x_T                  = None,        # optional custom starting noise
        log_every_t          = None,        # how often to store intermediates
        # Accept (and silently ignore) DDIM-specific kwargs for API parity
        callback             = None,
        img_callback         = None,
        eta                  = None,
        temperature          = None,
        noise_dropout        = None,
        score_corrector      = None,
        corrector_kwargs     = None,
        quantize_x0          = None,
        mask                 = None,
        x0                   = None,
        **kwargs,
    ):
        """
        Run the flow-matching ODE from t=1 (noise) to t=0 (data).

        Returns
        -------
        samples      : Tensor (batch_size, C, H, W)  — final denoised output
        intermediates: dict with "x_inter" and "pred_x0" lists
        """
        device = next(self.model.parameters()).device
        C, H, W = shape

        # Start from pure Gaussian noise at t = 1
        x = torch.randn((batch_size, C, H, W), device=device) if x_T is None else x_T

        # Time grid: integrate from t=1 down to t=0 in S equal steps
        # t_steps[i]   = current t  (starts at 1, ends just above 0)
        # t_steps[i+1] = next    t  (one step later, closer to 0)
        t_steps = torch.linspace(1.0, 0.0, S + 1, device=device)
        dt      = t_steps[:-1] - t_steps[1:]   # all positive, shape (S,)

        log_every = log_every_t or max(1, S // 10)
        intermediates = {"x_inter": [x.clone()], "pred_x0": []}

        desc = f"Flow ODE ({self.method}, {S} steps)"
        iterator = tqdm(range(S), desc=desc, disable=not verbose)

        for i in iterator:
            t_cur  = t_steps[i]       # scalar tensor, current t
            t_next = t_steps[i + 1]   # scalar tensor, next t
            step   = dt[i]            # scalar tensor, dt (positive)

            if self.method == "euler":
                x, x0_pred = self._euler_step(
                    x, t_cur, step, conditioning,
                    unconditional_guidance_scale, unconditional_conditioning,
                    batch_size,
                )
            else:  # midpoint / RK2
                x, x0_pred = self._midpoint_step(
                    x, t_cur, t_next, step, conditioning,
                    unconditional_guidance_scale, unconditional_conditioning,
                    batch_size,
                )

            if callback is not None:
                callback(i)
            if img_callback is not None:
                img_callback(x0_pred, i)

            if i % log_every == 0 or i == S - 1:
                intermediates["x_inter"].append(x.clone())
                intermediates["pred_x0"].append(x0_pred.clone())

        return x, intermediates

    # ------------------------------------------------------------------
    # Integration steps
    # ------------------------------------------------------------------

    def _euler_step(self, x, t_cur, dt, cond, scale, uncond, bs):
        """
        x_{t - dt} = x_t - dt * v(x_t, t)
        """
        v   = self._get_velocity(x, t_cur, cond, scale, uncond, bs)
        x0  = self._x0_from_velocity(x, v, t_cur)
        x   = x - dt * v
        return x, x0

    def _midpoint_step(self, x, t_cur, t_next, dt, cond, scale, uncond, bs):
        """
        Evaluate at t_cur, half-step to midpoint, evaluate at midpoint,
        full step using midpoint velocity (Heun / RK2).

        x_mid  = x_t  - (dt/2) * v(x_t,   t_cur)
        x_{t'} = x_t  - dt     * v(x_mid, t_mid)
        """
        # First evaluation at t_cur
        v1    = self._get_velocity(x, t_cur, cond, scale, uncond, bs)
        # Half Euler step to approximate midpoint
        t_mid = (t_cur + t_next) / 2.0
        x_mid = x - (dt / 2.0) * v1
        # Second evaluation at midpoint
        v2    = self._get_velocity(x_mid, t_mid, cond, scale, uncond, bs)
        # Predicted x_0 using midpoint velocity (more accurate than v1)
        x0    = self._x0_from_velocity(x, v2, t_cur)
        # Full step using midpoint velocity
        x     = x - dt * v2
        return x, x0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_velocity(self, x, t_cont, cond, guidance_scale, uncond_cond, batch_size):
        """
        Query the model for velocity at continuous time t_cont ∈ [0, 1].

        The UNet expects integer timesteps for its sinusoidal embedding, so we
        map t_cont → integer via _cont_to_int before calling apply_model.
        Applies classifier-free guidance if guidance_scale != 1.
        """
        t_int = self._cont_to_int(t_cont, batch_size, x.device)

        if uncond_cond is None or guidance_scale == 1.0:
            return self.model.apply_model(x, t_int, cond)

        # Classifier-free guidance: two forward passes, then interpolate
        v_cond   = self.model.apply_model(x, t_int, cond)
        v_uncond = self.model.apply_model(x, t_int, uncond_cond)
        return v_uncond + guidance_scale * (v_cond - v_uncond)

    def _cont_to_int(self, t_cont, batch_size: int, device) -> torch.Tensor:
        """
        Map a scalar continuous t ∈ [0, 1] to a batch of integer timesteps
        in [0, num_timesteps), compatible with the UNet's timestep embedding.
        """
        t_val = t_cont.item() if torch.is_tensor(t_cont) else float(t_cont)
        t_int = int(t_val * self.model.num_timesteps)
        t_int = max(0, min(t_int, self.model.num_timesteps - 1))
        return torch.full((batch_size,), t_int, dtype=torch.long, device=device)

    @staticmethod
    def _x0_from_velocity(x_t, v, t_cont):
        """
        Recover the predicted x_0 from the current point and velocity.

        Derivation: x_t = x_0 + t * v  →  x_0 = x_t - t * v
        (since v = epsilon - x_0 and x_t = (1-t)*x_0 + t*epsilon = x_0 + t*v)
        """
        t = t_cont.item() if torch.is_tensor(t_cont) else float(t_cont)
        return x_t - t * v