import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from MedSyn.src.train_super_res import Unet3D, GaussianDiffusion
from MedSyn.src.utils import *

from einops import rearrange
from lorad.dataloader import (
    FOV_FEATURE_DIM,
    GLOBAL_STD,
    GLOBAL_MEAN,
    TEXT_SEQ_LEN,
)  # noqa: F401

import kornia

def edge_map(volume_tensor):
    """
    Calculates the Sobel edge map for a 5D volume by processing it as a batch of 2D slices.
    """
    # Get the original 5D shape
    # B: Batch, C: Channel, D: Depth, H: Height, W: Width
    B, C, D, H, W = volume_tensor.shape

    # 1. Reshape the volume to look like a batch of 2D images
    # We merge the Batch and Depth dimensions together
    # New shape: (B * D, C, H, W)
    slices_2d = volume_tensor.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)

    # 2. Apply the 2D Sobel filter
    # The output will be a 4D tensor of edge maps
    edge_maps_2d = kornia.filters.sobel(slices_2d)

    # 3. Reshape the edge maps back to the original 5D volume format
    # First, view it as (B, D, C, H, W)
    # Then, permute the dimensions back to (B, C, D, H, W)
    final_edge_volume = edge_maps_2d.view(B, D, C, H, W).permute(0, 2, 1, 3, 4)

    return final_edge_volume

def edge_loss(pred, target):
    """
    Calculates the L1 loss between the Sobel edge maps of the prediction and target.
    """
    pred_edge = edge_map(pred)
    target_edge = edge_map(target)

    # Use L1 loss for robustness
    loss = torch.nn.functional.l1_loss(pred_edge, target_edge)
    return loss


def delta_along(volume_tensor, dim):
    """First-order forward difference along one spatial dimension."""
    if volume_tensor.ndim != 5:
        raise ValueError(
            "Expected a 5D volume shaped (B, C, D, H, W), "
            f"got shape {tuple(volume_tensor.shape)}"
        )
    if dim not in (2, 3, 4):
        raise ValueError(f"Spatial difference dim must be 2, 3, or 4; got {dim}")
    if volume_tensor.size(dim) < 2:
        raise ValueError(
            f"Cannot compute a first difference along dim {dim} "
            f"with size {volume_tensor.size(dim)}"
        )
    return (
        volume_tensor.narrow(dim, 1, volume_tensor.size(dim) - 1)
        - volume_tensor.narrow(dim, 0, volume_tensor.size(dim) - 1)
    )


def interslice_consistency_loss(pred, target):
    """Tri-planar first-difference L1 loss for (B, C, D, H, W) volumes."""
    if pred.shape != target.shape:
        raise ValueError(
            f"Prediction and target shapes must match, got {pred.shape} and {target.shape}"
        )
    return sum(
        F.l1_loss(delta_along(pred, dim), delta_along(target, dim))
        for dim in (2, 3, 4)
    ) / 3.0


class LatentOnlyUNet3D(Unet3D):
    """
    UNet3D variant for latent-only training (no low-res input concatenation).

    Adds:
    - learnable null text embedding for classifier-free guidance (CFG)
    - FOV MLP token prepended to the text cross-attention sequence
    - learnable null FOV token (used at inference when FOV is unknown)
    """

    def __init__(
        self,
        *args,
        fov_dim: int = FOV_FEATURE_DIM,
        text_seq_len: int = TEXT_SEQ_LEN,
        **kwargs,
    ):
        cond_dim = kwargs.get("cond_dim")
        if cond_dim is None:
            raise ValueError("LatentOnlyUNet3D requires cond_dim in unet_kwargs")

        super().__init__(*args, **kwargs)
        self.fov_dim = int(fov_dim)
        self.text_seq_len = int(text_seq_len)

        init_dim = self.init_conv0.out_channels
        init_kernel_size = self.init_conv0.kernel_size[0]
        init_padding = init_kernel_size // 2
        self.init_conv0 = nn.Conv3d(
            self.channels,
            init_dim,
            (init_kernel_size, init_kernel_size, init_kernel_size),
            padding=(init_padding, init_padding, init_padding),
        )

        if self.has_cond:
            self.null_cond_emb = nn.Parameter(torch.randn(1, self.text_seq_len, cond_dim))
            self.fov_mlp = nn.Sequential(
                nn.Linear(self.fov_dim, cond_dim),
                nn.GELU(),
                nn.Linear(cond_dim, cond_dim),
            )
            self.null_fov_token = nn.Parameter(torch.randn(1, 1, cond_dim))
        else:
            self.null_cond_emb = None
            self.fov_mlp = None
            self.null_fov_token = None

    def exists(self, val):
        return val is not None

    def _prepare_cross_attn_cond(
        self,
        cond,
        fov=None,
        null_cond_prob: float = 0.0,
        null_fov_prob: float = 0.0,
    ):
        if not self.has_cond:
            return cond
        if cond is None:
            raise ValueError("cond must be provided when has_cond=True")

        batch = cond.shape[0]
        device = cond.device

        if null_cond_prob > 0.0:
            mask = prob_mask_like((batch,), null_cond_prob, device)
            null_text = self.null_cond_emb.expand(batch, -1, -1)
            cond = torch.where(rearrange(mask, "b -> b 1 1"), null_text, cond)

        if fov is None:
            fov_token = self.null_fov_token.expand(batch, -1, -1)
        else:
            if fov.shape[-1] != self.fov_dim:
                raise ValueError(
                    f"Expected fov shape (B, {self.fov_dim}), got {tuple(fov.shape)}"
                )
            fov_token = self.fov_mlp(fov.float()).unsqueeze(1)
            if null_fov_prob > 0.0:
                mask = prob_mask_like((batch,), null_fov_prob, device)
                null_fov = self.null_fov_token.expand(batch, -1, -1)
                fov_token = torch.where(rearrange(mask, "b -> b 1 1"), null_fov, fov_token)

        return torch.cat([fov_token, cond], dim=1)

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale=1.0,
        fov=None,
        **kwargs,
    ):
        logits = self.forward(
            *args,
            cond=kwargs.get("cond"),
            fov=fov,
            null_cond_prob=0.0,
            null_fov_prob=0.0,
            **{k: v for k, v in kwargs.items() if k not in {"cond", "fov"}},
        )
        if cond_scale == 1.0 or not self.has_cond:
            return logits

        null_logits = self.forward(
            *args,
            cond=kwargs.get("cond"),
            fov=fov,
            null_cond_prob=1.0,
            null_fov_prob=1.0,
            **{k: v for k, v in kwargs.items() if k not in {"cond", "fov"}},
        )
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x,
        time,
        indexes=None,
        cond=None,
        fov=None,
        null_cond_prob=0.0,
        null_fov_prob=0.0,
        focus_present_mask=None,
        prob_focus_present=0.0,
    ):
        assert not (self.has_cond and not self.exists(cond)), (
            "cond must be passed in if cond_dim specified"
        )

        if self.has_cond:
            cond = self._prepare_cross_attn_cond(
                cond,
                fov=fov,
                null_cond_prob=null_cond_prob,
                null_fov_prob=null_fov_prob,
            )

        x = self.init_conv0(x)

        r = x.clone()
        t = self.time_mlp(time) if self.exists(self.time_mlp) else None

        h = []

        for idx, (block1, block2, spatial_attn, cross_attn, temporal_conv, downsample) in enumerate(self.downs):
            x = block1(x, t)
            x = block2(x, t)
            h.append(x)
            x = downsample(x)
            x = spatial_attn(x)
            x = cross_attn(x, kv=cond)
            x = temporal_conv(x, t)

        x = self.mid_block1(x, t)
        x = self.mid_spatial_attn(x)
        x = self.mid_temporal_conv(x, t)
        x = self.mid_block2(x, t)

        for block1, block2, spatial_attn, cross_attn, temporal_conv, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = upsample(x)
            x = block2(x, t)
            x = spatial_attn(x)
            x = cross_attn(x, kv=cond)
            x = temporal_conv(x, t)

        x = torch.cat((x, r), dim=1)
        return self.final_conv0(x)

class LatentDiffusion(GaussianDiffusion):
    def __init__(self, unet_kwargs, diffusion_kwargs):
        # prediction_type: "v" (default) or "x0". Prefer v-pred for stable sampling.
        self.prediction_type = diffusion_kwargs.pop("prediction_type", "v")
        self.null_cond_prob = float(diffusion_kwargs.pop("null_cond_prob", 0.1))
        self.null_fov_prob = float(diffusion_kwargs.pop("null_fov_prob", 0.15))
        unet = LatentOnlyUNet3D(**unet_kwargs)
        super().__init__(denoise_fn=unet, **diffusion_kwargs)
        self.log_var_l2 = nn.Parameter(torch.tensor(0.0))
        self.log_var_edge = nn.Parameter(torch.tensor(0.0))
        self.log_var_isc = nn.Parameter(torch.tensor(0.0))
        self.last_loss_components = {}

    def _model_out_to_x0(self, model_out, x_t, t_cont):
        """Convert network output to x0 given continuous t in [0, 1]."""
        if self.prediction_type == "v":
            return get_x0_from_v(model_out, x_t, t_cont)
        if self.prediction_type == "x0":
            return model_out
        if self.prediction_type == "eps":
            return get_x0_x_t(model_out, x_t, t_cont)
        raise ValueError(f"Unknown prediction_type={self.prediction_type!r}")

    def p_losses(
        self,
        x_start,
        t,
        indexes=None,
        cond=None,
        fov=None,
        noise=None,
        null_cond_prob=None,
        null_fov_prob=None,
        **kwargs,
    ):
        b, c, f, h, w, device = *x_start.shape, x_start.device
        x_noisy, eps = get_z_t(x_start, t)

        model_out = self.denoise_fn(
            x_noisy,
            t * self.num_timesteps,
            indexes=indexes,
            cond=cond,
            fov=fov,
            null_cond_prob=self.null_cond_prob if null_cond_prob is None else null_cond_prob,
            null_fov_prob=self.null_fov_prob if null_fov_prob is None else null_fov_prob,
            **kwargs,
        )

        if self.prediction_type == "v":
            target = get_v_from_x0_eps(x_start, eps, t)
            x0_pred = get_x0_from_v(model_out, x_noisy, t)
        elif self.prediction_type == "x0":
            target = x_start
            x0_pred = model_out
        elif self.prediction_type == "eps":
            target = eps
            x0_pred = get_x0_x_t(model_out, x_noisy, t)
        else:
            raise ValueError(f"Unknown prediction_type={self.prediction_type!r}")

        if self.loss_type == "l1":
            loss = F.l1_loss(target, model_out)
        elif self.loss_type == "l2":
            loss = F.mse_loss(target, model_out)
        elif self.loss_type == "edge+l2":
            # Primary objective on v/eps/x0; structural edge always on x0 space
            l2 = F.mse_loss(target, model_out, reduction="mean")
            edge = edge_loss(x0_pred, x_start)
            self.last_loss_components = {
                "l2": l2.detach(),
                "edge": edge.detach(),
            }
            loss = (
                torch.exp(-self.log_var_l2) * l2
                + torch.exp(-self.log_var_edge) * edge
                + self.log_var_l2
                + self.log_var_edge
            )
            return loss
        elif self.loss_type == "edge+l2+isc":
            # LiFT-C-style spatial consistency on x0, including the depth axis.
            l2 = F.mse_loss(target, model_out, reduction="mean")
            edge = edge_loss(x0_pred, x_start)
            isc = interslice_consistency_loss(x0_pred, x_start)
            self.last_loss_components = {
                "l2": l2.detach(),
                "edge": edge.detach(),
                "isc": isc.detach(),
            }
            loss = (
                torch.exp(-self.log_var_l2) * l2
                + torch.exp(-self.log_var_edge) * edge
                + torch.exp(-self.log_var_isc) * isc
                + self.log_var_l2
                + self.log_var_edge
                + self.log_var_isc
            )
            return loss
        else:
            raise NotImplementedError(f"Unknown loss_type={self.loss_type!r}")
        return loss

    def forward(self, x, *args, **kwargs):
        b, device, img_size = x.shape[0], x.device, self.image_size
        t = torch.rand((b,), device=device).float()
        return self.p_losses(x, t, *args, **kwargs)

    def p_mean_variance(
        self,
        x,
        t,
        clip_denoised: bool,
        indexes=None,
        cond=None,
        fov=None,
        cond_scale=1.0,
    ):
        model_out = self.denoise_fn.forward_with_cond_scale(
            x,
            t,
            indexes=indexes,
            cond=cond,
            fov=fov,
            cond_scale=cond_scale,
        )
        t_cont = t.float() / float(self.num_timesteps)
        x_recon = self._model_out_to_x0(model_out, x, t_cont)

        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x_recon, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim=-1
                )
                s.clamp_(min=1.)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))
            x_recon = x_recon.clamp(-s, s) / s

        model_mean, posterior_variance = get_z_t_via_z_tp1(
            x_recon,
            x,
            (t - 1) * 1.0 / (self.num_timesteps - 1.0),
            (t * 1.0) / (self.num_timesteps - 1.0),
        )
        return model_mean, posterior_variance

    @torch.inference_mode()
    def p_sample(
        self,
        x,
        t,
        indexes=None,
        cond=None,
        fov=None,
        cond_scale=1.0,
        clip_denoised=True,
    ):
        device = x.device
        B, _, F, _, _ = x.shape

        if not torch.is_tensor(t):
            t = torch.tensor([t], device=device, dtype=torch.long)
        t = t.to(device).long().view(-1)
        if t.numel() == 1 and B > 1:
            t = t.expand(B)

        model_mean, model_variance = self.p_mean_variance(
            x=x,
            t=t,
            indexes=indexes,
            clip_denoised=clip_denoised,
            cond=cond,
            fov=fov,
            cond_scale=cond_scale,
        )

        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(B, 1, 1, 1, 1).expand(B, 1, F, 1, 1)
        return model_mean + nonzero_mask * torch.sqrt(model_variance) * noise

    @torch.inference_mode()
    def p_sample_ddim(
        self,
        x,
        slice_id,
        cond,
        t,
        t_minus,
        clip_denoised: bool,
        index,
        fov=None,
        repeat_noise=False,
        use_original_steps=False,
        quantize_denoised=False,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
    ):
        model_out = self.denoise_fn.forward_with_cond_scale(
            x,
            t,
            indexes=slice_id,
            cond=cond,
            fov=fov,
            cond_scale=unconditional_guidance_scale,
        )
        t_cont = t.float() / float(self.num_timesteps)
        x_recon = self._model_out_to_x0(model_out, x, t_cont)

        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x_recon, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim=-1
                )
                s.clamp_(min=1.)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))
            x_recon = x_recon.clamp(-s, s) / s

        if t[0] < int(self.num_timesteps / self.ddim_timesteps):
            x = x_recon
        else:
            t_minus = torch.clip(t_minus, min=0.0)
            x = ddim_sample(
                x_recon,
                x,
                (t_minus * 1.0) / (self.num_timesteps),
                (t * 1.0) / (self.num_timesteps),
            )
        return x

    @torch.inference_mode()
    def p_sample_loop(
        self,
        shape,
        cond=None,
        fov=None,
        cond_scale=1.0,
        use_ddim=True,
        clip_denoised=False,
        device="cpu",
    ):
        bsz = shape[0]

        if use_ddim:
            time_steps = range(0, self.num_timesteps + 1, int(self.num_timesteps / self.ddim_timesteps))
        else:
            time_steps = range(0, self.num_timesteps)

        img = torch.randn(shape, device=device)
        indexes = []
        for b in range(bsz):
            index = np.arange(self.num_frames)
            indexes.append(torch.from_numpy(index))
        indexes = torch.stack(indexes, dim=0).long().to(device)
        for i, t in enumerate(reversed(time_steps)):
            time = torch.full((bsz,), t, device=device, dtype=torch.float32)

            if use_ddim:
                time_minus = time - int(self.num_timesteps / self.ddim_timesteps)
                img = self.p_sample_ddim(
                    x=img,
                    slice_id=indexes,
                    cond=cond,
                    fov=fov,
                    t=time,
                    t_minus=time_minus,
                    clip_denoised=clip_denoised,
                    index=len(time_steps) - i - 1,
                    unconditional_guidance_scale=cond_scale,
                )
            else:
                img = self.p_sample(
                    img,
                    time,
                    indexes=indexes,
                    cond=cond,
                    fov=fov,
                    cond_scale=cond_scale,
                    clip_denoised=clip_denoised,
                )

        return img
