import os
import sys
from datetime import datetime
from pathlib import Path

# MedSyn must be on PYTHONPATH (see README).
sys.path.append("./MedSyn/src/")
sys.path.append(".")

import torch
from torch.optim import AdamW
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import TQDMProgressBar
from transformers import AutoTokenizer, AutoModel
from train_super_res import EMA
from lorad.dataloader import (
    DEFAULT_STATS_PATH,
    denormalize_latent,
    get_dataloader,
)
from lorad.diffusion_model import LatentDiffusion
import imageio
import numpy as np
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from copy import deepcopy
from PIL import Image, ImageDraw, ImageFont


# ----------------------------
# GIF Visualization Utilities
# ----------------------------
LATENT_VIZ_UPSCALE = 4  # nearest-neighbor; latents are 96×96


def _as_dhw_float(volume) -> np.ndarray:
    """Torch/np volume → float32 (D, H, W)."""
    if isinstance(volume, torch.Tensor):
        volume = volume.detach().float().cpu().numpy()
    else:
        volume = np.asarray(volume, dtype=np.float32)
    volume = np.squeeze(volume)
    if volume.ndim == 4:  # (C, D, H, W) → first channel
        volume = volume[0]
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D latent volume after squeeze, got shape {volume.shape}")
    return volume.astype(np.float32, copy=False)


def _display_window_from_stats(stats_path: str, n_std: float = 3.0) -> tuple[float, float]:
    """Fixed denormalized window from bank mean/std (stable across steps/samples)."""
    from lorad.dataloader import load_latent_stats

    mean, std = load_latent_stats(stats_path)
    return float(mean - n_std * std), float(mean + n_std * std)


def _window_to_u8(
    volume: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
    percentile=(1.0, 99.0),
) -> tuple[np.ndarray, float, float]:
    """
    Map float volume to uint8 with a stable display window.

    Prefer shared vmin/vmax. Otherwise use robust percentiles — not raw
    min/max, which outliers crush into posterization.
    """
    v = volume.astype(np.float32, copy=False)
    if vmin is None or vmax is None:
        lo, hi = np.percentile(v, list(percentile))
        vmin = float(lo) if vmin is None else float(vmin)
        vmax = float(hi) if vmax is None else float(vmax)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        raise ValueError(f"Invalid display window vmin={vmin}, vmax={vmax}")
    scaled = (np.clip(v, vmin, vmax) - vmin) / (vmax - vmin)
    return (scaled * 255.0).astype(np.uint8), float(vmin), float(vmax)


def _upscale_gray(u8_hw: np.ndarray, scale: int = LATENT_VIZ_UPSCALE) -> np.ndarray:
    if scale <= 1:
        return u8_hw
    return np.asarray(
        Image.fromarray(u8_hw, mode="L").resize(
            (u8_hw.shape[1] * scale, u8_hw.shape[0] * scale),
            resample=Image.BILINEAR,
        )
    )


@rank_zero_only
def volume_to_axial_gif(
    volume,
    save_path,
    batch_text="",
    vmin: float | None = None,
    vmax: float | None = None,
    percentile=(1.0, 99.0),
    upscale: int = LATENT_VIZ_UPSCALE,
    per_slice_window: bool = False,
    slice_vmins: np.ndarray | None = None,
    slice_vmaxs: np.ndarray | None = None,
    slice_indices: np.ndarray | None = None,
):
    """
    Convert a 3D latent volume into an axial GIF.

    Grayscale + no-dither palette (GIF dither was causing checkerboard).
    Optional ``slice_indices`` skips empty pad slices.
    """
    vol = _as_dhw_float(volume)
    d = vol.shape[0]
    if slice_indices is None:
        slice_indices = np.arange(d)
    else:
        slice_indices = np.asarray(slice_indices, dtype=np.int64)
        if slice_indices.size == 0:
            raise ValueError("slice_indices is empty — no slices to visualize")

    if slice_vmins is not None and slice_vmaxs is not None:
        if len(slice_vmins) != d or len(slice_vmaxs) != d:
            raise ValueError("slice_vmins/vmaxs must match full depth")
        vmins = np.asarray(slice_vmins, dtype=np.float64)
        vmaxs = np.asarray(slice_vmaxs, dtype=np.float64)
    elif per_slice_window:
        vmins = np.empty(d, dtype=np.float64)
        vmaxs = np.empty(d, dtype=np.float64)
        for i in range(d):
            lo, hi = np.percentile(vol[i], list(percentile))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = float(vol[i].min()), float(vol[i].max())
            if hi <= lo:
                hi = lo + 1e-6
            vmins[i], vmaxs[i] = lo, hi
    else:
        _, vmin, vmax = _window_to_u8(vol, vmin=vmin, vmax=vmax, percentile=percentile)
        vmins = np.full(d, vmin, dtype=np.float64)
        vmaxs = np.full(d, vmax, dtype=np.float64)

    frames = []
    header_h = 18 if batch_text else 0
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11
        )
    except Exception:
        font = ImageFont.load_default()

    for i in slice_indices:
        i = int(i)
        lo, hi = float(vmins[i]), float(vmaxs[i])
        body = ((np.clip(vol[i], lo, hi) - lo) / (hi - lo) * 255.0).astype(np.uint8)
        body = _upscale_gray(body, scale=upscale)
        if header_h:
            canvas = Image.new("L", (body.shape[1], body.shape[0] + header_h), color=0)
            canvas.paste(Image.fromarray(body, mode="L"), (0, header_h))
            draw = ImageDraw.Draw(canvas)
            label = f"{batch_text[:70]}  z={i}/{d - 1}"
            draw.text((2, 2), label, fill=220, font=font)
            frames.append(np.asarray(canvas))
        else:
            frames.append(body)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    _save_gray_gif(save_path, frames)
    return float(np.min(vmins[slice_indices])), float(np.max(vmaxs[slice_indices])), vmins, vmaxs


def _save_gray_gif(save_path: str, frames: list[np.ndarray]) -> None:
    """Write grayscale GIF without dithering (Pillow dither → checkerboard)."""
    if not frames:
        raise ValueError(f"No frames to save at {save_path}")
    pal_frames = []
    for fr in frames:
        img = Image.fromarray(fr, mode="L")
        pal_frames.append(img.quantize(colors=256, method=Image.MEDIANCUT, dither=Image.Dither.NONE))
    pal_frames[0].save(
        save_path,
        save_all=True,
        append_images=pal_frames[1:],
        duration=100,
        loop=0,
        optimize=False,
        disposal=2,
    )


def _informative_slice_indices(
    volume: np.ndarray,
    min_std: float = 0.4,
    mid_fraction: float = 0.5,
) -> np.ndarray:
    """
    Keep mid-depth slices with real structure.

    Raises if no slice meets ``min_std`` — do not fall back to flat slices
    (per-slice windowing on those yields patchy square noise).
    """
    vol = _as_dhw_float(volume)
    d = vol.shape[0]
    mid = d // 2
    stds = vol.std(axis=(1, 2))
    keep = np.where(stds >= min_std)[0]
    if keep.size == 0:
        raise ValueError(
            f"No axial slices with std>={min_std} (max slice std={float(stds.max()):.4f}). "
            "Latent is near-constant along depth; not visualizing."
        )

    half = max(2, int(np.ceil(d * mid_fraction / 2.0)))
    band = keep[(keep >= mid - half) & (keep <= mid + half)]
    if band.size == 0:
        # Keep informative slices closest to mid-depth (still all above min_std).
        order = np.argsort(np.abs(keep.astype(np.float64) - mid))
        band = np.sort(keep[order[: max(3, min(len(keep), 2 * half + 1))]])
    return band.astype(np.int64)


@rank_zero_only
def visualize_latent_comparison_gif(
    original,
    reconstructed,
    batch_text="",
    save_path_prefix=None,
    stats_path=DEFAULT_STATS_PATH,
):
    """
    Save original / sample latent GIFs.

    Normalized latents, shared volume window on mid-depth informative slices
    (so GIF opens on readable anatomy). Skips viz if the GT latent has no
    structured axial slices (avoids patchy square noise from flat volumes).
    """
    assert save_path_prefix is not None, "save_path_prefix must be provided"
    del stats_path

    orig = _as_dhw_float(original)
    recon = _as_dhw_float(reconstructed)
    try:
        slice_indices = _informative_slice_indices(orig, min_std=0.4, mid_fraction=0.5)
    except ValueError as e:
        print(f"[Visualization] Skipping latent GIFs: {e}")
        return

    # Rotate so frame 0 is the mid-depth slice (IDE/GIF preview shows first frame).
    mid_z = orig.shape[0] // 2
    mid_local = int(np.argmin(np.abs(slice_indices.astype(np.float64) - mid_z)))
    slice_indices = np.concatenate(
        [slice_indices[mid_local:], slice_indices[:mid_local]]
    )

    # Shared window from informative GT slices only (not per-slice — that
    # amplifies float noise on low-contrast cuts into patchy squares).
    lo, hi = np.percentile(orig[slice_indices], [1.0, 99.0])
    if hi <= lo:
        print(
            f"[Visualization] Skipping latent GIFs: degenerate window "
            f"[{lo}, {hi}] on informative slices"
        )
        return
    vmins = np.full(orig.shape[0], lo, dtype=np.float64)
    vmaxs = np.full(orig.shape[0], hi, dtype=np.float64)

    orig_path = f"{save_path_prefix}_original.gif"
    recon_path = f"{save_path_prefix}_reconstructed.gif"
    volume_to_axial_gif(
        orig,
        orig_path,
        batch_text=(batch_text + " (Original)").strip(),
        slice_vmins=vmins,
        slice_vmaxs=vmaxs,
        slice_indices=slice_indices,
    )
    volume_to_axial_gif(
        recon,
        recon_path,
        batch_text=(batch_text + " (Sample)").strip(),
        slice_vmins=vmins,
        slice_vmaxs=vmaxs,
        slice_indices=slice_indices,
    )
    best_z = int(slice_indices[0])  # already rotated to mid-depth first
    mid_u8 = ((np.clip(orig[best_z], lo, hi) - lo) / (hi - lo) * 255.0).astype(np.uint8)
    mid_u8 = _upscale_gray(mid_u8, scale=LATENT_VIZ_UPSCALE)
    mid_path = f"{save_path_prefix}_original_mid.png"
    Image.fromarray(mid_u8, mode="L").save(mid_path)

    print(
        f"[Visualization] Saved latent GIFs (norm volume p1-p99 on mid band, "
        f"{len(slice_indices)}/{orig.shape[0]} slices):\n"
        f" - {orig_path}\n - {recon_path}\n - {mid_path} (z={best_z})"
    )


def _window_u8(vol_hu: np.ndarray, wl: float = -600, ww: float = 1500) -> np.ndarray:
    lo, hi = wl - ww / 2, wl + ww / 2
    x = np.clip(vol_hu, lo, hi)
    x = (x - lo) / (hi - lo + 1e-8)
    return (x * 255).astype(np.uint8)


def _decoder_to_hu(recon: np.ndarray) -> np.ndarray:
    """Map MedVAE CT decoder output (~[0,1] after clip) to HU for display."""
    x = np.clip(recon.astype(np.float32), 0.0, 1.0)
    return (x * 2000.0 - 1000.0).astype(np.float32)


@rank_zero_only
def visualize_ct_comparison_gif(
    gt_hu: np.ndarray,
    sample_hu: np.ndarray,
    save_path: str,
    batch_text: str = "",
    step: int = 2,
    scale: float = 0.35,
):
    """Side-by-side axial GIF: GT latent decode vs diffusion sample decode."""
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11
        )
    except Exception:
        font = ImageFont.load_default()

    g = _window_u8(gt_hu)
    s = _window_u8(sample_hu)
    d = min(g.shape[0], s.shape[0])
    frames = []
    for z in range(0, d, step):
        L = Image.fromarray(g[z]).convert("RGB")
        R = Image.fromarray(s[z]).convert("RGB")
        L = L.resize((max(1, int(L.width * scale)), max(1, int(L.height * scale))), Image.BILINEAR)
        R = R.resize((max(1, int(R.width * scale)), max(1, int(R.height * scale))), Image.BILINEAR)
        header = 26
        canvas = Image.new("RGB", (L.width * 2 + 4, L.height + header), (20, 20, 20))
        canvas.paste(L, (0, header))
        canvas.paste(R, (L.width + 4, header))
        draw = ImageDraw.Draw(canvas)
        title = (batch_text[:48] + "…") if len(batch_text) > 48 else batch_text
        draw.text((4, 2), f"GT decode  z={z}", fill=(255, 220, 100), font=font)
        draw.text((L.width + 8, 2), f"Sample decode  {title}", fill=(120, 220, 255), font=font)
        frames.append(np.asarray(canvas))

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(save_path, frames, duration=0.08, loop=0)
    print(f"[Visualization] Saved CT GIF: {save_path} ({len(frames)} frames)")


# ----------------------------
# Lightning Module
# ----------------------------

class LatentDiffusionLightningModule(LightningModule):
    def __init__(
        self,
        unet_kwargs,
        diffusion_kwargs,
        tokenizer_path,
        learning_rate=2e-4,
        vis_interval=1000,
        output_dir="checkpoints",
        ema_decay=0.995,
        step_start_ema=1000,
        update_ema_every=10,
        stats_path=DEFAULT_STATS_PATH,
        decode_vis=True,
        decode_every_n_epochs=10,
        medvae_gpu_dim=128,
        cond_scale=3.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["unet_kwargs", "diffusion_kwargs"])

        # CXR-BERT remote code imports configuration_cxrbert from this dir
        tok_dir = str(Path(tokenizer_path).resolve())
        if tok_dir not in sys.path:
            sys.path.insert(0, tok_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.text_encoder = AutoModel.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.text_encoder.eval()
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.diffusion = LatentDiffusion(unet_kwargs, diffusion_kwargs)
        self.ema = EMA(ema_decay)
        self.ema_model = deepcopy(self.diffusion)
        for p in self.ema_model.parameters():
            p.requires_grad = False

        self.update_ema_every = update_ema_every
        self.step_start_ema = step_start_ema
        self.learning_rate = learning_rate
        self.vis_interval = vis_interval
        self.output_dir = output_dir
        self.stats_path = stats_path
        self.decode_vis = decode_vis
        self.decode_every_n_epochs = int(decode_every_n_epochs)
        self.medvae_gpu_dim = medvae_gpu_dim
        self.cond_scale = float(cond_scale)
        self._medvae = None  # lazy-load on first CT visualization
        self._last_vis_batch = None

    def exists(self, val):
        return val is not None

    @staticmethod
    def _remove_medvae_checkpoint_state(checkpoint):
        """Exclude lazy visualization-only MedVAE weights from train checkpoints."""
        state_dict = checkpoint.get("state_dict", {})
        medvae_keys = [key for key in state_dict if key.startswith("_medvae.")]
        for key in medvae_keys:
            del state_dict[key]
        return len(medvae_keys)

    def on_save_checkpoint(self, checkpoint):
        removed = self._remove_medvae_checkpoint_state(checkpoint)
        if removed:
            print(
                f"[Checkpoint] Excluded {removed} visualization-only MedVAE parameters."
            )

    def on_load_checkpoint(self, checkpoint):
        removed = self._remove_medvae_checkpoint_state(checkpoint)
        if removed:
            print(
                f"[Checkpoint] Ignoring {removed} visualization-only MedVAE parameters."
            )

    def _get_medvae(self):
        if self._medvae is None:
            try:
                from medvae import MVAE
            except ImportError as e:
                raise ImportError(
                    "MedVAE is required when --decode_vis is enabled. "
                    "Install with: pip install -e ./MedVAE"
                ) from e

            print(
                f"[Visualization] Loading MedVAE (gpu_dim={self.medvae_gpu_dim}) "
                f"on {self.device} for CT decode..."
            )
            self._medvae = (
                MVAE(
                    model_name="medvae_4_1_3d",
                    modality="ct",
                    gpu_dim=self.medvae_gpu_dim,
                )
                .to(self.device)
                .eval()
            )
            for p in self._medvae.parameters():
                p.requires_grad = False
        return self._medvae

    def get_text_embeddings(self, texts, pad_id=0., eps=1e-6):
        tokens = self.tokenizer(
            texts,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tokens = tokens.to(self.device)
        with torch.no_grad():
            outputs = self.text_encoder(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
                output_hidden_states=True,
            )
            return outputs.hidden_states[-1]

    def training_step(self, batch, batch_idx):
        latents = batch["latent"].float().to(self.device)
        raw_texts = batch["text"]
        fov = batch["fov"].float().to(self.device)
        text_embeds = self.get_text_embeddings(raw_texts)
        loss = self.diffusion(latents, cond=text_embeds, fov=fov)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        for name, value in self.diffusion.last_loss_components.items():
            self.log(
                f"train_{name}",
                value,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )

        if (self.global_step >= self.step_start_ema) and (self.global_step % self.update_ema_every == 0):
            self.ema.update_model_average(self.ema_model, self.diffusion)

        # Cache for epoch-end MedVAE decode (CPU copies free GPU between epochs).
        self._last_vis_batch = (
            latents.detach().cpu(),
            text_embeds.detach().cpu(),
            fov.detach().cpu(),
            list(raw_texts) if isinstance(raw_texts, (list, tuple)) else [str(raw_texts)],
        )

        if (self.global_step % self.vis_interval == 0) and (batch_idx == 0):
            # Step path: latent GIFs only when epoch-scheduled decode is enabled.
            do_decode = self.decode_vis and self.decode_every_n_epochs <= 0
            self._visualize(
                latents,
                text_embeds,
                raw_texts,
                fov=fov,
                device=self.device,
                do_decode=do_decode,
                vis_prefix=os.path.join(self.output_dir, f"vis_step{self.global_step}"),
            )
        return loss

    def on_train_epoch_end(self):
        if not self.decode_vis or self.decode_every_n_epochs <= 0:
            return
        if self.current_epoch % self.decode_every_n_epochs != 0:
            return
        if self._last_vis_batch is None:
            raise RuntimeError(
                "MedVAE epoch decode scheduled but no training batch was cached yet"
            )
        latents, text_embeds, fov, raw_texts = self._last_vis_batch
        latents = latents.to(self.device)
        text_embeds = text_embeds.to(self.device)
        fov = fov.to(self.device)
        prefix = os.path.join(self.output_dir, f"vis_epoch{self.current_epoch}")
        self._visualize(
            latents,
            text_embeds,
            raw_texts,
            fov=fov,
            device=self.device,
            do_decode=True,
            vis_prefix=prefix,
            fail_loud=True,
        )

    def _visualize(
        self,
        latents,
        text_embeds,
        raw_texts,
        fov,
        device,
        do_decode=False,
        vis_prefix=None,
        fail_loud=False,
    ):
        self.ema_model.eval()
        with torch.no_grad():
            sample_norm = self.ema_model.p_sample_loop(
                cond=text_embeds[:1],
                fov=None,
                shape=(1, *latents.shape[1:]),
                device=device,
                use_ddim=True,
                clip_denoised=False,
                cond_scale=self.cond_scale,
            )
        if vis_prefix is None:
            vis_prefix = os.path.join(self.output_dir, f"vis_step{self.global_step}")
        text0 = raw_texts[0] if isinstance(raw_texts, (list, tuple)) else str(raw_texts)
        visualize_latent_comparison_gif(
            latents[0],
            sample_norm[0],
            batch_text=text0,
            save_path_prefix=vis_prefix,
            stats_path=self.stats_path,
        )

        if not do_decode:
            return

        try:
            medvae = self._get_medvae()
            gt_lat = denormalize_latent(latents[:1], stats_path=self.stats_path)
            sample_lat = denormalize_latent(sample_norm, stats_path=self.stats_path)
            with torch.no_grad():
                gt_dec = medvae.decode(gt_lat).detach().float().cpu().numpy().squeeze()
                sample_dec = medvae.decode(sample_lat).detach().float().cpu().numpy().squeeze()
            gt_hu = _decoder_to_hu(gt_dec)
            sample_hu = _decoder_to_hu(sample_dec)
            visualize_ct_comparison_gif(
                gt_hu,
                sample_hu,
                save_path=f"{vis_prefix}_ct_gt_vs_sample.gif",
                batch_text=text0,
            )
            z = gt_hu.shape[0] // 2
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12
                )
            except Exception:
                font = ImageFont.load_default()
            L = Image.fromarray(_window_u8(gt_hu[z])).convert("RGB")
            R = Image.fromarray(_window_u8(sample_hu[z])).convert("RGB")
            canvas = Image.new("RGB", (L.width * 2 + 4, L.height + 24), (20, 20, 20))
            canvas.paste(L, (0, 24))
            canvas.paste(R, (L.width + 4, 24))
            draw = ImageDraw.Draw(canvas)
            draw.text((4, 4), f"GT decode z={z}", fill=(255, 220, 100), font=font)
            draw.text((L.width + 8, 4), "Sample decode", fill=(120, 220, 255), font=font)
            mid_path = f"{vis_prefix}_ct_mid.png"
            canvas.save(mid_path)
            print(f"[Visualization] Saved CT mid PNG: {mid_path}")
        except Exception as e:
            raise RuntimeError(
                f"MedVAE decode failed during visualization (fail_loud={fail_loud}): {e}"
            ) from e
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def configure_optimizers(self):
        return AdamW(self.diffusion.parameters(), lr=self.learning_rate)


def _is_new_conditioning_state_key(key: str) -> bool:
    prefixes = (
        "diffusion.denoise_fn.null_cond_emb",
        "diffusion.denoise_fn.null_fov_token",
        "diffusion.denoise_fn.fov_mlp.",
        "ema_model.denoise_fn.null_cond_emb",
        "ema_model.denoise_fn.null_fov_token",
        "ema_model.denoise_fn.fov_mlp.",
    )
    return any(key.startswith(prefix) for prefix in prefixes)


def train(
    csv_path,
    tokenizer_path,
    unet_kwargs,
    diffusion_kwargs,
    output_dir="checkpoints",
    checkpoint=None,
    weights_checkpoint=None,
    batch_size=2,
    num_epochs=1000,
    learning_rate=2e-4,
    log_interval=100,
    vis_interval=1000,
    num_workers=1,
    target_shape=(48, 96, 96),
    latent_col="latent_path",
    stats_path=DEFAULT_STATS_PATH,
    decode_vis=True,
    decode_every_n_epochs=10,
    medvae_gpu_dim=128,
    gpus=1,
    seed=42,
    cond_scale=3.0,
):
    seed_everything(seed)
    if checkpoint is not None and weights_checkpoint is not None:
        raise ValueError(
            "Use either checkpoint (full Lightning resume) or weights_checkpoint "
            "(fresh optimizer finetune), not both."
        )

    dataloader = get_dataloader(
        csv_path=csv_path,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        target_shape=target_shape,
        latent_col=latent_col,
        stats_path=stats_path,
    )

    model = LatentDiffusionLightningModule(
        unet_kwargs=unet_kwargs,
        diffusion_kwargs=diffusion_kwargs,
        tokenizer_path=tokenizer_path,
        learning_rate=learning_rate,
        vis_interval=vis_interval,
        output_dir=output_dir,
        stats_path=stats_path,
        decode_vis=decode_vis,
        decode_every_n_epochs=decode_every_n_epochs,
        medvae_gpu_dim=medvae_gpu_dim,
        cond_scale=cond_scale,
    )

    if weights_checkpoint is not None:
        weights_path = Path(weights_checkpoint)
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"Weight checkpoint not found: {weights_path.resolve()}"
            )
        checkpoint_data = torch.load(
            weights_path, map_location="cpu", weights_only=False, mmap=True
        )
        if "state_dict" not in checkpoint_data:
            raise KeyError(
                f"Lightning checkpoint has no 'state_dict': {weights_path.resolve()}"
            )
        checkpoint_state = checkpoint_data["state_dict"]
        medvae_keys = [key for key in checkpoint_state if key.startswith("_medvae.")]
        model_state = {
            key: value
            for key, value in checkpoint_state.items()
            if not key.startswith("_medvae.")
        }
        incompatible = model.load_state_dict(
            model_state, strict=False
        )
        del checkpoint_data, checkpoint_state, model_state
        expected_missing = {
            "diffusion.log_var_isc",
            "ema_model.log_var_isc",
        }
        expected_missing.update(
            key for key in model.state_dict() if _is_new_conditioning_state_key(key)
        )
        missing = set(incompatible.missing_keys)
        unexpected = set(incompatible.unexpected_keys)
        if missing != expected_missing or unexpected:
            raise RuntimeError(
                "Checkpoint is incompatible with FOV/CFG/structural finetuning. "
                f"Expected only missing keys {sorted(expected_missing)}, "
                f"got missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        model.step_start_ema = 0
        print(
            f"Loaded model/EMA weights from {weights_path.resolve()} with fresh "
            "optimizer state; initialized new FOV/CFG parameters and log_var_isc=0. "
            f"Excluded {len(medvae_keys)} visualization-only MedVAE parameters; "
            "EMA updates start immediately."
        )

    logger = TensorBoardLogger(
        save_dir=os.path.join(output_dir, "runs"),
        name=datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    # Keep only two files: best (lowest train_loss_epoch) and last.ckpt.
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename="light_model_best",
        monitor="train_loss_epoch",
        mode="min",
        save_top_k=1,
        save_last=True,
        every_n_epochs=1,
        enable_version_counter=False,
    )

    trainer = Trainer(
        max_epochs=num_epochs,
        logger=logger,
        callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=10)],
        log_every_n_steps=log_interval,
        accelerator="gpu" if gpus else "cpu",
        devices=gpus,
        precision=16 if torch.cuda.is_available() else 32,
    )
    if checkpoint is None:
        if weights_checkpoint is None:
            print("No checkpoint provided, starting training from scratch.")
        else:
            print("Starting finetuning with fresh optimizer and epoch state.")
        trainer.fit(model, dataloader)
    else:
        trainer.fit(model, dataloader, ckpt_path=checkpoint)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="training_data.csv")
    parser.add_argument(
        "--tokenizer_path",
        default="./checkpoints/bert_tokenizer/",
        help="Path to CXR-BERT (or compatible) tokenizer + encoder directory.",
    )
    parser.add_argument("--checkpoint_path", required=False, default=None)
    parser.add_argument(
        "--weights_checkpoint",
        default=None,
        help=(
            "Load model/EMA weights from a Lightning checkpoint but start a new "
            "run with fresh optimizer and epoch state (for changed objectives)."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="runs/lorad",
        help="Directory for checkpoints and visualizations.",
    )
    parser.add_argument("--stats_path", default=DEFAULT_STATS_PATH)
    parser.add_argument("--vis_interval", type=int, default=1000)
    parser.add_argument(
        "--decode_vis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also decode one sample with MedVAE for CT-space GIFs (disable with --no-decode_vis).",
    )
    parser.add_argument(
        "--decode_every_n_epochs",
        type=int,
        default=10,
        help="MedVAE decode GT vs sample every N epochs (0 = decode on step vis_interval instead).",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1000,
        help="Training epochs.",
    )
    parser.add_argument("--medvae_gpu_dim", type=int, default=128)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument(
        "--prediction_type",
        choices=("v", "x0", "eps"),
        default="v",
        help="UNet target: v-pred (default, more stable), x0, or eps.",
    )
    parser.add_argument(
        "--loss_type",
        choices=("l1", "l2", "edge+l2", "edge+l2+isc"),
        default="edge+l2+isc",
        help=(
            "Training objective. edge+l2+isc adds tri-planar "
            "first-difference consistency on predicted x0 (LoRAD default)."
        ),
    )
    parser.add_argument(
        "--null_cond_prob",
        type=float,
        default=0.1,
        help="Classifier-free guidance dropout on text conditioning during training.",
    )
    parser.add_argument(
        "--null_fov_prob",
        type=float,
        default=0.15,
        help="Dropout probability replacing FOV features with the learned null token.",
    )
    parser.add_argument(
        "--cond_scale",
        type=float,
        default=3.0,
        help="Classifier-free guidance scale used for EMA visualization sampling.",
    )
    args = parser.parse_args()

    unet_kwargs = {
        "dim": 32,
        "channels": 1,
        "cond_dim": 768,
        "total_slices": 48,
        "use_bert_text_cond": True,
    }
    diffusion_kwargs = {
        "image_size": 96,
        "num_frames": 48,
        "text_use_bert_cls": True,
        "channels": 1,
        "ddim_timesteps": 75,
        "loss_type": args.loss_type,
        "prediction_type": args.prediction_type,
        "null_cond_prob": args.null_cond_prob,
        "null_fov_prob": args.null_fov_prob,
    }

    train(
        csv_path=args.csv_path,
        tokenizer_path=args.tokenizer_path,
        output_dir=args.output_dir,
        unet_kwargs=unet_kwargs,
        diffusion_kwargs=diffusion_kwargs,
        checkpoint=args.checkpoint_path,
        weights_checkpoint=args.weights_checkpoint,
        stats_path=args.stats_path,
        vis_interval=args.vis_interval,
        decode_vis=args.decode_vis,
        decode_every_n_epochs=args.decode_every_n_epochs,
        medvae_gpu_dim=args.medvae_gpu_dim,
        gpus=args.gpus,
        batch_size=args.batch_size,
        num_epochs=args.max_epochs,
        cond_scale=args.cond_scale,
    )
