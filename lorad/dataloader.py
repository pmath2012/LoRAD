"""
Dataloader for resize-pad MedVAE latents used by text-to-latent diffusion.

On-disk layout: (D, H, W) = (48, 96, 96) from encode grid (192, 384, 384).
Model tensor: (C, F, H, W) = (1, 48, 96, 96) — depth is frames (F).
No spatial Resize of latents.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai.transforms import Compose, EnsureChannelFirst, ToTensor, Transform
from torch.utils.data import DataLoader, Dataset

DEFAULT_STATS_PATH = "data/latent_mean_std.json"
# Fallback scalars if JSON is not provided (prefer loading from stats file).
GLOBAL_MEAN = -21.368939
GLOBAL_STD = 12.040871

# Encode (192,384,384) / 4 → latent (D,H,W)
DEFAULT_LATENT_SHAPE = (48, 96, 96)

# Per-axis (extent, center, activity): content footprint inside the fixed grid.
FOV_FEATURE_DIM = 9
TEXT_SEQ_LEN = 256


def compute_fov_features(volume: np.ndarray, min_std: float = 0.15) -> np.ndarray:
    """Derive latent-space FOV descriptors from anatomy extent along D/H/W."""
    vol = np.asarray(volume, dtype=np.float32)
    if vol.ndim == 4:
        vol = vol.squeeze(0)
    if vol.shape != DEFAULT_LATENT_SHAPE:
        raise ValueError(
            f"Expected latent shape {DEFAULT_LATENT_SHAPE}, got {vol.shape} for FOV features"
        )

    vol_std = float(vol.std())
    if vol_std < 1e-6:
        raise ValueError(
            f"Cannot compute FOV features on near-constant latent (std={vol_std})"
        )
    threshold = min_std * vol_std
    features: list[float] = []
    for axis in range(3):
        if axis == 0:
            profile = np.std(vol, axis=(1, 2))
        elif axis == 1:
            profile = np.std(vol, axis=(0, 2))
        else:
            profile = np.std(vol, axis=(0, 1))
        active = profile > threshold
        if not active.any():
            raise ValueError(
                f"No active slices along axis {axis} for FOV features "
                f"(threshold={threshold:.4f}, vol_std={vol_std:.4f})"
            )
        idx = np.where(active)[0]
        start, end = int(idx[0]), int(idx[-1])
        size = vol.shape[axis]
        extent = (end - start + 1) / size
        center = (start + end) / 2.0 / max(size - 1, 1)
        activity = float(profile[active].mean() / vol_std)
        features.extend([extent, center, activity])
    return np.asarray(features, dtype=np.float32)


class Normalize(Transform):
    def __init__(self, mean=None, std=None):
        self.mean = GLOBAL_MEAN if mean is None else mean
        self.std = GLOBAL_STD if std is None else std

    def __call__(self, data):
        data = data.astype(np.float32)
        return (data - self.mean) / (self.std + 1e-8)


def load_latent_stats(stats_path: str | None = DEFAULT_STATS_PATH):
    """Return (mean, std) from stats JSON. Requires the file when stats_path is set."""
    if not stats_path:
        return GLOBAL_MEAN, GLOBAL_STD
    p = Path(stats_path)
    if not p.is_file():
        raise FileNotFoundError(
            f"Missing latent stats at {p.resolve()}; expected mean/std JSON next to the .npy bank"
        )
    with open(p) as f:
        stats = json.load(f)
    return float(stats["mean"]), float(stats["std"])


def denormalize_latent(
    x,
    mean=None,
    std=None,
    stats_path: str | None = DEFAULT_STATS_PATH,
):
    """
    Invert dataloader Normalize: x * std + mean.

    Accepts numpy or torch tensors; returns same type.
    """
    if mean is None or std is None:
        m, s = load_latent_stats(stats_path)
        mean = m if mean is None else mean
        std = s if std is None else std
    return x * std + mean


def get_latent_transforms(mean=None, std=None):
    """
    Latent array transforms: channel + normalize only.

    Expected on-disk layout: (D, H, W) = (48, 96, 96).
    After EnsureChannelFirst → (1, D, H, W) which is (C, F, H, W) for the UNet.
    """
    return Compose(
        [
            EnsureChannelFirst(channel_dim="no_channel"),  # (1, D, H, W)
            Normalize(mean=mean, std=std),
            ToTensor(),
        ]
    )


class TextLatentDataset(Dataset):
    def __init__(self, dataframe, transforms=None, latent_col="latent_path"):
        self.df = dataframe.reset_index(drop=True)
        self.transforms = transforms
        self.latent_col = latent_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        path = row[self.latent_col]
        latent = np.load(path)
        if latent.ndim == 4:
            latent = latent.squeeze(0)
        if latent.shape != DEFAULT_LATENT_SHAPE:
            raise ValueError(
                f"Expected latent shape {DEFAULT_LATENT_SHAPE}, got {latent.shape} for {path}. "
                "Regenerate with generate_latents_resizepad.py; do not resize latents."
            )

        fov = compute_fov_features(latent)

        if self.transforms:
            latent = self.transforms(latent)

        text = row["impressions"] if "impressions" in row else row.get("text", "")

        return {
            "latent": latent.float(),
            "text": text,
            "fov": torch.from_numpy(fov),
        }


def _filter_ready_rows(df: pd.DataFrame, latent_col: str) -> pd.DataFrame:
    """Keep rows with finished latents that exist on disk and have axial structure."""
    out = df
    if "latent_done" in out.columns:
        done = (
            out["latent_done"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
        )
        out = out[done]
    paths = out[latent_col].astype(str)
    missing_paths = [p for p in paths if not Path(p).is_file()]
    if missing_paths:
        preview = ", ".join(missing_paths[:5])
        more = f" (+{len(missing_paths) - 5} more)" if len(missing_paths) > 5 else ""
        raise FileNotFoundError(
            f"{len(missing_paths)} latent files missing (CSV paths must resolve on disk). "
            f"Examples: {preview}{more}"
        )

    # Reject near-constant axial volumes — viz/training treat dim0 as depth (D,H,W).
    # Stretching those slices for display creates patchy square noise.
    bad_axial: list[str] = []
    for p in paths:
        vol = np.asarray(np.load(p, mmap_mode="r"), dtype=np.float32)
        if vol.ndim == 4:
            vol = np.squeeze(vol, axis=0)
        if tuple(vol.shape) != DEFAULT_LATENT_SHAPE:
            bad_axial.append(p)
            continue
        vol_std = float(vol.std())
        mean_slice_std = float(np.std(vol, axis=(1, 2)).mean())
        # Slices should carry a non-trivial fraction of the volume's energy.
        if vol_std < 1e-6 or mean_slice_std < 0.15 * vol_std:
            bad_axial.append(p)
    if bad_axial:
        preview = ", ".join(bad_axial[:5])
        more = f" (+{len(bad_axial) - 5} more)" if len(bad_axial) > 5 else ""
        raise RuntimeError(
            f"{len(bad_axial)}/{len(paths)} latents have near-constant axial slices "
            f"(mean in-slice std < 15% of volume std) — not valid (D,H,W)=(48,96,96) banks. "
            f"Examples: {preview}{more}. Remove/re-encode them before training."
        )
    return out.reset_index(drop=True)


def get_dataloader(
    csv_path,
    batch_size,
    shuffle,
    num_workers,
    target_shape=DEFAULT_LATENT_SHAPE,
    latent_col="latent_path",
    stats_path=DEFAULT_STATS_PATH,
):
    """
    DataLoader for fixed-shape resize-pad latents (48×96×96).

    Spatial Resize of latents is intentionally not applied.
    """
    if tuple(target_shape) != DEFAULT_LATENT_SHAPE:
        raise ValueError(
            f"Resize-pad latents are fixed at {DEFAULT_LATENT_SHAPE}; got target_shape={target_shape}"
        )

    df = pd.read_csv(csv_path)
    if latent_col not in df.columns:
        raise KeyError(
            f"CSV {csv_path} missing column {latent_col!r}. "
            "Run generate_latents_resizepad.py first."
        )

    df = _filter_ready_rows(df, latent_col)
    if len(df) == 0:
        raise RuntimeError(f"No usable latents in {csv_path} (col={latent_col})")

    mean, std = load_latent_stats(stats_path)
    print(
        f"[dataloader] n={len(df)} shape={DEFAULT_LATENT_SHAPE} "
        f"mean={mean:.6f} std={std:.6f} stats={stats_path}"
    )
    transforms = get_latent_transforms(mean=mean, std=std)
    dataset = TextLatentDataset(df, transforms=transforms, latent_col=latent_col)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
