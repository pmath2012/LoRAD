"""
LoRAD inference entrypoint.

Input:  prompts JSON — list of {"input_image_name": str, "report": str}
Output: <name>.nii.gz HU volumes, LAS orientation, spacing (1.0, 1.0, 1.8828) mm,
        shape (192, 384, 384) after MedVAE decode.

Expected model directory layout (LORAD_MODELS_DIR):
    light_model.ckpt
    bert_tokenizer/
    latent_mean_std.json
    hf_cache/                 — offline MedVAE Hugging Face cache (optional if online)

Paths can be overridden with environment variables:
    LORAD_MODELS_DIR, LORAD_INPUT, LORAD_OUTPUT, CFG_SCALE
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve repo root so sibling modules can be imported regardless of cwd.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "MedSyn" / "src"))

import numpy as np
import torch
import nibabel as nib

# ---------------------------------------------------------------------------
# Paths — override with env vars for local or containerized runs.
# ---------------------------------------------------------------------------
MODELS_DIR   = Path(os.environ.get("LORAD_MODELS_DIR", "/opt/app/models"))
CKPT_PATH    = MODELS_DIR / "light_model.ckpt"
TOKENIZER_PATH = MODELS_DIR / "bert_tokenizer"
STATS_PATH   = MODELS_DIR / "latent_mean_std.json"
HF_CACHE_DIR = MODELS_DIR / "hf_cache"

INPUT_PATH  = Path(os.environ.get("LORAD_INPUT", "/input/prompts.json"))
OUTPUT_DIR  = Path(os.environ.get("LORAD_OUTPUT", "/output"))

# ---------------------------------------------------------------------------
# Point HuggingFace hub at the pre-cached weights directory BEFORE any
# huggingface_hub import resolves its cache path (which happens at import time).
# This must be set at module level, not inside _load_medvae(), to guarantee
# it takes effect before `from medvae import MVAE` triggers any hub calls.
# ---------------------------------------------------------------------------
os.environ["HF_HOME"]               = str(HF_CACHE_DIR)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE_DIR / "hub")
os.environ["TRANSFORMERS_OFFLINE"]  = "1"
os.environ["HF_DATASETS_OFFLINE"]   = "1"
os.environ["HF_HUB_OFFLINE"]        = "1"

# Model architecture — must match training exactly.
LATENT_SHAPE = (48, 96, 96)   # (D, H, W) → model sees (1, 1, D, H, W)
UNET_KWARGS = {
    "dim": 32,
    "channels": 1,
    "cond_dim": 768,
    "total_slices": 48,
    "use_bert_text_cond": True,
}
DIFFUSION_KWARGS = {
    "image_size": 96,
    "num_frames": 48,
    "text_use_bert_cls": True,
    "channels": 1,
    "ddim_timesteps": 75,
    "prediction_type": "v",
}
MEDVAE_MODEL_NAME = "medvae_4_1_3d"
MEDVAE_MODALITY   = "ct"
MEDVAE_GPU_DIM    = 128

# Classifier-free guidance scale for text conditioning. FOV is always null at inference.
CFG_SCALE = float(os.environ.get("CFG_SCALE", "3.0"))
if CFG_SCALE < 1.0:
    raise ValueError(f"CFG_SCALE must be >= 1.0, got {CFG_SCALE}")


# ---------------------------------------------------------------------------
# 1. Eager prerequisite validation
# ---------------------------------------------------------------------------

def _validate_prerequisites() -> list[dict[str, str]]:
    """Crash immediately with a clear message if anything required is missing."""

    # GPU — no CPU fallback.
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU required but none found. "
            "Run with --gpus all (Docker) or on a machine with a CUDA GPU."
        )

    # Weight files / directories.
    required = [
        (CKPT_PATH,      "diffusion checkpoint"),
        (TOKENIZER_PATH, "CXR-BERT tokenizer directory"),
        (STATS_PATH,     "latent normalisation stats JSON"),
        (HF_CACHE_DIR,   "HuggingFace MedVAE cache directory"),
    ]
    for path, label in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {label}: {path}\n"
                f"Ensure weights.zip was mounted at /weights and unpacked to {MODELS_DIR}."
            )
    medvae_cache = HF_CACHE_DIR / "hub" / "models--stanfordmimi--MedVAE"
    if not medvae_cache.is_dir() or not any(medvae_cache.rglob("vae_4x_1c_3D.ckpt")):
        raise FileNotFoundError(
            f"MedVAE cache is incomplete under {medvae_cache}; "
            "expected vae_4x_1c_3D.ckpt."
        )

    # Input JSON — parse and validate structure before touching GPU.
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    raw = INPUT_PATH.read_text()
    try:
        prompts = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Cannot parse {INPUT_PATH}: {exc}") from exc

    if not isinstance(prompts, list) or len(prompts) == 0:
        raise ValueError(
            f"{INPUT_PATH} must be a non-empty JSON array; got: {type(prompts).__name__}"
        )

    for i, entry in enumerate(prompts):
        if not isinstance(entry, dict):
            raise ValueError(f"prompts[{i}] must be an object; got {type(entry).__name__}.")
        for key in ("input_image_name", "report"):
            if key not in entry:
                raise ValueError(
                    f"prompts[{i}] is missing required key {key!r}: {entry}"
                )
        if not isinstance(entry["input_image_name"], str) or not entry["input_image_name"].strip():
            raise ValueError(f"prompts[{i}]['input_image_name'] must be a non-empty string.")
        name = entry["input_image_name"].strip()
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError(
                f"prompts[{i}]['input_image_name'] must be a base name, not a path: {name!r}"
            )
        if not isinstance(entry["report"], str) or not entry["report"].strip():
            raise ValueError(f"prompts[{i}]['report'] must be a non-empty string.")

    # Output directory — check it is writable now, not mid-loop.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _sentinel = OUTPUT_DIR / ".write_test"
    try:
        _sentinel.touch()
        _sentinel.unlink()
    except OSError as exc:
        raise OSError(f"Output directory {OUTPUT_DIR} is not writable: {exc}") from exc

    print(
        f"[predict] Prerequisites OK — {len(prompts)} prompts, "
        f"GPU: {torch.cuda.get_device_name(0)}, "
        f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    )
    return prompts


# ---------------------------------------------------------------------------
# 2. Model loading — all eager, all raising on failure
# ---------------------------------------------------------------------------

def _load_diffusion_model(device: torch.device):
    """Load LatentDiffusionLightningModule from checkpoint. Raises on any mismatch."""
    # Import after sys.path is set.
    from transformers import AutoTokenizer, AutoModel
    from lorad.diffusion_model import LatentDiffusion
    tok_dir = str(TOKENIZER_PATH.resolve())
    if tok_dir not in sys.path:
        sys.path.insert(0, tok_dir)

    print(f"[predict] Loading CXR-BERT tokenizer from {TOKENIZER_PATH} …")
    tokenizer = AutoTokenizer.from_pretrained(
        str(TOKENIZER_PATH), trust_remote_code=True, local_files_only=True
    )
    text_encoder = AutoModel.from_pretrained(
        str(TOKENIZER_PATH), trust_remote_code=True, local_files_only=True
    )
    text_encoder.eval().to(device)
    for p in text_encoder.parameters():
        p.requires_grad = False

    print(f"[predict] Building LatentDiffusion model …")
    dk = dict(DIFFUSION_KWARGS)
    diffusion = LatentDiffusion(dict(UNET_KWARGS), dk)

    print(f"[predict] Loading checkpoint {CKPT_PATH} …")
    ckpt = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)

    # Prefer EMA weights for sampling; do not silently fall back to non-EMA.
    ema_state = {}
    for k, v in state.items():
        if k.startswith("ema_model."):
            ema_state[k[len("ema_model."):]] = v
    if not ema_state:
        raise RuntimeError(
            f"Checkpoint {CKPT_PATH} has no ema_model.* weights; "
            "LoRAD inference requires the trained EMA model."
        )
    diffusion.load_state_dict(ema_state, strict=True)
    print(f"[predict] EMA weights loaded ({len(ema_state)} keys).")

    diffusion.eval().to(device)
    for p in diffusion.parameters():
        p.requires_grad = False

    return tokenizer, text_encoder, diffusion


def _load_medvae(device: torch.device):
    """Load MedVAE decoder from pre-cached HF weights. Raises if cache missing."""
    from medvae import MVAE

    print(f"[predict] Loading MedVAE ({MEDVAE_MODEL_NAME}, {MEDVAE_MODALITY}) …")
    medvae = MVAE(model_name=MEDVAE_MODEL_NAME, modality=MEDVAE_MODALITY, gpu_dim=MEDVAE_GPU_DIM)
    medvae.eval().to(device)
    for p in medvae.parameters():
        p.requires_grad = False
    print("[predict] MedVAE loaded.")
    return medvae


# ---------------------------------------------------------------------------
# 3. Inference helpers
# ---------------------------------------------------------------------------

def _get_text_embedding(
    texts: list[str],
    tokenizer,
    text_encoder,
    device: torch.device,
) -> torch.Tensor:
    tokens = tokenizer(
        texts,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        outputs = text_encoder(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            output_hidden_states=True,
        )
    return outputs.hidden_states[-1]


def _load_latent_stats() -> tuple[float, float]:
    with open(STATS_PATH) as f:
        s = json.load(f)
    mean = float(s["mean"])
    std  = float(s["std"])
    if std <= 0:
        raise ValueError(f"Invalid std={std} in {STATS_PATH}; must be positive.")
    return mean, std


def _denormalize(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return x * std + mean


@torch.inference_mode()
def _sample(ema_model, cond: torch.Tensor, device: torch.device) -> torch.Tensor:
    shape = (1, 1, *LATENT_SHAPE)
    sample = ema_model.p_sample_loop(
        cond=cond,
        fov=None,
        shape=shape,
        device=device,
        use_ddim=True,
        clip_denoised=False,
        cond_scale=CFG_SCALE,
    )
    if sample.shape[1:] != (1, *LATENT_SHAPE):
        raise RuntimeError(
            f"Unexpected sample shape {tuple(sample.shape)}; expected (1, 1, {LATENT_SHAPE})"
        )
    if torch.isnan(sample).any() or torch.isinf(sample).any():
        raise RuntimeError(
            "Diffusion sample contains NaN/Inf values. "
            "Check the checkpoint and conditioning input."
        )
    return sample


DECODE_SHAPE = (192, 384, 384)   # MedVAE 4x upsamples (48,96,96) → (192,384,384)


@torch.inference_mode()
def _decode_to_hu(medvae, latent: torch.Tensor) -> np.ndarray:
    """Decode a (1,1,D,H,W) latent tensor → HU float32 volume (D,H,W) at DECODE_SHAPE."""
    decoded = medvae.decode(latent).detach().float().cpu().numpy().squeeze()
    if decoded.ndim != 3:
        raise RuntimeError(
            f"MedVAE decoder returned unexpected shape {decoded.shape}; expected 3-D (D,H,W)."
        )
    if decoded.shape != DECODE_SHAPE:
        raise RuntimeError(
            f"MedVAE decoder output shape {decoded.shape} does not match expected {DECODE_SHAPE}. "
            "Check that MEDVAE_MODEL_NAME='medvae_4_1_3d' and latent shape is (48,96,96)."
        )
    # Intensity align: map [p1, p99] → [0, 1], then to HU [-1000, 1000].
    p1, p99 = float(np.percentile(decoded, 1)), float(np.percentile(decoded, 99))
    span = p99 - p1
    if span < 1e-6:
        raise RuntimeError(
            f"Decoded volume has near-zero intensity range [{p1:.4f}, {p99:.4f}]; "
            "the model may have collapsed."
        )
    normalised = np.clip((decoded - p1) / span, 0.0, 1.0).astype(np.float32)
    hu = (normalised * 2000.0 - 1000.0).astype(np.float32)
    return hu


def _make_ctrate_affine(shape: tuple) -> np.ndarray:
    """
    Build a NIfTI affine matching CT-RATE's canonical header.

    CT-RATE volumes are in LAS orientation (Left-Anterior-Superior), encoded at
    (192, 384, 384) with effective voxel spacing (1.8828, 1.0, 1.0) mm along
    (D=z, H=y, W=x).  MedVAE decodes back to this shape.

    The affine maps voxel indices (i, j, k) → world RAS mm:
        x_RAS = -1 * i  (LR flipped: L→R)
        y_RAS = +1 * j  (AP: A→P)
        z_RAS = +spacing_z * k

    Origin is set so the volume centre is at the world origin (SPM convention).
    """
    d, h, w = shape
    sx, sy, sz = 1.0, 1.0, 1.8828  # mm per voxel  (x, y, z)
    # LAS: x increases Left→Right (negate), y increases Anterior→Posterior, z increases Superior→Inferior
    affine = np.array([
        [-sx,   0,   0,  (w - 1) * sx / 2],   # RAS x  (i-axis, flipped)
        [  0,  sy,   0, -(h - 1) * sy / 2],   # RAS y  (j-axis)
        [  0,   0,  sz, -(d - 1) * sz / 2],   # RAS z  (k-axis)
        [  0,   0,   0,                  1],
    ], dtype=np.float64)
    return affine


def _save_nifti(volume: np.ndarray, out_path: Path) -> None:
    """
    Save a (D,H,W) float32 HU volume as .nii.gz with the CT-RATE canonical
    affine (LAS orientation, spacing 1.88 × 1.0 × 1.0 mm).

    Matching CT-RATE-style spacing and orientation keeps decoded volumes
    comparable under common 3D / slice-wise image metrics.
    """
    affine = _make_ctrate_affine(volume.shape)
    img = nib.Nifti1Image(volume.astype(np.float32), affine=affine)
    # Set pixdim (voxel sizes) explicitly in the header so readers that ignore
    # the sform affine still see the correct spacing.
    hdr = img.header
    hdr.set_zooms((1.0, 1.0, 1.8828))  # (x, y, z) spacing in mm
    hdr.set_sform(affine, code=1)       # scanner RAS
    hdr.set_qform(affine, code=1)
    hdr['xyzt_units'] = 10              # mm (space) + seconds (time) — NIfTI code for mm
    nib.save(img, str(out_path))


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

def main() -> None:
    # --- Validate everything before touching the GPU ---
    prompts = _validate_prerequisites()

    device = torch.device("cuda:0")

    # --- Load latent normalisation stats ---
    lat_mean, lat_std = _load_latent_stats()
    print(f"[predict] Latent stats: mean={lat_mean:.4f} std={lat_std:.4f}")
    print(f"[predict] Sampling with CFG_SCALE={CFG_SCALE}, null FOV token")

    # --- Load all models eagerly ---
    tokenizer, text_encoder, ema_model = _load_diffusion_model(device)
    medvae = _load_medvae(device)

    print(f"[predict] Starting inference for {len(prompts)} prompt(s) …\n")

    for i, entry in enumerate(prompts):
        name   = entry["input_image_name"].strip()
        report = entry["report"].strip()
        out_path = OUTPUT_DIR / f"{name}.nii.gz"

        print(f"[{i+1}/{len(prompts)}] {name}")
        print(f"  report: {report[:120]}{'…' if len(report) > 120 else ''}")

        # Text embedding
        cond = _get_text_embedding([report], tokenizer, text_encoder, device)

        # DDIM sampling in normalised latent space
        sample_norm = _sample(ema_model, cond, device)

        # Denormalise to original latent range
        sample_lat = _denormalize(sample_norm, lat_mean, lat_std)

        # Decode latent → HU volume
        hu_volume = _decode_to_hu(medvae, sample_lat)
        print(f"  HU range: [{hu_volume.min():.1f}, {hu_volume.max():.1f}]  shape: {hu_volume.shape}")

        # Save
        _save_nifti(hu_volume, out_path)
        print(f"  → {out_path}")

        torch.cuda.empty_cache()

    print(f"\n[predict] Done. {len(prompts)} volume(s) written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
