# LoRAD

**LoRAD** (*Low-Resource Adaptive Diffusion*) is a text-to-CT generation method that learns a 3D latent diffusion model under limited paired report–volume data, with soft field-of-view (FOV) conditioning and structural consistency losses for more anatomically plausible volumes.

This repository contains the core training and inference code for LoRAD. 
---

## Method overview

LoRAD generates chest CT from free-text radiology reports in a compressed latent space:

1. **Encode** or **downsize** CT with a frozen foundation model MedVAE after a fixed resize–pad preprocess → latents of shape `(48, 96, 96)`.
2. **Diffuse** in latent space with a 3D U-Net conditioned on frozen CXR-BERT report embeddings (**v-prediction**).
3. **Regularize** predicted clean latents with:
   - axial **Sobel edge** L1 (structure in-plane), and
   - **tri-planar first-difference** L1 (inter-slice / 3D consistency).
4. **Condition on FOV** via a 9-D content-extent descriptor derived from each training latent (MLP → cross-attention token), with **null-FOV dropout** so inference can run from text alone.
5. **Sample** with **classifier-free guidance** (CFG); FOV is always the learned null token at test time.
6. **Decode** with MedVAE to HU and write NIfTI with a canonical LAS affine / spacing suitable for CT-RATE-style evaluation.

Loss terms are combined with Kendall-style learned log-variances. An EMA copy of the network is used for visualization and inference.

```
Report ──► CXR-BERT ──┐
                      ├─► 3D Latent U-Net (v-pred) ──► DDIM + CFG ──► MedVAE decode ──► CT
FOV token (train) ────┘         ▲
null FOV (infer) ───────────────┘
         + edge + inter-slice consistency on x₀
```

---

## Repository layout

```
LoRAD/
  README.md
  requirements.txt
  lorad/
    dataloader.py       # latent CSV loader + FOV features
    diffusion_model.py  # LatentOnlyUNet3D + LatentDiffusion (FOV, CFG, losses)
    train.py            # PyTorch Lightning training entrypoint
    infer.py            # report → NIfTI inference
```

External dependencies (not vendored here):

- [MedVAE](https://github.com/StanfordMIMI/MedVAE) — CT VAE encode/decode  
- A MedSyn-compatible 3D U-Net / Gaussian diffusion stack on `PYTHONPATH` (`MedSyn/src`, providing `train_super_res.Unet3D`, `GaussianDiffusion`, `EMA`, and utils)  
- A CXR-BERT (or compatible) text encoder directory for tokenization + embeddings  

---

## Installation

```bash
# Python 3.10+, CUDA PyTorch matching your GPU
pip install -r requirements.txt
pip install -e /path/to/MedVAE   # or the MedVAE install you prefer

export PYTHONPATH="/path/to/MedSyn/src:${PYTHONPATH}"
export PYTHONPATH="/path/to/LoRAD:${PYTHONPATH}"
```

---

## Data format

Training CSV must include at least:

| column | description |
|--------|-------------|
| `latent_path` | path to `.npy` latent of shape `(48, 96, 96)` |
| `impressions` | report text used for conditioning |

Optional: `latent_done` filter. Latents should be z-scored using a JSON stats file:

```json
{ "mean": <float>, "std": <float> }
```

FOV features are computed on the fly from each latent (no extra CSV columns required).

---

## Training

```bash
python -m lorad.train \
  --csv_path data/train.csv \
  --tokenizer_path checkpoints/bert_tokenizer \
  --stats_path data/latent_mean_std.json \
  --output_dir runs/lorad \
  --prediction_type v \
  --loss_type edge+l2+isc \
  --null_cond_prob 0.1 \
  --null_fov_prob 0.15 \
  --cond_scale 3.0 \
  --batch_size 2 \
  --max_epochs 500 \
  --decode_vis \
  --decode_every_n_epochs 10
```

Checkpoints saved under `--output_dir`:

- `light_model_best.ckpt` — lowest epoch train loss  
- `last.ckpt` — latest epoch (full Lightning state for resume)

Resume:

```bash
python -m lorad.train ... --checkpoint_path runs/lorad/last.ckpt --max_epochs 500
```

---

## Inference

Prepare a models directory:

```
models/
  light_model.ckpt      # EMA weights (Lightning ckpt with ema_model.*)
  bert_tokenizer/
  latent_mean_std.json
  hf_cache/             # offline MedVAE cache if running offline
```

```bash
export LORAD_MODELS_DIR=./models
export LORAD_INPUT=./prompts.json
export LORAD_OUTPUT=./outputs
export CFG_SCALE=3.0    # optional; default 3.0

python -m lorad.infer
```

`prompts.json` example:

```json
[
  {"input_image_name": "case_001", "report": "No acute cardiopulmonary process."}
]
```

Inference always uses the **null FOV** token (text + CFG only).

---

## Design notes (low-resource focus)

| Component | Role under limited data |
|-----------|-------------------------|
| Frozen MedVAE + text encoder | Avoid learning perception/language from scratch |
| Compact 3D U-Net in latent space | Fits ~hundreds of volumes on a single GPU |
| Edge + ISC losses | Encourage structure without dense organ masks |
| FOV token + null dropout | Disentangle coverage variation; still work without FOV at test |
| CFG | Sharper report adherence at sampling time |


## License

Code in this repository is released under the MIT License (see `LICENSE`).  
Third-party components (MedVAE, MedSyn, CXR-BERT, etc.) remain under their respective licenses.
