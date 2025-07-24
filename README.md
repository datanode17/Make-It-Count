# Make It Count: Enhanced Layout Guidance for Precise Object Counting in Diffusion Models

**Training‑Free Layout Guidance for Accurate Object Counting in Diffusion Models**

---

## Overview

**Make It Count** is a lightweight, plug‑in enhancement for text‑to‑image diffusion models (e.g., Stable Diffusion) that dramatically improves the model’s ability to generate the exact number of objects specified in the prompt—**without any additional training**. By introducing:

1. **Binary Forward Guidance**  
   Applies hard inside/outside masks (weights α > 1 inside, β < 1 outside) to sharpen token‑to‑bbox attention.

2. **Same‑Entity Masking (Backward Guidance)**  
   Ensures that multiple instances of the same object class do not bleed into one another by masking out same‑token regions during cross‑attention.

you can reduce counting Mean Absolute Error (MAE) by ~12 % and more than double the exact‑count rate compared to strong baselines (Layout Control, Attend‑and‑Excite, Stable Diffusion v1.5).

---

## Repository Structure

```text
.
├── makeitcount.ipynb        # Demo & evaluation notebook
├── transformer_2d.py        # 2D transformer block utilities
├── unet_2d_blocks.py        # U‑Net block definitions
├── unet_2d_condition.py     # Cross‑attention hooks & guidance code
└── README.md                # This file
