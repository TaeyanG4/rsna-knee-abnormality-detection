"""make_banner v1.0.3 (B, 2026-10-23): README banner for the public RSNA knee repo.

Uses the openly licensed Wikimedia Commons image "Knee MRI PD TSE FS Sagittal.jpg" (Ptrump16, CC BY-SA 4.0) as a faded
background element, so the banner is an adaptation under CC BY-SA 4.0 (credited in NOTICE.md). NOT competition data.
Output: docs/figures/banner.png (1600 x 460 px).
Changelog v1.0.3: larger text for readability at README width (label 19, headline 42, subtitle 21, medal 24, scores 20,
  credit 11 pt), brighter secondary text, banner 420 -> 460 px tall, MRI panel starts at x = 9.8.
Changelog v1.0.2: no team name (Taeyang); the competition name is the headline.
Changelog v1.0.1: MRI blended with a positional alpha (v1.0.0 scaled by intensity, too faint); medal chip without a number.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))  # repo root; this file lives in docs/figures/src
SRC = os.path.join(ROOT, 'downloads-wikimedia', 'sag', 'Knee_MRI_PD_TSE_FS_Sagittal.jpg')
OUT = os.path.join(ROOT, 'docs', 'figures', 'banner.png')
BG = '#0e2a2f'          # deep teal, close to the KneeOps sidebar
ACCENT = '#1baf7a'      # aqua (validated categorical slot 3)
SILVER = '#c9ccd1'
SUB = '#e3ecea'          # secondary text, brighter than v1.0.2's #d7e3e1

print("make_banner v1.0.3")
W, H, DPI = 16, 4.6, 100
fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=BG)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
ax.add_patch(plt.Rectangle((0, 0), W, H, color=BG, zorder=0))

img = mpimg.imread(SRC).astype(float)
if img.ndim == 3:
    img = img[..., :3].mean(axis=2)
img = img / img.max()
h, w = img.shape
# right-side MRI panel with a horizontal fade into the background
x0, x1 = 9.8, W
y0, y1 = 0, H
panel_w = x1 - x0
crop_h = int(w * (H / panel_w))
top = max(0, (h - crop_h) // 2)
crop = img[top:top + crop_h, :]
fade = np.clip(np.linspace(-0.2, 1.0, crop.shape[1]), 0, 1) ** 1.3
bg_rgb = np.array([int(BG[i:i + 2], 16) / 255 for i in (1, 3, 5)])
alpha = (0.92 * fade)[None, :, None]
tint = np.array([0.80, 0.95, 0.93])[None, None, :]
rgb = bg_rgb[None, None, :] * (1 - alpha) + (crop[..., None] ** 0.8) * tint * alpha
ax.imshow(np.clip(rgb, 0, 1), extent=(x0, x1, y0, y1), aspect='auto', zorder=1)

ax.text(0.7, 3.72, 'KAGGLE  ·  RSNA AI CHALLENGE 2026', color=ACCENT, fontsize=19, weight='bold', zorder=3)
ax.text(0.7, 2.78, 'RSNA Knee Abnormality Detection', color='white', fontsize=42, weight='bold', zorder=3)
ax.text(0.7, 2.06, '12 knee MRI findings  ·  multi-series DICOM  ·  macro AUC', color=SUB, fontsize=21, zorder=3)
# medal chip
ax.add_patch(plt.Circle((1.04, 0.98), 0.34, color=SILVER, zorder=3))
ax.text(1.6, 1.12, 'Silver medal', color='white', fontsize=24, weight='bold', zorder=3)
ax.text(1.6, 0.55, 'private 0.943  ·  rank 312 / 6,425  ·  public 0.958', color=SUB, fontsize=20, zorder=3)
ax.text(W - 0.15, 0.14, 'MRI: Ptrump16, Wikimedia Commons, CC BY-SA 4.0 (not competition data)', color='#b4c8c5',
        fontsize=11, ha='right', zorder=3)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=DPI, facecolor=BG)
print('written', OUT)
