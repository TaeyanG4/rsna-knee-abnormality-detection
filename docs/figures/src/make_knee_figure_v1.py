"""make_knee_figure v1.0.0 (B, 2026-10-23): composite of two openly licensed Wikimedia Commons knee MRI images.

NOT competition data. Inputs (downloaded with Taeyang's approval, sizes 229,749 / 28,536 bytes):
  downloads-wikimedia/sag/Knee_MRI_PD_TSE_FS_Sagittal.jpg  - Ptrump16, CC BY-SA 4.0
  downloads-wikimedia/cor/Proton_density_MRI_of_a_grade_2_medial_meniscal_tear.jpg - Lefevre et al. 2016, CC BY 4.0
Output: docs/figures/knee_mri_examples.png (an adaptation; CC BY-SA 4.0 share-alike applies to it).
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))  # repo root; this file lives in docs/figures/src
DL = os.path.join(ROOT, 'downloads-wikimedia')  # first download the two files listed in NOTICE.md into sag/ and cor/
OUT = os.path.join(ROOT, 'docs', 'figures', 'knee_mri_examples.png')
INK, INK2, SURF = '#0b0b0b', '#52514e', '#fcfcfb'

panels = [
    (os.path.join(DL, 'sag', 'Knee_MRI_PD_TSE_FS_Sagittal.jpg'),
     'Sagittal proton-density, fat-suppressed',
     'Ptrump16, Wikimedia Commons, CC BY-SA 4.0'),
    (os.path.join(DL, 'cor', 'Proton_density_MRI_of_a_grade_2_medial_meniscal_tear.jpg'),
     'Coronal proton-density: grade 2 medial meniscal tear',
     'Lefevre et al., Radiol Res Pract 2016, Wikimedia Commons, CC BY 4.0'),
]

print('make_knee_figure v1.0.0')
fig, axes = plt.subplots(1, 2, figsize=(9, 4.9), dpi=150, facecolor=SURF)
for ax, (path, title, credit) in zip(axes, panels):
    img = mpimg.imread(path)
    ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
    ax.set_title(title, fontsize=10, color=INK)
    ax.set_xlabel(credit, fontsize=7.5, color=INK2)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
fig.suptitle('Knee MRI examples (openly licensed images, not competition data)', fontsize=11, color=INK, x=0.01, ha='left')
fig.tight_layout()
fig.savefig(OUT, facecolor=SURF)
print('written', OUT)
