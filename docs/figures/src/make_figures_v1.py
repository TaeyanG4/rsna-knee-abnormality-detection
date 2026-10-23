"""make_figures v1.0.2 (B, 2026-10-23): README figures for the public RSNA knee repo.

Inputs: data/submissions-raw.csv (kaggle competitions submissions --csv, our own scores only).
Outputs (docs/figures/): score_timeline.png, public_vs_private.png, pipeline_overview.png.
No competition data is read.
Changelog v1.0.2: submissions without a description (37 temporary/teammate uploads named submission.csv) are excluded (Taeyang).
Changelog v1.0.1: separate final-pair annotation offsets; wider first pipeline box. Colours: dataviz reference palette slots 1-2 (validated, light surface).
"""
import csv, io, os, collections
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))  # repo root; this file lives in docs/figures/src
OUT = os.path.join(ROOT, 'docs', 'figures')
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': GRID,
                     'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF})


def load():
    t = open(os.path.join(ROOT, 'docs', 'data', 'submissions-raw.csv'), encoding='utf-8').read()
    rows = list(csv.DictReader(io.StringIO(t[t.find('ref,'):])))
    out = []
    for r in rows:
        if not (r.get('description') or '').strip():
            continue
        pub, prv = (r.get('publicScore') or '').strip(), (r.get('privateScore') or '').strip()
        if pub in ('', 'None') or prv in ('', 'None'):
            continue
        out.append((datetime.fromisoformat(r['date'][:19]), float(pub), float(prv), r['ref']))
    return sorted(out)


FINAL = {'57415972': ('final A (x3-v2)', (-95, -34)), '57475409': ('final B (P25F flat hedge)', (-40, -58))}


def timeline(rows):
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150)
    d = [r[0] for r in rows]
    ax.plot(d, [r[1] for r in rows], 'o', ms=5, color=BLUE, mec=SURF, mew=1, label='public LB')
    ax.plot(d, [r[2] for r in rows], 's', ms=5, color=ORANGE, mec=SURF, mew=1, label='private LB')
    for y, name in [(0.943, 'silver cut-off 0.943 (rank 321)'), (0.954, 'gold cut-off 0.954')]:
        ax.axhline(y, color=INK2, lw=0.8, ls=(0, (3, 3)), zorder=0)
        ax.text(d[0], y + 0.0008, name, color=INK2, fontsize=8, va='bottom')
    for r in rows:
        if r[3] in FINAL:
            ax.annotate(FINAL[r[3]][0], (r[0], r[2]), xytext=FINAL[r[3]][1], textcoords='offset points',
                        fontsize=8, color=INK, arrowprops=dict(arrowstyle='-', color=INK2, lw=0.7))
    ax.set_ylabel('macro AUC (12 findings)')
    ax.set_title('Our 87 scored pipeline submissions: public vs private score over time', loc='left', color=INK, fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.grid(axis='y', color=GRID, lw=0.6)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    lo = min(min(r[1], r[2]) for r in rows)
    ax.set_ylim(max(0.85, lo - 0.005), 0.962)
    ax.legend(frameon=False, loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'score_timeline.png'))
    plt.close(fig)


def scatter(rows):
    c = collections.Counter((r[1], r[2]) for r in rows if r[1] >= 0.94)
    fig, ax = plt.subplots(figsize=(6.2, 5), dpi=150)
    xs, ys, ns = zip(*[(k[0], k[1], v) for k, v in c.items()])
    ax.scatter(xs, ys, s=[30 + 25 * n for n in ns], color=BLUE, alpha=0.85, edgecolor=SURF, linewidth=1.5)
    for x, y, n in zip(xs, ys, ns):
        if n >= 3:
            ax.text(x, y, str(n), color='white', fontsize=8, ha='center', va='center', weight='bold')
    ax.set_xlabel('public LB (390 studies, 3 decimals)')
    ax.set_ylabel('private LB (3 decimals)')
    ax.set_title('Public ties hide little: almost every 0.957-0.958\npublic submission landed at private 0.942-0.943',
                 loc='left', color=INK, fontsize=10.5)
    ax.grid(color=GRID, lw=0.6)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.text(0.01, 0.01, 'bubble size / number = submissions with that score pair (public >= 0.940 shown)',
            transform=ax.transAxes, fontsize=7.5, color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'public_vs_private.png'))
    plt.close(fig)


def box(ax, x, y, w, h, text, fc, ec, fs=8.5, color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.06',
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=color, linespacing=1.35)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='-|>', color=INK2, lw=1))


def pipeline():
    fig, ax = plt.subplots(figsize=(10, 5.4), dpi=150)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.6); ax.axis('off')
    pub_fc, own_fc = '#e3eefb', '#fdeadf'
    box(ax, 0.1, 4.45, 2.85, 0.9, 'Hidden test MRI study\n(multi-series DICOM, ~1,300 studies)', '#f0efec', GRID)
    box(ax, 3.2, 4.45, 3.0, 0.9, 'Shared decode / render\n(one pass per study, T4 x2)', '#f0efec', GRID)
    arrow(ax, 2.95, 4.9, 3.2, 4.9)
    box(ax, 0.1, 2.55, 3.2, 1.35, 'P  (public members)\nRaptor CoAtNet R384 + R224 (SWA)\nR6 ConvNeXt-S + flip TTA\nper-target public weights', pub_fc, BLUE)
    box(ax, 3.5, 2.55, 2.6, 1.35, 'O  (our older bundle)\nown ConvNeXt / CoAtNet\nmembers (S_R7, EXP143v2,\nEXP155 ...)', own_fc, ORANGE)
    box(ax, 6.3, 2.55, 3.55, 1.35, 'K1  (our model)\nConvNeXt-S 2.5D MIL, trio:\nk1_v2 + v3-G seed0 + seed1\nfail-loud runtime, 8.8 h guard', own_fc, ORANGE)
    for x in (1.7, 4.8, 8.05):
        arrow(ax, 4.7, 4.45, x, 3.9)
    box(ax, 0.6, 0.95, 4.1, 1.05, 'x3-v2 (final A)\nINF3 = rank(.85 r(P) + .15 r(O))\nfinal = rank(.8 r(INF3) + .2 r(K1, stride 3))', '#ffffff', INK2, fs=8.2)
    box(ax, 5.2, 0.95, 4.4, 1.05, 'P25F (final B, hedge)\nfinal = rank(.75 r(P_flat) + .25 r(K1, stride 2))\nno O; flat P weights (.5 for all targets)', '#ffffff', INK2, fs=8.2)
    arrow(ax, 1.7, 2.55, 2.0, 2.0); arrow(ax, 4.8, 2.55, 3.6, 2.0); arrow(ax, 8.05, 2.55, 3.9, 2.0)
    arrow(ax, 1.7, 2.55, 6.4, 2.0); arrow(ax, 8.05, 2.55, 8.1, 2.0)
    ax.text(5.0, 0.35, 'per-target rank averaging (12 findings, average ties); final pair = two structurally different families',
            ha='center', fontsize=8.5, color=INK2)
    ax.text(0.1, 5.5, 'Serving pipeline of the two final submissions (public 0.958 / 0.957, private 0.943 / 0.943)',
            fontsize=11, color=INK)
    fig.savefig(os.path.join(OUT, 'pipeline_overview.png'), bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    print('make_figures v1.0.2')
    rows = load()
    print('scored submissions', len(rows))
    timeline(rows); scatter(rows); pipeline()
    print('written to', OUT)
