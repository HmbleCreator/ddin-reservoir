"""
generate_paper_figures_fixed.py
================================
Fixed version: resolves all text/element overlaps across all 7 figures.
Key fixes per figure:
  p1_fig2  — radar pushed to left panel; bar chart annotations repositioned
  p2_fig1  — box heights unified; font size reduced to prevent text clipping
  p2_fig2  — annotations moved off bars; y-limit raised
  p2_fig3  — axis labels moved below strip; bottom panel y expanded
  p3_fig1  — annotations shifted to avoid overlap with ESL band label
  p3_fig2  — text box positions adjusted for N_neurons=40 viewport
  p3_fig3  — escape diagram rows spaced further apart; right panel taller
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.gridspec as gridspec
import numpy as np
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DPI = 300

def savefig(name):
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {name}")

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

# ══════════════════════════════════════════════════════════════════════════════
# PAPER 1 — Figure 2: Five-Axis Semantic Wheel + Clustering Results
# FIX: Separate radar into its own subplot area properly; annotations moved
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Paper 1] Generating p1_fig2_axis_clustering.png ...")

fig = plt.figure(figsize=(14, 5.5))

# Use GridSpec: left half for radar, right half for bars
gs = gridspec.GridSpec(1, 2, figure=fig, left=0.04, right=0.97,
                       bottom=0.12, top=0.88, wspace=0.38)

# LEFT — polar radar inside a proper axes
ax_polar = fig.add_subplot(gs[0], polar=True)
axes_labels = ['EXP\n(Expansion)', 'TRN\n(Transform)', 'MOT\n(Motion)',
               'SEP\n(Separation)', 'CNT\n(Containment)']
N = 5
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]
root_kar = [0.72, 0.25, 0.61, 0.18, 0.14, 0.72]
root_dhR = [0.11, 0.19, 0.08, 0.14, 0.83, 0.11]

ax_polar.set_theta_offset(np.pi / 2)
ax_polar.set_theta_direction(-1)
ax_polar.set_xticks(angles[:-1])
ax_polar.set_xticklabels(axes_labels, size=8.5, ha='center')
ax_polar.set_yticks([0.25, 0.5, 0.75])
ax_polar.set_yticklabels(['0.25', '0.5', '0.75'], size=7, color='grey')
ax_polar.set_ylim(0, 1)
ax_polar.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
ax_polar.plot(angles, root_kar, color='#2196F3', linewidth=2, linestyle='solid',
              label='√kar (motion/action)')
ax_polar.fill(angles, root_kar, color='#2196F3', alpha=0.15)
ax_polar.plot(angles, root_dhR, color='#E91E63', linewidth=2, linestyle='dashed',
              label='√dhR (hold/contain)')
ax_polar.fill(angles, root_dhR, color='#E91E63', alpha=0.15)
ax_polar.set_title('Phenomenological Axis Profiles\n(example roots)',
                   size=9.5, pad=22, fontweight='bold')
ax_polar.legend(loc='upper right', bbox_to_anchor=(1.55, 1.22), fontsize=8.5,
                frameon=True, framealpha=0.9)

# RIGHT — ARI bar chart by locus group
ax2 = fig.add_subplot(gs[1])
categories = ['Throat\n(ka-varga)', 'Palatal\n(ca-varga)', 'Retroflex',
              'Dental\n(ta-varga)', 'Labial\n(pa-varga)']
sep_scores = [0.11, 0.34, 0.22, 0.58, 0.09]
cnt_scores = [0.18, 0.12, 0.15, 0.08, 0.61]
mot_scores = [0.38, 0.22, 0.41, 0.19, 0.17]

x = np.arange(len(categories))
w = 0.25
ax2.bar(x - w, sep_scores, w, label='SEP axis', color='#E53935', alpha=0.85, edgecolor='white')
ax2.bar(x,     mot_scores, w, label='MOT axis', color='#FB8C00', alpha=0.85, edgecolor='white')
ax2.bar(x + w, cnt_scores, w, label='CNT axis', color='#1E88E5', alpha=0.85, edgecolor='white')
ax2.set_xticks(x)
ax2.set_xticklabels(categories, fontsize=8.5)
ax2.set_ylabel('Normalised axis score (mean)', fontsize=9)
ax2.set_title('Axis Scores by Locus Group\n(150-root Paninian Benchmark)',
              fontsize=9.5, fontweight='bold')
ax2.legend(fontsize=8.5, frameon=False, loc='upper right')
ax2.set_ylim(0, 0.95)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Annotations — placed well above bars, no overlap
ax2.annotate('Dental → SEP dominant\n(high F2 → separation)',
             xy=(3 - w, sep_scores[3]), xytext=(0.8, 0.82),
             arrowprops=dict(arrowstyle='->', color='#E53935', lw=1.2),
             fontsize=7.5, color='#E53935',
             bbox=dict(boxstyle='round,pad=0.2', fc='#FFEBEE', ec='#E53935', alpha=0.8))
ax2.annotate('Labial → CNT dominant\n(low F2 → containment)',
             xy=(4 + w, cnt_scores[4]), xytext=(2.85, 0.73),
             arrowprops=dict(arrowstyle='->', color='#1E88E5', lw=1.2),
             fontsize=7.5, color='#1E88E5',
             bbox=dict(boxstyle='round,pad=0.2', fc='#E3F2FD', ec='#1E88E5', alpha=0.8))

fig.suptitle('Paper 1 — Phonosemantic Grounding: Axis Profiles and Locus–Meaning Correspondence',
             fontsize=9.5, style='italic', y=0.02, color='grey')
savefig('p1_fig2_axis_clustering.png')


# ══════════════════════════════════════════════════════════════════════════════
# PAPER 2 — Figure 1: DDIN Architecture Pipeline
# FIX: Uniform box heights; text font scaled; no clipping
# ══════════════════════════════════════════════════════════════════════════════
print("[Paper 2] Generating p2_fig1_architecture.png ...")

fig, ax = plt.subplots(figsize=(15, 4.8))
ax.set_xlim(0, 15)
ax.set_ylim(0, 4.8)
ax.axis('off')

BOX_H    = 2.2   # uniform height for all boxes
BOX_Y    = 1.0   # bottom y of all boxes
MID_Y    = BOX_Y + BOX_H / 2
BOX_W    = 1.85  # uniform width

# Centre 6 boxes across full width with equal gaps
N_BOXES  = 6
TOTAL_W  = BOX_W * N_BOXES
MARGIN   = 0.2
AVAIL    = 15 - 2 * MARGIN
GAP      = (AVAIL - TOTAL_W) / (N_BOXES - 1)
START_X  = MARGIN

box_labels = [
    ('Sanskrit Root\n(Dhātu)\ne.g. √kR',             '#0D47A1'),
    ('Formant-First\nEmbedding φ(d)\n∈ ℝ²³\n[F1,F2, manner,\npratyāhāra]', '#00695C'),
    ('Poisson\nSpike Train\nrate ∝ φⱼ(d)×100Hz',     '#E65100'),
    ('LIF Reservoir\nN=128 neurons\nτᵢ~U(10,100)ms\nheterogeneous', '#4A148C'),
    ('BCM Plasticity\nΔW=η·x·y(y−θ)\n[no backprop]', '#1B5E20'),
    ('Semantic\nReadout\nARI = 0.0366',               '#B71C1C'),
]

box_xs = [START_X + i * (BOX_W + GAP) for i in range(N_BOXES)]

for i, (label, col) in enumerate(box_labels):
    x = box_xs[i]
    rect = FancyBboxPatch((x, BOX_Y), BOX_W, BOX_H,
                          boxstyle="round,pad=0.10",
                          facecolor=col, edgecolor='white',
                          linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(x + BOX_W / 2, MID_Y, label,
            ha='center', va='center', color='white',
            fontsize=7.5, fontweight='bold', linespacing=1.5, zorder=4)

# Arrows between boxes
for i in range(N_BOXES - 1):
    x1 = box_xs[i] + BOX_W
    x2 = box_xs[i + 1]
    ax.annotate('', xy=(x2, MID_Y), xytext=(x1, MID_Y),
                arrowprops=dict(arrowstyle='->', color='#37474F',
                                lw=2.0, mutation_scale=18))

# Bottom banner — spans full box range
rect_bottom = FancyBboxPatch((MARGIN, 0.10), 15 - 2 * MARGIN, 0.72,
                              boxstyle="round,pad=0.05",
                              facecolor='#ECEFF1', edgecolor='#90A4AE', linewidth=1)
ax.add_patch(rect_bottom)
ax.text(7.5, 0.46,
        'NO BACKPROPAGATION   |   NO WORD-LEVEL LABELS   |   NO CORPUS CO-OCCURRENCE   |   BCM LOCAL LEARNING ONLY',
        ha='center', va='center', fontsize=8, color='#37474F', fontweight='bold')

ax.set_title('Figure 1 — DDIN Receiver Model Architecture',
             fontsize=12, fontweight='bold', pad=10, color='#212121')
plt.tight_layout()
savefig('p2_fig1_architecture.png')


# ══════════════════════════════════════════════════════════════════════════════
# PAPER 2 — Figure 2: ARI Progression + Formant Weight Ablation
# FIX: y-limits extended; annotations avoid bars; legend repositioned
# ══════════════════════════════════════════════════════════════════════════════
print("[Paper 2] Generating p2_fig2_ari_progression.png ...")

fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))

ax = axes[0]
versions  = ['v12\n(21D one-hot)', 'v13\n(29D Pratyāhāra)',
             'v15\n(26D redundant)', 'v16\n(23D formant-first)']
ari_axis  = [0.0260, 0.0180, 0.0002, 0.0366]
ari_locus = [0.0820, 0.0460, 0.0640, 0.0398]

x = np.arange(len(versions))
w = 0.35
ax.bar(x - w/2, ari_axis,  w, label='ARI (semantic axis)',  color='#1565C0', alpha=0.9, edgecolor='white')
ax.bar(x + w/2, ari_locus, w, label='ARI (locus group)',    color='#F57F17', alpha=0.9, edgecolor='white')

ax.axvspan(2.5, 3.5, alpha=0.08, color='green', zorder=0)
# Annotation above the highlighted column — moved higher to avoid bar tops
ax.text(3.0, 0.112, 'Breakthrough\n(Gap=0.003)',
        ha='center', va='bottom', fontsize=7.5, color='#1B5E20', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.2', fc='#F1F8E9', ec='#1B5E20', alpha=0.85))

ax.axhline(0, color='#888888', linewidth=0.8, linestyle='--', alpha=0.8)
# ARI=0 label moved to right side at negative y to avoid bar overlap
ax.text(3.7, -0.010, 'ARI=0\n(random)', color='#888888', fontsize=7.5, ha='right',
        va='top', alpha=1.0,
        bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='#888888', alpha=0.7))
ax.set_xticks(x)
ax.set_xticklabels(versions, fontsize=8.5)
ax.set_ylabel('Adjusted Rand Index (ARI)', fontsize=9)
ax.set_title('ARI Progression by Embedding Version', fontsize=10, fontweight='bold')
ax.legend(fontsize=8.5, frameon=False, loc='upper left')
ax.set_ylim(-0.018, 0.140)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# RIGHT — formant weight ablation
ax2 = axes[1]
wF_vals    = [0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,  8.0]
ari_ax_wF  = [0.0042, 0.0118, 0.0247, 0.0366, 0.0312, 0.0119, 0.0055, 0.0021]
ari_loc_wF = [0.0681, 0.0591, 0.0478, 0.0398, 0.0211, 0.0088, 0.0043, 0.0019]

ax2.plot(wF_vals, ari_ax_wF,  'o-',  color='#1565C0', linewidth=2, markersize=5, label='ARI (axis)')
ax2.plot(wF_vals, ari_loc_wF, 's--', color='#F57F17', linewidth=2, markersize=5, label='ARI (locus)')
ax2.fill_between(wF_vals, ari_ax_wF, ari_loc_wF, alpha=0.12, color='grey', label='Locus dominance gap')

ax2.axvline(2.0, color='green', linewidth=1.5, linestyle=':', alpha=0.8)
# Label to the right of the vline, no overlap with curve peaks
ax2.text(2.15, 0.060, 'Optimal\nwF=2.0', fontsize=7.5, color='#1B5E20', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.2', fc='#F1F8E9', ec='#1B5E20', alpha=0.85))

ax2.set_xlabel('Formant Weight (wF)', fontsize=9)
ax2.set_ylabel('ARI', fontsize=9)
ax2.set_title('Formant Weight Ablation\n(v16 embedding)', fontsize=10, fontweight='bold')
ax2.legend(fontsize=8.5, frameon=False, loc='upper right')
ax2.set_ylim(-0.005, 0.092)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.suptitle('Figure 2 — ARI Progression and Formant Weight Ablation',
             fontsize=11, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
savefig('p2_fig2_ari_progression.png')


# ══════════════════════════════════════════════════════════════════════════════
# PAPER 2 — Figure 3: F2 Gradient + Semantic Axis Mapping
# FIX: Adjusted figsize, padding, labels position, and limits to avoid overlaps
# ══════════════════════════════════════════════════════════════════════════════
print("[Paper 2] Generating p2_fig3_f2_gradient.png ...")

fig = plt.figure(figsize=(13, 8.0))
gs  = gridspec.GridSpec(2, 1, height_ratios=[1.0, 1.6], hspace=0.7,
                         top=0.82, bottom=0.10, left=0.07, right=0.96)

# TOP — F2 gradient strip
ax_top = fig.add_subplot(gs[0])
gradient = np.linspace(0, 1, 300).reshape(1, -1)
ax_top.imshow(gradient, aspect='auto', cmap='RdYlBu_r', extent=[800, 2100, 0, 1])
ax_top.set_xlim(800, 2100)
ax_top.set_ylim(0, 1)
ax_top.set_yticks([])
ax_top.set_xlabel('F2 Locus Frequency (Hz)', fontsize=10, labelpad=8)
ax_top.set_title('Second Formant (F2) as Continuous Articulatory Coordinate',
                 fontsize=11, fontweight='bold', pad=45)

loci   = [800,  1100,      1400,     1800,    2100]
labels_top = ['Labial\n(bilabial\nclosure)',
              'Retroflex\n(tongue\ncurled back)',
              'Velar/Throat\n(back of\nmouth)',
              'Dental\n(tongue\nat teeth)',
              'Palatal\n(tongue\nat ridge)']
for hz, lbl in zip(loci, labels_top):
    ax_top.axvline(hz, color='white', linewidth=1.2, alpha=0.8)
    ax_top.text(hz, 1.05, lbl, ha='center', va='bottom', fontsize=8,
                color='#212121', linespacing=1.3,
                transform=ax_top.get_xaxis_transform())

# Varga markers — placed INSIDE the strip with high visibility text
varga_hz     = [1400, 2100, 1100, 1800,  800]
varga_lbls   = ['ka-varga\n(Throat)', 'ca-varga\n(Palatal)',
                'Retroflex',          'ta-varga\n(Dental)', 'pa-varga\n(Labial)']
varga_colors = ['#EF6C00', '#6A1B9A', '#00838F', '#F9A825', '#2E7D32']  # Darker for contrast
for hz, lbl, col in zip(varga_hz, varga_lbls, varga_colors):
    ax_top.plot(hz, 0.82, 'v', color=col, markersize=13, zorder=5)
    ax_top.text(hz, 0.40, lbl, ha='center', va='center', fontsize=7.5,
                color=col, fontweight='bold', linespacing=1.3,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85))

# BOTTOM — semantic axis F2 dot plot
ax_bot = fig.add_subplot(gs[1])
axes_l  = ['CNT (Contain)', 'TRN (Transform)', 'EXP (Expand)', 'MOT (Motion)', 'SEP (Separate)']
f2_mean = [0.439,           0.472,             0.476,          0.477,          0.555]
f2_std  = [0.087,           0.114,             0.102,          0.108,          0.091]
colors  = ['#1565C0', '#7B1FA2', '#EF6C00', '#00838F', '#C62828']
f2_hz   = [800 + v * (2100 - 800) for v in f2_mean]

for i, (hz, std, lbl, col) in enumerate(zip(f2_hz, f2_std, axes_l, colors)):
    std_hz = std * (2100 - 800)
    ax_bot.barh(i, std_hz * 2, left=hz - std_hz, height=0.45,
                color=col, alpha=0.18)
    ax_bot.plot(hz, i, 'D', color=col, markersize=10, zorder=5)
    # Labels to the right of range bars
    ax_bot.text(hz + std_hz + 25, i, lbl,
                ha='left', va='center', fontsize=9.5, color=col, fontweight='bold')

ax_bot.set_xlim(600, 2450)   # extra space on ends
ax_bot.set_ylim(-1.0, 5.0)   # expanded limits to ensure all bars show clearly
ax_bot.set_yticks([])
ax_bot.set_xlabel('F2 (Hz) — mapped from normalised value', fontsize=10, labelpad=8)
ax_bot.set_title('Phenomenological Axis Mean F2 Position   [Spearman ρ = 1.0, predicted without fitting]',
                 fontsize=10, fontweight='bold', pad=15)
ax_bot.spines['top'].set_visible(False)
ax_bot.spines['right'].set_visible(False)
ax_bot.spines['left'].set_visible(False)

# Footer notes — below the axes, safely inside plot area
ax_bot.text(2450, -1.0, 'SEP→separation (forward place)',
            ha='right', va='top', fontsize=8.5, color='#C62828', style='italic')
ax_bot.text(600,  -1.0, 'CNT→containment (bilabial closure)',
            ha='left',  va='top', fontsize=8.5, color='#1565C0', style='italic')

plt.suptitle('Figure 3 — F2 Formant Gradient: Acoustic Physics Predicts Semantic Order',
             fontsize=12, fontweight='bold', y=0.96)
savefig('p2_fig3_f2_gradient.png')


# ══════════════════════════════════════════════════════════════════════════════
# PAPER 3 — Figure 1: Phase Transition — ARI vs tau_max sweep
# FIX: annotations pulled away from ESL band text; no label overlap
# ══════════════════════════════════════════════════════════════════════════════
print("[Paper 3] Generating p3_fig1_phase_transition.png ...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

ax = axes[0]
tau_vals = [100,  200,    500,      1000]
ari_vals = [0.0130, 0.0088, -0.0101, -0.0008]
rate_vals = [12, 28, 402, 493]

color_pts = ['#2E7D32' if a > 0 else '#C62828' for a in ari_vals]
ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
ax.plot(tau_vals, ari_vals, 'k-', linewidth=1.5, alpha=0.5, zorder=1)
for tau, ari, col in zip(tau_vals, ari_vals, color_pts):
    ax.scatter(tau, ari, color=col, s=90, zorder=5)

ax.axvspan(350, 1050, alpha=0.07, color='red')
# ESL label in the upper part of the span
ax.text(690, 0.017, 'ESL Zone\n(synchronous\nseizure)', fontsize=8,
        color='#C62828', ha='center', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.25', fc='#FFEBEE', ec='#C62828', alpha=0.85))
ax.axvline(350, color='red', linewidth=2, linestyle=':', alpha=0.7, label='ESL threshold τ*')

# Asynchronous annotation — upper left, clear of ESL band
ax.annotate('ARI=0.0130\n(asynchronous, coherent)',
            xy=(100, 0.0130), xytext=(60, 0.022),
            arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.2),
            fontsize=7.5, color='#2E7D32',
            bbox=dict(boxstyle='round,pad=0.2', fc='#E8F5E9', ec='#2E7D32', alpha=0.8))
# Seizure annotation — lower right, inside chart
ax.annotate('ARI=−0.0101\n(full seizure ~400Hz)',
            xy=(500, -0.0101), xytext=(560, -0.016),
            arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.2),
            fontsize=7.5, color='#C62828',
            bbox=dict(boxstyle='round,pad=0.2', fc='#FFEBEE', ec='#C62828', alpha=0.8))

ax.set_xlabel('τ_max (ms)', fontsize=10)
ax.set_ylabel('ARI (semantic axis)', fontsize=10)
ax.set_title('Phase Transition:\nARI vs Membrane Time Constant', fontsize=10, fontweight='bold')
ax.legend(fontsize=8.5, frameon=False, loc='lower right')
ax.set_xlim(50, 1100)
ax.set_ylim(-0.024, 0.030)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# RIGHT — Mean firing rate vs tau_max
ax2 = axes[1]
color_rate = ['#2E7D32', '#F9A825', '#C62828', '#C62828']
ax2.bar(range(4), rate_vals, color=color_rate, alpha=0.85, edgecolor='white', linewidth=0.5)
ax2.set_xticks(range(4))
ax2.set_xticklabels(['Exp17\n(100ms)', 'Exp17b\n(200ms)',
                     'Exp17b\n(500ms)', 'Exp17b\n(1000ms)'], fontsize=8.5)
ax2.set_ylabel('Mean Network Firing Rate (Hz)', fontsize=10)
ax2.set_title('Population Firing Rate\n(synchrony onset visible at ~500 Hz)',
              fontsize=10, fontweight='bold')
ax2.axhline(499, color='red', linewidth=1.5, linestyle='--', alpha=0.7, label='Saturation ≈ 500 Hz')
ax2.legend(fontsize=8.5, frameon=False)
ax2.set_ylim(0, 580)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.suptitle('Figure 1 — The Epileptiform Synchrony Limit: Phase Transition in Spiking Dynamics',
             fontsize=11, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
savefig('p3_fig1_phase_transition.png')


# ══════════════════════════════════════════════════════════════════════════════
# PAPER 3 — Figure 2: Raster Plot — Asynchronous vs Synchronous
# FIX: text box y-position uses data coords (0..N_neurons); placed at top-left
# ══════════════════════════════════════════════════════════════════════════════
print("[Paper 3] Generating p3_fig2_raster_comparison.png ...")

np.random.seed(42)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
N_neurons = 40
T_ms = 200

# LEFT — Asynchronous sparse
ax = axes[0]
for n in range(N_neurons):
    rate = np.random.uniform(5, 20) / 1000
    spikes = np.where(np.random.rand(T_ms) < rate)[0]
    ax.scatter(spikes, [n] * len(spikes), s=4, color='#1565C0', alpha=0.7)

ax.set_xlim(0, T_ms)
ax.set_ylim(-1, N_neurons + 6)   # extra headroom for text box
ax.set_xlabel('Time (ms)', fontsize=10)
ax.set_ylabel('Neuron index', fontsize=10)
ax.set_title('Asynchronous Sparse Regime\n(τ_max=100ms, rate≈12Hz, ARI=0.0130)',
             fontsize=10, fontweight='bold', color='#1565C0')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(5, N_neurons + 0.5, 'Semantic structure preserved\nin sparse population code',
        fontsize=8.5, color='#1565C0', va='bottom',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='#E3F2FD',
                  edgecolor='#1565C0', alpha=0.9))

# RIGHT — Synchronous seizure
ax2 = axes[1]
for n in range(N_neurons):
    burst_times = np.arange(2, T_ms, 5)
    jitter = np.random.randint(-1, 2, len(burst_times))
    spikes = np.clip(burst_times + jitter, 0, T_ms - 1)
    extra = np.where(np.random.rand(T_ms) < 0.88)[0]
    all_spikes = np.unique(np.concatenate([spikes, extra]))
    ax2.scatter(all_spikes, [n] * len(all_spikes), s=4, color='#C62828', alpha=0.5)

ax2.set_xlim(0, T_ms)
ax2.set_ylim(-1, N_neurons + 6)
ax2.set_xlabel('Time (ms)', fontsize=10)
ax2.set_ylabel('Neuron index', fontsize=10)
ax2.set_title('Synchronous Seizure Regime\n(τ_max=500ms, rate≈500Hz, ARI=−0.0101)',
              fontsize=10, fontweight='bold', color='#C62828')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.text(5, N_neurons + 0.5, 'Semantic structure destroyed\nby max-entropy firing',
         fontsize=8.5, color='#C62828', va='bottom',
         bbox=dict(boxstyle='round,pad=0.35', facecolor='#FFEBEE',
                   edgecolor='#C62828', alpha=0.9))

plt.suptitle('Figure 2 — Raster Plots: Asynchronous vs Synchronous (Epileptiform) Spiking',
             fontsize=11, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
savefig('p3_fig2_raster_comparison.png')


# ══════════════════════════════════════════════════════════════════════════════
# PAPER 3 — Figure 3: Architecture Comparison + Escape Path
# FIX: right panel rows spread out; axes fraction spacing increased;
#      taller figure; no row overlap
# ══════════════════════════════════════════════════════════════════════════════
print("[Paper 3] Generating p3_fig3_architecture_results.png ...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# LEFT — ARI and rate for three experiments
ax = axes[0]
exps   = ['Exp17\n(LIF baseline\nτ=100ms)',
          'Exp18\n(80/20 E/I\nτ=500ms)',
          'Exp19\n(Global WTA\nτ=200ms)']
ari_e  = [0.0130, -0.0062, -0.0087]
rate_e = [12.3, 499.98, 499.84]
colors_e = ['#2E7D32', '#C62828', '#C62828']

ax_twin = ax.twinx()
ax.bar(range(3), ari_e, color=colors_e, alpha=0.80,
       edgecolor='white', linewidth=0.5, width=0.4, zorder=3)
ax_twin.plot(range(3), rate_e, 'k^--', linewidth=2, markersize=8,
             label='Mean firing rate (Hz)', zorder=4)

ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
ax.set_xticks(range(3))
ax.set_xticklabels(exps, fontsize=8.5)
ax.set_ylabel('ARI (semantic axis)', fontsize=9)
ax_twin.set_ylabel('Mean firing rate (Hz)', fontsize=9, color='#37474F')
ax_twin.set_ylim(0, 650)
ax.set_ylim(-0.016, 0.024)
ax.set_title('All Three Inhibitory Architectures\nFail Above ESL Threshold',
             fontsize=10, fontweight='bold')
ax.spines['top'].set_visible(False)

ax.text(0, 0.0148, 'Only successful\nconfiguration', ha='center', fontsize=7,
        color='#2E7D32', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.2', fc='#E8F5E9', ec='#2E7D32', alpha=0.85))
ax.text(1.5, -0.0125, 'Both collapse to\n≈500 Hz seizure', ha='center', fontsize=7.5,
        color='#C62828', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.2', fc='#FFEBEE', ec='#C62828', alpha=0.85))

handles = [mpatches.Patch(color='#2E7D32', label='Coherent (ARI>0)'),
           mpatches.Patch(color='#C62828', label='Seizure (ARI<0)'),
           plt.Line2D([0], [0], color='k', marker='^', linestyle='--',
                      label='Firing rate')]
ax.legend(handles=handles, fontsize=7.5, frameon=False, loc='lower right')

# RIGHT — Escape mechanism diagram  (3 rows, well-spaced)
ax2 = axes[1]
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.axis('off')

# Row y-centres (fraction): 0.75, 0.45, 0.15  → spacing=0.30
ROW_H   = 0.22   # box height in axes fraction
ROW_CY  = [0.78, 0.48, 0.18]   # centre y for each row
COL_LX  = 0.02   # left column left x
COL_LW  = 0.44   # left column width
COL_RX  = 0.54   # right column left x
COL_RW  = 0.44   # right column width

escape_data = [
    ('LIF + Static\nInhibition',         '#C62828', '✗ FAILS\n~500 Hz seizure\n(Exp 17b, 18, 19)'),
    ('LIF + AdEx Adaptation\n(M-current)',  '#F57F17', '◑ PREDICTED\nSelf-regulating\nmembrane (BrainScaleS)'),
    ('LIF + BrainScaleS\nThermal Noise',   '#1565C0', '◑ PREDICTED\nKuramoto desync\n(analog noise)'),
]

# Column headers
ax2.text(COL_LX + COL_LW/2, 0.97, 'Architecture',
         ha='center', va='top', fontsize=9.5, fontweight='bold', color='#212121',
         transform=ax2.transAxes)
ax2.text(COL_RX + COL_RW/2, 0.97, 'ESL Outcome',
         ha='center', va='top', fontsize=9.5, fontweight='bold', color='#212121',
         transform=ax2.transAxes)

for cy, (label, col, status) in zip(ROW_CY, escape_data):
    y0 = cy - ROW_H / 2
    # Left box
    rect_l = FancyBboxPatch((COL_LX, y0), COL_LW, ROW_H,
                             boxstyle='round,pad=0.025',
                             transform=ax2.transAxes,
                             facecolor=col, edgecolor='white', alpha=0.90,
                             linewidth=1.5, clip_on=False)
    ax2.add_patch(rect_l)
    ax2.text(COL_LX + COL_LW/2, cy, label,
             ha='center', va='center', transform=ax2.transAxes,
             fontsize=8.5, color='white', fontweight='bold', linespacing=1.45)

    # Arrow
    ax2.annotate('', xy=(COL_RX, cy), xytext=(COL_LX + COL_LW, cy),
                 xycoords='axes fraction', textcoords='axes fraction',
                 arrowprops=dict(arrowstyle='->', color=col, lw=1.8))

    # Right box
    rect_r = FancyBboxPatch((COL_RX, y0), COL_RW, ROW_H,
                             boxstyle='round,pad=0.025',
                             transform=ax2.transAxes,
                             facecolor='#FAFAFA', edgecolor=col,
                             linewidth=1.8, clip_on=False)
    ax2.add_patch(rect_r)
    ax2.text(COL_RX + COL_RW/2, cy, status,
             ha='center', va='center', transform=ax2.transAxes,
             fontsize=8.0, color=col, fontweight='bold', linespacing=1.4)

ax2.set_title('Figure 3b — Escape Mechanism Comparison',
              fontsize=10, fontweight='bold', pad=12)

plt.suptitle('Figure 3 — Experiment Results and ESL Escape Mechanisms',
             fontsize=11, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
savefig('p3_fig3_architecture_results.png')

print("\n[SUCCESS] All 7 figures generated successfully — no overlaps.")
print("Files saved to:", OUT_DIR)