"""
DDIN v4 — Structured Dhātu Extraction
======================================

Building directly on DDIN v3 (exp03).

What changed from v3:
  - v3: broke symmetry, proved sparsity, saw semantic differentiation
  - v4: NOW we extract STRUCTURE — who does what

Core insight (mindset shift):
  BEFORE: Dhātu = whole-system behavior
  NOW:    Dhātu = localized subgraph + dynamical role

The three steps:
  1. Cluster neurons by: activity patterns + differential response + connectivity
  2. Extract subgraphs as Dhātu candidates
  3. Label each Dhātu by its dynamical role

DDIN layers — where we are:
  Parā      → pre-symbolic ground            [done v1/v2]
  Paśyantī  → coherent attractor states      [done v3]
  Madhyamā  → structured Dhātu subgraphs     [← THIS EXPERIMENT]
  Vaikharī  → output / language              [next]
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # works in Kaggle / headless environments
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False
    print("[Warning] networkx not available. Inter-Dhātu graph will be skipped.")

from scipy.signal import find_peaks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print("="*60)


# ────────────────────────────────────────────────────────────────
# 1.  HETEROGENEOUS LIQUID SYSTEM  (identical to v3)
# ────────────────────────────────────────────────────────────────

class HeterogeneousLiquidSystem(nn.Module):
    """
    Per-neuron alpha (decay) and beta (input sensitivity) break symmetry
    so that neurons can specialise into distinct functional roles.
    """
    def __init__(self, dim=64):
        super().__init__()
        self.W     = nn.Parameter(torch.randn(dim, dim) * 0.05)
        self.alpha = nn.Parameter(torch.rand(dim) * 0.6 + 0.1)   # [0.1, 0.7]
        self.beta  = nn.Parameter(torch.rand(dim) * 0.4 + 0.05)  # [0.05, 0.45]

    def forward(self, x, u, dt=0.1):
        dx = -self.alpha * x + torch.tanh(x @ self.W) + self.beta * u
        return x + dt * dx


# ────────────────────────────────────────────────────────────────
# 2.  INPUT GENERATOR  (identical to v3, plus "ramp")
# ────────────────────────────────────────────────────────────────

def generate_input(seq_len, dim, mode="sine"):
    t = torch.linspace(0, 10, seq_len).to(device)

    if mode == "sine":
        freqs = torch.linspace(0.8, 1.2, dim).to(device)
        return torch.sin(t.unsqueeze(1) * freqs.unsqueeze(0))

    elif mode == "cosine":
        freqs = torch.linspace(0.8, 1.2, dim).to(device)
        return torch.cos(t.unsqueeze(1) * freqs.unsqueeze(0))

    elif mode == "constant":
        levels = torch.linspace(0.2, 0.8, dim).to(device)
        return levels.unsqueeze(0).repeat(seq_len, 1)

    elif mode == "impulse":
        u = torch.zeros(seq_len, dim).to(device)
        for i in range(dim):
            start = 5 + (i % 10) * 2
            u[start:start + 3, i] = 1.0
        return u

    elif mode == "noise":
        return torch.randn(seq_len, dim).to(device) * 0.3

    elif mode == "ramp":
        t_norm = t / t.max()
        return t_norm.unsqueeze(1) * torch.linspace(0.1, 1.0, dim).to(device).unsqueeze(0)

    raise ValueError(f"Unknown mode: {mode}")


# ────────────────────────────────────────────────────────────────
# 3.  TRAINING  (same two-phase protocol as v3)
# ────────────────────────────────────────────────────────────────

dim      = 64
seq_len  = 200
EPOCHS   = 400
PRUNE_START = 200
PRUNE_EVERY = 20
PRUNE_THRESH = 0.02

model     = HeterogeneousLiquidSystem(dim=dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

loss_history = []
synapse_history = []

print("Phase 1: Learning (epochs 0–200)")
print("Phase 2: Pruning  (epochs 200–400)")
print("="*50)

for epoch in range(EPOCHS):
    optimizer.zero_grad()

    x = torch.zeros(1, dim).to(device)
    u = generate_input(seq_len, dim, mode="sine")

    pred_loss      = 0.0
    energy_loss    = 0.0
    smoothness_loss = 0.0

    for t_step in range(seq_len - 1):
        x_next   = model(x,      u[t_step])
        x_future = model(x_next, u[t_step + 1])

        pred_loss       += torch.mean((x_future - x_next.detach()) ** 2)
        energy_loss     += torch.mean(torch.abs(x))
        smoothness_loss += torch.mean((x_next - x) ** 2)
        x = x_next

    if epoch < PRUNE_START:
        loss = pred_loss + 0.1 * smoothness_loss
    else:
        loss = (pred_loss
                + 0.1  * smoothness_loss
                + 0.02 * energy_loss
                + 0.05 * torch.sum(torch.abs(model.W)))

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # hard pruning every PRUNE_EVERY epochs in phase 2
    if epoch >= PRUNE_START and epoch % PRUNE_EVERY == 0:
        with torch.no_grad():
            model.W[torch.abs(model.W) < PRUNE_THRESH] = 0.0

    active = torch.count_nonzero(torch.abs(model.W) > 0.01).item()
    synapse_history.append(active)
    loss_history.append(loss.item())

    if epoch % 50 == 0:
        phase = "Learning" if epoch < PRUNE_START else "Pruning "
        print(f"  Epoch {epoch:3d} ({phase}) | loss={loss.item():.4f} | synapses={active}")

sparsity = 1.0 - synapse_history[-1] / (dim * dim)
print(f"\nFinal sparsity: {sparsity:.1%}   |   Active synapses: {synapse_history[-1]}")


# ────────────────────────────────────────────────────────────────
# 4.  COLLECT TRAJECTORIES FOR ALL MODES
# ────────────────────────────────────────────────────────────────

MODES = ["sine", "cosine", "constant", "impulse", "noise", "ramp"]

def collect_trajectory(mode, steps=300):
    x = torch.zeros(1, dim).to(device)
    u = generate_input(steps, dim, mode=mode)
    states = []
    with torch.no_grad():
        for t_step in range(steps):
            x = model(x, u[t_step])
            states.append(x.cpu().numpy())
    return np.array(states).squeeze()   # (steps, dim)

print("\nCollecting trajectories...")
trajs = {mode: collect_trajectory(mode) for mode in MODES}
for mode, traj in trajs.items():
    print(f"  {mode:10s}  shape={traj.shape}")


# ────────────────────────────────────────────────────────────────
# 5.  PER-NEURON FEATURE VECTORS
# ────────────────────────────────────────────────────────────────

print("\nComputing per-neuron features...")

mean_acts = {mode: np.mean(np.abs(trajs[mode]), axis=0) for mode in MODES}
var_acts  = {mode: np.var(trajs[mode],          axis=0) for mode in MODES}

# 5a. Semantic differentiation: mean |act_A – act_B| across mode pairs
MODE_PAIRS = [
    ("sine",    "cosine"),
    ("sine",    "constant"),
    ("sine",    "impulse"),
    ("cosine",  "impulse"),
    ("constant","noise"),
]
diff_matrix = np.stack([
    np.abs(mean_acts[m1] - mean_acts[m2]) for m1, m2 in MODE_PAIRS
], axis=0)                          # (n_pairs, dim)
mean_diff = np.mean(diff_matrix, axis=0)   # (dim,)

# 5b. Connectivity: in+out strength from weight matrix
W = model.W.detach().cpu().numpy()
out_strength = np.sum(np.abs(W), axis=1)
in_strength  = np.sum(np.abs(W), axis=0)
hub_score    = out_strength + in_strength

# 5c. Temporal periodicity — autocorrelation peak beyond lag 5
def autocorr_peak_strength(signal, lag_start=5, lag_end=60):
    signal = signal - np.mean(signal)
    if np.std(signal) < 1e-6:
        return 0.0
    ac = np.correlate(signal, signal, mode='full')
    ac = ac[len(ac)//2:]
    ac /= (ac[0] + 1e-8)
    segment = ac[lag_start:lag_end]
    if len(segment) == 0:
        return 0.0
    peaks, _ = find_peaks(segment, height=0.2)
    return float(ac[peaks[0] + lag_start]) if len(peaks) > 0 else 0.0

periodicity = np.array([
    autocorr_peak_strength(trajs["sine"][:, i]) for i in range(dim)
])

# 5d. Impulse sensitivity: special — does this neuron fire after an impulse?
impulse_sensitivity = mean_acts["impulse"]

# Assemble feature matrix: (dim, 8)
feature_matrix = np.column_stack([
    mean_acts["sine"],        # col 0: general excitability
    mean_acts["cosine"],      # col 1: cosine response
    impulse_sensitivity,      # col 2: impulse response
    var_acts["sine"],         # col 3: temporal variability
    var_acts["impulse"],      # col 4: post-impulse variability
    mean_diff,                # col 5: semantic differentiation
    hub_score,                # col 6: network connectivity
    periodicity,              # col 7: temporal periodicity
])
print(f"Feature matrix: {feature_matrix.shape}  →  ({dim} neurons × 8 features)")

# Normalize
scaler = StandardScaler()
F_norm = scaler.fit_transform(feature_matrix)


# ────────────────────────────────────────────────────────────────
# 6.  CLUSTER NEURONS → DHĀTU CANDIDATES
# ────────────────────────────────────────────────────────────────

print("\nClustering neurons into Dhātu candidates...")

N_DHATU = 5   # 5 dhātu — matches the 5 articulation loci of Sanskrit

kmeans = KMeans(n_clusters=N_DHATU, random_state=42, n_init=30, max_iter=500)
cluster_labels = kmeans.fit_predict(F_norm)

cluster_sizes = [int(np.sum(cluster_labels == k)) for k in range(N_DHATU)]
print(f"Cluster sizes: {cluster_sizes}")


# ────────────────────────────────────────────────────────────────
# 7.  CHARACTERISE EACH DHĀTU  (dynamical role)
# ────────────────────────────────────────────────────────────────

# Thresholds (relative to global means)
GLOBAL_MEAN_DIFF       = np.mean(mean_diff)
GLOBAL_MEAN_HUB        = np.mean(hub_score)
GLOBAL_MEAN_PERIODICITY = np.mean(periodicity)
GLOBAL_MEAN_IMPULSE    = np.mean(impulse_sensitivity)

def classify_role(mean_sine, mean_impulse, mean_var, mean_d, mean_per, mean_hub):
    """
    Heuristic role classifier.
    Returns (role_long, role_short)
    """
    if mean_impulse > GLOBAL_MEAN_IMPULSE * 1.3 and mean_var < 0.05:
        return "sthā  (stable / fixed-point attractor)", "STABLE"
    elif mean_per > GLOBAL_MEAN_PERIODICITY * 1.4:
        return "gam   (motion / periodic oscillation)",  "MOTION"
    elif mean_d > GLOBAL_MEAN_DIFF * 1.5:
        return "śru   (discriminating / semantic)",       "SEMANTIC"
    elif mean_hub > GLOBAL_MEAN_HUB * 1.5:
        return "sam   (integrating / hub node)",          "HUB"
    else:
        return "sandhi (transitional / mixed mode)",      "TRANSIT"

print("\nDhātu characterisation:")
print("-"*60)

dhatu_profiles = []
for k in range(N_DHATU):
    mask    = cluster_labels == k
    neurons = np.where(mask)[0]

    ms  = float(np.mean(mean_acts["sine"][mask]))
    mc  = float(np.mean(mean_acts["cosine"][mask]))
    mi  = float(np.mean(impulse_sensitivity[mask]))
    mv  = float(np.mean(var_acts["sine"][mask]))
    md  = float(np.mean(mean_diff[mask]))
    mp  = float(np.mean(periodicity[mask]))
    mh  = float(np.mean(hub_score[mask]))

    role_long, role_short = classify_role(ms, mi, mv, md, mp, mh)

    profile = dict(
        id=k, neurons=neurons.tolist(), n_neurons=len(neurons),
        role_long=role_long, role_short=role_short,
        mean_sine=ms, mean_cosine=mc, mean_impulse=mi,
        mean_var=mv, mean_diff=md, periodicity=mp, hub=mh,
    )
    dhatu_profiles.append(profile)

    print(f"\n  Dhātu {k}  →  {role_long}")
    print(f"    neurons      : {neurons.tolist()}")
    print(f"    size         : {len(neurons)}")
    print(f"    sine_act     : {ms:.4f}")
    print(f"    impulse_act  : {mi:.4f}")
    print(f"    diff_score   : {md:.4f}  (global mean={GLOBAL_MEAN_DIFF:.4f})")
    print(f"    periodicity  : {mp:.4f}  (global mean={GLOBAL_MEAN_PERIODICITY:.4f})")
    print(f"    hub_score    : {mh:.4f}  (global mean={GLOBAL_MEAN_HUB:.4f})")


# ────────────────────────────────────────────────────────────────
# 8.  EXTRACT SUBGRAPHS (F_unfold → Dhātu subgraphs)
# ────────────────────────────────────────────────────────────────

EDGE_THRESHOLD = 0.02

def extract_subgraph_info(cluster_id, W, labels, threshold=EDGE_THRESHOLD):
    """
    For a Dhātu cluster:
      - internal edges: within the cluster
      - external edges: to other clusters (cross-Dhātu communication)
    """
    neurons = np.where(labels == cluster_id)[0]
    internal, external = [], []

    for i in neurons:
        for j in range(len(labels)):
            if abs(W[i, j]) > threshold:
                weight = float(W[i, j])
                if j in neurons:
                    internal.append((int(i), int(j), weight))
                else:
                    external.append((cluster_id, int(labels[j]), weight))

    return internal, external

subgraph_data = {}
inter_weights = np.zeros((N_DHATU, N_DHATU))

print("\nSubgraph extraction:")
for k in range(N_DHATU):
    internal, external = extract_subgraph_info(k, W, cluster_labels)
    subgraph_data[k] = dict(internal=internal, external=external)

    # accumulate inter-Dhātu weights
    for (src, tgt, wt) in external:
        if src != tgt:
            inter_weights[src, tgt] += abs(wt)

    print(f"  Dhātu {k}: {len(internal):3d} internal edges | "
          f"{len(external):3d} external connections")

print("\nInter-Dhātu communication matrix:")
print("     " + "  ".join(f" D{j}" for j in range(N_DHATU)))
for i in range(N_DHATU):
    row = "  ".join(f"{inter_weights[i,j]:4.2f}" for j in range(N_DHATU))
    role = dhatu_profiles[i]['role_short'][:6]
    print(f"  D{i} [{role}]:  {row}")


# ────────────────────────────────────────────────────────────────
# 9.  PCA (2D embedding for visualisation)
# ────────────────────────────────────────────────────────────────

pca = PCA(n_components=2)
F_2d = pca.fit_transform(F_norm)


# ────────────────────────────────────────────────────────────────
# 10.  VISUALISATION
# ────────────────────────────────────────────────────────────────

DHATU_COLORS = ['#ff6b6b', '#ffd93d', '#6bcb77', '#4d96ff', '#c77dff']
BAR_COLORS   = [DHATU_COLORS[cluster_labels[i]] for i in range(dim)]

fig = plt.figure(figsize=(22, 15))
fig.patch.set_facecolor('#0d0d0d')
fig.suptitle(
    "DDIN v4  —  Dhātu Extraction: Structured Intelligence Emergence\n"
    "Madhyamā layer: from attractor states → localized functional subgraphs",
    fontsize=13, color='white', fontweight='bold', y=0.98
)

def dark_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor('#111111')
    ax.set_title(title, color='white', fontsize=8.5, pad=4)
    if xlabel: ax.set_xlabel(xlabel, color='#888888', fontsize=7)
    if ylabel: ax.set_ylabel(ylabel, color='#888888', fontsize=7)
    ax.tick_params(colors='#888888', labelsize=6)
    for spine in ax.spines.values():
        spine.set_color('#333333')


# ── (1,1) Neuron Feature Space (PCA) ─────────────────────────────
ax1 = fig.add_subplot(3, 4, 1)
dark_ax(ax1, "Neuron Feature Space (PCA)\ncolored by Dhātu cluster")
for k in range(N_DHATU):
    m = cluster_labels == k
    ax1.scatter(F_2d[m, 0], F_2d[m, 1],
                c=DHATU_COLORS[k], s=55, alpha=0.9, zorder=3,
                label=f"D{k}: {dhatu_profiles[k]['role_short']}")
ax1.legend(fontsize=6, facecolor='#222222', labelcolor='white',
           framealpha=0.8, loc='best')

# ── (1,2) Weight Matrix sorted by Dhātu ──────────────────────────
ax2 = fig.add_subplot(3, 4, 2)
dark_ax(ax2, "Weight Matrix (sorted by Dhātu)\nbright = strong connection")
sort_order = np.argsort(cluster_labels)
W_sorted   = np.abs(W[np.ix_(sort_order, sort_order)])
ax2.imshow(W_sorted, cmap='hot', vmin=0, vmax=0.15, aspect='auto')
# cluster boundary lines
cum = 0
for k in range(N_DHATU):
    sz = int(np.sum(cluster_labels == k))
    ax2.axhline(cum - 0.5, color=DHATU_COLORS[k], lw=1.2, alpha=0.9)
    ax2.axvline(cum - 0.5, color=DHATU_COLORS[k], lw=1.2, alpha=0.9)
    cum += sz
ax2.axis('off')

# ── (1,3) Semantic differentiation per neuron ────────────────────
ax3 = fig.add_subplot(3, 4, 3)
dark_ax(ax3, "Semantic Differentiation\n|act_sine − act_cosine| per neuron",
        "Neuron", "Δ activity")
ax3.bar(range(dim), mean_diff, color=BAR_COLORS, alpha=0.85)
ax3.axhline(GLOBAL_MEAN_DIFF, color='white', lw=0.8, linestyle='--', alpha=0.5)

# ── (1,4) Hub score per neuron ───────────────────────────────────
ax4 = fig.add_subplot(3, 4, 4)
dark_ax(ax4, "Hub Score (in + out strength)\ncolored by Dhātu", "Neuron", "Σ |W|")
ax4.bar(range(dim), hub_score, color=BAR_COLORS, alpha=0.85)
ax4.axhline(GLOBAL_MEAN_HUB, color='white', lw=0.8, linestyle='--', alpha=0.5)

# ── (2,1–2,5) Phase-space per Dhātu ─────────────────────────────
for k in range(N_DHATU):
    ax = fig.add_subplot(3, 4, 5 + k)
    role = dhatu_profiles[k]['role_short']
    dark_ax(ax, f"Dhātu {k}  [{role}]\nphase space (2 neurons)")

    neurons_k = np.where(cluster_labels == k)[0]
    if len(neurons_k) >= 2:
        n0, n1 = neurons_k[0], neurons_k[1]
        ax.plot(trajs["sine"][:, n0],    trajs["sine"][:, n1],
                '-', color='cyan',    lw=0.7, alpha=0.7, label='sine')
        ax.plot(trajs["cosine"][:, n0],  trajs["cosine"][:, n1],
                '-', color='magenta', lw=0.7, alpha=0.7, label='cosine')
        ax.plot(trajs["impulse"][:, n0], trajs["impulse"][:, n1],
                '-', color='yellow',  lw=0.7, alpha=0.7, label='impulse')
    ax.set_xlabel(f"neuron {neurons_k[0] if len(neurons_k)>0 else '?'}",
                  color='#888888', fontsize=6)
    ax.set_ylabel(f"neuron {neurons_k[1] if len(neurons_k)>1 else '?'}",
                  color='#888888', fontsize=6)
    ax.tick_params(colors='#888888', labelsize=5)
    for spine in ax.spines.values():
        spine.set_color(DHATU_COLORS[k])
        spine.set_linewidth(1.5)
    if k == 0:
        ax.legend(fontsize=5, facecolor='#222222', labelcolor='white', loc='best')

# ── (3,1) Inter-Dhātu communication graph ────────────────────────
ax10 = fig.add_subplot(3, 4, 10)
dark_ax(ax10, "Inter-Dhātu Communication\n(Madhyamā grammar sketch)")

if HAS_NX:
    import networkx as nx
    G_inter = nx.DiGraph()
    for k in range(N_DHATU):
        G_inter.add_node(k)
    for i in range(N_DHATU):
        for j in range(N_DHATU):
            if i != j and inter_weights[i, j] > 0.05:
                G_inter.add_edge(i, j, weight=float(inter_weights[i, j]))

    pos = nx.circular_layout(G_inter)
    nx.draw_networkx_nodes(G_inter, pos, ax=ax10,
                           node_color=DHATU_COLORS[:N_DHATU],
                           node_size=900, alpha=0.95)
    labels_nx = {k: f"D{k}\n{dhatu_profiles[k]['role_short'][:5]}"
                 for k in range(N_DHATU)}
    nx.draw_networkx_labels(G_inter, pos, labels=labels_nx, ax=ax10,
                            font_color='white', font_size=6, font_weight='bold')
    if G_inter.edges():
        ews = [G_inter[u][v]['weight'] for u, v in G_inter.edges()]
        max_ew = max(ews) if ews else 1.0
        nx.draw_networkx_edges(G_inter, pos, ax=ax10,
                               width=[3.5 * w / max_ew for w in ews],
                               edge_color='#cccccc', alpha=0.6,
                               arrows=True, arrowsize=12,
                               connectionstyle='arc3,rad=0.15')
    ax10.axis('off')
else:
    # Fallback: heat-map of the inter_weights matrix
    ax10.imshow(inter_weights, cmap='hot', aspect='auto')
    ax10.set_xticks(range(N_DHATU))
    ax10.set_yticks(range(N_DHATU))

# ── (3,2) Training loss + synapse count ──────────────────────────
ax11 = fig.add_subplot(3, 4, 11)
dark_ax(ax11, "Training Loss", "Epoch", "Loss")
ax11.plot(loss_history, color='#88ccff', lw=0.8, alpha=0.9)
ax11.axvline(x=PRUNE_START, color='yellow', lw=1, ls='--', alpha=0.7,
             label='Pruning start')
ax11.legend(fontsize=6, facecolor='#222222', labelcolor='white')

ax11b = ax11.twinx()
ax11b.plot(synapse_history, color='#ff9944', lw=0.8, alpha=0.7)
ax11b.set_ylabel("Active synapses", color='#ff9944', fontsize=6)
ax11b.tick_params(colors='#ff9944', labelsize=5)

# ── (3,3) Dhātu profile bar chart ────────────────────────────────
ax12 = fig.add_subplot(3, 4, 12)
dark_ax(ax12, "Dhātu Feature Profiles\n(normalised)", "", "value")
metrics     = ['sine', 'impulse', 'diff', 'period', 'hub/10']
metric_vals = lambda p: [
    p['mean_sine'], p['mean_impulse'], p['mean_diff'],
    p['periodicity'], p['hub'] / 10.0
]
x     = np.arange(len(metrics))
width = 0.15
for k, p in enumerate(dhatu_profiles):
    ax12.bar(x + k * width, metric_vals(p), width=width,
             color=DHATU_COLORS[k], alpha=0.85, label=f"D{k}")
ax12.set_xticks(x + width * 2)
ax12.set_xticklabels(metrics, rotation=30, fontsize=6, color='#888888')
ax12.legend(fontsize=6, facecolor='#222222', labelcolor='white')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('ddin_v4_dhatu_extraction.png', dpi=150, bbox_inches='tight',
            facecolor='#0d0d0d')
plt.show()
print("\nSaved: ddin_v4_dhatu_extraction.png")


# ────────────────────────────────────────────────────────────────
# 11.  FINAL SUMMARY
# ────────────────────────────────────────────────────────────────

total_internal = sum(len(subgraph_data[k]['internal']) for k in range(N_DHATU))
total_external = sum(len(subgraph_data[k]['external']) for k in range(N_DHATU))
final_synapses = synapse_history[-1]
final_sparsity = 1.0 - final_synapses / (dim * dim)

print("\n" + "="*60)
print("DDIN v4  —  DHĀTU EXTRACTION  —  FINAL SUMMARY")
print("="*60)

print(f"""
Network
  Neurons  : {dim}
  Dhātu    : {N_DHATU}
  Sparsity : {final_sparsity:.1%}
  Synapses : {final_synapses}

Structure
  Internal edges (within Dhātu)   : {total_internal}
  External connections (cross-Dhātu): {total_external}
""")

print("Dhātu Roles:")
for p in dhatu_profiles:
    print(f"  D{p['id']}  [{p['role_short']:8s}]  {p['n_neurons']:2d} neurons  "
          f"→  {p['role_long']}")

print("""
Where you stand in the DDIN stack
  ✔  Parā      : pre-symbolic ground          (v1 / v2)
  ✔  Paśyantī  : coherent attractor states    (v3)
  ✔  Madhyamā  : structured Dhātu subgraphs   ← YOU ARE HERE  (v4)
  ◻  Vaikharī  : output / language            (next)

The Madhyamā layer is now real:
  - Each Dhātu is a localized subgraph with a distinct dynamical role.
  - The inter-Dhātu communication matrix above is the first sketch
    of the DDIN's internal grammar.
  - This is no longer a single attractor field.
    It is a network of networks — an organism, not a simulation.

Next milestone:
  Build the Vaikharī projection head:
    Dhātu activation pattern → symbolic label / token
  This closes the loop from dynamics to language.
""")
print("="*60)
