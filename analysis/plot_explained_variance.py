import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# Load explained variance data
# =========================================================
# Assumes explained_variance_ratio.npy is in the same folder
evr = np.load("../outputs/multi_pc_ablation/explained_variance_ratio.npy")

# Basic validation
if evr.ndim != 1:
    raise ValueError("explained_variance_ratio.npy must contain a 1D array.")

if len(evr) == 0:
    raise ValueError("explained_variance_ratio.npy is empty.")

pc_indices = np.arange(1, len(evr) + 1)
cumulative_evr = np.cumsum(evr)

# =========================================================
# Optional focus: show only the first N components clearly
# while still computing cumulative variance over all PCs
# =========================================================
max_display = min(15, len(evr))   # adjust if you want more PCs shown
pc_display = pc_indices[:max_display]
evr_display = evr[:max_display]
cum_display = cumulative_evr[:max_display]

# =========================================================
# Plot styling
# =========================================================
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11
})

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

# =========================================================
# (a) Explained variance ratio per component
# =========================================================
axes[0].bar(pc_display, evr_display, alpha=0.85, width=0.7)
axes[0].plot(pc_display, evr_display, marker="o", linewidth=1.8)

axes[0].set_title("(a) Explained variance ratio")
axes[0].set_xlabel("Principal component index")
axes[0].set_ylabel("Explained variance ratio")
axes[0].set_xticks(pc_display)
axes[0].grid(True, axis="y", alpha=0.3)

# Mark the first few components to emphasize low-rank structure
highlight_n = min(3, len(pc_display))
for i in range(highlight_n):
    axes[0].annotate(
        f"PC{i+1}",
        xy=(pc_display[i], evr_display[i]),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        fontsize=10
    )

# =========================================================
# (b) Cumulative explained variance
# =========================================================
axes[1].plot(pc_display, cum_display, marker="o", linewidth=2.2)
axes[1].set_title("(b) Cumulative explained variance")
axes[1].set_xlabel("Principal component index")
axes[1].set_ylabel("Cumulative explained variance")
axes[1].set_xticks(pc_display)
axes[1].set_ylim(0, 1.05)
axes[1].grid(True, alpha=0.3)

# Optional reference lines
thresholds = [0.8, 0.9, 0.95]
for t in thresholds:
    axes[1].axhline(t, linestyle="--", linewidth=1.0, alpha=0.6)

# Annotate the first component count reaching 90%, if it exists
above_90 = np.where(cumulative_evr >= 0.9)[0]
if len(above_90) > 0:
    idx_90 = int(above_90[0])
    if idx_90 < max_display:
        axes[1].annotate(
            f"90% by PC{idx_90 + 1}",
            xy=(pc_indices[idx_90], cumulative_evr[idx_90]),
            xytext=(10, -20),
            textcoords="offset points",
            fontsize=10
        )

# =========================================================
# Overall figure title
# =========================================================
fig.suptitle(
    "Explained Variance Spectrum of the Gender Subspace PCA",
    fontsize=15,
    y=1.02
)

# Tight layout and save
plt.tight_layout()
plt.savefig("explained_variance_spectrum.png", dpi=300, bbox_inches="tight")
plt.show()