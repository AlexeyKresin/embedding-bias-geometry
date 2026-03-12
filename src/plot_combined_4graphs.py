import pandas as pd
import matplotlib.pyplot as plt

# Input files
pc_file = "../outputs/pc_removal_sweep_results.csv"
weat_file = "../outputs/weat_pc_sweep_results.csv"

# Load data
pc = pd.read_csv(pc_file)
weat = pd.read_csv(weat_file)

# General style for poster readability
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11
})

fig, axs = plt.subplots(4, 1, figsize=(7, 13), sharex=True)

# 1. Direct Bias
axs[0].plot(pc["k"], pc["direct_bias"], marker="o", linewidth=2.2, markersize=8)
axs[0].set_ylabel("Direct Bias")
axs[0].set_title("Direct Bias vs Removed Gender Components")
axs[0].grid(True, alpha=0.3)

# 2. WEAT
axs[1].plot(weat["k"], weat["weat_effect_size"], marker="o", linewidth=2.2, markersize=8)
axs[1].set_ylabel("WEAT (Cohen's d)")
axs[1].set_title("WEAT Association Bias vs Removed Components")
axs[1].grid(True, alpha=0.3)

# 3. Mean Displacement
axs[2].plot(pc["k"], pc["mean_displacement"], marker="o", linewidth=2.2, markersize=8)
axs[2].set_ylabel("Mean Displacement")
axs[2].set_title("Mean Vector Displacement vs Removed Components")
axs[2].grid(True, alpha=0.3)

# 4. Neighbor Stability
axs[3].plot(pc["k"], pc["nn_stability_at10"], marker="o", linewidth=2.2, markersize=8)
axs[3].set_ylabel("Neighbor Overlap@10")
axs[3].set_xlabel("Number of gender principal components removed (k)")
axs[3].set_title("Semantic Stability vs Removed Components")
axs[3].grid(True, alpha=0.3)

# Show only tested k values
axs[3].set_xticks(pc["k"])

# Overall title
fig.suptitle(
    "Bias and Geometry Respond Differently to Progressive Debiasing",
    fontsize=17,
    y=0.995
)

plt.tight_layout(rect=[0, 0, 1, 0.985])

# Save figure
plt.savefig("../figures/combined_bias_analysis_spire_4panel.png", dpi=300, bbox_inches="tight")
plt.show()