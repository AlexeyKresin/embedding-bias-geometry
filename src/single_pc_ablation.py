import numpy as np
import matplotlib.pyplot as plt

# Example data (replace with your actual values)
pc_indices = np.arange(1, 11)

direct_bias = np.array([0.00, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
weat = np.array([1.73, 1.66, 1.73, 1.72, 1.74, 1.74, 1.75, 1.74, 1.73, 1.74])

mean_displacement = np.array([0.00, 0.01, 0.01, 0.005, 0.003, 0.004, 0.004, 0.003, 0.004, 0.003])
neighbor_stability = np.array([0.98, 0.94, 0.96, 0.99, 0.995, 0.99, 0.992, 0.993, 0.991, 0.998])


# ✅ Normalization
def normalize(x):
    if np.max(x) == np.min(x):
        return np.zeros_like(x)
    return (x - np.min(x)) / (np.max(x) - np.min(x))


direct_bias_n = normalize(direct_bias)
weat_n = normalize(weat)
mean_disp_n = normalize(mean_displacement)
stability_n = normalize(neighbor_stability)


# 🎨 Colors (distinct per metric)
colors = {
    "direct_bias": "#1f77b4",      # blue
    "weat": "#ff7f0e",             # orange
    "mean_disp": "#2ca02c",        # green
    "stability": "#d62728"         # red
}


# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (a) Bias metrics
axes[0].plot(pc_indices, direct_bias_n, marker='o',
             color=colors["direct_bias"], label='Direct Bias')

axes[0].plot(pc_indices, weat_n, marker='o',
             color=colors["weat"], label='WEAT')

axes[0].set_title('(a) Bias metrics (normalized)')
axes[0].set_xlabel('PC index')
axes[0].set_ylabel('Normalized value')
axes[0].grid(True, alpha=0.3)
axes[0].legend()


# (b) Geometry metrics
axes[1].plot(pc_indices, mean_disp_n, marker='o',
             color=colors["mean_disp"], label='Mean displacement')

axes[1].plot(pc_indices, stability_n, marker='o',
             color=colors["stability"], label='Neighbor stability@10')

axes[1].set_title('(b) Geometry metrics (normalized)')
axes[1].set_xlabel('PC index')
axes[1].set_ylabel('Normalized value')
axes[1].grid(True, alpha=0.3)
axes[1].legend()


plt.tight_layout()

# ✅ Save as high-quality PNG
plt.savefig("single_pc_ablation_normalized.png", dpi=300, bbox_inches='tight')

plt.show()