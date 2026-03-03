import pandas as pd
import matplotlib.pyplot as plt

pc_file = "outputs/pc_removal_sweep_results.csv"
weat_file = "outputs/weat_pc_sweep_results.csv"

pc = pd.read_csv(pc_file)
weat = pd.read_csv(weat_file)

fig, axs = plt.subplots(3, 1, figsize=(6, 10), sharex=True)

# Direct bias
axs[0].plot(pc["k"], pc["direct_bias"], marker="o")
axs[0].set_ylabel("Direct bias")
axs[0].set_title("Projection bias vs k")
axs[0].grid(True)

# WEAT
axs[1].plot(weat["k"], weat["weat_effect_size"], marker="o")
axs[1].set_ylabel("WEAT (Cohen's d)")
axs[1].set_title("Association bias vs k")
axs[1].grid(True)

# Neighbor stability
axs[2].plot(pc["k"], pc["nn_stability_at10"], marker="o")
axs[2].set_ylabel("Neighbor stability @10")
axs[2].set_xlabel("k (number of gender PCs removed)")
axs[2].set_title("Semantic stability vs k")
axs[2].grid(True)

plt.tight_layout()
plt.savefig("figures/combined_bias_analysis.png", dpi=300)
plt.show()