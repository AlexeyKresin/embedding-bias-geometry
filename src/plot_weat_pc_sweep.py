import os
import csv
import matplotlib.pyplot as plt

CSV_PATH = os.path.join("outputs", "weat_pc_sweep_results.csv")
OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

k = []
weat = []

with open(CSV_PATH, "r", encoding="utf8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        k.append(int(r["k"]))
        weat.append(float(r["weat_effect_size"]))

plt.figure()
plt.plot(k, weat, marker="o")
plt.xlabel("k (number of gender PCs removed)")
plt.ylabel("WEAT effect size (Cohen's d)")
plt.title("WEAT under k-PC removal (GloVe 300d)")
plt.grid(True, alpha=0.3)

out = os.path.join(OUT_DIR, "weat_vs_k.png")
plt.savefig(out, dpi=200, bbox_inches="tight")
plt.close()

print("Saved:", out)