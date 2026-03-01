import os
import csv
import matplotlib.pyplot as plt

CSV_PATH = os.path.join("outputs", "pc_removal_sweep_results.csv")
OUT_DIR = os.path.join("figures")
os.makedirs(OUT_DIR, exist_ok=True)

def read_rows(path):
    rows = []
    with open(path, "r", encoding="utf8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "k": int(r["k"]),
                "direct_bias": float(r["direct_bias"]),
                "mean_displacement": float(r["mean_displacement"]),
                "nn_stability_at10": float(r["nn_stability_at10"]),
            })
    rows.sort(key=lambda x: x["k"])
    return rows

rows = read_rows(CSV_PATH)
k = [r["k"] for r in rows]
bias = [r["direct_bias"] for r in rows]
disp = [r["mean_displacement"] for r in rows]
stab = [r["nn_stability_at10"] for r in rows]

# Plot 1: Direct bias vs k
plt.figure()
plt.plot(k, bias, marker="o")
plt.xlabel("k (number of gender PCs removed)")
plt.ylabel("Direct bias (mean |cos(w, g)|)")
plt.title("Direct bias vs k (PC removal)")
plt.grid(True, alpha=0.3)
p1 = os.path.join(OUT_DIR, "bias_vs_k.png")
plt.savefig(p1, dpi=200, bbox_inches="tight")
plt.close()

# Plot 2: Mean displacement vs k
plt.figure()
plt.plot(k, disp, marker="o")
plt.xlabel("k (number of gender PCs removed)")
plt.ylabel("Mean displacement (1 - cos(w, w'))")
plt.title("Semantic displacement vs k (PC removal)")
plt.grid(True, alpha=0.3)
p2 = os.path.join(OUT_DIR, "displacement_vs_k.png")
plt.savefig(p2, dpi=200, bbox_inches="tight")
plt.close()

# Plot 3: Neighbor stability vs k
plt.figure()
plt.plot(k, stab, marker="o")
plt.xlabel("k (number of gender PCs removed)")
plt.ylabel("Neighbor stability@10 (mean overlap)")
plt.title("Neighbor stability vs k (PC removal)")
plt.grid(True, alpha=0.3)
p3 = os.path.join(OUT_DIR, "nn_stability_vs_k.png")
plt.savefig(p3, dpi=200, bbox_inches="tight")
plt.close()

# Plot 4 (key): Bias vs displacement
plt.figure()
plt.plot(disp, bias, marker="o")
for i, kk in enumerate(k):
    plt.annotate(str(kk), (disp[i], bias[i]), textcoords="offset points", xytext=(5, 5))
plt.xlabel("Mean displacement (1 - cos(w, w'))")
plt.ylabel("Direct bias (mean |cos(w, g)|)")
plt.title("Bias vs displacement (trade-off curve)")
plt.grid(True, alpha=0.3)
p4 = os.path.join(OUT_DIR, "bias_vs_displacement.png")
plt.savefig(p4, dpi=200, bbox_inches="tight")
plt.close()

print("Saved plots:")
print(" -", p1)
print(" -", p2)
print(" -", p3)
print(" -", p4)