import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# -----------------------------
# Configuration
# -----------------------------
EMBEDDING_FILE = "../data/glove.6B.300d.txt"   # adjust if needed
OUTPUT_CSV = "../outputs/bias_spectrum.csv"
OUTPUT_FIG = "../figures/bias_spectrum.png"

# Definitional gender pairs
DEFINITIONAL_PAIRS = [
    ("man", "woman"),
    ("he", "she"),
    ("father", "mother"),
    ("king", "queen"),
    ("boy", "girl"),
    ("brother", "sister"),
    ("son", "daughter"),
    ("husband", "wife"),
    ("male", "female"),
    ("uncle", "aunt"),
]

# Neutral words for projection analysis
NEUTRAL_WORDS = [
    "doctor", "nurse", "teacher", "engineer", "scientist", "manager",
    "lawyer", "artist", "writer", "student", "parent", "child",
    "leader", "worker", "designer", "researcher", "assistant", "clerk",
    "chef", "pilot", "driver", "farmer", "journalist", "professor"
]


# -----------------------------
# Utilities
# -----------------------------
def load_glove_embeddings(path):
    embeddings = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            vec = np.asarray(parts[1:], dtype=np.float32)
            embeddings[word] = vec
    return embeddings


def normalize(v):
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


# -----------------------------
# Main analysis
# -----------------------------
def main():
    print("Loading embeddings...")
    embeddings = load_glove_embeddings(EMBEDDING_FILE)
    print(f"Loaded {len(embeddings)} embeddings.")

    # Keep only definitional pairs that exist in vocab
    valid_pairs = []
    pair_diff_vectors = []

    for a, b in DEFINITIONAL_PAIRS:
        if a in embeddings and b in embeddings:
            diff = embeddings[a] - embeddings[b]
            pair_diff_vectors.append(diff)
            valid_pairs.append((a, b))
        else:
            print(f"Skipping missing pair: ({a}, {b})")

    if len(pair_diff_vectors) < 2:
        raise ValueError("Not enough valid definitional pairs found for PCA.")

    X = np.vstack(pair_diff_vectors)

    # PCA on definitional pair differences
    pca = PCA()
    pca.fit(X)

    components = pca.components_                     # shape: [num_pcs, dim]
    explained = pca.explained_variance_ratio_       # shape: [num_pcs]

    # Pair projection strength per PC
    # Mean absolute projection of definitional differences onto each PC
    pair_proj_strength = []
    for i, pc in enumerate(components):
        vals = [abs(np.dot(diff, pc)) for diff in pair_diff_vectors]
        pair_proj_strength.append(np.mean(vals))

    # Neutral word projection strength per PC
    valid_neutral_words = [w for w in NEUTRAL_WORDS if w in embeddings]
    if len(valid_neutral_words) == 0:
        raise ValueError("No neutral words found in embeddings.")

    neutral_proj_strength = []
    for i, pc in enumerate(components):
        vals = [abs(np.dot(embeddings[w], pc)) for w in valid_neutral_words]
        neutral_proj_strength.append(np.mean(vals))

    # Save results
    results = pd.DataFrame({
        "pc_index": np.arange(1, len(components) + 1),
        "explained_variance_ratio": explained,
        "pair_projection_mean_abs": pair_proj_strength,
        "neutral_projection_mean_abs": neutral_proj_strength,
    })

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_FIG), exist_ok=True)

    results.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved spectrum table to {OUTPUT_CSV}")

    # Plot top PCs only for readability
    top_k = min(10, len(results))
    plot_df = results.iloc[:top_k]

    fig, axs = plt.subplots(3, 1, figsize=(7, 10), sharex=True)

    axs[0].bar(plot_df["pc_index"], plot_df["explained_variance_ratio"])
    axs[0].set_ylabel("Variance ratio")
    axs[0].set_title("Gender Bias Spectrum: Variance Explained by Each PC")
    axs[0].grid(True, axis="y", alpha=0.3)

    axs[1].bar(plot_df["pc_index"], plot_df["pair_projection_mean_abs"])
    axs[1].set_ylabel("Mean |projection|")
    axs[1].set_title("Alignment of Definitional Pair Differences with Each PC")
    axs[1].grid(True, axis="y", alpha=0.3)

    axs[2].bar(plot_df["pc_index"], plot_df["neutral_projection_mean_abs"])
    axs[2].set_ylabel("Mean |projection|")
    axs[2].set_xlabel("Principal component index")
    axs[2].set_title("Average Neutral-Word Projection onto Each Gender PC")
    axs[2].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_FIG, dpi=300, bbox_inches="tight")
    plt.show()

    print("\nTop PCs summary:")
    print(results.head(10).round(4))


if __name__ == "__main__":
    main()