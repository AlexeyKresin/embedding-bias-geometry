import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# -----------------------------------
# Config
# -----------------------------------
EMBEDDING_FILE = "../data/glove.6B.300d.txt"   # adjust if needed
OUTPUT_FIG = "../figures/gender_bias_prestige_jobs.png"

WORDS = [
    "he", "she",
    "man", "woman",
    "doctor", "nurse",
    "engineer", "teacher",
    "ceo", "director"
]

COLORS = {
    "male": "#1f77b4",
    "female": "#d62728",
    "neutral": "#7f7f7f"
}

# Keep this simple and intuitive for the poster
MALE_ASSOCIATED = {"he", "man", "engineer", "ceo", "director"}
FEMALE_ASSOCIATED = {"she", "woman", "nurse", "teacher"}
NEUTRAL_OR_MIXED = {"doctor"}

PAIR_LINES = [
    ("doctor", "nurse"),
    ("engineer", "teacher"),
    ("ceo", "director"),
]

# -----------------------------------
# Utilities
# -----------------------------------
def load_glove_subset(path, target_words):
    target_words = set(target_words)
    embeddings = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            if word in target_words:
                embeddings[word] = np.asarray(parts[1:], dtype=np.float32)
    return embeddings


def word_color(word):
    if word in MALE_ASSOCIATED:
        return COLORS["male"]
    if word in FEMALE_ASSOCIATED:
        return COLORS["female"]
    return COLORS["neutral"]


def get_label_offset(word, x, y, xmin, xmax, ymin, ymax):
    dx = 0.06
    dy = 0.06
    ha = "left"
    va = "bottom"

    if x > xmax - 0.45:
        dx = -0.08
        ha = "right"

    if y > ymax - 0.45:
        dy = -0.08
        va = "top"

    if x < xmin + 0.35:
        dx = 0.08
        ha = "left"

    if y < ymin + 0.35:
        dy = 0.08
        va = "bottom"

    return dx, dy, ha, va


# -----------------------------------
# Main
# -----------------------------------
def main():
    print("Loading embeddings...")
    embeddings = load_glove_subset(EMBEDDING_FILE, WORDS)

    found_words = [w for w in WORDS if w in embeddings]
    missing_words = [w for w in WORDS if w not in embeddings]

    print(f"Found {len(found_words)} / {len(WORDS)} words")
    if missing_words:
        print("Missing:", missing_words)

    if len(found_words) < 6:
        raise ValueError("Not enough words found for a useful plot.")

    X = np.vstack([embeddings[w] for w in found_words])

    # PCA projection for visualization only
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    coords = {w: X_2d[i] for i, w in enumerate(found_words)}

    # Gender arrow based on female -> male anchor words
    female_anchor_words = [w for w in ["she", "woman"] if w in coords]
    male_anchor_words = [w for w in ["he", "man"] if w in coords]

    if female_anchor_words and male_anchor_words:
        female_center = np.mean([coords[w] for w in female_anchor_words], axis=0)
        male_center = np.mean([coords[w] for w in male_anchor_words], axis=0)
        arrow_start = female_center
        arrow_vec = male_center - female_center
    else:
        arrow_start = np.array([0.0, 0.0])
        arrow_vec = np.array([1.0, 0.0])

    arrow_scale = 0.55
    arrow_end = arrow_start + arrow_scale * arrow_vec

    # Plot limits with padding
    all_x = [coords[w][0] for w in found_words]
    all_y = [coords[w][1] for w in found_words]
    xmin, xmax = min(all_x) - 0.35, max(all_x) + 0.35
    ymin, ymax = min(all_y) - 0.35, max(all_y) + 0.35

    # Figure
    plt.figure(figsize=(8.8, 6.4))

    # Points + labels
    for word in found_words:
        x, y = coords[word]
        c = word_color(word)

        plt.scatter(
            x, y,
            s=160,
            color=c,
            edgecolor="black",
            linewidth=0.9,
            zorder=3
        )

        dx, dy, ha, va = get_label_offset(word, x, y, xmin, xmax, ymin, ymax)
        plt.text(
            x + dx, y + dy, word,
            fontsize=13,
            ha=ha, va=va
        )

    # Dashed lines
    for w1, w2 in PAIR_LINES:
        if w1 in coords and w2 in coords:
            x1, y1 = coords[w1]
            x2, y2 = coords[w2]
            plt.plot(
                [x1, x2], [y1, y2],
                linestyle="--",
                linewidth=1.2,
                alpha=0.35,
                color="black",
                zorder=2
            )

    # -----------------------------------
    # Centered gender-direction guide
    # -----------------------------------
    # Use the female->male anchor direction, but draw it in the middle
    direction = arrow_vec / np.linalg.norm(arrow_vec) if np.linalg.norm(arrow_vec) > 0 else np.array([1.0, 0.0])

    # Put guide near the center of the plot
    center_x = (xmin + xmax) / 2.0
    center_y = (ymin + ymax) / 2.0

    # Arrow length for display only
    guide_len = 0.9

    # Female side arrow
    plt.arrow(
        center_x, center_y,
        -guide_len * direction[0], -guide_len * direction[1],
        head_width=0.10,
        head_length=0.14,
        linewidth=2.0,
        color="#d62728",
        length_includes_head=True,
        zorder=4
    )

    # Male side arrow
    plt.arrow(
        center_x, center_y,
        guide_len * direction[0], guide_len * direction[1],
        head_width=0.10,
        head_length=0.14,
        linewidth=2.0,
        color="#1f77b4",
        length_includes_head=True,
        zorder=4
    )

    # Center label
    plt.text(
        center_x,
        center_y + 0.18,
        "gender direction",
        fontsize=12,
        ha="center",
        va="bottom",
        color="black"
    )

    # Female label
    plt.text(
        center_x - 1.05 * guide_len * direction[0],
        center_y - 1.05 * guide_len * direction[1],
        "female-associated",
        fontsize=11,
        ha="right",
        va="top",
        color="#d62728"
    )

    # Male label
    plt.text(
        center_x + 1.05 * guide_len * direction[0],
        center_y + 1.05 * guide_len * direction[1],
        "male-associated",
        fontsize=11,
        ha="left",
        va="bottom",
        color="#1f77b4"
    )

    # Style
    plt.axhline(0, linewidth=0.8, alpha=0.15, color="gray")
    plt.axvline(0, linewidth=0.8, alpha=0.15, color="gray")
    plt.grid(True, alpha=0.08)

    plt.title("Gender Bias in Word Embeddings", fontsize=20, pad=12)
    plt.xlabel("")
    plt.ylabel("")
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    # Legend
    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], marker='o', color='w', label='Male-associated',
               markerfacecolor=COLORS["male"], markeredgecolor="black", markersize=11),
        Line2D([0], [0], marker='o', color='w', label='Female-associated',
               markerfacecolor=COLORS["female"], markeredgecolor="black", markersize=11),
        Line2D([0], [0], marker='o', color='w', label='Neutral / mixed',
               markerfacecolor=COLORS["neutral"], markeredgecolor="black", markersize=11),
    ]
    plt.legend(handles=legend_items, loc="upper right", fontsize=11, frameon=True)

    plt.text(
        0.02, 0.02,
        "2D PCA projection for visualization only",
        transform=plt.gca().transAxes,
        fontsize=10,
        alpha=0.8
    )

    os.makedirs(os.path.dirname(OUTPUT_FIG), exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved figure to: {OUTPUT_FIG}")


if __name__ == "__main__":
    main()