#!/usr/bin/env python3

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors


# =========================================================
# Defaults
# =========================================================

WORD2VEC_RELATIVE = "data/GoogleNews-vectors-negative300.bin"

DEFINITIONAL_PAIRS_JSON = "data/definitional_pairs.json"
DIRECT_BIAS_WORDS_JSON = "data/direct_bias_words.json"
WEAT_JSON = "data/weat_sets.json"

DEFAULT_OUTPUT_CSV = "outputs/single_pc_ablation/word2vec/single_pc_ablation_results.csv"
DEFAULT_OUTPUT_PNG = "outputs/single_pc_ablation/word2vec/single_pc_ablation_plot.png"

DEFAULT_NEIGHBOR_K = 10
DEFAULT_NEIGHBOR_SAMPLE_SIZE = 200
DEFAULT_RANDOM_SEED = 42


@dataclass
class WeatSets:
    X: list[str]
    Y: list[str]
    A: list[str]
    B: list[str]


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return get_repo_root() / path


def load_json(path: str | Path):
    with open(resolve_path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def load_word2vec_bin(path: str | Path) -> tuple[list[str], np.ndarray]:
    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Word2Vec file not found: {path}")

    print(f"Loading Word2Vec binary from: {path}")
    model = KeyedVectors.load_word2vec_format(str(path), binary=True)

    vocab = list(model.index_to_key)
    X = np.vstack([model[w] for w in vocab]).astype(np.float32)

    print(f"Loaded {len(vocab):,} Word2Vec vectors with dimension {X.shape[1]}")
    return vocab, X


def normalize_rows(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return mat / norms


def normalize_vector(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < eps:
        raise ValueError("Cannot normalize near-zero vector")
    return v / norm


def remove_single_pc(X: np.ndarray, g: np.ndarray) -> np.ndarray:
    g = normalize_vector(g).astype(np.float32)
    projections = X @ g
    return X - projections[:, None] * g[None, :]


def mean_displacement_from_pc(X0: np.ndarray, g: np.ndarray) -> float:
    """
    For single-PC removal, Xp = X0 - outer(X0 @ g, g).
    The displacement vector is (Xp - X0) = -outer(X0 @ g, g).
    Since g is unit length, ||Xp - X0|| = |X0 @ g|.
    This avoids allocating the full Xp - X0 matrix.
    """
    g = normalize_vector(g).astype(np.float32)
    projections = X0 @ g
    return float(np.mean(np.abs(projections)))


def load_definitional_pairs(path: str | Path) -> list[tuple[str, str]]:
    data = load_json(path)

    if isinstance(data, list):
        pairs_raw = data
    elif isinstance(data, dict) and "pairs" in data:
        pairs_raw = data["pairs"]
    else:
        raise ValueError("Definitional pairs JSON must be a list or {'pairs': [...]}.")

    pairs = []
    for item in pairs_raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"Invalid definitional pair: {item}")
        pairs.append((str(item[0]), str(item[1])))

    return pairs


def compute_gender_pcs_from_definitional_pairs(
    vocab: Sequence[str],
    X: np.ndarray,
    pairs: Sequence[tuple[str, str]],
) -> np.ndarray:
    vocab_to_idx = {w: i for i, w in enumerate(vocab)}

    diffs = []
    missing = []

    for a, b in pairs:
        if a in vocab_to_idx and b in vocab_to_idx:
            diffs.append(X[vocab_to_idx[a]] - X[vocab_to_idx[b]])
        else:
            missing.append((a, b))

    if not diffs:
        raise ValueError("No definitional pairs found in Word2Vec vocabulary.")

    D = np.vstack(diffs).astype(np.float64)
    D = D - D.mean(axis=0, keepdims=True)

    _, _, vt = np.linalg.svd(D, full_matrices=False)
    pcs = normalize_rows(vt)

    if missing:
        print(f"Warning: skipped {len(missing)} missing definitional pairs.", file=sys.stderr)

    return pcs


def load_direct_bias_words(path: str | Path) -> list[str]:
    data = load_json(path)

    if isinstance(data, list):
        return [str(x) for x in data]

    if isinstance(data, dict) and "words" in data:
        return [str(x) for x in data["words"]]

    raise ValueError("Direct bias JSON must be a list or {'words': [...]}.")


def load_weat_sets(path: str | Path) -> WeatSets:
    data = load_json(path)
    required = ["X", "Y", "A", "B"]

    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"WEAT JSON missing keys: {missing}")

    return WeatSets(
        X=[str(w) for w in data["X"]],
        Y=[str(w) for w in data["Y"]],
        A=[str(w) for w in data["A"]],
        B=[str(w) for w in data["B"]],
    )


def direct_bias(
    vocab: Sequence[str],
    X: np.ndarray,
    gender_direction: np.ndarray,
    target_words: Sequence[str],
) -> float:
    vocab_to_idx = {w: i for i, w in enumerate(vocab)}
    keep = [vocab_to_idx[w] for w in target_words if w in vocab_to_idx]

    if not keep:
        raise ValueError("No direct-bias target words found in vocabulary.")

    Xn = normalize_rows(X)
    g = normalize_vector(gender_direction)

    vals = np.abs(Xn[keep] @ g)
    return float(np.mean(vals))


def association(w: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
    w_n = normalize_vector(w)
    A_n = normalize_rows(A)
    B_n = normalize_rows(B)
    return float((A_n @ w_n).mean() - (B_n @ w_n).mean())


def weat_effect_size(vocab: Sequence[str], X: np.ndarray, sets: WeatSets) -> float:
    vocab_to_idx = {w: i for i, w in enumerate(vocab)}

    def get_vectors(words):
        found = [vocab_to_idx[w] for w in words if w in vocab_to_idx]
        if not found:
            raise ValueError(f"No WEAT words found from set: {words[:5]}")
        return X[found]

    X_words = get_vectors(sets.X)
    Y_words = get_vectors(sets.Y)
    A_words = get_vectors(sets.A)
    B_words = get_vectors(sets.B)

    s_X = np.array([association(w, A_words, B_words) for w in X_words])
    s_Y = np.array([association(w, A_words, B_words) for w in Y_words])

    denom = np.std(np.concatenate([s_X, s_Y]), ddof=1)
    if np.isclose(denom, 0.0):
        return 0.0

    return float((s_X.mean() - s_Y.mean()) / denom)


def topk_neighbors(X: np.ndarray, query_indices: np.ndarray, k: int) -> list[set[int]]:
    Xn = normalize_rows(X)
    neigh_sets = []

    for idx in query_indices:
        sims = Xn[idx] @ Xn.T
        sims[idx] = -np.inf
        topk = np.argpartition(-sims, kth=k)[:k]
        neigh_sets.append(set(map(int, topk.tolist())))

    return neigh_sets


def neighbor_stability_at_k(
    X0: np.ndarray,
    X1: np.ndarray,
    k: int = 10,
    sample_size: int | None = 200,
    seed: int = 42,
) -> float:
    n = X0.shape[0]

    if sample_size is None or sample_size >= n:
        query_indices = np.arange(n)
    else:
        rng = np.random.default_rng(seed)
        query_indices = np.sort(rng.choice(n, size=sample_size, replace=False))

    neigh0 = topk_neighbors(X0, query_indices, k)
    neigh1 = topk_neighbors(X1, query_indices, k)

    overlaps = [len(a & b) / float(k) for a, b in zip(neigh0, neigh1)]
    return float(np.mean(overlaps))


def save_plots(df: pd.DataFrame, out_png: str | Path) -> None:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(df["pc"], df["direct_bias"], marker="o", label="Direct Bias")
    axes[0].plot(df["pc"], df["weat"], marker="o", label="WEAT")
    axes[0].set_ylabel("Metric value")
    axes[0].set_title("Word2Vec Single-PC Ablation: Bias Metrics")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(df["pc"], df["mean_displacement"], marker="o", label="Mean displacement")
    axes[1].plot(df["pc"], df["neighbor_stability_at_10"], marker="o", label="Neighbor stability@10")
    axes[1].set_xlabel("PC index")
    axes[1].set_ylabel("Metric value")
    axes[1].set_title("Word2Vec Single-PC Ablation: Geometry Metrics")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_experiment(args):
    vocab, X0 = load_word2vec_bin(args.embedding_path)

    pairs = load_definitional_pairs(args.definitional_pairs_json)
    pcs = compute_gender_pcs_from_definitional_pairs(vocab, X0, pairs)

    if args.pc_end > pcs.shape[0]:
        raise ValueError(f"Requested PC{args.pc_end}, but only {pcs.shape[0]} PCs are available.")

    direct_bias_words = load_direct_bias_words(args.direct_bias_words_json)
    weat_sets = load_weat_sets(args.weat_json)

    baseline_gender_direction = pcs[0]
    baseline_direct_bias = direct_bias(vocab, X0, baseline_gender_direction, direct_bias_words)
    baseline_weat = weat_effect_size(vocab, X0, weat_sets)

    rows = []

    print("\nBaseline metrics on original Word2Vec embedding")
    print(f"Direct Bias: {baseline_direct_bias:.6f}")
    print(f"WEAT:        {baseline_weat:.6f}\n")

    for pc in range(args.pc_start, args.pc_end + 1):
        g = pcs[pc - 1]
        Xp = remove_single_pc(X0, g)

        db = direct_bias(vocab, Xp, baseline_gender_direction, direct_bias_words)
        weat = weat_effect_size(vocab, Xp, weat_sets)
        disp = mean_displacement_from_pc(X0, g)
        stab = neighbor_stability_at_k(
            X0,
            Xp,
            k=args.neighbor_k,
            sample_size=args.neighbor_sample_size,
            seed=args.random_seed,
        )

        rows.append({
            "pc": pc,
            "direct_bias": db,
            "weat": weat,
            "mean_displacement": disp,
            "neighbor_stability_at_10": stab,
            "delta_direct_bias": db - baseline_direct_bias,
            "delta_weat": weat - baseline_weat,
            "delta_mean_displacement": disp,
            "delta_neighbor_stability": stab - 1.0,
        })

        print(
            f"PC{pc:02d} | direct_bias={db:.6f} | "
            f"weat={weat:.6f} | mean_disp={disp:.6f} | "
            f"neighbor_stability@10={stab:.6f}"
        )

    df = pd.DataFrame(rows)

    output_csv = resolve_path(args.output_csv)
    output_png = resolve_path(args.output_png)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    save_plots(df, output_png)

    print(f"\nSaved CSV  -> {output_csv}")
    print(f"Saved plot -> {output_png}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Word2Vec single-PC ablation experiment.")

    parser.add_argument(
        "--embedding-path",
        type=str,
        default=WORD2VEC_RELATIVE,
        help="Path to Word2Vec GoogleNews .bin file."
    )

    parser.add_argument("--definitional-pairs-json", type=str, default=DEFINITIONAL_PAIRS_JSON)
    parser.add_argument("--direct-bias-words-json", type=str, default=DIRECT_BIAS_WORDS_JSON)
    parser.add_argument("--weat-json", type=str, default=WEAT_JSON)

    parser.add_argument("--pc-start", type=int, default=1)
    parser.add_argument("--pc-end", type=int, default=10)

    parser.add_argument("--neighbor-k", type=int, default=DEFAULT_NEIGHBOR_K)
    parser.add_argument("--neighbor-sample-size", type=int, default=DEFAULT_NEIGHBOR_SAMPLE_SIZE)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)

    parser.add_argument("--output-csv", type=str, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-png", type=str, default=DEFAULT_OUTPUT_PNG)

    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())