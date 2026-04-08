#!/usr/bin/env python3
"""
multi_pc_ablation.py

Research-oriented multi-PC ablation experiment for bias analysis in static word
embeddings. The script estimates a fixed PCA basis from definitional difference
vectors and removes the top-k PCs (for k = 0..N) from all embedding vectors.

Metrics computed for each k:
    - Direct Bias
    - WEAT score
    - Mean displacement vs original embedding
    - Neighbor stability@10

Design choices:
    - Fixed PCA basis computed once from the original embedding space
    - Consistent metric evaluation across all k
    - No dataset/metric definition changes
    - Emphasis on clarity and research reproducibility

Author: OpenAI assistant draft for research collaboration
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# =========================
# Data containers
# =========================

VectorDict = Dict[str, np.ndarray]


@dataclass
class ExperimentRow:
    k: int
    direct_bias: float
    weat_score: float
    mean_displacement: float
    neighbor_stability_at_10: float
    cumulative_explained_variance: float
    incremental_explained_variance: float


# =========================
# Utility functions
# =========================

def l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return L2-normalized vector."""
    norm = np.linalg.norm(vec)
    if norm < eps:
        return vec.copy()
    return vec / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """Cosine similarity between two vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def ensure_dir(path: Path) -> None:
    """Create directory if needed."""
    path.mkdir(parents=True, exist_ok=True)


# =========================
# Embedding loading
# =========================

def load_txt_embeddings(
    embedding_path: Path,
    max_vocab: Optional[int] = None,
    normalize_on_load: bool = False,
    encoding: str = "utf-8",
) -> VectorDict:
    """
    Load text embeddings in standard GloVe/word2vec-text format.

    Supported line formats:
        word val1 val2 ...
    or optionally header:
        vocab_size dim

    Notes:
        - If your file has a header, this loader tries to detect it.
        - Multiword tokens are not supported here.
    """
    word_to_vec: VectorDict = {}

    with embedding_path.open("r", encoding=encoding, errors="ignore") as f:
        first_line = f.readline().strip().split()
        has_header = False

        if len(first_line) == 2:
            try:
                int(first_line[0])
                int(first_line[1])
                has_header = True
            except ValueError:
                has_header = False

        if not has_header:
            # Rewind and process first line as data
            f.seek(0)

        for idx, line in enumerate(f):
            parts = line.rstrip().split()
            if len(parts) < 10:
                # Skip malformed or empty lines
                continue

            word = parts[0]
            try:
                vec = np.asarray([float(x) for x in parts[1:]], dtype=np.float32)
            except ValueError:
                continue

            if normalize_on_load:
                vec = l2_normalize(vec)

            word_to_vec[word] = vec

            if max_vocab is not None and len(word_to_vec) >= max_vocab:
                break

    if not word_to_vec:
        raise ValueError(f"No embeddings loaded from: {embedding_path}")

    return word_to_vec


# =========================
# JSON loading
# =========================

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_definitional_pairs(data) -> List[Tuple[str, str]]:
    """
    Expected input examples:
        [
          ["she", "he"],
          ["woman", "man"]
        ]

    or
        {
          "definitional_pairs": [
            ["she", "he"],
            ["woman", "man"]
          ]
        }
    """
    if isinstance(data, dict):
        if "definitional_pairs" in data:
            data = data["definitional_pairs"]
        else:
            raise ValueError(
                "Unsupported definitional_pairs.json format. "
                "Expected list or dict with key 'definitional_pairs'."
            )

    pairs: List[Tuple[str, str]] = []
    for item in data:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"Invalid definitional pair: {item}")
        pairs.append((str(item[0]), str(item[1])))
    return pairs


def extract_direct_bias_words(data) -> List[str]:
    """
    Expected input examples:
        ["doctor", "nurse", "engineer"]

    or
        {"words": ["doctor", "nurse", "engineer"]}
    """
    if isinstance(data, dict):
        if "words" in data:
            data = data["words"]
        else:
            raise ValueError(
                "Unsupported direct_bias_words.json format. "
                "Expected list or dict with key 'words'."
            )

    words = [str(x) for x in data]
    return words


def extract_weat_sets(data) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Expected formats:

    Option A:
        {
          "X": [...],
          "Y": [...],
          "A": [...],
          "B": [...]
        }

    Option B:
        {
          "target_1": [...],
          "target_2": [...],
          "attribute_1": [...],
          "attribute_2": [...]
        }

    If your current file has different field names, adapt only this function.
    """
    if not isinstance(data, dict):
        raise ValueError("weat_sets.json must be a dict.")

    if all(k in data for k in ("X", "Y", "A", "B")):
        return (
            [str(w) for w in data["X"]],
            [str(w) for w in data["Y"]],
            [str(w) for w in data["A"]],
            [str(w) for w in data["B"]],
        )

    alt_keys = ("target_1", "target_2", "attribute_1", "attribute_2")
    if all(k in data for k in alt_keys):
        return (
            [str(w) for w in data["target_1"]],
            [str(w) for w in data["target_2"]],
            [str(w) for w in data["attribute_1"]],
            [str(w) for w in data["attribute_2"]],
        )

    raise ValueError(
        "Unsupported weat_sets.json format. "
        "Expected keys (X,Y,A,B) or (target_1,target_2,attribute_1,attribute_2)."
    )


# =========================
# Vocabulary filtering
# =========================

def filter_words_in_vocab(words: Sequence[str], word_to_vec: VectorDict) -> List[str]:
    return [w for w in words if w in word_to_vec]


def filter_pairs_in_vocab(
    pairs: Sequence[Tuple[str, str]], word_to_vec: VectorDict
) -> List[Tuple[str, str]]:
    return [(a, b) for (a, b) in pairs if a in word_to_vec and b in word_to_vec]


# =========================
# PCA / subspace construction
# =========================

def build_definitional_difference_matrix(
    word_to_vec: VectorDict,
    definitional_pairs: Sequence[Tuple[str, str]],
    center: bool = False,
    normalize_differences: bool = False,
) -> np.ndarray:
    """
    Build matrix D whose rows are definitional difference vectors:
        d_i = v(a_i) - v(b_i)

    Important:
        Keep this aligned with your single-PC pipeline.
    """
    diffs = []
    for a, b in definitional_pairs:
        diff = word_to_vec[a] - word_to_vec[b]
        if normalize_differences:
            diff = l2_normalize(diff)
        diffs.append(diff)

    D = np.vstack(diffs)

    if center:
        D = D - D.mean(axis=0, keepdims=True)

    return D


def fit_pca_components(
    D: np.ndarray,
    max_components: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit PCA using SVD on matrix D.

    Returns:
        components: shape (m, dim), rows are orthonormal principal directions
        explained_variance_ratio: shape (m,)
    """
    if D.ndim != 2:
        raise ValueError("D must be a 2D matrix.")

    n_samples, dim = D.shape
    m = min(n_samples, dim)

    if max_components is not None:
        m = min(m, max_components)

    # Full SVD
    U, S, Vt = np.linalg.svd(D, full_matrices=False)

    components = Vt[:m]
    eigenvalues = (S ** 2) / max(n_samples - 1, 1)

    total_variance = float(np.sum(eigenvalues))
    if total_variance <= 0:
        explained_variance_ratio = np.zeros(m, dtype=np.float64)
    else:
        explained_variance_ratio = eigenvalues[:m] / total_variance

    return components, explained_variance_ratio


# =========================
# Debiasing / projection removal
# =========================

def remove_top_k_subspace(
    word_to_vec: VectorDict,
    components: np.ndarray,
    k: int,
) -> VectorDict:
    """
    Remove the projection onto span(u_1, ..., u_k):
        x' = x - U_k^T(U_k x)  if components are row vectors
    where U_k has shape (k, dim).
    """
    if k < 0 or k > components.shape[0]:
        raise ValueError(f"k must be between 0 and {components.shape[0]}")

    if k == 0:
        return {w: v.copy() for w, v in word_to_vec.items()}

    U_k = components[:k]  # shape (k, dim)

    debiased: VectorDict = {}
    for word, vec in word_to_vec.items():
        # coefficients shape: (k,)
        coeffs = U_k @ vec
        projection = U_k.T @ coeffs
        debiased[word] = vec - projection

    return debiased


# =========================
# Bias direction and metrics
# =========================

def compute_gender_direction_from_pairs_mean(
    word_to_vec: VectorDict,
    definitional_pairs: Sequence[Tuple[str, str]],
    normalize_direction: bool = True,
) -> np.ndarray:
    """
    Mean definitional direction:
        g = average(v(a) - v(b))

    This is often used for direct bias.
    If your single-PC code instead uses PC1 as the direction for direct bias,
    replace this function accordingly.
    """
    diffs = [word_to_vec[a] - word_to_vec[b] for a, b in definitional_pairs]
    g = np.mean(np.vstack(diffs), axis=0)
    return l2_normalize(g) if normalize_direction else g


def compute_direct_bias(
    word_to_vec: VectorDict,
    words: Sequence[str],
    gender_direction: np.ndarray,
    c: float = 1.0,
) -> float:
    """
    Direct Bias as average |cos(w, g)|^c over evaluation words.

    By default c=1.0.
    If your single-PC setup uses a different exponent, pass it via CLI.
    """
    valid_words = [w for w in words if w in word_to_vec]
    if not valid_words:
        return float("nan")

    vals = []
    for w in valid_words:
        score = abs(cosine_similarity(word_to_vec[w], gender_direction)) ** c
        vals.append(score)

    return float(np.mean(vals))


def association(
    word_to_vec: VectorDict,
    w: str,
    A: Sequence[str],
    B: Sequence[str],
) -> float:
    """
    WEAT association:
        s(w, A, B) = mean_{a in A} cos(w,a) - mean_{b in B} cos(w,b)
    """
    mean_a = np.mean([cosine_similarity(word_to_vec[w], word_to_vec[a]) for a in A])
    mean_b = np.mean([cosine_similarity(word_to_vec[w], word_to_vec[b]) for b in B])
    return float(mean_a - mean_b)


def compute_weat_score(
    word_to_vec: VectorDict,
    X: Sequence[str],
    Y: Sequence[str],
    A: Sequence[str],
    B: Sequence[str],
) -> float:
    """
    Standard WEAT effect size:
        (mean s(x,A,B) - mean s(y,A,B)) / std_dev over X union Y
    """
    Xv = [w for w in X if w in word_to_vec]
    Yv = [w for w in Y if w in word_to_vec]
    Av = [w for w in A if w in word_to_vec]
    Bv = [w for w in B if w in word_to_vec]

    if not Xv or not Yv or not Av or not Bv:
        return float("nan")

    s_x = np.array([association(word_to_vec, x, Av, Bv) for x in Xv], dtype=np.float64)
    s_y = np.array([association(word_to_vec, y, Av, Bv) for y in Yv], dtype=np.float64)

    numerator = float(np.mean(s_x) - np.mean(s_y))
    pooled = np.concatenate([s_x, s_y])
    denom = float(np.std(pooled))

    if abs(denom) < 1e-12:
        return 0.0

    return numerator / denom


def compute_mean_displacement(
    original: VectorDict,
    transformed: VectorDict,
    words: Optional[Sequence[str]] = None,
    normalize_by_original_norm: bool = False,
) -> float:
    """
    Mean L2 displacement between original and transformed vectors.

    If words is None, computes over the intersection of vocabularies.
    """
    if words is None:
        words = sorted(set(original.keys()) & set(transformed.keys()))

    vals = []
    for w in words:
        if w not in original or w not in transformed:
            continue
        disp = np.linalg.norm(transformed[w] - original[w])
        if normalize_by_original_norm:
            base = np.linalg.norm(original[w])
            if base > 1e-12:
                disp /= base
        vals.append(float(disp))

    if not vals:
        return float("nan")

    return float(np.mean(vals))


def build_matrix_for_vocab(
    word_to_vec: VectorDict,
    vocab: Sequence[str],
    normalize_rows: bool = True,
) -> np.ndarray:
    """
    Build matrix of shape (|vocab|, dim).
    """
    rows = []
    for w in vocab:
        vec = word_to_vec[w]
        if normalize_rows:
            vec = l2_normalize(vec)
        rows.append(vec)
    return np.vstack(rows)


def top_k_neighbors_from_similarity_row(
    sims: np.ndarray,
    self_index: int,
    k: int,
) -> List[int]:
    """
    Return indices of top-k neighbors excluding self.
    """
    sims = sims.copy()
    sims[self_index] = -np.inf
    if k >= len(sims) - 1:
        order = np.argsort(-sims)
        return [int(i) for i in order if i != self_index]
    idx = np.argpartition(-sims, kth=k)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return [int(i) for i in idx if i != self_index][:k]


def compute_neighbor_stability_at_k(
    original: VectorDict,
    transformed: VectorDict,
    eval_vocab: Sequence[str],
    k_neighbors: int = 10,
) -> float:
    """
    Average Jaccard overlap between original and transformed top-k nearest neighbors.

    Important:
        This uses eval_vocab as the search space for neighbors.
        For research consistency, keep the same eval_vocab across all k.

    Returns:
        Mean overlap in [0,1]
    """
    vocab = [w for w in eval_vocab if w in original and w in transformed]
    if len(vocab) <= k_neighbors + 1:
        return float("nan")

    orig_mat = build_matrix_for_vocab(original, vocab, normalize_rows=True)
    trans_mat = build_matrix_for_vocab(transformed, vocab, normalize_rows=True)

    orig_sim = orig_mat @ orig_mat.T
    trans_sim = trans_mat @ trans_mat.T

    overlaps = []
    for i in range(len(vocab)):
        n1 = set(top_k_neighbors_from_similarity_row(orig_sim[i], i, k_neighbors))
        n2 = set(top_k_neighbors_from_similarity_row(trans_sim[i], i, k_neighbors))
        union = n1 | n2
        inter = n1 & n2
        if not union:
            overlaps.append(1.0)
        else:
            overlaps.append(len(inter) / len(union))

    return float(np.mean(overlaps))


# =========================
# Experiment runner
# =========================

def run_multi_pc_ablation(
    original_embeddings: VectorDict,
    definitional_pairs: Sequence[Tuple[str, str]],
    direct_bias_words: Sequence[str],
    weat_X: Sequence[str],
    weat_Y: Sequence[str],
    weat_A: Sequence[str],
    weat_B: Sequence[str],
    max_k: int,
    direct_bias_c: float = 1.0,
    pca_center: bool = False,
    normalize_definitional_diffs: bool = False,
    displacement_words: Optional[Sequence[str]] = None,
    neighbor_eval_vocab: Optional[Sequence[str]] = None,
    pca_max_components: Optional[int] = None,
    normalize_gender_direction: bool = True,
) -> Tuple[List[ExperimentRow], np.ndarray, np.ndarray]:
    """
    Main multi-PC experiment using a fixed PCA basis.
    """
    valid_pairs = filter_pairs_in_vocab(definitional_pairs, original_embeddings)
    if not valid_pairs:
        raise ValueError("No definitional pairs remain after vocabulary filtering.")

    direct_bias_words = filter_words_in_vocab(direct_bias_words, original_embeddings)
    weat_X = filter_words_in_vocab(weat_X, original_embeddings)
    weat_Y = filter_words_in_vocab(weat_Y, original_embeddings)
    weat_A = filter_words_in_vocab(weat_A, original_embeddings)
    weat_B = filter_words_in_vocab(weat_B, original_embeddings)

    if displacement_words is None:
        displacement_words = sorted(original_embeddings.keys())
    else:
        displacement_words = filter_words_in_vocab(displacement_words, original_embeddings)

    if neighbor_eval_vocab is None:
        neighbor_vocab_set = set(direct_bias_words)
        neighbor_vocab_set.update(weat_X)
        neighbor_vocab_set.update(weat_Y)
        neighbor_vocab_set.update(weat_A)
        neighbor_vocab_set.update(weat_B)
        for a, b in valid_pairs:
            neighbor_vocab_set.add(a)
            neighbor_vocab_set.add(b)

        neighbor_eval_vocab = sorted(
            w for w in neighbor_vocab_set if w in original_embeddings
        )
    else:
        neighbor_eval_vocab = filter_words_in_vocab(neighbor_eval_vocab, original_embeddings)

    D = build_definitional_difference_matrix(
        original_embeddings,
        valid_pairs,
        center=pca_center,
        normalize_differences=normalize_definitional_diffs,
    )

    components, explained_ratio = fit_pca_components(
        D,
        max_components=pca_max_components,
    )

    max_k = min(max_k, components.shape[0])

    # Keep direct bias definition aligned with original setup.
    # If your single-PC pipeline uses another bias direction, replace here.
    original_gender_direction = compute_gender_direction_from_pairs_mean(
        original_embeddings,
        valid_pairs,
        normalize_direction=normalize_gender_direction,
    )

    rows: List[ExperimentRow] = []

    for k in range(0, max_k + 1):
        transformed = remove_top_k_subspace(original_embeddings, components, k)

        # Important:
        # Keep the direct-bias direction fixed from the original setup unless your
        # single-PC code explicitly recomputes it after debiasing.
        db = compute_direct_bias(
            transformed,
            direct_bias_words,
            original_gender_direction,
            c=direct_bias_c,
        )

        weat = compute_weat_score(
            transformed,
            weat_X,
            weat_Y,
            weat_A,
            weat_B,
        )

        disp = compute_mean_displacement(
            original_embeddings,
            transformed,
            words=displacement_words,
            normalize_by_original_norm=False,
        )

        stab = compute_neighbor_stability_at_k(
            original_embeddings,
            transformed,
            eval_vocab=neighbor_eval_vocab,
            k_neighbors=10,
        )

        incremental = 0.0 if k == 0 else float(explained_ratio[k - 1])
        cumulative = 0.0 if k == 0 else float(np.sum(explained_ratio[:k]))

        rows.append(
            ExperimentRow(
                k=k,
                direct_bias=db,
                weat_score=weat,
                mean_displacement=disp,
                neighbor_stability_at_10=stab,
                cumulative_explained_variance=cumulative,
                incremental_explained_variance=incremental,
            )
        )

        print(
            f"k={k:02d} | "
            f"direct_bias={db:.6f} | "
            f"weat={weat:.6f} | "
            f"mean_disp={disp:.6f} | "
            f"neighbor_stability@10={stab:.6f}"
        )

    return rows, components, explained_ratio


# =========================
# Saving outputs
# =========================

def save_results_csv(rows: Sequence[ExperimentRow], out_path: Path) -> None:
    ensure_dir(out_path.parent)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(asdict(rows[0]).keys()) if rows else [
                "k",
                "direct_bias",
                "weat_score",
                "mean_displacement",
                "neighbor_stability_at_10",
                "cumulative_explained_variance",
                "incremental_explained_variance",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def save_text_report(
    rows: Sequence[ExperimentRow],
    explained_ratio: np.ndarray,
    out_path: Path,
) -> None:
    ensure_dir(out_path.parent)

    with out_path.open("w", encoding="utf-8") as f:
        f.write("Multi-PC Ablation Report\n")
        f.write("=" * 80 + "\n\n")

        f.write("Explained variance ratio by component:\n")
        for i, val in enumerate(explained_ratio, start=1):
            f.write(f"PC{i:02d}: {val:.6f}\n")

        f.write("\n")
        f.write("Metric table:\n")
        f.write(
            "k | direct_bias | weat_score | mean_displacement | "
            "neighbor_stability@10 | cumulative_var | incremental_var\n"
        )
        f.write("-" * 110 + "\n")

        for r in rows:
            f.write(
                f"{r.k:2d} | "
                f"{r.direct_bias:.6f} | "
                f"{r.weat_score:.6f} | "
                f"{r.mean_displacement:.6f} | "
                f"{r.neighbor_stability_at_10:.6f} | "
                f"{r.cumulative_explained_variance:.6f} | "
                f"{r.incremental_explained_variance:.6f}\n"
            )

        f.write("\n")
        f.write("Interpretation guide:\n")
        f.write("- Direct Bias should generally decrease as k increases.\n")
        f.write("- WEAT may decrease more slowly or non-monotonically.\n")
        f.write("- Mean displacement is the geometric cost of debiasing.\n")
        f.write("- Neighbor stability@10 reflects local semantic preservation.\n")


# =========================
# CLI
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-PC ablation experiments on static word embeddings."
    )

    parser.add_argument(
        "--embedding-path",
        type=str,
        required=True,
        help="Path to text embedding file (e.g., GloVe .txt).",
    )
    parser.add_argument(
        "--definitional-pairs-json",
        type=str,
        required=True,
        help="Path to definitional_pairs.json",
    )
    parser.add_argument(
        "--direct-bias-words-json",
        type=str,
        required=True,
        help="Path to direct_bias_words.json",
    )
    parser.add_argument(
        "--weat-json",
        type=str,
        required=True,
        help="Path to weat_sets.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/multi_pc_ablation",
        help="Directory to save results.",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=10,
        help="Maximum number of top PCs to remove.",
    )
    parser.add_argument(
        "--max-vocab",
        type=int,
        default=None,
        help="Optional vocab cap for loading embeddings.",
    )
    parser.add_argument(
        "--normalize-on-load",
        action="store_true",
        help="L2-normalize embeddings when loaded.",
    )
    parser.add_argument(
        "--pca-center",
        action="store_true",
        help="Center definitional difference matrix before PCA.",
    )
    parser.add_argument(
        "--normalize-definitional-diffs",
        action="store_true",
        help="L2-normalize each definitional difference vector before PCA.",
    )
    parser.add_argument(
        "--direct-bias-c",
        type=float,
        default=1.0,
        help="Exponent c in Direct Bias = mean(|cos(w,g)|^c).",
    )
    parser.add_argument(
        "--pca-max-components",
        type=int,
        default=None,
        help="Optional cap on number of PCA components retained.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    embedding_path = Path(args.embedding_path)
    definitional_pairs_json = Path(args.definitional_pairs_json)
    direct_bias_words_json = Path(args.direct_bias_words_json)
    weat_json = Path(args.weat_json)
    output_dir = Path(args.output_dir)

    ensure_dir(output_dir)

    print("Loading embeddings...")
    embeddings = load_txt_embeddings(
        embedding_path=embedding_path,
        max_vocab=args.max_vocab,
        normalize_on_load=args.normalize_on_load,
    )
    print(f"Loaded {len(embeddings):,} embeddings.")

    print("Loading config files...")
    definitional_pairs = extract_definitional_pairs(load_json(definitional_pairs_json))
    direct_bias_words = extract_direct_bias_words(load_json(direct_bias_words_json))
    weat_X, weat_Y, weat_A, weat_B = extract_weat_sets(load_json(weat_json))

    print("Running multi-PC ablation...")
    rows, components, explained_ratio = run_multi_pc_ablation(
        original_embeddings=embeddings,
        definitional_pairs=definitional_pairs,
        direct_bias_words=direct_bias_words,
        weat_X=weat_X,
        weat_Y=weat_Y,
        weat_A=weat_A,
        weat_B=weat_B,
        max_k=args.max_k,
        direct_bias_c=args.direct_bias_c,
        pca_center=args.pca_center,
        normalize_definitional_diffs=args.normalize_definitional_diffs,
        displacement_words=None,
        neighbor_eval_vocab=None,
        pca_max_components=args.pca_max_components,
        normalize_gender_direction=True,
    )

    csv_path = output_dir / "multi_pc_ablation_results.csv"
    txt_path = output_dir / "multi_pc_ablation_report.txt"
    pcs_path = output_dir / "pca_components.npy"
    evr_path = output_dir / "explained_variance_ratio.npy"

    print("Saving outputs...")
    save_results_csv(rows, csv_path)
    save_text_report(rows, explained_ratio, txt_path)
    np.save(pcs_path, components)
    np.save(evr_path, explained_ratio)

    print(f"Results CSV written to: {csv_path}")
    print(f"Report written to:      {txt_path}")
    print(f"PCA components saved:   {pcs_path}")
    print(f"Explained variance:     {evr_path}")


if __name__ == "__main__":
    main()