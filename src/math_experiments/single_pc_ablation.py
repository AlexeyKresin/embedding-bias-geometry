#!/usr/bin/env python3
"""
single_pc_ablation.py
=====================

Single-PC ablation experiment for gender subspace analysis in static word embeddings.

What this script does
---------------------
For each gender principal component g_i (default: PCs 1..10), it removes ONLY that
component from the embedding matrix:

    X' = X - (X @ g_i[:, None]) * g_i[None, :]

and recomputes:
    - Direct Bias
    - WEAT
    - Mean displacement from the original embedding
    - Neighbor stability@10 relative to the original embedding

It saves:
    - CSV results
    - PNG plots

Design goal
-----------
Reuse the existing project where possible, but avoid forcing a project redesign.
The script supports two integration styles:

1) Preferred: point the script at your existing project functions by editing the
   CONFIG / ADAPTER section near the top.
2) Fallback: use the standalone helper implementations included here.

Important notes
---------------
- This script does NOT recompute PCA after each ablation.
- It removes one PC at a time, not cumulatively.
- It keeps the same vocabulary / evaluation sets across runs.
- It verifies that each PC is normalized, and normalizes if needed.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# CONFIG / ADAPTERS
# =============================================================================
# Edit ONLY this section to connect the script to your current project.
# If you already have project functions, point to them here using:
#     "package.module:function_name"
#
# If left as None, the script uses its own helper implementation.
#
# Expected signatures if you wire your own functions:
#   load_embedding() -> (vocab: list[str], X: np.ndarray[shape=(n_words, dim)])
#   load_gender_pcs() -> np.ndarray[shape=(n_pcs, dim)] or (dim, n_pcs)
#   direct_bias(vocab, X, gender_direction, **kwargs) -> float
#   weat(vocab, X, **kwargs) -> float
#
# Example:
#   LOAD_EMBEDDING_FN = "src.embedding_utils:load_glove_matrix"
#   LOAD_GENDER_PCS_FN = "src.gender_subspace:load_gender_pcs"
#   DIRECT_BIAS_FN = "src.metrics:compute_direct_bias"
#   WEAT_FN = "src.metrics:compute_weat"
# =============================================================================
# By default this script can read the raw GloVe text file directly.
# You can still override it with your own project loader function later.
LOAD_EMBEDDING_FN: str | None = "__main__:load_glove_txt"
LOAD_GENDER_PCS_FN: str | None = None
DIRECT_BIAS_FN: str | None = None
WEAT_FN: str | None = None

# Optional local data files if your project stores arrays directly.
# Supported formats:
#   - embedding .npz with keys: X, vocab
#   - gender PCs .npy or .npz with key: pcs
EMBEDDING_NPZ: str | None = None
GENDER_PCS_FILE: str | None = None

# Raw GloVe fallback location, relative to repo root.
GLOVE_TXT_RELATIVE: str = "data/glove.6B.300d.txt"

# Optional fallback source for computing gender PCs directly from definitional pairs.
# Supported JSON formats:
#   1) [["she", "he"], ["woman", "man"], ...]
#   2) {"pairs": [["she", "he"], ["woman", "man"], ...]}
DEFINITIONAL_PAIRS_JSON = "data/definitional_pairs.json"

# Optional evaluation sets. If you already have them in your project, keep these as None
# and wire the project functions above. Otherwise, you can pass JSON files from CLI.
DIRECT_BIAS_WORDS_JSON = "data/direct_bias_words.json"
WEAT_JSON = "data/weat_sets.json"

# Output defaults
DEFAULT_OUTPUT_CSV = "single_pc_ablation_results.csv"
DEFAULT_OUTPUT_PNG = "single_pc_ablation_plot.png"
DEFAULT_DISCOVERY_TXT = "single_pc_ablation_discovery.txt"

# Neighborhood metric defaults
DEFAULT_NEIGHBOR_K = 10
DEFAULT_NEIGHBOR_SAMPLE_SIZE = 2000  # set None to use full vocabulary
DEFAULT_RANDOM_SEED = 42


# =============================================================================
# Data containers
# =============================================================================
@dataclass
class ExperimentAssets:
    vocab: list[str]
    X: np.ndarray               # (n_words, dim)
    pcs: np.ndarray             # (n_pcs, dim)
    vocab_to_idx: dict[str, int]


@dataclass
class WeatSets:
    X: list[str]
    Y: list[str]
    A: list[str]
    B: list[str]


# =============================================================================
# Small utilities
# =============================================================================

def get_repo_root() -> Path:
    """
    Infer repo root from this file location:
    repo_root / src / ... / single_pc_ablation.py
    """
    return Path(__file__).resolve().parents[2]


def resolve_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return get_repo_root() / path


def load_glove_txt(path: str | Path | None = None) -> tuple[list[str], np.ndarray]:
    """
    Load GloVe vectors from the raw text file.
    Default path: repo_root / data / glove.6B.300d.txt
    """
    glove_path = resolve_path(path or GLOVE_TXT_RELATIVE)
    if not glove_path.exists():
        raise FileNotFoundError(
            f"GloVe text file not found: {glove_path}\n"
            "Update GLOVE_TXT_RELATIVE or LOAD_EMBEDDING_FN in the config section."
        )

    vocab: list[str] = []
    vectors: list[np.ndarray] = []
    with open(glove_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            parts = line.rstrip().split(" ")
            if len(parts) < 301:
                continue
            word = parts[0]
            try:
                vec = np.asarray(parts[1:], dtype=np.float64)
            except ValueError:
                continue
            if vec.shape[0] != 300:
                continue
            vocab.append(word)
            vectors.append(vec)

    if not vectors:
        raise RuntimeError(f"No vectors were loaded from: {glove_path}")

    X = np.vstack(vectors)
    return vocab, X
def resolve_callable(spec: str | None) -> Callable | None:
    """Resolve 'package.module:function' into a callable."""
    if not spec:
        return None
    if ":" not in spec:
        raise ValueError(f"Function spec must look like 'package.module:function', got: {spec}")
    module_name, fn_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, fn_name)
    if not callable(fn):
        raise TypeError(f"Resolved object is not callable: {spec}")
    return fn


def ensure_2d_pcs(pcs: np.ndarray, emb_dim: int) -> np.ndarray:
    """Accept PCs as (n_pcs, dim) or (dim, n_pcs) and return (n_pcs, dim)."""
    pcs = np.asarray(pcs, dtype=np.float64)
    if pcs.ndim == 1:
        pcs = pcs.reshape(1, -1)
    if pcs.ndim != 2:
        raise ValueError(f"Expected 2D PC array, got shape {pcs.shape}")

    if pcs.shape[1] == emb_dim:
        return pcs
    if pcs.shape[0] == emb_dim:
        return pcs.T
    raise ValueError(f"PC shape {pcs.shape} is incompatible with embedding dim {emb_dim}")


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
    """Remove only one principal component g from every row of X."""
    g = normalize_vector(np.asarray(g, dtype=np.float64))
    return X - np.outer(X @ g, g)


def mean_displacement(X0: np.ndarray, X1: np.ndarray) -> float:
    return float(np.linalg.norm(X1 - X0, axis=1).mean())


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A_n = normalize_rows(A)
    B_n = normalize_rows(B)
    return A_n @ B_n.T


def topk_neighbors(X: np.ndarray, query_indices: np.ndarray, k: int) -> list[set[int]]:
    """Return top-k neighbor id sets for each query index, excluding self."""
    Xn = normalize_rows(X)
    neigh_sets: list[set[int]] = []
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
    sample_size: int | None = 2000,
    seed: int = 42,
) -> float:
    """
    Average overlap of top-k cosine neighbors between original and modified spaces.
    Returns a score in [0, 1].
    """
    n = X0.shape[0]
    if sample_size is None or sample_size >= n:
        query_indices = np.arange(n)
    else:
        rng = np.random.default_rng(seed)
        query_indices = np.sort(rng.choice(n, size=sample_size, replace=False))

    neigh0 = topk_neighbors(X0, query_indices, k)
    neigh1 = topk_neighbors(X1, query_indices, k)

    overlaps = []
    for a, b in zip(neigh0, neigh1):
        overlaps.append(len(a & b) / float(k))
    return float(np.mean(overlaps))


# =============================================================================
# Simple standalone metric implementations (used only if no project fn is wired)
# =============================================================================
def direct_bias_standalone(
    vocab: Sequence[str],
    X: np.ndarray,
    gender_direction: np.ndarray,
    target_words: Sequence[str],
    c: float = 1.0,
) -> float:
    """
    Simple direct bias: mean(|cos(w, g)|^c) over target words.
    Assumes gender_direction is a single unit vector.
    """
    idx = {w: i for i, w in enumerate(vocab)}
    keep = [idx[w] for w in target_words if w in idx]
    if not keep:
        raise ValueError("No target words found for Direct Bias calculation")

    Xn = normalize_rows(X)
    g = normalize_vector(gender_direction)
    vals = np.abs(Xn[keep] @ g) ** c
    return float(np.mean(vals))


def association(w: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
    w_n = normalize_vector(w)
    A_n = normalize_rows(A)
    B_n = normalize_rows(B)
    return float((A_n @ w_n).mean() - (B_n @ w_n).mean())


def weat_effect_size_standalone(vocab: Sequence[str], X: np.ndarray, sets: WeatSets) -> float:
    """Caliskan-style WEAT effect size."""
    idx = {w: i for i, w in enumerate(vocab)}

    def get(words: Sequence[str]) -> np.ndarray:
        found = [idx[w] for w in words if w in idx]
        if not found:
            raise ValueError(f"None of the words were found in vocab: {words[:5]}")
        return X[found]

    X_words = get(sets.X)
    Y_words = get(sets.Y)
    A_words = get(sets.A)
    B_words = get(sets.B)

    s_X = np.array([association(w, A_words, B_words) for w in X_words], dtype=np.float64)
    s_Y = np.array([association(w, A_words, B_words) for w in Y_words], dtype=np.float64)
    denom = np.std(np.concatenate([s_X, s_Y]), ddof=1)
    if np.isclose(denom, 0.0):
        return 0.0
    return float((s_X.mean() - s_Y.mean()) / denom)


# =============================================================================
# Loading helpers
# =============================================================================
def load_embedding_from_npz(path: str | Path) -> tuple[list[str], np.ndarray]:
    data = np.load(path, allow_pickle=True)
    if "X" not in data or "vocab" not in data:
        raise KeyError(f"Embedding NPZ must contain keys 'X' and 'vocab': {path}")
    X = np.asarray(data["X"], dtype=np.float64)
    vocab_raw = data["vocab"].tolist()
    vocab = [str(w) for w in vocab_raw]
    return vocab, X


def load_pcs_from_file(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=True), dtype=np.float64)
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        if "pcs" not in data:
            raise KeyError(f"PC NPZ must contain key 'pcs': {path}")
        return np.asarray(data["pcs"], dtype=np.float64)
    raise ValueError(f"Unsupported PC file type: {path}")


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_assets(definitional_pairs_json: str | None = None) -> ExperimentAssets:
    load_embedding = resolve_callable(LOAD_EMBEDDING_FN)
    load_gender_pcs = resolve_callable(LOAD_GENDER_PCS_FN)

    if load_embedding is not None:
        vocab, X = load_embedding()
    elif EMBEDDING_NPZ:
        vocab, X = load_embedding_from_npz(resolve_path(EMBEDDING_NPZ))
    else:
        raise RuntimeError(
            "No embedding source configured. Set LOAD_EMBEDDING_FN or EMBEDDING_NPZ."
        )

    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Embedding matrix must be 2D, got shape {X.shape}")

    if load_gender_pcs is not None:
        pcs = load_gender_pcs()
    elif GENDER_PCS_FILE:
        pcs = load_pcs_from_file(resolve_path(GENDER_PCS_FILE))
    else:
        pairs_path = definitional_pairs_json or DEFINITIONAL_PAIRS_JSON
        if not pairs_path:
            raise RuntimeError(
                "No gender PC source configured. Set LOAD_GENDER_PCS_FN or GENDER_PCS_FILE, "
                "or provide --definitional-pairs-json."
            )
        pairs = load_definitional_pairs(resolve_path(pairs_path))
        pcs = compute_gender_pcs_from_definitional_pairs(vocab, X, pairs)

    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Embedding matrix must be 2D, got shape {X.shape}")

    pcs = ensure_2d_pcs(np.asarray(pcs, dtype=np.float64), emb_dim=X.shape[1])
    pcs = normalize_rows(pcs)

    vocab = list(map(str, vocab))
    if len(vocab) != X.shape[0]:
        raise ValueError(f"len(vocab)={len(vocab)} does not match X rows={X.shape[0]}")

    vocab_to_idx = {w: i for i, w in enumerate(vocab)}
    return ExperimentAssets(vocab=vocab, X=X, pcs=pcs, vocab_to_idx=vocab_to_idx)



def load_definitional_pairs(path: str | Path) -> list[tuple[str, str]]:
    data = load_json(resolve_path(path))
    if isinstance(data, list):
        pairs_raw = data
    elif isinstance(data, dict) and "pairs" in data:
        pairs_raw = data["pairs"]
    else:
        raise ValueError(
            "Definitional pairs JSON must be either a list of pairs or {'pairs': [...]}."
        )

    pairs: list[tuple[str, str]] = []
    for item in pairs_raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"Invalid definitional pair entry: {item}")
        a, b = str(item[0]), str(item[1])
        pairs.append((a, b))
    if not pairs:
        raise ValueError("No definitional pairs found")
    return pairs


def compute_gender_pcs_from_definitional_pairs(
    vocab: Sequence[str],
    X: np.ndarray,
    pairs: Sequence[tuple[str, str]],
) -> np.ndarray:
    """
    Compute gender PCs once from definitional pair difference vectors using SVD.
    This is only a fallback when no precomputed PC loader/file is configured.
    """
    vocab_to_idx = {w: i for i, w in enumerate(vocab)}
    diffs = []
    missing = []
    for a, b in pairs:
        if a in vocab_to_idx and b in vocab_to_idx:
            diffs.append(X[vocab_to_idx[a]] - X[vocab_to_idx[b]])
        else:
            missing.append((a, b))
    if not diffs:
        raise ValueError(
            "None of the definitional pairs were found in the embedding vocabulary."
        )

    D = np.vstack(diffs).astype(np.float64)
    D = D - D.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(D, full_matrices=False)
    pcs = vt  # shape: (n_components, dim)

    if missing:
        print(
            f"Warning: skipped {len(missing)} definitional pairs not found in vocab.",
            file=sys.stderr,
        )
    return pcs

# =============================================================================
# Discovery / inspection helpers
# =============================================================================
def find_code_candidates(project_root: Path) -> list[tuple[str, int, str]]:
    """
    Heuristic search to help locate relevant existing code.
    Returns tuples: (relative_path, line_number, matched_line).
    """
    patterns = [
        "glove", "embedding", "load_embedding", "word2vec",
        "pca", "principal component", "gender subspace", "definitional",
        "direct bias", "direct_bias",
        "weat", "association",
        "neighbor stability", "neighbour stability", "stability@10",
        "displacement",
    ]

    hits: list[tuple[str, int, str]] = []
    for path in project_root.rglob("*.py"):
        if any(part.startswith(".") for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        lines = text.splitlines()
        for i, line in enumerate(lines, start=1):
            lower = line.lower()
            if any(p in lower for p in patterns):
                hits.append((str(path.relative_to(project_root)), i, line.strip()))
    return hits


def write_discovery_report(project_root: Path, out_path: Path) -> None:
    hits = find_code_candidates(project_root)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("Single-PC ablation discovery report\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Project root: {project_root}\n\n")
        if not hits:
            f.write("No candidate .py matches found.\n")
            return
        current_file = None
        for rel_path, line_no, line in hits:
            if rel_path != current_file:
                current_file = rel_path
                f.write(f"\n[{rel_path}]\n")
            f.write(f"  L{line_no}: {line}\n")


# =============================================================================
# Evaluation set loaders
# =============================================================================
def load_direct_bias_words(cli_path: str | None) -> list[str]:
    path = cli_path or DIRECT_BIAS_WORDS_JSON
    if not path:
        raise RuntimeError(
            "No Direct Bias word list provided. Pass --direct-bias-words-json or set DIRECT_BIAS_WORDS_JSON."
        )
    data = load_json(resolve_path(path))
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict) and "words" in data:
        return [str(x) for x in data["words"]]
    raise ValueError("Direct Bias JSON must be either a list or {'words': [...]}.")


def load_weat_sets(cli_path: str | None) -> WeatSets:
    path = cli_path or WEAT_JSON
    if not path:
        raise RuntimeError(
            "No WEAT set file provided. Pass --weat-json or set WEAT_JSON."
        )
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


# =============================================================================
# Metric wrapper functions
# =============================================================================
def compute_direct_bias(
    vocab: list[str],
    X: np.ndarray,
    gender_direction: np.ndarray,
    target_words: list[str],
) -> float:
    project_fn = resolve_callable(DIRECT_BIAS_FN)
    if project_fn is not None:
        sig = inspect.signature(project_fn)
        kwargs = {}
        if "gender_direction" in sig.parameters:
            kwargs["gender_direction"] = gender_direction
        elif "g" in sig.parameters:
            kwargs["g"] = gender_direction
        if "target_words" in sig.parameters:
            kwargs["target_words"] = target_words
        elif "words" in sig.parameters:
            kwargs["words"] = target_words
        return float(project_fn(vocab, X, **kwargs))

    return direct_bias_standalone(vocab, X, gender_direction, target_words)


def compute_weat(vocab: list[str], X: np.ndarray, weat_sets: WeatSets) -> float:
    project_fn = resolve_callable(WEAT_FN)
    if project_fn is not None:
        sig = inspect.signature(project_fn)
        kwargs = {}
        for name in ["X_words", "targets_x", "x_words", "X_set"]:
            if name in sig.parameters:
                kwargs[name] = weat_sets.X
                break
        for name in ["Y_words", "targets_y", "y_words", "Y_set"]:
            if name in sig.parameters:
                kwargs[name] = weat_sets.Y
                break
        for name in ["A_words", "attrs_a", "a_words", "A_set"]:
            if name in sig.parameters:
                kwargs[name] = weat_sets.A
                break
        for name in ["B_words", "attrs_b", "b_words", "B_set"]:
            if name in sig.parameters:
                kwargs[name] = weat_sets.B
                break
        return float(project_fn(vocab, X, **kwargs))

    return weat_effect_size_standalone(vocab, X, weat_sets)


# =============================================================================
# Plotting
# =============================================================================
def save_plots(df: pd.DataFrame, out_png: str | Path) -> None:
    out_png = Path(out_png)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(df["pc"], df["direct_bias"], marker="o", label="Direct Bias")
    axes[0].plot(df["pc"], df["weat"], marker="o", label="WEAT")
    axes[0].set_ylabel("Metric value")
    axes[0].set_title("Single-PC ablation metrics by PC index")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(df["pc"], df["mean_displacement"], marker="o", label="Mean displacement")
    axes[1].plot(df["pc"], df["neighbor_stability_at_10"], marker="o", label="Neighbor stability@10")
    axes[1].set_xlabel("PC index")
    axes[1].set_ylabel("Metric value")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    plt.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)

    delta_png = out_png.with_name(out_png.stem + "_deltas" + out_png.suffix)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["pc"], df["delta_direct_bias"], marker="o", label="Δ Direct Bias")
    ax.plot(df["pc"], df["delta_weat"], marker="o", label="Δ WEAT")
    ax.plot(df["pc"], df["delta_mean_displacement"], marker="o", label="Δ Mean displacement")
    ax.plot(df["pc"], df["delta_neighbor_stability"], marker="o", label="Δ Neighbor stability")
    ax.axhline(0.0, linewidth=1.0, alpha=0.5)
    ax.set_xlabel("PC index")
    ax.set_ylabel("Delta from original")
    ax.set_title("Single-PC ablation deltas relative to original embedding")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    fig.savefig(delta_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main experiment
# =============================================================================
def run_experiment(
    pc_start: int,
    pc_end: int,
    output_csv: str | Path,
    output_png: str | Path,
    direct_bias_words_json: str | None,
    weat_json: str | None,
    definitional_pairs_json: str | None,
    neighbor_k: int,
    neighbor_sample_size: int | None,
    random_seed: int,
) -> pd.DataFrame:
    assets = build_assets(definitional_pairs_json)
    vocab, X0, pcs = assets.vocab, assets.X, assets.pcs

    if pc_end > pcs.shape[0]:
        raise ValueError(f"Requested PCs up to {pc_end}, but only {pcs.shape[0]} PCs are available")

    direct_bias_words = load_direct_bias_words(direct_bias_words_json)
    weat_sets = load_weat_sets(weat_json)

    # Baseline metrics on original embedding.
    baseline_gender_direction = pcs[0]
    baseline_direct_bias = compute_direct_bias(vocab, X0, baseline_gender_direction, direct_bias_words)
    baseline_weat = compute_weat(vocab, X0, weat_sets)
    baseline_mean_displacement = 0.0
    baseline_neighbor_stability = 1.0

    rows = []
    for pc in range(pc_start, pc_end + 1):
        g = normalize_vector(pcs[pc - 1])
        Xp = remove_single_pc(X0, g)

        direct_bias = compute_direct_bias(vocab, Xp, baseline_gender_direction, direct_bias_words)
        weat = compute_weat(vocab, Xp, weat_sets)
        disp = mean_displacement(X0, Xp)
        stab = neighbor_stability_at_k(
            X0,
            Xp,
            k=neighbor_k,
            sample_size=neighbor_sample_size,
            seed=random_seed,
        )

        rows.append(
            {
                "pc": pc,
                "direct_bias": direct_bias,
                "weat": weat,
                "mean_displacement": disp,
                "neighbor_stability_at_10": stab,
                "delta_direct_bias": direct_bias - baseline_direct_bias,
                "delta_weat": weat - baseline_weat,
                "delta_mean_displacement": disp - baseline_mean_displacement,
                "delta_neighbor_stability": stab - baseline_neighbor_stability,
            }
        )
        print(
            f"PC{pc:02d} | direct_bias={direct_bias:.6f} | weat={weat:.6f} | "
            f"mean_disp={disp:.6f} | neighbor_stability@10={stab:.6f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    save_plots(df, output_png)

    print("\nBaseline metrics on original embedding")
    print(f"  direct_bias:            {baseline_direct_bias:.6f}")
    print(f"  weat:                   {baseline_weat:.6f}")
    print(f"  mean_displacement:      {baseline_mean_displacement:.6f}")
    print(f"  neighbor_stability@10:  {baseline_neighbor_stability:.6f}")

    if not df.empty:
        best_db = df.loc[df["delta_direct_bias"].idxmin()]
        best_weat = df.loc[df["delta_weat"].idxmin()]
        biggest_shift = df.loc[df["mean_displacement"].idxmax()]

        print("\nSummary")
        print(f"  Biggest Direct Bias reduction: PC{int(best_db['pc'])} (Δ={best_db['delta_direct_bias']:.6f})")
        print(f"  Biggest WEAT reduction:        PC{int(best_weat['pc'])} (Δ={best_weat['delta_weat']:.6f})")
        print(f"  Biggest geometry shift:        PC{int(biggest_shift['pc'])} (mean displacement={biggest_shift['mean_displacement']:.6f})")

    print(f"\nSaved CSV  -> {output_csv}")
    print(f"Saved plot -> {output_png}")
    print(f"Saved plot -> {Path(output_png).with_name(Path(output_png).stem + '_deltas' + Path(output_png).suffix)}")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-PC ablation on gender subspace PCs.")
    parser.add_argument("--project-root", type=str, default=".", help="Project root for code discovery report")
    parser.add_argument("--discover-only", action="store_true", help="Only scan project files and write discovery report")
    parser.add_argument("--discovery-out", type=str, default=DEFAULT_DISCOVERY_TXT)

    parser.add_argument("--pc-start", type=int, default=1)
    parser.add_argument("--pc-end", type=int, default=10)
    parser.add_argument("--output-csv", type=str, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-png", type=str, default=DEFAULT_OUTPUT_PNG)

    parser.add_argument("--direct-bias-words-json", type=str, default=None)
    parser.add_argument("--weat-json", type=str, default=None)
    parser.add_argument(
        "--definitional-pairs-json",
        type=str,
        default=None,
        help="JSON file with definitional gender pairs. Used only if no gender PC loader/file is configured.",
    )

    parser.add_argument("--neighbor-k", type=int, default=DEFAULT_NEIGHBOR_K)
    parser.add_argument("--neighbor-sample-size", type=int, default=DEFAULT_NEIGHBOR_SAMPLE_SIZE)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()

    discovery_out = Path(args.discovery_out)
    write_discovery_report(project_root, discovery_out)
    print(f"Discovery report written -> {discovery_out}")

    if args.discover_only:
        return

    run_experiment(
        pc_start=args.pc_start,
        pc_end=args.pc_end,
        output_csv=args.output_csv,
        output_png=args.output_png,
        direct_bias_words_json=args.direct_bias_words_json,
        weat_json=args.weat_json,
        definitional_pairs_json=args.definitional_pairs_json,
        neighbor_k=args.neighbor_k,
        neighbor_sample_size=args.neighbor_sample_size,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
