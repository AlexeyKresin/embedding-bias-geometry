import os
import csv
import sys
import numpy as np
from numpy.linalg import norm
from sklearn.decomposition import PCA
from gensim.models import KeyedVectors

# Usage:
#   python src/weat_pc_sweep.py path/to/embedding.txt
# Default:
#   python src/weat_pc_sweep.py

EMBEDDING_PATH = sys.argv[1] if len(sys.argv) > 1 else "glove.6B.300d.txt"

PAIRS_PATH = "wordlists/definitional_pairs.txt"
X_PATH = "wordlists/weat_male.txt"
Y_PATH = "wordlists/weat_female.txt"
A_PATH = "wordlists/weat_career.txt"
B_PATH = "wordlists/weat_family.txt"

K_VALUES = [0, 1, 2, 3, 5, 8, 10]

def unit(v):
    n = norm(v)
    return v if n == 0 else v / n

def cosine(a, b):
    na = norm(a); nb = norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def load_words(path):
    words = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            w = line.strip()
            if w:
                words.append(w)
    return words

def load_pairs(path):
    pairs = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            w1, w2 = line.split()
            pairs.append((w1, w2))
    return pairs

def compute_gender_pcs(model, pairs, n_components=20):
    diffs = []
    kept = []
    for w1, w2 in pairs:
        if w1 in model and w2 in model:
            diffs.append(model[w1] - model[w2])
            kept.append((w1, w2))
    diffs = np.array(diffs)
    if diffs.shape[0] < 2:
        raise ValueError("Not enough definitional pairs found in vocab.")
    pca = PCA(n_components=min(n_components, diffs.shape[0]))
    pca.fit(diffs)
    return pca.components_, pca.explained_variance_ratio_, kept

def remove_top_k(v, pcs, k):
    out = v.astype(np.float64, copy=True)
    for i in range(k):
        pc = pcs[i]
        out = out - np.dot(out, pc) * pc
    return out

def s_association(w_vec, A_vecs, B_vecs):
    # s(w, A, B) = mean cos(w,a) - mean cos(w,b)
    a = np.mean([cosine(w_vec, av) for av in A_vecs])
    b = np.mean([cosine(w_vec, bv) for bv in B_vecs])
    return a - b

def weat_effect_size(X_vecs, Y_vecs, A_vecs, B_vecs):
    # Cohen's d effect size (Caliskan et al.)
    sx = np.array([s_association(xv, A_vecs, B_vecs) for xv in X_vecs], dtype=np.float64)
    sy = np.array([s_association(yv, A_vecs, B_vecs) for yv in Y_vecs], dtype=np.float64)
    mean_diff = float(np.mean(sx) - np.mean(sy))
    pooled = np.concatenate([sx, sy])
    std = float(np.std(pooled, ddof=1)) if pooled.size > 1 else 0.0
    if std == 0.0:
        return 0.0
    return mean_diff / std

def get_vecs(model, words, pcs, k):
    vecs = []
    kept = []
    for w in words:
        if w in model:
            v = unit(model[w].astype(np.float64))
            v2 = unit(remove_top_k(v, pcs, k))
            vecs.append(v2)
            kept.append(w)
    return vecs, kept

def main():
    print("Loading embeddings:", EMBEDDING_PATH)
    model = KeyedVectors.load_word2vec_format(EMBEDDING_PATH, binary=False, no_header=True)

    pairs = load_pairs(PAIRS_PATH)
    pcs, evr, kept_pairs = compute_gender_pcs(model, pairs, n_components=20)
    print("Gender PCA EVR top 10:", evr[:10])
    print(f"Definitional pairs used: {len(kept_pairs)}/{len(pairs)}")

    X = load_words(X_PATH)
    Y = load_words(Y_PATH)
    A = load_words(A_PATH)
    B = load_words(B_PATH)

    os.makedirs("outputs", exist_ok=True)
    out_csv = os.path.join("outputs", "weat_pc_sweep_results.csv")

    rows = []
    for k in K_VALUES:
        A_vecs, A_kept = get_vecs(model, A, pcs, k)
        B_vecs, B_kept = get_vecs(model, B, pcs, k)
        X_vecs, X_kept = get_vecs(model, X, pcs, k)
        Y_vecs, Y_kept = get_vecs(model, Y, pcs, k)

        # Basic sanity: need at least 2 items in each set ideally
        if min(len(A_vecs), len(B_vecs), len(X_vecs), len(Y_vecs)) < 2:
            print(f"k={k}: not enough words found in vocab (X={len(X_vecs)},Y={len(Y_vecs)},A={len(A_vecs)},B={len(B_vecs)})")
            effect = float("nan")
        else:
            effect = weat_effect_size(X_vecs, Y_vecs, A_vecs, B_vecs)

        print(f"k={k} WEAT_effect_size={effect:.6f}  (kept: X={len(X_kept)}, Y={len(Y_kept)}, A={len(A_kept)}, B={len(B_kept)})")
        rows.append((k, effect, len(X_kept), len(Y_kept), len(A_kept), len(B_kept)))

    with open(out_csv, "w", encoding="utf8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "weat_effect_size", "X_kept", "Y_kept", "A_kept", "B_kept"])
        for r in rows:
            w.writerow(r)

    print("Saved:", out_csv)

if __name__ == "__main__":
    main()