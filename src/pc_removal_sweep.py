import os
import numpy as np
from sklearn.decomposition import PCA
from gensim.models import KeyedVectors
from numpy.linalg import norm

EMBEDDING_PATH = "glove.6B.300d.txt"
PAIRS_PATH = "wordlists/definitional_pairs.txt"
NEUTRAL_PATH = "wordlists/neutral_occupations.txt"

# Experiment settings
K_VALUES = [0, 1, 2, 3, 5, 8, 10]
TOP_N = 10               # neighbors@10
CANDIDATE_VOCAB = 50000  # limit for neighbor search to keep it fast

def unit(v):
    n = norm(v)
    return v if n == 0 else (v / n)

def cosine(a, b):
    na = norm(a); nb = norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

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

def load_words(path):
    words = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            w = line.strip()
            if w:
                words.append(w)
    return words

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

def direct_bias(model, neutral_words, g_dir, pcs, k):
    g = unit(g_dir.astype(np.float64))
    vals = []
    for w in neutral_words:
        if w not in model:
            continue
        v = unit(model[w].astype(np.float64))
        v2 = unit(remove_top_k(v, pcs, k))
        vals.append(abs(cosine(v2, g)))
    return float(np.mean(vals)) if vals else float("nan")

def mean_displacement(model, neutral_words, pcs, k):
    vals = []
    for w in neutral_words:
        if w not in model:
            continue
        v = unit(model[w].astype(np.float64))
        v2 = unit(remove_top_k(v, pcs, k))
        vals.append(1.0 - cosine(v, v2))
    return float(np.mean(vals)) if vals else float("nan")

def build_candidate_matrix(model, max_words):
    words = model.index_to_key[:max_words]
    mat = np.array([unit(model[w].astype(np.float64)) for w in words], dtype=np.float64)
    return words, mat

def top_neighbors_from_matrix(query_vec, cand_words, cand_mat, topn=10):
    # cosine similarity with pre-normalized matrix = dot product
    sims = cand_mat @ query_vec
    # exclude exact match by setting to -inf if present handled outside
    idx = np.argpartition(-sims, topn)[:topn]
    idx = idx[np.argsort(-sims[idx])]
    return [cand_words[i] for i in idx]

def neighbor_stability(model, neutral_words, pcs, k, cand_words, cand_mat, topn=10):
    overlaps = []
    cand_index = {w: i for i, w in enumerate(cand_words)}  # for excluding self if needed

    for w in neutral_words:
        if w not in model:
            continue

        v = unit(model[w].astype(np.float64))
        v2 = unit(remove_top_k(v, pcs, k))

        # Temporarily exclude the word itself (if it is in candidate list)
        if w in cand_index:
            i = cand_index[w]
            orig = cand_mat[i].copy()
            cand_mat[i] = 0.0  # makes dot product 0; not perfect but prevents self being top-1

            nb_before = top_neighbors_from_matrix(v, cand_words, cand_mat, topn=topn)
            nb_after  = top_neighbors_from_matrix(v2, cand_words, cand_mat, topn=topn)

            cand_mat[i] = orig
        else:
            nb_before = top_neighbors_from_matrix(v, cand_words, cand_mat, topn=topn)
            nb_after  = top_neighbors_from_matrix(v2, cand_words, cand_mat, topn=topn)

        overlap = len(set(nb_before).intersection(nb_after)) / float(topn)
        overlaps.append(overlap)

    return float(np.mean(overlaps)) if overlaps else float("nan")

def main():
    print("Loading embeddings...")
    model = KeyedVectors.load_word2vec_format(EMBEDDING_PATH, binary=False, no_header=True)

    pairs = load_pairs(PAIRS_PATH)
    neutral = load_words(NEUTRAL_PATH)

    print("Computing gender PCs...")
    pcs, evr, kept = compute_gender_pcs(model, pairs, n_components=20)
    g_dir = pcs[0]
    print("EVR top 10:", evr[:10])
    print(f"Definitional pairs used: {len(kept)}/{len(pairs)}")

    print(f"Building candidate vocab matrix (top {CANDIDATE_VOCAB} words) for neighbor stability...")
    cand_words, cand_mat = build_candidate_matrix(model, CANDIDATE_VOCAB)

    os.makedirs("outputs", exist_ok=True)
    out_csv = os.path.join("outputs", "pc_removal_sweep_results.csv")

    rows = []
    for k in K_VALUES:
        print(f"\n=== k={k} ===")
        db = direct_bias(model, neutral, g_dir, pcs, k)
        disp = mean_displacement(model, neutral, pcs, k)
        stab = neighbor_stability(model, neutral, pcs, k, cand_words, cand_mat, topn=TOP_N)
        print(f"direct_bias={db:.6f}  mean_displacement={disp:.6f}  nn_stability@{TOP_N}={stab:.4f}")
        rows.append((k, db, disp, stab))

    with open(out_csv, "w", encoding="utf8") as f:
        f.write("k,direct_bias,mean_displacement,nn_stability_at10\n")
        for k, db, disp, stab in rows:
            f.write(f"{k},{db},{disp},{stab}\n")

    print("\nSaved:", out_csv)

if __name__ == "__main__":
    main()