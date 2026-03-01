import numpy as np
from sklearn.decomposition import PCA
from gensim.models import KeyedVectors
from numpy.linalg import norm

EMBEDDING_PATH = "glove.6B.300d.txt"
PAIRS_PATH = "wordlists/definitional_pairs.txt"
NEUTRAL_PATH = "wordlists/neutral_occupations.txt"
K_VALUES = [0, 1, 2, 3]

def unit(v):
    n = norm(v)
    return v if n == 0 else v / n

def cosine(a, b):
    na = norm(a); nb = norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def load_pairs(path):
    pairs = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            if line.strip():
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

def compute_pcs(model, pairs, n_components=10):
    diffs = []
    for w1, w2 in pairs:
        if w1 in model and w2 in model:
            diffs.append(model[w1] - model[w2])
    diffs = np.array(diffs)
    pca = PCA(n_components=min(n_components, diffs.shape[0]))
    pca.fit(diffs)
    return pca.components_, pca.explained_variance_ratio_

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

def main():
    print("Loading embeddings...")
    model = KeyedVectors.load_word2vec_format(EMBEDDING_PATH, binary=False, no_header=True)

    pairs = load_pairs(PAIRS_PATH)
    neutral = load_words(NEUTRAL_PATH)

    pcs, evr = compute_pcs(model, pairs, n_components=10)
    g_dir = pcs[0]
    print("EVR top 10:", evr)

    for k in K_VALUES:
        db = direct_bias(model, neutral, g_dir, pcs, k)
        print(f"k={k} direct_bias={db:.6f}")

if __name__ == "__main__":
    main()