import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


# =========================================================
# CONFIG
# =========================================================

EMBEDDING_FILE = "../data/glove.6B.300d.txt"
OUTPUT_CSV = "../outputs/single_pc_ablation_results.csv"

WORDLIST_DIR = "../wordlists"

NEUTRAL_FILE = os.path.join(WORDLIST_DIR, "neutral_occupations.txt")
WEAT_CAREER_FILE = os.path.join(WORDLIST_DIR, "weat_career.txt")
WEAT_FAMILY_FILE = os.path.join(WORDLIST_DIR, "weat_family.txt")
WEAT_MALE_FILE = os.path.join(WORDLIST_DIR, "weat_male.txt")
WEAT_FEMALE_FILE = os.path.join(WORDLIST_DIR, "weat_female.txt")  # make sure this exists

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

TOP_K_PCS = 10
NN_K = 10


# =========================================================
# UTILS
# =========================================================

def load_word_list(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_glove_embeddings(path):
    embeddings = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            vec = np.asarray(parts[1:], dtype=np.float32)
            embeddings[word] = vec
    return embeddings


def normalize_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def cosine_similarity_vec_to_mat(vec, mat):
    vec_norm = np.linalg.norm(vec)
    if vec_norm == 0:
        return np.zeros(len(mat))
    mat_norms = np.linalg.norm(mat, axis=1)
    mat_norms[mat_norms == 0] = 1.0
    return (mat @ vec) / (mat_norms * vec_norm)


def project_off_pc(vec, pc):
    return vec - np.dot(vec, pc) * pc


def build_gender_pca(embeddings, definitional_pairs):
    diffs = []
    valid_pairs = []

    for a, b in definitional_pairs:
        if a in embeddings and b in embeddings:
            diffs.append(embeddings[a] - embeddings[b])
            valid_pairs.append((a, b))
        else:
            print(f"Skipping missing definitional pair: ({a}, {b})")

    if len(diffs) < 2:
        raise ValueError("Not enough valid definitional pairs found for PCA.")

    X = np.vstack(diffs)
    pca = PCA()
    pca.fit(X)

    return pca, valid_pairs, X


def direct_bias(neutral_vectors, gender_direction):
    vals = []
    gnorm = np.linalg.norm(gender_direction)
    for w in neutral_vectors:
        denom = np.linalg.norm(w) * gnorm
        if denom == 0:
            vals.append(0.0)
        else:
            vals.append(abs(np.dot(w, gender_direction) / denom))
    return float(np.mean(vals))


def weat_association(w, A, B):
    return np.mean(cosine_similarity_vec_to_mat(w, A)) - np.mean(cosine_similarity_vec_to_mat(w, B))


def weat_effect_size(X, Y, A, B):
    s_X = np.array([weat_association(x, A, B) for x in X])
    s_Y = np.array([weat_association(y, A, B) for y in Y])

    numerator = np.mean(s_X) - np.mean(s_Y)
    denom = np.std(np.concatenate([s_X, s_Y]), ddof=1)

    if denom == 0:
        return 0.0

    return float(numerator / denom)


def mean_displacement(original_vectors, transformed_vectors):
    vals = []
    for w, w2 in zip(original_vectors, transformed_vectors):
        denom = np.linalg.norm(w) * np.linalg.norm(w2)
        if denom == 0:
            vals.append(0.0)
        else:
            cosv = np.dot(w, w2) / denom
            vals.append(1.0 - cosv)
    return float(np.mean(vals))


def neighbor_stability(words, original_embs, transformed_embs, k=10):
    X0 = np.vstack([original_embs[w] for w in words])
    X1 = np.vstack([transformed_embs[w] for w in words])

    X0n = normalize_matrix(X0)
    X1n = normalize_matrix(X1)

    n_neighbors = min(k + 1, len(words))

    nn0 = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    nn1 = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")

    nn0.fit(X0n)
    nn1.fit(X1n)

    idx0 = nn0.kneighbors(X0n, return_distance=False)
    idx1 = nn1.kneighbors(X1n, return_distance=False)

    overlaps = []
    for i in range(len(words)):
        n0 = [j for j in idx0[i] if j != i][:k]
        n1 = [j for j in idx1[i] if j != i][:k]

        if k == 0:
            overlaps.append(1.0)
        else:
            overlaps.append(len(set(n0).intersection(set(n1))) / k)

    return float(np.mean(overlaps))


# =========================================================
# MAIN
# =========================================================

def main():
    print("Loading embeddings...")
    embeddings = load_glove_embeddings(EMBEDDING_FILE)
    print(f"Loaded {len(embeddings)} embeddings.")

    print("Loading word lists...")
    neutral_words_raw = load_word_list(NEUTRAL_FILE)
    X_words_raw = load_word_list(WEAT_CAREER_FILE)
    Y_words_raw = load_word_list(WEAT_FAMILY_FILE)
    A_words_raw = load_word_list(WEAT_MALE_FILE)
    B_words_raw = load_word_list(WEAT_FEMALE_FILE)

    # Filter all lists to words present in embeddings
    neutral_words = [w for w in neutral_words_raw if w in embeddings]
    X_words = [w for w in X_words_raw if w in embeddings]
    Y_words = [w for w in Y_words_raw if w in embeddings]
    A_words = [w for w in A_words_raw if w in embeddings]
    B_words = [w for w in B_words_raw if w in embeddings]

    print(f"Neutral words in vocab: {len(neutral_words)} / {len(neutral_words_raw)}")
    print(f"WEAT career words in vocab: {len(X_words)} / {len(X_words_raw)}")
    print(f"WEAT family words in vocab: {len(Y_words)} / {len(Y_words_raw)}")
    print(f"WEAT male words in vocab: {len(A_words)} / {len(A_words_raw)}")
    print(f"WEAT female words in vocab: {len(B_words)} / {len(B_words_raw)}")

    if not neutral_words:
        raise ValueError("No neutral words found in embeddings.")
    if not X_words or not Y_words or not A_words or not B_words:
        raise ValueError("One or more WEAT sets are empty after vocabulary filtering.")

    # PCA on definitional pair differences
    pca, valid_pairs, diff_matrix = build_gender_pca(embeddings, DEFINITIONAL_PAIRS)
    components = pca.components_
    max_pc = min(TOP_K_PCS, len(components))

    print(f"Valid definitional pairs used for PCA: {len(valid_pairs)}")
    print(f"Testing single-PC removal for top {max_pc} PCs")

    # For consistency with your earlier Direct Bias setup:
    # use PC1 as the reference gender direction
    gender_direction = components[0]

    # Baseline vectors
    neutral_vecs_orig = [embeddings[w] for w in neutral_words]

    X_orig = np.vstack([embeddings[w] for w in X_words])
    Y_orig = np.vstack([embeddings[w] for w in Y_words])
    A_orig = np.vstack([embeddings[w] for w in A_words])
    B_orig = np.vstack([embeddings[w] for w in B_words])

    baseline_direct = direct_bias(neutral_vecs_orig, gender_direction)
    baseline_weat = weat_effect_size(X_orig, Y_orig, A_orig, B_orig)

    print(f"Baseline Direct Bias: {baseline_direct:.6f}")
    print(f"Baseline WEAT: {baseline_weat:.6f}")

    # Common pool for geometry metrics
    analysis_words = sorted(set(neutral_words + X_words + Y_words + A_words + B_words))
    analysis_orig = {w: embeddings[w] for w in analysis_words}

    results = []

    for i in range(max_pc):
        pc = components[i]

        transformed = {}
        for w in analysis_words:
            transformed[w] = project_off_pc(embeddings[w], pc)

        # Direct Bias
        neutral_vecs_new = [transformed[w] for w in neutral_words]
        direct_new = direct_bias(neutral_vecs_new, gender_direction)

        # WEAT
        X_new = np.vstack([transformed[w] for w in X_words])
        Y_new = np.vstack([transformed[w] for w in Y_words])
        A_new = np.vstack([transformed[w] for w in A_words])
        B_new = np.vstack([transformed[w] for w in B_words])
        weat_new = weat_effect_size(X_new, Y_new, A_new, B_new)

        # Geometry
        orig_mat = np.vstack([analysis_orig[w] for w in analysis_words])
        new_mat = np.vstack([transformed[w] for w in analysis_words])

        disp = mean_displacement(orig_mat, new_mat)
        stability = neighbor_stability(analysis_words, analysis_orig, transformed, k=NN_K)

        results.append({
            "pc_index": i + 1,
            "direct_bias_after_removing_only_pc": direct_new,
            "delta_direct_bias": baseline_direct - direct_new,
            "weat_after_removing_only_pc": weat_new,
            "delta_weat": baseline_weat - weat_new,
            "mean_displacement": disp,
            "nn_stability_at10": stability,
        })

        print(
            f"PC{i+1}: "
            f"DirectBias={direct_new:.6f}, "
            f"ΔDirectBias={baseline_direct - direct_new:.6f}, "
            f"WEAT={weat_new:.6f}, "
            f"ΔWEAT={baseline_weat - weat_new:.6f}, "
            f"Disp={disp:.6f}, "
            f"NN@10={stability:.6f}"
        )

    df = pd.DataFrame(results)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved results to: {OUTPUT_CSV}")
    print("\nSingle-PC ablation summary:")
    print(df.round(6))


if __name__ == "__main__":
    main()