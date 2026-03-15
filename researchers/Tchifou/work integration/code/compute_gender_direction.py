import numpy as np
from sklearn.decomposition import PCA
from gensim.models import KeyedVectors

EMBEDDING_PATH = "glove.6B.300d.txt"   # we will download next

def load_word_pairs(path):
    pairs = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            w1, w2 = line.strip().split()
            pairs.append((w1, w2))
    return pairs

print("Loading embeddings (this will be slow first time)...")
model = KeyedVectors.load_word2vec_format(EMBEDDING_PATH, binary=False, no_header=True)

pairs = load_word_pairs("wordlists/definitional_pairs.txt")

diff_vectors = []

for w1, w2 in pairs:
    if w1 in model and w2 in model:
        diff_vectors.append(model[w1] - model[w2])
    else:
        print("Missing:", w1, w2)

diff_vectors = np.array(diff_vectors)

print("Running PCA...")
pca = PCA(n_components=10)
pca.fit(diff_vectors)

gender_direction = pca.components_[0]

print("Explained variance ratio:")
print(pca.explained_variance_ratio_)

np.save("gender_direction.npy", gender_direction)
print("Saved gender direction vector to gender_direction.npy")