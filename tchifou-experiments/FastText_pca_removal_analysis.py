import numpy as np
import pandas as pd
import os
import io
import requests
from sklearn.decomposition import PCA
from gensim.models import KeyedVectors
import gensim.downloader as api
from scipy.stats import spearmanr

# --- 1. PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
INPUT_DIR = os.path.join(DATA_DIR, 'input')
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')

for folder in [INPUT_DIR, OUTPUT_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# File Paths - Updated for FastText
MODEL_FILENAME = os.path.join(INPUT_DIR, 'fasttext-wiki-news-300.bin')
WORDSIM_FILENAME = os.path.join(INPUT_DIR, 'wordsim353_combined.csv')
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'fasttext_debiasing_results.csv')

def load_model():
    """Loads the FastText model from input folder or downloads it."""
    if os.path.exists(MODEL_FILENAME):
        print(f"Loading local FastText model from {MODEL_FILENAME}...")
        # Note: use load_facebook_vectors for .bin or load_word2vec_format for .vec
        return KeyedVectors.load_word2vec_format(MODEL_FILENAME, binary=False)
    else:
        print(f"Model not found in {INPUT_DIR}. Downloading FastText (Wiki News)...")
        # 'fasttext-wiki-news-subwords-300' is a standard high-quality option
        model = api.load('fasttext-wiki-news-subwords-300')
        print(f"Saving model to {MODEL_FILENAME}...")
        model.save_word2vec_format(MODEL_FILENAME, binary=False)
        return model

def load_wordsim353():
    """Loads WordSim-353 and saves it to the data/input folder."""
    if os.path.exists(WORDSIM_FILENAME):
        print(f"Loading WordSim-353 from: {WORDSIM_FILENAME}")
        df = pd.read_csv(WORDSIM_FILENAME)
    else:
        url = "https://raw.githubusercontent.com/infofreund/wordsim353/master/combined.csv"
        print("Downloading WordSim-353 to input folder...")
        try:
            response = requests.get(url)
            df = pd.read_csv(io.StringIO(response.text))
            df.to_csv(WORDSIM_FILENAME, index=False)
        except Exception as e:
            print(f"Error loading WordSim-353: {e}")
            return []

    pairs = []
    for _, row in df.iterrows():
        pairs.append(((str(row['Word 1']).lower(), str(row['Word 2']).lower()), row['Human (mean)']))
    return pairs

# --- 2. CORE LOGIC (Remains mathematically identical for FastText) ---

def get_gender_subspace(model, pairs):
    diff_vectors = []
    for female, male in pairs:
        if female in model and male in model:
            diff_vectors.append(model[female] - model[male])
    pca = PCA()
    pca.fit(diff_vectors)
    return pca.components_

def remove_pc_projections(embeddings, pcs_to_remove):
    debiased_embeddings = embeddings.copy()
    for v in pcs_to_remove:
        v = v / np.linalg.norm(v) 
        projections = np.outer(debiased_embeddings @ v, v)
        debiased_embeddings -= projections
    
    norms = np.linalg.norm(debiased_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1 
    return debiased_embeddings / norms

# --- 3. KPI FUNCTIONS (Compatible with any unit-norm vectors) ---

def calculate_direct_bias(embeddings, word_indices, gender_direction):
    neutral_vectors = embeddings[word_indices]
    cos_sims = neutral_vectors @ gender_direction
    return np.mean(np.abs(cos_sims))

def calculate_mvd(original_embeddings, debiased_embeddings):
    distances = np.linalg.norm(original_embeddings - debiased_embeddings, axis=1)
    return np.mean(distances)

def evaluate_semantic_geometry(debiased_vectors, model, eval_pairs):
    sims = []
    gold_standard = []
    for (w1, w2), score in eval_pairs:
        if w1 in model and w2 in model:
            idx1, idx2 = model.key_to_index[w1], model.key_to_index[w2]
            sims.append(np.dot(debiased_vectors[idx1], debiased_vectors[idx2]))
            gold_standard.append(score)
    if len(sims) < 2: return 0.0
    return spearmanr(sims, gold_standard).correlation

def calculate_weat_score(embeddings, model, target_A, target_B, attr_X, attr_Y):
    def s_w(w_idx, X_idx, Y_idx):
        mean_X = np.mean([np.dot(embeddings[w_idx], embeddings[x]) for x in X_idx])
        mean_Y = np.mean([np.dot(embeddings[w_idx], embeddings[y]) for y in Y_idx])
        return mean_X - mean_Y

    A_idx = [model.key_to_index[w] for w in target_A if w in model]
    B_idx = [model.key_to_index[w] for w in target_B if w in model]
    X_idx = [model.key_to_index[w] for w in attr_X if w in model]
    Y_idx = [model.key_to_index[w] for w in attr_Y if w in model]

    if not (A_idx and B_idx and X_idx and Y_idx): return 0.0

    scores_A = [s_w(a, X_idx, Y_idx) for a in A_idx]
    scores_B = [s_w(b, X_idx, Y_idx) for b in B_idx]
    return (np.mean(scores_A) - np.mean(scores_B)) / np.std(scores_A + scores_B)

# --- 4. EXECUTION ---

model = load_model()
sem_pairs = load_wordsim353()

gender_pairs = [('he', 'she'), ('man', 'woman'), ('male', 'female'), ('boy', 'girl'), ('father', 'mother'), ('son', 'daughter'), 
                ('brother', 'sister'), ('king', 'queen'), ('husband', 'wife'), ('actor', 'actress'), ('uncle', 'aunt'), 
                ('gentleman', 'lady'), ('grandfather', 'grandmother'), ('prince', 'princess'), ('monk', 'nun')]
gender_components = get_gender_subspace(model, gender_pairs)
gender_dir = gender_components[0]

neutral_words = ['doctor', 'nurse', 'engineer', 'teacher', 'scientist', 'programmer', 'manager', 'lawyer', 'mathematician', 'homemaker', 'receptionist', 'librarian', 
                 'surgeon', 'chef', 'journalist', 'architect', 'accountant', 'designer', 'assistant', 'boss']
neutral_idx = [model.key_to_index[w] for w in neutral_words if w in model.key_to_index]

# WEAT Sets (Science vs Art mapped to Male/Female attributes)
X= ['science', 'technology', 'physics', 'chemistry', 'einstein', 'nasa']
Y= ['poetry', 'art', 'dance', 'literature', 'novel', 'symphony']
A= ['man', 'male', 'he', 'him', 'boy', 'brother']
B= ['woman', 'female', 'she', 'her', 'girl', 'sister']

# FastText models from api.load often come pre-normalized, but we ensure it here
original_vectors = model.get_normed_vectors()
results = []

print("\nStarting Progressive PCA Removal Analysis on FastText...")
for n in range(11):
    current_pcs = gender_components[:n]
    debiased_vectors = remove_pc_projections(original_vectors, current_pcs)
    
    db = calculate_direct_bias(debiased_vectors, neutral_idx, gender_dir)
    mvd = calculate_mvd(original_vectors, debiased_vectors)
    weat = calculate_weat_score(debiased_vectors, model, X, Y, A, B)
    sem_geo = evaluate_semantic_geometry(debiased_vectors, model, sem_pairs)
    
    results.append({
        'pcs_removed': n,
        'direct_bias': db,
        'mvd': mvd,
        'weat_score': weat,
        'semantic_geometry': sem_geo
    })
    print(f"PCs Removed: {n:2} | DB: {db:.4f} | MVD: {mvd:.4f} | WEAT: {weat:.4f} | Sem: {sem_geo:.4f}")

# --- 5. EXPORT ---
df_results = pd.DataFrame(results)
df_results.to_csv(OUTPUT_CSV, index=False)

print("\n--- Analysis Complete ---")
print(f"FastText Results saved in: {OUTPUT_DIR}")