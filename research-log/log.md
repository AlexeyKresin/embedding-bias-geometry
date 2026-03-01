\# Research Log — Embedding Bias Geometry



\## Hypotheses

H1: Gender bias is multi-dimensional (spans multiple PCs)

H2: Removing PC1 leaves residual bias

H3: Removing more PCs reduces bias but harms semantic meaning

H4: There is a measurable bias–destructiveness trade-off frontier



\## 2026-03-01

\- Initialized repository structure.

\- Next: freeze wordlists (P, N) and implement baseline runner.


### 2026-03-01 — Gender subspace PCA (GloVe 6B 300d)

**Setup**
- Embedding: GloVe 6B, 300d
- Definitional pairs: `wordlists/definitional_pairs.txt`
- Method: PCA on difference vectors (w₁ − w₂)

**Results**
Explained variance ratios (top 10 PCs):
[0.2998, 0.1467, 0.1071, 0.0943, 0.0764, 0.0544, 0.0373, 0.0369, 0.0327, 0.0252]

Cumulative variance:
- PC1: 29.98%
- PC1–3: ~55%
- PC1–5: ~72%

**Reproducibility details**

- Script: src/compute_gender_direction.py
- Python environment: research_env (Python 3.12, gensim 4.3.2, sklearn 1.3.2)
- Embedding file: glove.6B.300d.txt
- Output artifact: gender_direction.npy (saved in project root)
- Number of definitional pairs found in vocabulary: (leave blank for now)



**Interpretation (preliminary)**
- Gender-related structure is not concentrated in a single principal component.
- Multiple components beyond PC1 contribute substantial variance.
- This challenges the assumption that a single “gender direction” fully captures bias under this setup.

**Claims we can safely make**
- The gender subspace estimated from definitional pairs is multi-component in GloVe 300d.
- Removing only PC1 is likely insufficient to eliminate all gender-correlated structure.

**Claims we avoid for now**
- Generalization to all embeddings or all wordlists.
- Claims about downstream task performance or fairness improvements.

**Next decisive experiment**
- Perform k-PC removal (k = 1, 2, 3, 5, 8, 10).
- Measure bias reduction (direct bias / WEAT) versus semantic impact (Destructiveness Index).
- Plot bias–destructiveness trade-off curves.

### 2026-03-01 — PCA variance ratios Cumulative top -5 = 0.724

### 2026-03-01 — k-PC removal sweep

We removed the top k principal components of the gender subspace (k = 0,1,2,3,5,8,10) and measured:
- direct bias
- mean displacement
- neighbor stability@10

Key observation:
Direct bias drops to 0 after removing only the first principal component (k=1).  
However, both displacement and neighbor stability continue to change monotonically as k increases.

Interpretation:
The standard direct bias metric becomes insensitive to remaining gender-related structure after removal of the first component. Additional components affect semantic structure without affecting the bias metric.

This suggests that fairness metrics based on a single gender direction underestimate residual structured bias.

### Plots Analysis at \figures
Observation:
Direct bias becomes zero after removing only the first principal component.

However, semantic displacement and neighbor instability continue to increase as additional components are removed.

Conclusion:
The direct bias metric detects only the primary gender direction (PC1). Higher-order gender components still influence semantic geometry but are not captured by the metric.

Implication:
Current fairness evaluation may certify embeddings as unbiased while meaningful representational changes continue to occur.

