# Protocol (v0.1)

Embedding:
- GloVe 6B 300d

Wordlists:
- definitional_pairs: wordlists/definitional_pairs.txt
- neutral_words: wordlists/neutral_occupations.txt

Gender subspace:
- PCA on definitional pair difference vectors
- report explained variance ratios (top 10)

Experiments (next):
- PC removal sweep: k ∈ {0,1,2,3,5,8,10}
Metrics:
- direct_bias: mean |cos(w, g)| over neutral words
- DI_displacement: mean (1 - cos(w, w'))
- NN_stability@10: overlap of top-10 neighbors before vs after
Outputs:
- outputs/pc_sweep.csv