# embedding-bias-geometry

Geometric analysis of bias and semantic distortion in word embeddings.

Official arXiv preprint: https://arxiv.org/abs/2606.07964

**What Does Debiasing Really Remove? A Geometric Study of PCA-Based Gender Debiasing in Word Embeddings**
Alexey Kresin, Tchifou M. Dieffi, Tomer Caspi

## Abstract

Bias mitigation techniques are widely used to reduce unwanted social biases in word embeddings. In this work, we investigate what these methods actually remove from the embedding space. We show that direct gender bias is concentrated in a low-dimensional subspace and can be substantially reduced by removing the leading principal component. However, increasing debiasing strength progressively alters the geometric structure of embeddings, affecting semantic relationships. Our findings highlight the trade-off between fairness and preservation of linguistic information in static word embeddings.

## Main Findings

* Direct gender bias is dominated by the first principal component.
* Associative bias (WEAT) is more distributed and exhibits non-monotonic behavior.
* Increasing debiasing strength progressively distorts embedding geometry.
* Similar patterns are observed across GloVe, Word2Vec, and FastText.
* Fairness interventions involve trade-offs and should be evaluated beyond a single bias metric.

## Goals

* SPIRE 2026 (Hood College): early framework and pilot evidence
* arXiv publication ✅
* AAAI 2027: geometry-first evaluation and bias–utility trade-off analysis

## Repo structure

* `src/` core code (load embeddings, debias transforms, metrics)
* `experiments/` runnable scripts and configurations
* `research-log/` weekly research logbook
* `wordlists/` definitional pairs, neutral words, evaluation vocabularies
* `figures/` exported plots
* `paper/` manuscript sources and LaTeX files

## Citation

If you use this work, please cite:

```bibtex
@article{kresin2026debiasing,
  title={What Does Debiasing Really Remove? A Geometric Study of PCA-Based Gender Debiasing in Word Embeddings},
  author={Kresin, Alexey and Dieffi, Tchifou M. and Caspi, Tomer},
  journal={arXiv preprint arXiv:2606.07964},
  year={2026}
}
```
