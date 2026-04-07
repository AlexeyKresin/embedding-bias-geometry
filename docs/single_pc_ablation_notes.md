# Single-PC Ablation Integration Notes

This script was written to minimize project changes and reuse existing code where possible.

## What the script expects

You need to connect it to your current project in one of two ways.

### Option A: Reuse existing project functions

Edit the config section near the top of `single_pc_ablation.py`:

- `LOAD_EMBEDDING_FN`
- `LOAD_GENDER_PCS_FN`
- `DIRECT_BIAS_FN`
- `WEAT_FN`

Each should look like:

```python
LOAD_EMBEDDING_FN = "src.some_module:some_function"
```

Expected return values / signatures are documented in the file.

### Option B: Point to saved arrays

If your project already saves arrays, set:

- `EMBEDDING_NPZ` to an `.npz` file containing `X` and `vocab`
- `GENDER_PCS_FILE` to `.npy` or `.npz` containing `pcs`

Then provide evaluation sets on the command line:

```bash
python single_pc_ablation.py \
  --direct-bias-words-json data/direct_bias_words.json \
  --weat-json data/weat_sets.json
```

## WEAT JSON format

```json
{
  "X": ["career", "corporation", "salary"],
  "Y": ["home", "parents", "children"],
  "A": ["man", "male", "he"],
  "B": ["woman", "female", "she"]
}
```

## Direct Bias word list JSON

Either:

```json
["doctor", "nurse", "engineer"]
```

or:

```json
{"words": ["doctor", "nurse", "engineer"]}
```

## Typical run

```bash
python single_pc_ablation.py \
  --project-root . \
  --direct-bias-words-json data/direct_bias_words.json \
  --weat-json data/weat_sets.json
```

## Discovery mode

To inspect the existing repository for likely code locations first:

```bash
python single_pc_ablation.py --project-root . --discover-only
```

This writes `single_pc_ablation_discovery.txt`.

## Outputs

- `single_pc_ablation_results.csv`
- `single_pc_ablation_plot.png`
- `single_pc_ablation_plot_deltas.png`
- `single_pc_ablation_discovery.txt`

## Metrics / assumptions

- Removes exactly one PC at a time.
- Does not recompute PCA.
- Uses the same vocabulary and evaluation sets across runs.
- Normalizes each PC before ablation.
- Mean displacement is the mean L2 shift per word.
- Neighbor stability@10 is the mean overlap of cosine top-10 neighbors vs. the original space.
- If project metric functions are not wired, the script uses built-in helpers for Direct Bias and WEAT effect size.
