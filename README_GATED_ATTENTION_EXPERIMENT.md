# Cached-feature gated attention experiment

Put `gated_attention_experiment.py` next to the existing `fourbranch.py` and
`fusion_heads.py`. Use the same CUDA PyTorch environment as the original project.
No new encoder weights, image augmentation, feature extraction, or extra Python
packages are needed. This trains fusion heads on existing NPZ caches only.

## Architecture

Each selected branch has its own LayerNorm -> Linear -> GELU projection to 128
dimensions. Branch identity embeddings are added. A four-head self-attention
block with pre-normalization, residual connections and a feed-forward network
models interactions among branches. A gated attention pooling layer computes
`softmax(w^T(tanh(Vh) * sigmoid(Uh)))` across the contextualized branch tokens.
The weighted sum is passed to the usual 256-unit GELU/dropout classifier.

This is a new attention head, distinct from the original `dynamic_gated` MLP.
Pooling weights describe how contextualized tokens are pooled; after attention,
each token can already contain information from other branches. They are not
causal or independent branch-importance estimates. Attention does not guarantee
removal of redundant information or better performance.

## Windows CMD: same four branches, six fusion baselines

Frozen cache:

```cmd
python gated_attention_experiment.py --cache cache_embeddings --fusion all --out outputs/gated_frozen_s42 --predict-test
```

LoRA cache:

```cmd
python gated_attention_experiment.py --cache cache_embeddings_lora_s42 --fusion all --out outputs/gated_lora_s42 --predict-test
```

Each command trains concat, projected_concat, static_weighted, dynamic_gated,
gated_concat and gated_attention. It uses the same seed and training defaults
for each mode. Parameter counts differ, so this is a practical architecture
comparison rather than a matched-parameter-capacity experiment.

## Only the new attention head

```cmd
python gated_attention_experiment.py --cache cache_embeddings --fusion gated_attention --branches densenet raddino chexfound chexzero --out outputs/gated_attention_frozen_s42 --predict-test
```

For B234, use `--branches raddino chexfound chexzero`; for B134, use
`--branches densenet chexfound chexzero`. Do not compare results from different
branch subsets as if they were a controlled fusion-head ablation.

## Defaults and data checks

AdamW, learning rate 0.001, weight decay 0.0001, batch size 64, maximum 80 epochs,
patience 15, dropout 0.35, hidden size 256, projection size 128, gate hidden size
64, seed 42, standard CE, square-root-frequency resampling with replacement.
No auxiliary COVID head is used. Selection uses four-class validation macro-F1.

Each NPZ must contain `x`, `names`, `y`. Each split must match its official CSV
in sample order and labels. This runner expects one row per image, not a
multi-view cache. Features must be finite and have consistent dimensions.
Split sample IDs must not overlap. Test caches must have empty labels.
These checks do not establish patient-level independence or checkpoint provenance;
use the same verified caches as the original experiment.

## Outputs

A unique directory is created for every invocation, preserving previous runs:

```text
outputs/gated_frozen_s42/<timestamp_random>/
    run_config.json
    summary.csv
    gated_attention/
        best.pt
        best_val_report.json
        val_probabilities.csv
        val_gate_weights.csv
        val_gate_summary.csv
        history.csv
        curves.png
        confusion_matrix.csv
        confusion_matrix.png
        submission.csv
        submission_probabilities.csv
    <other requested fusion modes>/...
```

The summary uses scores in [0,1], not percent. Upload `submission.csv`, not
`submission_probabilities.csv`. Test labels and Public leaderboard scores are
not used for early stopping or mode ranking. Predictions use the validation-best
checkpoint. Cache and source hashes are recorded in `run_config.json`.

The original files are not modified. The extended model factory is installed
only in this process. Consequently, original `fourbranch.py predict` cannot
load the new `gated_attention` mode by itself; use `--predict-test` in this runner.

## Synthetic test

```cmd
python test_gated_attention_experiment.py
```

This checks gradient propagation, normalized pooling weights, training all six
modes, checkpoint reload, validation probability reproduction, submission order
and one-hot predictions, and rejection of an intentionally misaligned cache.
