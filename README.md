# Eval-awareness in Qwen3.5-4B: probing, base rates, and steering

Code for a ~20h MATS 12.0 application project on **eval awareness** — whether a
model internally represents "I am being evaluated" vs "I am in deployment",
whether a probe for it survives realistic base rates, and whether the direction
is causally load-bearing.

Model: `Qwen/Qwen3.5-4B` (revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`),
bf16 on one A10G. 32 layers, d_model 2560, hybrid attention (8 full-attention
layers at 3, 7, ..., 31; the other 24 are GatedDeltaNet linear attention).

## Headline results

| | |
|---|---|
| Probe, layer 12, pair-grouped 5-fold CV | **0.960 ± 0.060** (AUC 0.990) |
| Same probe, leave-one-framing-template-out | **0.880** |
| Baseline: random direction / shuffled labels / prompt length only | 0.540 / 0.502 / 0.494 |
| Ask the model directly, **explicit** framings | AUC **0.984** (≈ probe) |
| Ask the model directly, **implicit** framings | AUC **0.560** (probe: 0.930) |
| Probe precision at a 1% base rate | **0.327** (~2 false alarms per true hit) |
| Steering ±1 norm: refusal rate | 4% → **35%**, but a random direction gives **22%** (p = 0.0596) |

Short version: framing is easy to decode, the probe only beats "just ask the
model" on implicit cues, it is not usable as a monitor at realistic prevalence,
and we could not show the direction is causally load-bearing.

## Layout

```
src/01_build_dataset.py            50 matched eval/deployment prompt pairs
src/02a_inspect_model.py           module-tree / shape verification (saves nothing)
src/02_extract_activations.py      residual stream, all 32 layers, last token
src/03_probe_and_baselines.py      probes + baselines A/C/D + LOTO + base rates
src/03b_blackbox_baseline.py       baseline B: ask the model (with acquiescence check)
src/04a_steering_smoketest.py      verifies the steering hook before using it
src/04_steering.py                 E1 self-report, E2 damage, E3 behaviour
src/04b_score_refusals.py          post-hoc refusal metric on saved generations
src/04c_refusal_replication.py     pre-specified test of the refusal hypothesis
src/05_random_samples_for_writeup.py   randomly sampled raw examples
src/05b_refusal_content_analysis.py    exploratory: are refusals content-specific?

data/contrastive_prompts.jsonl     the dataset (100 prompts, 50 pairs)
results/*.json                     all numbers quoted in the write-up
figures/*.png                      the four figures
logs/*.txt                         full unedited console output + raw generations
```

## Reproducing

```bash
pip install torch transformers nnsight scikit-learn matplotlib einops torchvision
python src/01_build_dataset.py
python src/02a_inspect_model.py          # optional, verification only
python src/02_extract_activations.py     # ~26s, writes activations/ (63MB, gitignored)
python src/03_probe_and_baselines.py     # CPU only
python src/03b_blackbox_baseline.py
python src/04_steering.py                # ~30 min
python src/04b_score_refusals.py
python src/04c_refusal_replication.py    # ~15 min
```

Scripts assert their own invariants (shapes, no pair leaking across CV folds,
steering hook is a no-op at coefficient 0) and print a summary of what they did.

## Two bugs worth knowing about if you build on this

1. **This tokenizer defaults to `padding_side="right"`.** Batching prompts of
   different lengths puts a pad token at position −1, so you read the residual
   stream above padding instead of above the real final token. It inflated the
   layer-0 eval/deployment difference 60× (0.019 → 1.20) and would have produced
   a near-perfect, meaningless probe.
2. **Batching perturbs activations even with left padding** (up to 3.5% relative
   L2 here), because 24 of 32 layers are recurrent and do not cleanly ignore pad
   tokens. Extraction therefore runs at batch size 1 with no padding.

Also: nnsight 0.7 rewrites the body of a `with model.trace(...)` block, so names
bound by a list comprehension inside it do not survive. And `LanguageModel`
refuses this model (registered multimodal) — use `VisionLanguageModel`.
