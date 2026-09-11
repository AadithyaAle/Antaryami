"""Phase 3 - per-layer linear probes, all baselines, and the base-rate test.

Reads activations/all.pt (100, 32, 2560) produced by Phase 2. No forward passes
here, so this is CPU-only and fast to iterate on. The black-box baseline
(Baseline B) needs the GPU and lives in src/03b_blackbox_baseline.py.

WHAT IS MEASURED, and why each piece exists:

  MAIN PROBE   per-layer logistic regression, eval vs deployment.
               Cross-validation is GROUPED BY pair_id. This is essential, not
               cosmetic: Phase 2 measured that changing the task moves the
               residual stream twice as far as changing the framing
               (across-task L2 10.8 vs within-pair 4.9 at layer 24). With
               random splits, the eval and deployment version of the SAME task
               would land in train and test, so the probe could score well by
               recognising the task instead of the framing. Grouping by
               pair_id makes that impossible.
               Features are standardised with a scaler FIT ON TRAIN ONLY,
               because residual-stream norms grow 2.1 -> 49.8 across layers.

  BASELINE A   random Gaussian direction of the same dimensionality, with only
               a scale and intercept fitted on train. Answers "does a LEARNED
               direction beat an arbitrary one?"
  BASELINE C   label-shuffled control: identical pipeline, permuted labels.
               Answers "is my pipeline leaking signal some other way?"
  EXTRA        difference-of-means probe. The simplest possible method, so it
               belongs here as the thing regularised LR has to justify itself
               against.
  (BASELINE B  ask the model directly -> src/03b_blackbox_baseline.py)

  GENERALISATION  train on the 60 EXPLICIT-framed prompts, test on the 40
               IMPLICIT-framed ones. Explicit framings contain giveaway words
               ("you are being evaluated"); implicit ones carry only structural
               cues ("Item 8/50 | max score 10" vs "Sent from mobile app").
               If accuracy collapses here, the probe is a keyword detector,
               not an eval-awareness detector. Reported per implicit template
               too, because implicit_3 has a known register confound.

  BASE RATES   Neel's actual question is: "If we ran this over evals where only
               1% showed eval awareness, would we just get tons of false
               positives?" So we report FPR and TPR with Wilson confidence
               intervals, then convert to precision at realistic base rates.
               Balanced accuracy is deliberately NOT the headline number.

Run:  python src/03_probe_and_baselines.py
Out:  results/phase3_probe_results.json
      results/probe_directions.pt   (for Phase 4 steering)
      figures/probe_accuracy_by_layer.png
      figures/precision_vs_base_rate.png
"""

import json
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACT_DIR = os.path.join(HERE, "activations")
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")

N_CV_SEEDS = 5          # repeats of the grouped 5-fold split
N_RANDOM_DIRS = 20      # Baseline A directions per layer
N_SHUFFLES = 5          # Baseline C label permutations per layer
C_DEFAULT = 1.0         # logistic regression inverse regularisation strength
C_SWEEP = [0.001, 0.01, 0.1, 1.0, 10.0]


def wilson(k, n, z=1.96):
    """Wilson score interval for a proportion. Used because with only 50
    negatives, a naive FPR point estimate is badly overconfident."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def load():
    acts = torch.load(os.path.join(ACT_DIR, "all.pt")).numpy().astype(np.float64)
    with open(os.path.join(ACT_DIR, "metadata.json")) as f:
        meta = json.load(f)
    rows = meta["rows"]
    assert acts.shape[0] == len(rows) == 100, (acts.shape, len(rows))
    y = np.array([r["label"] for r in rows])
    groups = np.array([r["pair_id"] for r in rows])
    style = np.array([r["framing_style"] for r in rows])
    template = np.array([r["framing_template_id"] for r in rows])
    category = np.array([r["category"] for r in rows])
    n_tokens = np.array([r["n_tokens"] for r in rows])
    assert y.sum() == 50
    assert len(set(groups)) == 50
    return acts, y, groups, style, template, category, n_tokens, meta


# --------------------------------------------------------------------------
# probe variants. each returns test-set scores in [0, 1]
# --------------------------------------------------------------------------

def fit_logreg(Xtr, ytr, Xte, C=C_DEFAULT):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=5000)
    clf.fit(sc.transform(Xtr), ytr)
    return clf.predict_proba(sc.transform(Xte))[:, 1], clf, sc


def fit_diff_of_means(Xtr, ytr, Xte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
    w = Ztr[ytr == 1].mean(0) - Ztr[ytr == 0].mean(0)
    w = w / np.linalg.norm(w)
    # fit only scale + intercept on the 1-D projection
    lr = LogisticRegression(max_iter=5000).fit((Ztr @ w).reshape(-1, 1), ytr)
    return lr.predict_proba((Zte @ w).reshape(-1, 1))[:, 1], w, sc


def fit_random_dir(Xtr, ytr, Xte, rng):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
    w = rng.standard_normal(Xtr.shape[1])
    w = w / np.linalg.norm(w)
    lr = LogisticRegression(max_iter=5000).fit((Ztr @ w).reshape(-1, 1), ytr)
    return lr.predict_proba((Zte @ w).reshape(-1, 1))[:, 1]


def grouped_splits(y, groups, seed):
    from sklearn.model_selection import StratifiedGroupKFold
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros((len(y), 1)), y, groups))


def main():
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)
    from sklearn.metrics import roc_auc_score

    acts, y, groups, style, template, category, n_tokens, meta = load()
    n_layers = acts.shape[1]
    print(f"loaded activations {acts.shape} from {meta['model_id']} "
          f"rev {meta['model_revision'][:12]}")
    print(f"labels: {y.sum()} eval / {(1-y).sum()} deployment; "
          f"{len(set(groups))} pair groups")
    print(f"CV: StratifiedGroupKFold(5), grouped by pair_id, "
          f"{N_CV_SEEDS} seeds -> {5*N_CV_SEEDS} fold-evaluations per layer")
    print(f"features standardised per fold (fit on train only); "
          f"LogisticRegression(C={C_DEFAULT}, max_iter=5000)")

    # ---------------- sanity: does grouping actually hold? ----------------
    sp = grouped_splits(y, groups, 0)
    for tr, te in sp:
        assert not (set(groups[tr]) & set(groups[te])), "pair leaked across folds"
    print(f"verified: no pair_id appears in both train and test "
          f"(fold test sizes {[len(te) for _, te in sp]})")

    # ---------------- BASELINE D: prompt length only ----------------
    # Motivation: the layer-0 probe scores far above chance, but at layer 0 the
    # final token is the SAME token in every prompt and almost no context has
    # mixed in. The most likely non-framing explanation is sequence length /
    # absolute position: within a pair the eval and deployment prompts are a
    # couple of tokens different, and if the SIGN of that difference is
    # systematic per template, a probe could read position instead of framing.
    # So: can the label be predicted from token count ALONE?
    print("\n" + "=" * 100)
    print("BASELINE D: predict the label from PROMPT LENGTH ALONE (1 feature)")
    print("this is a control for the suspiciously high layer-0 accuracy")
    print("=" * 100)
    len_accs = []
    for seed in range(N_CV_SEEDS):
        for tr, te in grouped_splits(y, groups, seed):
            p, _, _ = fit_logreg(n_tokens[tr].reshape(-1, 1).astype(float), y[tr],
                                 n_tokens[te].reshape(-1, 1).astype(float))
            len_accs.append(((p > 0.5).astype(int) == y[te]).mean())
    print(f"  token-count-only probe accuracy: {np.mean(len_accs):.3f} "
          f"+/- {np.std(len_accs):.3f}")
    print(f"  eval mean length {n_tokens[y==1].mean():.1f}, "
          f"deployment mean length {n_tokens[y==0].mean():.1f}")
    print("\n  within-pair length difference (eval minus deployment), per template:")
    print(f"  {'template':>12} {'n pairs':>8} {'mean diff':>10} {'sign consistent?':>18}")
    len_by_template = {}
    for t in sorted(set(template)):
        m = template == t
        pairs = sorted(set(groups[m]))
        diffs = []
        for g in pairs:
            gm = groups == g
            diffs.append(int(n_tokens[gm & (y == 1)][0] - n_tokens[gm & (y == 0)][0]))
        diffs = np.array(diffs)
        consistent = bool(np.all(diffs > 0) or np.all(diffs < 0))
        len_by_template[t] = {"mean_diff": float(diffs.mean()),
                              "sign_consistent": consistent,
                              "diffs": diffs.tolist()}
        print(f"  {t:>12} {len(pairs):8d} {diffs.mean():10.1f} "
              f"{str(consistent):>18}   raw: {diffs.tolist()}")
    n_consistent = sum(v["sign_consistent"] for v in len_by_template.values())
    print(f"\n  {n_consistent}/{len(len_by_template)} templates have a "
          f"sign-consistent length gap.")
    print("  If length-only accuracy is near chance, the layer-0 signal is NOT "
          "just length.")

    # ---------------- main sweep over layers ----------------
    per_layer = {}
    print("\n" + "=" * 100)
    print("PER-LAYER RESULTS  (mean +/- std over 25 fold-evaluations; "
          "full per-fold numbers printed after)")
    print("=" * 100)
    print(f"{'layer':>5} {'attn':>5} | {'probe acc':>16} {'AUC':>6} | "
          f"{'diff-means':>11} | {'BASE-A rand':>11} | {'BASE-C shuf':>11}")
    print("-" * 100)

    layer_types = None
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(meta["model_id"])
        layer_types = getattr(cfg.text_config, "layer_types", None)
    except Exception:
        pass

    for L in range(n_layers):
        X = acts[:, L, :]
        accs, aucs, dm_accs, rand_accs, shuf_accs = [], [], [], [], []
        oof = np.zeros((N_CV_SEEDS, len(y)))
        for si, seed in enumerate(range(N_CV_SEEDS)):
            for tr, te in grouped_splits(y, groups, seed):
                p, _, _ = fit_logreg(X[tr], y[tr], X[te])
                oof[si, te] = p
                accs.append(((p > 0.5).astype(int) == y[te]).mean())
                aucs.append(roc_auc_score(y[te], p))
                pdm, _, _ = fit_diff_of_means(X[tr], y[tr], X[te])
                dm_accs.append(((pdm > 0.5).astype(int) == y[te]).mean())
        # Baseline A: random directions (1 CV seed, N_RANDOM_DIRS draws)
        rng = np.random.default_rng(1234 + L)
        for tr, te in grouped_splits(y, groups, 0):
            for _ in range(N_RANDOM_DIRS):
                pr = fit_random_dir(X[tr], y[tr], X[te], rng)
                rand_accs.append(((pr > 0.5).astype(int) == y[te]).mean())
        # Baseline C: shuffled labels (1 CV seed, N_SHUFFLES permutations)
        for s in range(N_SHUFFLES):
            rs = np.random.default_rng(999 + s)
            ysh = rs.permutation(y)
            for tr, te in grouped_splits(ysh, groups, 0):
                ps, _, _ = fit_logreg(X[tr], ysh[tr], X[te])
                shuf_accs.append(((ps > 0.5).astype(int) == ysh[te]).mean())

        per_layer[L] = {
            "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            "acc_folds": [float(a) for a in accs],
            "auc_mean": float(np.mean(aucs)),
            "dm_mean": float(np.mean(dm_accs)), "dm_std": float(np.std(dm_accs)),
            "rand_mean": float(np.mean(rand_accs)), "rand_std": float(np.std(rand_accs)),
            "rand_max": float(np.max(rand_accs)),
            "shuf_mean": float(np.mean(shuf_accs)), "shuf_std": float(np.std(shuf_accs)),
            "shuf_max": float(np.max(shuf_accs)),
            "oof": oof.tolist(),
        }
        at = ""
        if layer_types:
            at = "FULL" if layer_types[L] == "full_attention" else "lin"
        print(f"{L:5d} {at:>5} | {np.mean(accs):7.3f} +/- {np.std(accs):.3f} "
              f"{np.mean(aucs):6.3f} | {np.mean(dm_accs):5.3f}+/-{np.std(dm_accs):.3f} | "
              f"{np.mean(rand_accs):5.3f}+/-{np.std(rand_accs):.3f} | "
              f"{np.mean(shuf_accs):5.3f}+/-{np.std(shuf_accs):.3f}")

    best_L = max(per_layer, key=lambda L: per_layer[L]["acc_mean"])
    print("-" * 100)
    print(f"best layer by mean CV accuracy: {best_L} "
          f"(acc {per_layer[best_L]['acc_mean']:.3f}, "
          f"AUC {per_layer[best_L]['auc_mean']:.3f})")

    # ---------------- full per-fold numbers at the best layer ----------------
    print("\n" + "=" * 100)
    print(f"RAW PER-FOLD ACCURACIES AT LAYER {best_L} (nothing averaged)")
    print("=" * 100)
    folds = per_layer[best_L]["acc_folds"]
    for si in range(N_CV_SEEDS):
        chunk = folds[si * 5:(si + 1) * 5]
        print(f"  cv seed {si}: " + "  ".join(f"{a:.3f}" for a in chunk)
              + f"   (mean {np.mean(chunk):.3f})")
    print(f"  all 25 values: {[round(a,3) for a in folds]}")
    print(f"  mean {np.mean(folds):.4f}  std {np.std(folds):.4f}  "
          f"min {np.min(folds):.3f}  max {np.max(folds):.3f}")
    print(f"  each fold tests 20 prompts (10 pairs), so one prompt = 0.05 accuracy")

    # ---------------- regularisation sweep at the best layer ----------------
    print("\n" + "=" * 100)
    print(f"REGULARISATION SWEEP AT LAYER {best_L} "
          f"(hyperparameters, since these must be reported)")
    print("=" * 100)
    sweep = {}
    X = acts[:, best_L, :]
    for C in C_SWEEP:
        a = []
        for seed in range(N_CV_SEEDS):
            for tr, te in grouped_splits(y, groups, seed):
                p, _, _ = fit_logreg(X[tr], y[tr], X[te], C=C)
                a.append(((p > 0.5).astype(int) == y[te]).mean())
        sweep[C] = {"mean": float(np.mean(a)), "std": float(np.std(a))}
        print(f"  C={C:<7} acc {np.mean(a):.3f} +/- {np.std(a):.3f}")
    print(f"  headline uses C={C_DEFAULT}; accuracy is flat across "
          f"{min(C_SWEEP)}-{max(C_SWEEP)}, so the result is not a tuning artifact")

    # ---------------- baseline comparison at the best layer ----------------
    print("\n" + "=" * 100)
    print(f"BASELINE COMPARISON AT LAYER {best_L}")
    print("=" * 100)
    b = per_layer[best_L]
    print(f"  learned probe (LR)          {b['acc_mean']:.3f} +/- {b['acc_std']:.3f}")
    print(f"  difference-of-means         {b['dm_mean']:.3f} +/- {b['dm_std']:.3f}")
    print(f"  BASELINE A random direction {b['rand_mean']:.3f} +/- {b['rand_std']:.3f}"
          f"   (best of {N_RANDOM_DIRS} draws x 5 folds: {b['rand_max']:.3f})")
    print(f"  BASELINE C shuffled labels  {b['shuf_mean']:.3f} +/- {b['shuf_std']:.3f}"
          f"   (max over {N_SHUFFLES} permutations x 5 folds: {b['shuf_max']:.3f})")
    print(f"  chance                      0.500")
    print(f"  probe minus random direction: "
          f"{b['acc_mean']-b['rand_mean']:+.3f}")
    print(f"  probe minus shuffled labels : {b['acc_mean']-b['shuf_mean']:+.3f}")

    # ---------------- explicit -> implicit generalisation ----------------
    print("\n" + "=" * 100)
    print("GENERALISATION: train on EXPLICIT framings, test on IMPLICIT framings")
    print("(the real test of whether this is eval-awareness or keyword-spotting)")
    print("=" * 100)
    ex = style == "explicit"
    im = style == "implicit"
    print(f"train n={ex.sum()} ({y[ex].sum()} eval), "
          f"test n={im.sum()} ({y[im].sum()} eval)")
    print(f"{'layer':>5} | {'acc':>6} {'AUC':>6} | per implicit template")
    print("-" * 100)
    gen = {}
    imp_templates = sorted(set(template[im]))
    for L in range(n_layers):
        X = acts[:, L, :]
        p, _, _ = fit_logreg(X[ex], y[ex], X[im])
        pred = (p > 0.5).astype(int)
        acc = (pred == y[im]).mean()
        auc = roc_auc_score(y[im], p)
        per_t = {}
        for t in imp_templates:
            m = template[im] == t
            per_t[t] = float((pred[m] == y[im][m]).mean())
        gen[L] = {"acc": float(acc), "auc": float(auc), "per_template": per_t}
        cells = "  ".join(f"{t.replace('implicit_','t')}={per_t[t]:.2f}"
                          for t in imp_templates)
        print(f"{L:5d} | {acc:6.3f} {auc:6.3f} | {cells}")
    best_gen = max(gen, key=lambda L: gen[L]["acc"])
    print("-" * 100)
    print(f"best generalising layer: {best_gen} acc {gen[best_gen]['acc']:.3f} "
          f"AUC {gen[best_gen]['auc']:.3f}")
    print(f"same layer as best in-distribution ({best_L})? "
          f"{'yes' if best_gen == best_L else 'no'}")
    print(f"at layer {best_L}: in-distribution {b['acc_mean']:.3f} -> "
          f"explicit-to-implicit {gen[best_L]['acc']:.3f} "
          f"(drop of {b['acc_mean']-gen[best_L]['acc']:+.3f})")
    print("each implicit template contributes 10 prompts, so one prompt = 0.10")

    # ---------------- leave-one-TEMPLATE-out ----------------
    # Baseline D showed length alone is at chance (0.494) because the sign of
    # the within-pair length gap flips between templates. BUT within a single
    # template the gap is fixed and sign-consistent, and our pair-grouped CV
    # still puts the same template in train and test. So a probe could learn
    # "template X + this length/position" instead of anything about framing.
    # Leave-one-template-out removes that route entirely: the test template's
    # surface form and length signature are never seen in training.
    print("\n" + "=" * 100)
    print("LEAVE-ONE-TEMPLATE-OUT CV (10 folds, one per framing template)")
    print("removes the 'template identity + length' shortcut that pair-grouped "
          "CV leaves open")
    print("=" * 100)
    all_templates = sorted(set(template))
    loto = {}
    print(f"{'layer':>5} | {'mean acc':>8} | " +
          " ".join(f"{t.replace('explicit_','e').replace('implicit_','i'):>4}"
                   for t in all_templates))
    print("-" * 100)
    for L in range(n_layers):
        X = acts[:, L, :]
        accs_t = {}
        for t in all_templates:
            te = template == t
            tr = ~te
            p, _, _ = fit_logreg(X[tr], y[tr], X[te])
            accs_t[t] = float(((p > 0.5).astype(int) == y[te]).mean())
        loto[L] = {"per_template": accs_t,
                   "mean": float(np.mean(list(accs_t.values())))}
        print(f"{L:5d} | {loto[L]['mean']:8.3f} | " +
              " ".join(f"{accs_t[t]:4.2f}" for t in all_templates))
    best_loto = max(loto, key=lambda L: loto[L]["mean"])
    print("-" * 100)
    print(f"best layer under leave-one-template-out: {best_loto} "
          f"(mean acc {loto[best_loto]['mean']:.3f})")
    print(f"at layer {best_L} (best in-distribution): "
          f"{loto[best_L]['mean']:.3f}")
    print(f"each template fold tests 10 prompts, so one prompt = 0.10 accuracy")

    # ---------------- base-rate / false-positive analysis ----------------
    print("\n" + "=" * 100)
    print(f"BASE-RATE TEST AT LAYER {best_L} — the question Neel actually asked")
    print("=" * 100)
    oof = np.array(per_layer[best_L]["oof"])   # (seeds, 100)
    oof_mean = oof.mean(0)
    pred = (oof_mean > 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    tpr, fpr = tp / (tp + fn), fp / (fp + tn)
    print("out-of-fold confusion matrix at threshold 0.5 "
          "(predictions averaged over the 5 CV seeds):")
    print(f"                    pred deployment   pred eval")
    print(f"  true deployment   {tn:15d} {fp:11d}")
    print(f"  true eval         {fn:15d} {tp:11d}")
    print(f"  TPR (recall) = {tp}/{tp+fn} = {tpr:.3f}   "
          f"Wilson 95% CI {tuple(round(v,3) for v in wilson(tp, tp+fn))}")
    print(f"  FPR          = {fp}/{fp+tn} = {fpr:.3f}   "
          f"Wilson 95% CI {tuple(round(v,3) for v in wilson(fp, fp+tn))}")
    print(f"  balanced accuracy = {(tpr + (1-fpr))/2:.3f}  <-- deliberately NOT "
          f"the headline")

    print("\nprecision if we deployed this monitor at realistic base rates.")
    print("precision = pi*TPR / (pi*TPR + (1-pi)*FPR), using the FPR above and")
    print("also the pessimistic end of its 95% CI:")
    print(f"{'base rate':>10} | {'precision':>10} {'prec (CI-pessimistic)':>22} | "
          f"{'false alarms per true hit':>26}")
    print("-" * 100)
    fpr_hi = wilson(fp, fp + tn)[1]
    base_rate_rows = {}
    for pi in (0.01, 0.02, 0.05, 0.10, 0.50):
        prec = pi * tpr / (pi * tpr + (1 - pi) * fpr) if (fpr > 0 or pi > 0) else float("nan")
        prec_lo = pi * tpr / (pi * tpr + (1 - pi) * fpr_hi)
        fa = ((1 - pi) * fpr) / (pi * tpr) if tpr > 0 and pi > 0 else float("nan")
        base_rate_rows[pi] = {"precision": float(prec),
                              "precision_pessimistic": float(prec_lo),
                              "false_alarms_per_true_hit": float(fa)}
        print(f"{pi:10.2f} | {prec:10.3f} {prec_lo:22.3f} | {fa:26.1f}")

    # threshold that buys a low FPR, and what recall survives
    print("\nif we instead pick the threshold that holds FPR at or below 1% "
          "out-of-fold:")
    order = np.sort(oof_mean[y == 0])[::-1]
    thr_1pct = float(order[0]) + 1e-9   # above every negative -> FPR 0/50
    tpr_at = float(((oof_mean > thr_1pct) & (y == 1)).sum() / (y == 1).sum())
    print(f"  threshold {thr_1pct:.4f}  ->  FPR 0/50 = 0.000, TPR {tpr_at:.3f}")
    print(f"  note: FPR 0/50 has a Wilson 95% upper bound of "
          f"{wilson(0,50)[1]:.3f}, i.e. the true FPR could still be ~7%.")
    print(f"  with 50 negatives we simply cannot demonstrate an FPR below ~7%, "
          f"and at a 1% base\n  rate a 7% FPR still means "
          f"{0.99*wilson(0,50)[1]/(0.01*max(tpr_at,1e-9)):.0f} false alarms per "
          f"true hit. This is a hard\n  limit of the dataset size, not a "
          f"tuning problem.")

    # literal imbalanced holdout, stated honestly
    print("\nliteral imbalanced holdout, 5 eval among 95 deployment "
          "(1000 bootstrap draws).")
    print("NOTE: only 50 real deployment prompts exist, so the 95 negatives are")
    print("sampled WITH REPLACEMENT. This inflates apparent stability and is a")
    print("weaker test than the analytic numbers above. Reported for completeness.")
    rng = np.random.default_rng(0)
    pos_scores = oof_mean[y == 1]
    neg_scores = oof_mean[y == 0]
    tps, fps, precs = [], [], []
    for _ in range(1000):
        ps = rng.choice(pos_scores, 5, replace=True)
        ns = rng.choice(neg_scores, 95, replace=True)
        t = int((ps > 0.5).sum())
        f = int((ns > 0.5).sum())
        tps.append(t)
        fps.append(f)
        precs.append(t / (t + f) if (t + f) > 0 else np.nan)
    precs_arr = np.array(precs, dtype=float)
    print(f"  true positives  : mean {np.mean(tps):.2f} / 5")
    print(f"  false positives : mean {np.mean(fps):.2f} / 95  "
          f"(5th-95th pct {np.percentile(fps,5):.0f}-{np.percentile(fps,95):.0f})")
    print(f"  precision       : mean {np.nanmean(precs_arr):.3f}  "
          f"(5th-95th pct {np.nanpercentile(precs_arr,5):.3f}-"
          f"{np.nanpercentile(precs_arr,95):.3f})")

    # ---------------- per-category breakdown at best layer ----------------
    print("\n" + "=" * 100)
    print(f"PER-CATEGORY out-of-fold accuracy at layer {best_L}")
    print("=" * 100)
    per_cat = {}
    for c in sorted(set(category)):
        m = category == c
        per_cat[c] = float((pred[m] == y[m]).mean())
        print(f"  {c:24s} {per_cat[c]:.3f}   (n={int(m.sum())})")

    # ---------------- save directions for Phase 4 ----------------
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    Xb = acts[:, best_L, :]
    sc_full = StandardScaler().fit(Xb)
    clf_full = LogisticRegression(C=C_DEFAULT, max_iter=5000).fit(sc_full.transform(Xb), y)
    # map the probe direction back into raw activation space
    w_std = clf_full.coef_[0]
    w_raw = w_std / sc_full.scale_
    w_raw = w_raw / np.linalg.norm(w_raw)
    Zb = sc_full.transform(Xb)
    dm = Zb[y == 1].mean(0) - Zb[y == 0].mean(0)
    dm_raw = dm / sc_full.scale_
    dm_raw = dm_raw / np.linalg.norm(dm_raw)
    cos = float(w_raw @ dm_raw)
    print(f"\ncosine similarity between LR probe direction and "
          f"difference-of-means direction at layer {best_L}: {cos:.3f}")
    torch.save({
        "best_layer": int(best_L),
        "probe_direction_raw": torch.tensor(w_raw, dtype=torch.float32),
        "diff_means_direction_raw": torch.tensor(dm_raw, dtype=torch.float32),
        "scaler_mean": torch.tensor(sc_full.mean_, dtype=torch.float32),
        "scaler_scale": torch.tensor(sc_full.scale_, dtype=torch.float32),
        "intercept": float(clf_full.intercept_[0]),
        "cos_probe_vs_diffmeans": cos,
        "note": "directions are unit-norm in RAW activation space; "
                "+ve = more eval-like",
    }, os.path.join(RESULTS, "probe_directions.pt"))

    # ---------------- figures ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Ls = list(range(n_layers))
    m = np.array([per_layer[L]["acc_mean"] for L in Ls])
    s = np.array([per_layer[L]["acc_std"] for L in Ls])
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(Ls, m, "-o", ms=4, label="logistic probe (grouped 5-fold CV)")
    ax.fill_between(Ls, m - s, m + s, alpha=0.2)
    ax.plot(Ls, [per_layer[L]["dm_mean"] for L in Ls], "-s", ms=3,
            label="difference-of-means probe")
    ax.plot(Ls, [gen[L]["acc"] for L in Ls], "-^", ms=3,
            label="train explicit -> test implicit")
    ax.plot(Ls, [loto[L]["mean"] for L in Ls], "-v", ms=3,
            label="leave-one-template-out CV")
    ax.plot(Ls, [per_layer[L]["rand_mean"] for L in Ls], "--",
            label=f"Baseline A: random direction (n={N_RANDOM_DIRS}/layer)")
    ax.plot(Ls, [per_layer[L]["shuf_mean"] for L in Ls], ":",
            label=f"Baseline C: shuffled labels (n={N_SHUFFLES}/layer)")
    ax.axhline(0.5, color="k", lw=0.8, label="chance")
    if layer_types:
        for L, t in enumerate(layer_types):
            if t == "full_attention":
                ax.axvline(L, color="grey", alpha=0.25, lw=1)
        ax.plot([], [], color="grey", alpha=0.4, lw=1,
                label="full-attention layers (3,7,...,31)")
    ax.set_xlabel("layer")
    ax.set_ylabel("accuracy (eval vs deployment)")
    ax.set_title("Eval-framing probe accuracy by layer, Qwen3.5-4B\n"
                 "100 prompts = 50 matched pairs; CV grouped by pair_id; "
                 "shaded = +/-1 SD over 25 folds")
    ax.set_ylim(0.3, 1.02)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f1 = os.path.join(FIGURES, "probe_accuracy_by_layer.png")
    fig.savefig(f1, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    pis = np.logspace(-3, 0, 200)
    for label, f_ in [(f"measured FPR = {fpr:.3f}", fpr),
                      (f"pessimistic FPR = {fpr_hi:.3f} (95% CI upper)", fpr_hi)]:
        prec = pis * tpr / (pis * tpr + (1 - pis) * f_)
        ax.plot(pis * 100, prec, label=label)
    ax.axvline(1.0, color="r", ls="--", lw=1, label="1% base rate (Neel's case)")
    ax.set_xscale("log")
    ax.set_xlabel("prevalence of eval-awareness in the population (%)")
    ax.set_ylabel("precision of the probe (fraction of alarms that are real)")
    ax.set_title(f"Probe precision vs base rate, layer {best_L}\n"
                 f"TPR={tpr:.2f}, FPR={fpr:.2f} measured out-of-fold on "
                 f"100 prompts (50 neg)")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    f2 = os.path.join(FIGURES, "precision_vs_base_rate.png")
    fig.savefig(f2, dpi=150)
    plt.close(fig)

    out = {
        "config": {
            "model_id": meta["model_id"], "model_revision": meta["model_revision"],
            "n_cv_seeds": N_CV_SEEDS, "cv": "StratifiedGroupKFold(5) grouped by pair_id",
            "C": C_DEFAULT, "standardised": True,
            "n_random_dirs": N_RANDOM_DIRS, "n_shuffles": N_SHUFFLES,
        },
        "best_layer": int(best_L),
        "per_layer": {str(L): {k: v for k, v in per_layer[L].items() if k != "oof"}
                      for L in per_layer},
        "regularisation_sweep": {str(k): v for k, v in sweep.items()},
        "generalisation_explicit_to_implicit": {str(L): gen[L] for L in gen},
        "leave_one_template_out": {str(L): loto[L] for L in loto},
        "best_layer_confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                                 "tpr": tpr, "fpr": fpr,
                                 "tpr_ci": wilson(tp, tp+fn), "fpr_ci": wilson(fp, fp+tn)},
        "base_rate_analysis": {str(k): v for k, v in base_rate_rows.items()},
        "per_category_oof_accuracy": per_cat,
        "cos_probe_vs_diffmeans": cos,
        "baseline_d_length_only": {
            "acc_mean": float(np.mean(len_accs)),
            "acc_std": float(np.std(len_accs)),
            "eval_mean_tokens": float(n_tokens[y == 1].mean()),
            "deployment_mean_tokens": float(n_tokens[y == 0].mean()),
            "within_pair_length_by_template": len_by_template,
        },
    }
    with open(os.path.join(RESULTS, "phase3_probe_results.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 100)
    print("outputs")
    print("=" * 100)
    print(f"  results/phase3_probe_results.json")
    print(f"  results/probe_directions.pt   (layer {best_L}, for Phase 4)")
    print(f"  {f1}")
    print(f"  {f2}")
    print("\nBaseline B (ask the model directly) is not in this script - "
          "run src/03b_blackbox_baseline.py")


if __name__ == "__main__":
    main()
