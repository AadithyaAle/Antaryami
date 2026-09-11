"""Phase 2 - extract residual-stream activations for all 100 prompts.

For every prompt: one forward pass, recording the output of all 32 decoder
layers at the LAST token position. Saves per-prompt tensors plus one combined
tensor so that no later phase ever needs another forward pass.

Two decisions here are not cosmetic. Both came from measurement in
src/02a_inspect_model.py plus scratch tests, not from assumption:

  1. BATCH SIZE 1, NO PADDING.
     This model's tokenizer defaults to padding_side="right". With right
     padding, position -1 of a shorter sequence is a PAD token, so we would
     be reading the residual stream above padding instead of above the real
     final token. That bug silently inflates the eval-vs-deployment
     difference at layer 0 from 0.02 to 1.20 relative L2 - i.e. it would
     have produced a strong, entirely spurious "signal".
     Left padding fixes the position, but a second measurement showed that
     batching with left padding still perturbs activations by up to 3.5%
     relative L2 versus running each prompt alone. That is expected here:
     24 of the 32 layers are linear/recurrent attention (GatedDeltaNet),
     and recurrent layers do not cleanly ignore pad tokens the way softmax
     attention does. A 3.5% batch-composition-dependent artifact is small,
     but it is the kind of thing that can correlate with condition and fake
     a result. 100 prompts is cheap, so we remove the confound entirely by
     running one prompt at a time with no padding.

  2. THE CHAT TEMPLATE IS APPLIED, AND IT ENDS IN A THINK-OPEN TAG.
     Qwen3.5's template appends the assistant header plus an opening think
     tag, so the final tokens are identical for all 100 prompts. Combined
     with Phase 1's prefix-only framing, this means the extraction position
     is lexically identical across every prompt in the dataset, and any
     difference we later find must come from context. It is also exactly
     the position where the model is about to begin reasoning, which is the
     natural place to steer in Phase 4.

Run:   python src/02_extract_activations.py [--force]
Out:   activations/{id}.pt        per prompt, shape (32, 2560), float32
       activations/all.pt         shape (100, 32, 2560), float32
       activations/metadata.json  prompt order, labels, model revision, shapes
"""

import json
import os
import sys
import time

import torch

MODEL_ID = "Qwen/Qwen3.5-4B"
LAYER_PATH = "model.language_model.layers"  # verified in 02a_inspect_model.py
EXPECTED_N_LAYERS = 32
EXPECTED_HIDDEN = 2560

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data", "contrastive_prompts.jsonl")
ACT_DIR = os.path.join(HERE, "activations")


def load_records():
    with open(DATA) as f:
        records = [json.loads(line) for line in f]
    assert len(records) == 100, len(records)
    return records


def resolve(obj, path):
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def main():
    force = "--force" in sys.argv
    all_path = os.path.join(ACT_DIR, "all.pt")
    if os.path.exists(all_path) and not force:
        print(f"{all_path} already exists; refusing to redo forward passes.")
        print("Pass --force to re-extract.")
        return

    os.makedirs(ACT_DIR, exist_ok=True)
    records = load_records()

    from huggingface_hub import HfApi
    from nnsight import VisionLanguageModel

    # nnsight's LanguageModel rejects this model (registered multimodal), so
    # VisionLanguageModel is required even though we only ever feed it text.
    print(f"loading {MODEL_ID} ...")
    t0 = time.time()
    lm = VisionLanguageModel(MODEL_ID, dtype=torch.bfloat16,
                             device_map="cuda", dispatch=True)
    layers = resolve(lm, LAYER_PATH)
    tokenizer = lm.tokenizer
    n_layers = len(layers)
    assert n_layers == EXPECTED_N_LAYERS, n_layers
    print(f"loaded in {time.time()-t0:.1f}s; n_layers={n_layers}")

    try:
        revision = HfApi().model_info(MODEL_ID).sha
    except Exception:
        revision = "unknown"
    print(f"model revision: {revision}")

    def render(prompt):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )

    # Every prompt must share the same final tokens, otherwise the extraction
    # position is not comparable across prompts. Check before spending compute.
    tails = {tuple(tokenizer(render(r["prompt"]))["input_ids"][-6:]) for r in records}
    assert len(tails) == 1, f"extraction position is not lexically matched: {tails}"
    tail_str = [tokenizer.decode([t]) for t in next(iter(tails))]
    print(f"shared final 6 tokens for all 100 prompts: {tail_str}")

    def last_token_acts(text):
        """Return (n_layers, hidden) float32 activations at the final token."""
        enc = tokenizer(text, return_tensors="pt")
        input_ids = enc["input_ids"].to("cuda")
        attention_mask = enc["attention_mask"].to("cuda")
        assert attention_mask.shape[0] == 1
        assert int(attention_mask.sum()) == attention_mask.shape[1], \
            "unexpected padding in a batch-size-1 forward pass"
        # nnsight 0.7 rewrites the trace body; names bound by a list
        # comprehension inside it do not survive, so append to an outer list.
        saved = []
        with torch.no_grad():
            with lm.trace({"input_ids": input_ids,
                           "attention_mask": attention_mask}):
                for i in range(n_layers):
                    saved.append(layers[i].output.save())
        acts = []
        for h in saved:
            h = h[0] if isinstance(h, tuple) else h
            assert h.ndim == 3 and h.shape[0] == 1, tuple(h.shape)
            assert h.shape[-1] == EXPECTED_HIDDEN, tuple(h.shape)
            acts.append(h[0, -1, :])
        out = torch.stack(acts).float().cpu()
        assert out.shape == (n_layers, EXPECTED_HIDDEN), tuple(out.shape)
        assert torch.isfinite(out).all(), "non-finite activations"
        assert out.abs().sum() > 0, "all-zero activations"
        return out

    print(f"\nextracting, batch size 1, {len(records)} prompts ...")
    t0 = time.time()
    per_prompt = []
    meta_rows = []
    for k, r in enumerate(records):
        text = render(r["prompt"])
        n_tokens = len(tokenizer(text)["input_ids"])
        acts = last_token_acts(text)
        torch.save(acts, os.path.join(ACT_DIR, f"{r['id']}.pt"))
        per_prompt.append(acts)
        meta_rows.append({
            "index": k,
            "id": r["id"],
            "pair_id": r["pair_id"],
            "category": r["category"],
            "condition": r["condition"],
            "label": 1 if r["condition"] == "eval" else 0,
            "framing_style": r["framing_style"],
            "framing_template_id": r["framing_template_id"],
            "n_tokens": n_tokens,
        })
        if (k + 1) % 20 == 0 or k == 0:
            print(f"  {k+1:3d}/{len(records)}  {r['id']:38s} "
                  f"n_tokens={n_tokens:3d}  ({time.time()-t0:.1f}s)")

    all_acts = torch.stack(per_prompt)
    assert all_acts.shape == (100, n_layers, EXPECTED_HIDDEN), tuple(all_acts.shape)
    assert torch.isfinite(all_acts).all()
    torch.save(all_acts, all_path)

    metadata = {
        "model_id": MODEL_ID,
        "model_revision": revision,
        "dtype_forward": "bfloat16",
        "dtype_saved": "float32",
        "layer_path": LAYER_PATH,
        "n_layers": n_layers,
        "hidden_size": EXPECTED_HIDDEN,
        "n_prompts": len(records),
        "position": "last token of the chat-templated prompt",
        "shared_final_tokens": tail_str,
        "batch_size": 1,
        "padding": "none",
        "shape": list(all_acts.shape),
        "rows": meta_rows,
    }
    with open(os.path.join(ACT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - t0
    print(f"\ndone in {elapsed:.1f}s ({elapsed/len(records):.2f}s per prompt)")
    print(f"  all.pt shape      : {tuple(all_acts.shape)}  float32  "
          f"({all_acts.numel()*4/1e6:.1f} MB)")
    print(f"  per-prompt tensors: {len(records)} files in activations/")
    print(f"  metadata          : activations/metadata.json")

    # ---------------- post-hoc checks on what we just wrote ----------------
    print("\nchecks:")
    reloaded = torch.load(all_path)
    assert torch.equal(reloaded, all_acts)
    print("  reload of all.pt matches in-memory tensor: ok")

    labels = torch.tensor([m["label"] for m in meta_rows])
    assert int(labels.sum()) == 50, int(labels.sum())
    print(f"  label balance: {int(labels.sum())} eval / "
          f"{int((1-labels).sum())} deployment")

    norms = all_acts.norm(dim=-1)
    print("  residual-stream L2 norm by layer (mean over 100 prompts):")
    for i in range(0, n_layers, 4):
        print(f"    layer {i:2d}: {norms[:, i].mean():8.2f}")
    print(f"    layer {n_layers-1:2d}: {norms[:, n_layers-1].mean():8.2f}")

    # Determinism spot-check: re-run 3 prompts and compare.
    print("  determinism spot-check (3 prompts re-run):")
    for k in (0, 50, 99):
        again = last_token_acts(render(records[k]["prompt"]))
        md = (again - per_prompt[k]).abs().max().item()
        print(f"    {records[k]['id']:38s} max abs diff {md:.3e}")

    # Crude paired signal check: within-pair distance vs across-pair distance.
    # NOT a result, just confirmation the data is not degenerate.
    by_pair = {}
    for m, a in zip(meta_rows, per_prompt):
        by_pair.setdefault(m["pair_id"], {})[m["condition"]] = a
    layer = n_layers - 8
    evals = torch.stack([p["eval"][layer] for p in by_pair.values()])
    deps = torch.stack([p["deployment"][layer] for p in by_pair.values()])
    within = (evals - deps).norm(dim=-1).mean().item()
    # mean off-diagonal distance among the 50 eval prompts
    dd = torch.cdist(evals, evals)
    across = dd.sum().item() / (50 * 49)
    print(f"  at layer {layer}: mean within-pair (eval vs deployment) L2 = "
          f"{within:.3f}")
    print(f"  at layer {layer}: mean eval-to-eval (different task)  L2 = "
          f"{across:.3f}")
    print("  if within-pair distance were as large as across-task distance, "
          "framing would be\n  a weak effect next to task identity. Interpret "
          "in Phase 3, not here.")
    print("\n  (sanity numbers only - no probing claims until Phase 3)")


if __name__ == "__main__":
    main()
