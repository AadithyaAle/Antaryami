"""Phase 4, step 0 - verify the steering hook does what I think it does.

Checks, in order:
  1. the hook fires on the right module, the right number of times
  2. with coefficient 0 the output is bit-identical to no hook at all
     (i.e. the hook itself is not perturbing anything)
  3. the activation at the steered layer actually moves by the intended amount
  4. generation with a large coefficient produces different text
Nothing here is a result.
"""
import json
import os

import torch

MODEL_ID = "Qwen/Qwen3.5-4B"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAYER = 12


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    # NOTE: the module path depends on which AutoModel class loaded the model.
    # nnsight's VisionLanguageModel exposes model.language_model.layers (the
    # vision tower is present), while AutoModelForCausalLM returns a
    # Qwen3_5TextModel where the layers sit at model.layers. Resolve, don't
    # assume - this bit me here.
    layers = None
    for path in ("model.language_model.layers", "model.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            print(f"  {path:32s} -> absent")
            continue
        print(f"  {path:32s} -> ModuleList len={len(obj)}")
        if layers is None:
            layers, layer_path = obj, path
    assert layers is not None, "could not find decoder layers"
    print(f"using {layer_path}; n_layers={len(layers)}  steering layer={LAYER}")

    d = torch.load(os.path.join(HERE, "results", "probe_directions.pt"))
    print(f"directions file keys: {list(d.keys())}")
    print(f"best_layer recorded in file: {d['best_layer']}")
    vec = d["probe_direction_raw"].to("cuda", torch.bfloat16)
    print(f"probe direction shape {tuple(vec.shape)}, "
          f"norm {vec.float().norm().item():.4f}")

    calls = {"n": 0, "shapes": []}

    def make_hook(coeff):
        def hook(module, inputs, output):
            calls["n"] += 1
            if isinstance(output, tuple):
                h = output[0]
                calls["shapes"].append(tuple(h.shape))
                return (h + coeff * vec,) + output[1:]
            calls["shapes"].append(tuple(output.shape))
            return output + coeff * vec
        return hook

    text = tok.apply_chat_template([{"role": "user", "content": "Name three colours."}],
                                   tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    enc = tok(text, return_tensors="pt").to("cuda")

    # --- 1/2: coefficient 0 must be a no-op ---
    with torch.no_grad():
        clean = model(**enc).logits[0, -1].float().clone()
    h = layers[LAYER].register_forward_hook(make_hook(0.0))
    calls["n"] = 0
    with torch.no_grad():
        zeroed = model(**enc).logits[0, -1].float().clone()
    h.remove()
    print(f"\nhook fired {calls['n']} time(s) for one forward pass; "
          f"activation shapes {calls['shapes']}")
    print(f"coeff=0 max |logit diff| vs no hook: "
          f"{(clean - zeroed).abs().max().item():.3e}  "
          f"(must be 0.000e+00)")
    assert torch.equal(clean, zeroed), "hook perturbs output at coeff 0"

    # --- 3: does the activation move by the intended amount? ---
    grabbed = {}

    def grab(module, inputs, output):
        h_ = output[0] if isinstance(output, tuple) else output
        grabbed["h"] = h_[0, -1, :].float().clone()

    g = layers[LAYER].register_forward_hook(grab)
    with torch.no_grad():
        model(**enc)
    base = grabbed["h"].clone()
    g.remove()

    coeff = 5.0
    h1 = layers[LAYER].register_forward_hook(make_hook(coeff))
    g = layers[LAYER].register_forward_hook(grab)   # runs after the steer hook
    with torch.no_grad():
        model(**enc)
    steered = grabbed["h"].clone()
    h1.remove()
    g.remove()
    delta = (steered - base)
    print(f"\nintended delta norm (coeff {coeff} x unit vector): {coeff:.3f}")
    print(f"observed delta norm at layer {LAYER} last token: "
          f"{delta.norm().item():.3f}")
    print(f"cosine(observed delta, steering direction): "
          f"{torch.nn.functional.cosine_similarity(delta, vec.float(), dim=0).item():.4f}")

    # --- reference scale: how big is the residual stream here anyway? ---
    acts = torch.load(os.path.join(HERE, "activations", "all.pt"))
    mean_norm = acts[:, LAYER, :].norm(dim=-1).mean().item()
    print(f"\nmean residual-stream norm at layer {LAYER} over the 100 dataset "
          f"prompts: {mean_norm:.3f}")
    print(f"so coefficient {coeff} is {coeff/mean_norm:.2f}x the typical "
          f"residual norm - steering strengths should be quoted in these units")

    # --- 4: does generation change? ---
    print("\ngeneration check (greedy, 40 new tokens):")
    for c in (0.0, +2 * mean_norm, -2 * mean_norm):
        hh = layers[LAYER].register_forward_hook(make_hook(c))
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=40, do_sample=False)
        hh.remove()
        gen = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"  coeff {c:+8.2f} ({c/mean_norm:+.1f} norms): "
              f"{gen[:110]!r}")


if __name__ == "__main__":
    main()
