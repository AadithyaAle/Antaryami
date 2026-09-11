"""Phase 2, step 0 - inspect the model before running anything expensive.

Nothing here is a result. This exists because the brief's model id was wrong,
the model is a hybrid-attention multimodal architecture, and installed nnsight
is 0.7 (not the 0.2 the brief assumed). So every assumption the extraction
script will rely on gets printed and checked here first:

  1. which AutoModel class actually loads Qwen3.5-4B
  2. the real module path to the decoder layers (vision tower means it is
     nested, so it must not be guessed)
  3. n_layers / hidden_size read off the LOADED model, not the config file
  4. what the chat template puts at the end of the prompt (this determines
     what "last token" means for extraction)
  5. a single 2-prompt trace, to confirm the hook returns the shape we expect
     and that an eval/deployment pair actually differs at that position

Run:  python src/02a_inspect_model.py
"""

import json
import os

import torch

MODEL_ID = "Qwen/Qwen3.5-4B"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_prompts():
    path = os.path.join(HERE, "data", "contrastive_prompts.jsonl")
    with open(path) as f:
        return [json.loads(line) for line in f]


def module_tree(module, prefix="", depth=0, max_depth=3, max_children=6):
    """Print a truncated module tree so we can see the real attribute path."""
    children = list(module.named_children())
    for i, (name, child) in enumerate(children):
        if i >= max_children:
            print(f"{'  ' * depth}... (+{len(children) - max_children} more)")
            break
        n_params = sum(p.numel() for p in child.parameters(recurse=True))
        line = f"{'  ' * depth}{name}: {type(child).__name__}"
        if isinstance(child, torch.nn.ModuleList):
            line += f"  [len={len(child)}]"
        line += f"  ({n_params/1e6:.1f}M params)"
        print(line)
        # don't expand every one of 32 identical decoder layers
        if depth < max_depth:
            if isinstance(child, torch.nn.ModuleList) and len(child) > 2:
                print(f"{'  ' * (depth+1)}[0]: {type(child[0]).__name__} "
                      f"(remaining {len(child)-1} layers elided)")
                module_tree(child[0], depth=depth + 2, max_depth=max_depth)
            else:
                module_tree(child, depth=depth + 1, max_depth=max_depth)


def main():
    from transformers import AutoConfig, AutoTokenizer

    print("=" * 78)
    print("1. WHICH AutoModel CLASS LOADS THIS MODEL")
    print("=" * 78)
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    print(f"config class      : {type(cfg).__name__}")
    print(f"architectures     : {cfg.architectures}")
    # the text half of the config is where n_layers/hidden_size live
    text_cfg = getattr(cfg, "text_config", cfg)
    print(f"text config class : {type(text_cfg).__name__}")
    print(f"config n_layers   : {text_cfg.num_hidden_layers}")
    print(f"config hidden_size: {text_cfg.hidden_size}")

    import transformers
    candidates = ["AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModel"]
    chosen = None
    for name in candidates:
        auto_cls = getattr(transformers, name, None)
        if auto_cls is None:
            continue
        mapping = getattr(auto_cls, "_model_mapping", None)
        ok = mapping is not None and type(cfg) in mapping
        print(f"  {name:32s} supports this config: {ok}")
        if ok and chosen is None:
            chosen = (name, auto_cls)
    assert chosen is not None, "no AutoModel class maps this config"
    print(f"-> using {chosen[0]}")

    print()
    print("=" * 78)
    print("2. LOADING VIA NNSIGHT + MODULE TREE")
    print("=" * 78)
    # nnsight.LanguageModel REFUSES this model: it is registered with
    # AutoModelForImageTextToText, and nnsight 0.7 requires VisionLanguageModel
    # for multimodal architectures. We only ever feed it text.
    import nnsight
    from nnsight import VisionLanguageModel
    print(f"nnsight {nnsight.__version__} -> using VisionLanguageModel "
          "(LanguageModel rejects multimodal configs)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    lm = VisionLanguageModel(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="cuda",
        dispatch=True,  # load weights now, so shapes below are real
    )
    print()
    module_tree(lm._model if hasattr(lm, "_model") else lm.model)

    print()
    print("=" * 78)
    print("3. RESOLVING THE DECODER-LAYER PATH")
    print("=" * 78)
    inner = lm._model if hasattr(lm, "_model") else lm.model
    found = None
    for path in ["model.language_model.layers", "model.layers",
                 "language_model.model.layers", "language_model.layers"]:
        obj = inner
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            print(f"  {path:34s} -> absent")
            continue
        print(f"  {path:34s} -> ModuleList len={len(obj)}  "
              f"layer type={type(obj[0]).__name__}")
        if found is None:
            found = (path, obj)
    assert found is not None, "could not locate decoder layers"
    layer_path, layers = found

    # nnsight hooks must go on the ENVOY (lm.<path>), not the raw torch module
    envoy = lm
    for part in layer_path.split("."):
        envoy = getattr(envoy, part)
    print(f"   envoy for hooking : lm.{layer_path} -> {type(envoy).__name__}")
    n_layers = len(layers)
    hidden_size = text_cfg.hidden_size
    print(f"-> decoder layers at: {layer_path}  (n_layers={n_layers})")
    assert n_layers == text_cfg.num_hidden_layers, (n_layers, text_cfg.num_hidden_layers)

    # hybrid attention: confirm the full/linear split the config claimed
    layer_types = getattr(text_cfg, "layer_types", None)
    if layer_types:
        from collections import Counter
        print(f"   layer_types: {dict(Counter(layer_types))}")
        print(f"   full_attention at layers: "
              f"{[i for i, t in enumerate(layer_types) if t == 'full_attention']}")

    print()
    print("=" * 78)
    print("4. WHAT THE CHAT TEMPLATE PUTS AT THE END")
    print("=" * 78)
    records = load_prompts()
    pair = [r for r in records if r["pair_id"] == "math_00"]
    pair = {r["condition"]: r for r in pair}
    rendered = {}
    for cond, r in pair.items():
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}],
            tokenize=False, add_generation_prompt=True,
        )
        rendered[cond] = text
    print("--- eval-framed, fully rendered ---")
    print(repr(rendered["eval"]))
    print()
    ids = {c: tokenizer(t)["input_ids"] for c, t in rendered.items()}
    for cond in ("eval", "deployment"):
        tail = [tokenizer.decode([t]) for t in ids[cond][-6:]]
        print(f"{cond:11s} n_tokens={len(ids[cond]):3d}  last 6 tokens={tail}")
    assert ids["eval"][-6:] == ids["deployment"][-6:], \
        "tail tokens differ between conditions - would be a lexical confound"
    print("-> tails are IDENTICAL across conditions, so the last-token position "
          "is lexically matched and any difference must come from context.")

    print()
    print("=" * 78)
    print("5. ONE 2-PROMPT TRACE TO CHECK THE HOOK AND SHAPES")
    print("=" * 78)
    prompts = [rendered["eval"], rendered["deployment"]]
    # left-pad so that index -1 is the true last token for both sequences
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # NOTE (nnsight 0.7): the body of a trace block is re-executed by nnsight's
    # frame rewriting, and names bound by a list COMPREHENSION inside the block
    # do not survive it (NameError). Appending into a list created outside the
    # block does work. Verified in scratch tests.
    saved = []
    with lm.trace(prompts):
        for i in range(n_layers):
            saved.append(envoy[i].output.save())

    print(f"type of layers[0].output : {type(saved[0])}")
    first = saved[0]
    if isinstance(first, tuple):
        print(f"  tuple of len {len(first)}; element 0 shape {first[0].shape}")
        get = lambda s: s[0]
    else:
        print(f"  tensor shape {first.shape}")
        get = lambda s: s
    batch, seq, hid = get(first).shape
    print(f"batch={batch} seq={seq} hidden={hid}")
    assert batch == 2, batch
    assert hid == hidden_size, (hid, hidden_size)

    stacked = torch.stack([get(s)[:, -1, :] for s in saved], dim=1)
    print(f"stacked last-token activations shape: {tuple(stacked.shape)}  "
          f"(expected (2, {n_layers}, {hidden_size}))")
    assert stacked.shape == (2, n_layers, hidden_size), stacked.shape
    assert torch.isfinite(stacked.float()).all(), "non-finite activations"

    d = (stacked[0] - stacked[1]).float()
    per_layer = d.norm(dim=-1) / stacked[0].float().norm(dim=-1).clamp(min=1e-6)
    print("\nrelative L2 difference between the eval and deployment member of "
          "pair math_00, per layer (a pure sanity check, n=1 pair):")
    for i in range(0, n_layers, 4):
        print(f"  layer {i:2d}: {per_layer[i].item():.4f}")
    assert per_layer.max() > 0, "activations identical - hook is not reading context"

    print()
    print("=" * 78)
    print("SUMMARY OF VERIFIED FACTS FOR THE EXTRACTION SCRIPT")
    print("=" * 78)
    print(f"  automodel class : {chosen[0]}")
    print(f"  layer path      : {layer_path}")
    print(f"  n_layers        : {n_layers}")
    print(f"  hidden_size     : {hidden_size}")
    print(f"  layer output    : {'tuple, take [0]' if isinstance(first, tuple) else 'tensor'}")
    print(f"  target all.pt   : (100, {n_layers}, {hidden_size})")
    print("  nothing was saved to disk by this script")


if __name__ == "__main__":
    main()
