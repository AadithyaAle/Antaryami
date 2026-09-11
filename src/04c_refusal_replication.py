"""Phase 4c - pre-specified test of the refusal hypothesis.

HYPOTHESIS (formed in 04b by reading generations, therefore NOT yet evidence):
  Steering +1 norm along the probe direction at layer 12 increases the refusal
  rate relative to (a) no steering and (b) a random direction of the SAME norm.

Observed in 04b on the 20 controversial-advice prompts: 15% baseline -> 35% at
+0.5 -> 45% at +1.0 for the probe direction, but the random control also
reached 30% at +1.0. With n=20 that difference (9/20 vs 6/20) is nowhere near
significant, so the hypothesis needs a real test.

THIS SCRIPT IS THAT TEST, and everything about it is fixed before running:
  - prompts   : all 100, so 80 of them are from categories that played no part
                in forming the hypothesis
  - conditions: unsteered, probe_lr @ +1.0 norms, random_control @ +1.0 norms
  - metric    : imported unchanged from 04b_score_refusals.py
  - statistic : two-sided Fisher exact test, probe vs random and probe vs
                unsteered
  - decoding  : greedy, so the only thing that varies is the intervention
  - the prediction is directional: probe > random. If probe <= random, the
    hypothesis is dead and the honest answer is that the earlier pattern was
    small-sample noise.

Run:  python src/04c_refusal_replication.py
Out:  results/phase4_refusal_replication.json
      logs/phase4c_generations.txt
"""

import importlib.util
import json
import os

import torch

MODEL_ID = "Qwen/Qwen3.5-4B"
LAYER = 12
COEFF_NORMS = 1.0
MAX_NEW_TOKENS = 80
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")
LOGS = os.path.join(HERE, "logs")


def load_metric():
    """Import is_refusal from 04b unchanged - the metric must not be retuned."""
    path = os.path.join(HERE, "src", "04b_score_refusals.py")
    spec = importlib.util.spec_from_file_location("scorer", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.is_refusal, mod.has_safe_redirect, mod.REFUSAL_PATTERNS


def fisher(a, b, c, d):
    """Two-sided Fisher exact test on [[a,b],[c,d]] without scipy."""
    from math import comb
    n = a + b + c + d
    r1, c1 = a + b, a + c

    def p_table(x):
        return (comb(r1, x) * comb(n - r1, c1 - x)) / comb(n, c1)

    lo = max(0, c1 - (n - r1))
    hi = min(r1, c1)
    p_obs = p_table(a)
    return sum(p_table(x) for x in range(lo, hi + 1)
               if p_table(x) <= p_obs + 1e-12)


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    is_refusal, has_safe_redirect, patterns = load_metric()
    print(f"metric imported from 04b unchanged: {len(patterns)} refusal patterns")

    with open(os.path.join(HERE, "data", "contrastive_prompts.jsonl")) as f:
        records = [json.loads(line) for line in f]
    print(f"prompts: {len(records)} (all categories)")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    layers = model.model.layers
    dirs = torch.load(os.path.join(RESULTS, "probe_directions.pt"))
    acts = torch.load(os.path.join(HERE, "activations", "all.pt"))
    mean_norm = acts[:, LAYER, :].norm(dim=-1).mean().item()

    g = torch.Generator().manual_seed(0)   # same seed as 04_steering.py
    rand_vec = torch.randn(2560, generator=g)
    rand_vec = rand_vec / rand_vec.norm()

    CONDITIONS = {
        "unsteered": None,
        "probe_lr@+1": dirs["probe_direction_raw"],
        "random_control@+1": rand_vec,
    }

    handle = {"h": None}

    def steer(vec):
        if handle["h"] is not None:
            handle["h"].remove()
            handle["h"] = None
        if vec is None:
            return
        v = vec.to("cuda", torch.bfloat16) * (COEFF_NORMS * mean_norm)

        def hook(module, inputs, output):
            if isinstance(output, tuple):
                return (output[0] + v,) + output[1:]
            return output + v
        handle["h"] = layers[LAYER].register_forward_hook(hook)

    def generate(prompt):
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = tok(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, enc["input_ids"].shape[1]:],
                          skip_special_tokens=True)

    log = open(os.path.join(LOGS, "phase4c_generations.txt"), "w")
    results = {}
    flags = {}
    print(f"\ncoefficient {COEFF_NORMS} norms = {COEFF_NORMS*mean_norm:.2f} raw; "
          f"max_new_tokens={MAX_NEW_TOKENS}; greedy")
    for cname, vec in CONDITIONS.items():
        steer(vec)
        refus, redir = [], []
        for r in records:
            g_ = generate(r["prompt"])
            refus.append(is_refusal(g_))
            redir.append(has_safe_redirect(g_))
            log.write("=" * 88 + f"\ncondition={cname} id={r['id']} "
                      f"refusal={refus[-1]}\n" + "-" * 88 + "\n")
            log.write(g_.strip() + "\n\n")
        steer(None)
        flags[cname] = refus
        results[cname] = {
            "n": len(records),
            "refusals": int(sum(refus)),
            "refusal_rate": float(sum(refus) / len(records)),
            "safe_redirects": int(sum(redir)),
            "by_category": {},
            "by_condition_framing": {},
        }
        for cat in sorted({r["category"] for r in records}):
            idx = [i for i, r in enumerate(records) if r["category"] == cat]
            results[cname]["by_category"][cat] = \
                float(sum(refus[i] for i in idx) / len(idx))
        for fr in ("eval", "deployment"):
            idx = [i for i, r in enumerate(records) if r["condition"] == fr]
            results[cname]["by_condition_framing"][fr] = \
                float(sum(refus[i] for i in idx) / len(idx))
        print(f"  {cname:20s} refusals {results[cname]['refusals']:3d}/100 "
              f"({results[cname]['refusal_rate']:.0%})")
    log.close()

    print("\n" + "=" * 96)
    print("PRE-SPECIFIED TEST RESULTS (n=100 prompts per condition)")
    print("=" * 96)
    print(f"{'condition':>20} {'refusal':>10} | " +
          " ".join(f"{c[:12]:>13}" for c in
                   sorted({r['category'] for r in records})))
    print("-" * 96)
    cats = sorted({r["category"] for r in records})
    for cname in CONDITIONS:
        row = results[cname]
        print(f"{cname:>20} {row['refusals']:>4d}/100 {row['refusal_rate']:>4.0%} | " +
              " ".join(f"{row['by_category'][c]:>12.0%} " for c in cats))
    print("-" * 96)

    pr = results["probe_lr@+1"]["refusals"]
    rc = results["random_control@+1"]["refusals"]
    un = results["unsteered"]["refusals"]
    n = 100
    p_vs_rand = fisher(pr, n - pr, rc, n - rc)
    p_vs_base = fisher(pr, n - pr, un, n - un)
    print(f"\nprobe_lr@+1 vs random_control@+1 : {pr}/100 vs {rc}/100, "
          f"Fisher exact two-sided p = {p_vs_rand:.4f}")
    print(f"probe_lr@+1 vs unsteered         : {pr}/100 vs {un}/100, "
          f"Fisher exact two-sided p = {p_vs_base:.4f}")
    verdict = ("SUPPORTED" if (pr > rc and p_vs_rand < 0.05)
               else "NOT SUPPORTED")
    print(f"\nprediction was probe > random. Result: {verdict}")
    if verdict == "NOT SUPPORTED":
        print("  -> the refusal shift seen in 04b is not attributable to the "
              "probe direction\n     specifically; a random vector of the same "
              "norm does comparable work.")

    print("\nrefusal rate split by the framing of the prompt itself:")
    for cname in CONDITIONS:
        b = results[cname]["by_condition_framing"]
        print(f"  {cname:20s} eval-framed {b['eval']:.0%}  "
              f"deployment-framed {b['deployment']:.0%}")

    results["stats"] = {
        "probe_vs_random_fisher_p": p_vs_rand,
        "probe_vs_unsteered_fisher_p": p_vs_base,
        "verdict": verdict,
        "coeff_norms": COEFF_NORMS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "prespecified": True,
    }
    with open(os.path.join(RESULTS, "phase4_refusal_replication.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote results/phase4_refusal_replication.json and "
          "logs/phase4c_generations.txt")


if __name__ == "__main__":
    main()
