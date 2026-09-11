"""Phase 5 - pull RANDOMLY SELECTED raw examples for the write-up.

Neel's write-up guidance: "If bad data would sink your project, show me the
data... include some randomly selected qualitative examples in the write-up,
ideally just after the executive summary. Randomly selected, not cherry-picked!"

Two things in this project would sink it if they were bad:
  1. the hand-written contrastive prompt pairs (everything rests on them being
     genuinely matched and on neither side sounding fake)
  2. the regex refusal metric in 04b/04c (the Phase 4 behavioural numbers are
     only meaningful if its judgements are correct)

So this samples both with a FIXED SEED, prints whatever comes out, and makes no
attempt to pick flattering cases. For the refusal metric it samples flagged and
unflagged generations separately so both error directions are visible.

No new experiments - this only reformats data already on disk.

Run:  python src/05_random_samples_for_writeup.py
Out:  logs/random_samples_for_writeup.txt (and stdout)
"""

import json
import os
import random
import re

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 20260911

def main():
    rng = random.Random(SEED)
    out_lines = []

    def emit(s=""):
        print(s)
        out_lines.append(s)

    emit(f"RANDOMLY SELECTED EXAMPLES  (seed={SEED}, nothing cherry-picked)")
    emit("=" * 88)

    # ---------- 1. random prompt pairs ----------
    with open(os.path.join(HERE, "data", "contrastive_prompts.jsonl")) as f:
        records = [json.loads(line) for line in f]
    by_pair = {}
    for r in records:
        by_pair.setdefault(r["pair_id"], {})[r["condition"]] = r
    pair_ids = sorted(by_pair)
    chosen = rng.sample(pair_ids, 3)

    emit()
    emit("PART 1: three randomly chosen prompt pairs out of 50")
    emit("=" * 88)
    for pid in chosen:
        p = by_pair[pid]
        emit()
        emit(f"--- pair {pid}  (category={p['eval']['category']}, "
             f"framing={p['eval']['framing_style']}, "
             f"template={p['eval']['framing_template_id']}) ---")
        for cond in ("eval", "deployment"):
            emit(f"[{cond.upper()}]")
            emit(p[cond]["prompt"])
            emit()
        same = p["eval"]["task_core"] == p["deployment"]["task_core"]
        emit(f"task text identical across the two framings: {same}")

    # ---------- 2. random refusal-metric judgements ----------
    gen_path = os.path.join(HERE, "logs", "phase4c_generations.txt")
    header = re.compile(r"^condition=(\S+) id=(\S+) refusal=(True|False)$")
    entries, cur, buf = [], None, []
    with open(gen_path) as f:
        for line in f:
            m = header.match(line.strip())
            if m:
                if cur:
                    entries.append((*cur, "\n".join(buf).strip()))
                cur = (m.group(1), m.group(2), m.group(3) == "True")
                buf = []
            elif cur is not None and not set(line.strip()) <= {"=", "-"}:
                buf.append(line.rstrip("\n"))
    if cur:
        entries.append((*cur, "\n".join(buf).strip()))

    steered = [e for e in entries if e[0] == "probe_lr@+1"]
    flagged = [e for e in steered if e[2]]
    unflagged = [e for e in steered if not e[2]]
    emit()
    emit("=" * 88)
    emit("PART 2: randomly chosen refusal-metric judgements, probe_lr@+1 "
         "condition")
    emit(f"({len(flagged)} of {len(steered)} were flagged as refusals; "
         f"sampling 3 flagged + 3 unflagged so both error directions show)")
    emit("=" * 88)
    for label, pool, k in (("FLAGGED AS REFUSAL", flagged, 3),
                           ("NOT FLAGGED", unflagged, 3)):
        emit()
        emit(f"### {label} ###")
        for cond, pid, flag, text in rng.sample(pool, min(k, len(pool))):
            emit()
            emit(f"[{pid}]  metric says refusal={flag}")
            snippet = " ".join(text.split())[:300]
            emit(f"  {snippet} ...")

    path = os.path.join(HERE, "logs", "random_samples_for_writeup.txt")
    with open(path, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    emit()
    emit(f"[written to {path}]")


if __name__ == "__main__":
    main()
