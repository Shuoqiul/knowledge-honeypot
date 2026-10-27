"""E0.7: benign false-trigger rate of the released detector on real benchmark
queries, with the origin tag masked (a deployed defender cannot read it).

Pure CPU / stdlib. Does not import torch and never touches the network for the
victim: it only calls HoneypotTeacher.suspicion_score, exactly as the attacker
loop does. Reads the bundled benchmark caches in data/ so no download is needed.

Usage:
    python scripts/e0_7_benign_false_trigger.py \
        --domain medical \
        --out results/e0_7/e0_7_benign_false_trigger.json

Design (mirrors the audit's reasoning):
  - Each benign query is treated as belonging to a session of declared length L.
  - We sweep the step within a session from 0..(L-1) to find the first step at
    which a *fixed* query text would cross tau (the score has a time term that
    grows with step/L).
  - We report the trigger rate as a function of (tau, L) and the distribution of
    first-trigger steps. source is None, i.e. tag masked.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from honeypot.honeypot_teacher import HoneypotTeacher


def load_rows(domain: str, limit: int):
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    if domain == "medical":
        cache = os.path.join(base, "medmcqa_eval_cache.json")
        if os.path.exists(cache):
            with open(cache, encoding="utf-8") as f:
                rows = json.load(f)
            return [r.get("question", "") for r in rows][:limit if limit else None]
        raise FileNotFoundError(
            "medmcqa cache not found; MedQA needs the HF hub. Run this on a node "
            "with data/ populated, or pass --questions_file."
        )
    if domain == "financial":
        cache = os.path.join(base, "finqa_test_cache.json")
        with open(cache, encoding="utf-8") as f:
            rows = json.load(f)
        return [r.get("question", "") for r in rows][:limit if limit else None]
    raise ValueError(domain)


def first_trigger_step(score_fn, question, length, tau):
    for step in range(length):
        if score_fn(question, None, step, length) >= tau:
            return step
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="medical", choices=["medical", "financial"])
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows in cache")
    ap.add_argument("--questions_file", default=None, help="optional newline-delimited questions")
    ap.add_argument("--lengths", default="50,100,200,500")
    ap.add_argument("--taus", default="0.4,0.5,0.6,0.7,0.8")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.questions_file:
        with open(args.questions_file, encoding="utf-8") as f:
            questions = [ln.strip() for ln in f if ln.strip()]
        if args.limit:
            questions = questions[: args.limit]
    else:
        questions = load_rows(args.domain, args.limit)

    # An HKG with no nodes avoids any file dependency; we only use the score.
    teacher = HoneypotTeacher(domain=args.domain, base_query=lambda *a, **k: "", hkg_size=0)
    score = teacher.suspicion_score

    lengths = [int(x) for x in args.lengths.split(",")]
    taus = [float(x) for x in args.taus.split(",")]

    table = []
    for L in lengths:
        for tau in taus:
            triggered = 0
            steps = []
            for q in questions:
                s0 = first_trigger_step(score, q, L, tau)
                if s0 is not None:
                    triggered += 1
                    steps.append(s0 / L)
            steps.sort()
            table.append({
                "session_length": L,
                "tau": tau,
                "n": len(questions),
                "triggered": triggered,
                "trigger_rate": triggered / max(1, len(questions)),
                "median_first_trigger_frac": steps[len(steps) // 2] if steps else None,
            })
            print(f"L={L:>3} tau={tau:.2f} -> trigger_rate={triggered/max(1,len(questions)):.4f} "
                  f"(median frac {table[-1]['median_first_trigger_frac']})")

    # Also report the no-time-term base score distribution (step=0 dominates).
    base = [score(q, None, 0, 1) for q in questions]
    base_min = min(base) if base else None
    base_mean = sum(base) / len(base) if base else None

    payload = {
        "experiment": "E0.7 benign false trigger (source masked)",
        "domain": args.domain,
        "n_questions": len(questions),
        "source": None,
        "base_score": {"min": base_min, "mean": base_mean},
        "table": table,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
