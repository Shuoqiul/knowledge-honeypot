"""E0.8: accuracy loss for BENIGN users who get falsely triggered and substituted.

E0.7 measured only the *trigger probability* (does a benign session ever cross
tau). E0.8 measures what the paper actually claims: does substituting benign
users' answers hurt their *task accuracy*.

Construction (session-length driven, "A"):
  - Take benign benchmark questions with GOLD labels (MedQA).
  - Lay them into a session of declared length L. Walk steps 0..L-1; at each
    step the benign query shares the same *content* score as any query
    (tag masked), so the detector fires at step t0 (predicted by suspicion_score
    with the time term). Queries at step >= t0 are answered with the DEFENSE
    substitution (HKG node or junk); queries before t0 get the real victim answer.
  - Metric: task accuracy vs GOLD. Two arms:
       * CONTROL: every query answered by the real victim (no defense).
       * TRIGGERED: queries >= t0 replaced (HKG or junk), others real.
    The benign-user harm = accuracy(CONTROL) - accuracy(TRIGGERED).

This is accuracy vs gold (what a benign user wants), NOT Agreement vs victim.
No attacker, no student, no ICL. Only the local vLLM victim is queried.
"""
import argparse
import json
import os
import re
import sys

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate import load_medqa, normalize_choice          # gold-labelled MedQA
from honeypot.honeypot_teacher import HoneypotTeacher


def make_client(base_url):
    return OpenAI(api_key="EMPTY", base_url=base_url)


def ask(client, model, question, max_tokens=16):
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": question}],
        temperature=0.0, max_tokens=max_tokens,
    )
    return r.choices[0].message.content.strip()


def first_trigger(score, question, L, tau):
    for step in range(L):
        if score(question, None, step, L) >= tau:
            return step
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="medical")
    ap.add_argument("--benchmark", default="medqa")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--lengths", default="50,100,200,500")
    ap.add_argument("--taus", default="0.6,0.7,0.8")
    ap.add_argument("--defense", choices=["hkg", "junk"], default="hkg")
    ap.add_argument("--hkg_path", default="honeypot_hkg/medical_hkg_1500_repaired.json")
    ap.add_argument("--base_url", default=os.getenv("VICTIM_BASE_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--victim_model", default="llama-3.1-8b-instruct")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = load_medqa(args.limit)
    client = make_client(args.base_url)

    # Detector (no HKG needed for scoring; use empty to avoid file load) + a
    # substitution teacher for the DEFENSE answer text.
    scorer = HoneypotTeacher(domain=args.domain, base_query=lambda *a, **k: "", hkg_size=0)
    score = scorer.suspicion_score
    sub = HoneypotTeacher(
        domain=args.domain, base_query=lambda *a, **k: "", mode="full",
        hkg_size=1500, hkg_path=args.hkg_path,
        junk_control=(args.defense == "junk"), mask_source=True,
    )

    # CONTROL: real victim accuracy on each question (once).
    control_correct = []
    for row in rows:
        ans = ask(client, args.victim_model, row["question"])
        control_correct.append(normalize_choice(ans) == row["answer"])
    ctrl_acc = sum(control_correct) / len(rows)

    lengths = [int(x) for x in args.lengths.split(",")]
    taus = [float(x) for x in args.taus.split(",")]

    table = []
    for L in lengths:
        for tau in taus:
            # For a benign session of length L, where does it trigger? Use the
            # median question as representative (same detector for all).
            trig_steps = [first_trigger(score, r["question"], L, tau) for r in rows]
            n_triggered_sessions = sum(1 for t in trig_steps if t is not None)

            # TRIGGERED arm: a query is substituted iff its session triggered at
            # or before this question's position. To make a per-question accuracy
            # we use a representative session: questions at index i in [0,lim)
            # are placed at step i of a length-L session.
            # Place questions 0..(min(n,L)-1) at steps 0.. of an L-session. Only
            # these in-session questions can be substituted; questions beyond L
            # are outside the session and always real (excluded from the "in-
            # session accuracy" that measures benign harm).
            n_in = min(len(rows), L)
            ok_in = 0
            sub_count = 0
            for i in range(n_in):
                row = rows[i]
                t0 = trig_steps[i]
                substituted = (t0 is not None) and (i >= t0)
                if substituted:
                    # Force the defense answer once the session has triggered at
                    # or before this step (deterministic "remainder substituted").
                    if args.defense == "hkg":
                        ans = sub._hkg_answer(sub.hkg.next_node(), include_breadcrumb=True)
                    else:
                        ans = sub._junk_answer()
                    sub_count += 1
                else:
                    ans = ask(client, args.victim_model, row["question"])
                ok_in += int(normalize_choice(ans) == row["answer"])
            triggered_acc_in = ok_in / max(1, n_in)

            # Control accuracy over the SAME in-session subset for a fair delta.
            ctrl_acc_in = sum(control_correct[:n_in]) / max(1, n_in)
            table.append({
                "session_length": L, "tau": tau,
                "in_session_n": n_in,
                "sessions_triggered_frac": n_triggered_sessions / len(rows),
                "substituted_frac_in_session": sub_count / max(1, n_in),
                "control_accuracy_in_session": ctrl_acc_in,
                "triggered_accuracy_in_session": triggered_acc_in,
                "accuracy_loss_pp_in_session": (ctrl_acc_in - triggered_acc_in) * 100,
            })
            print(f"L={L:>3} tau={tau} trig_sess={n_triggered_sessions/len(rows):.2f} "
                  f"sub_in={sub_count/max(1,n_in):.2f} "
                  f"ctrl_in={ctrl_acc_in:.3f} trig_in={triggered_acc_in:.3f} "
                  f"loss={(ctrl_acc_in-triggered_acc_in)*100:.1f}pp")

    payload = {"experiment": "E0.8 benign accuracy loss under false trigger",
               "defense": args.defense, "benchmark": args.benchmark,
               "n": len(rows), "metric": "task accuracy vs gold",
               "control_accuracy": ctrl_acc, "table": table}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(payload, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
