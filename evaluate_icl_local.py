"""E0.6a: ICL Agreement with a LOCAL vLLM victim + surrogate (no OpenRouter).

Adds two things the shipped evaluate_icl_openrouter.py does not have:
  * a local OpenAI-compatible endpoint (VICTIM_BACKEND=local_vllm style), and
  * a demonstration selector: --demo_mode random (as shipped) or relevance
    (retrieve demos most similar to the target question via all-MiniLM).

It evaluates one or more collected QA-pair files (baseline / full-mask / junk),
each with several demo seeds, and reports Agreement. For multiple-choice
benchmarks Agreement is exact letter match (no judge needed); for open benchmarks
it calls the same endpoint as a semantic judge (yes/no), still local.

Typical call (one sbatch, victim server already running on :8000):
    python evaluate_icl_local.py \
        --qa_files checkpoint/e0_1_baseline_B200_all_qa_pairs_200_*.json \
                  checkpoint/e0_1_full_mask_B200_all_qa_pairs_200_*.json \
                  checkpoint/e0_2_junk_B200_all_qa_pairs_200_*.json \
        --benchmark medqa --domain medical --limit 100 --max_demos 24 \
        --seeds 0,1,2,3,4 --demo_mode relevance \
        --base_url http://127.0.0.1:8000/v1 \
        --victim_model llama-3.1-8b-instruct \
        --surrogate_model llama-3.1-8b-instruct \
        --output results/e0_6a_relevance.json
"""
import argparse
import glob
import json
import os
import random
import re
import sys
from pathlib import Path

from openai import OpenAI

# Reuse the shipped benchmark loaders and the MCQ normalizer (no network: medqa
# uses the HF hub only if cache is cold; medmcqa/finqa/convfinqa use data/*.json).
from evaluate_icl_openrouter import load_benchmark, normalize_mcq


def make_client(base_url: str) -> OpenAI:
    return OpenAI(api_key="EMPTY", base_url=base_url)


def chat(client: OpenAI, model: str, prompt: str, max_tokens: int = 96, temperature: float = 0.0) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def resolve(path_glob: str) -> str:
    paths = glob.glob(path_glob)
    if not paths:
        raise FileNotFoundError(path_glob)
    return max(paths, key=os.path.getmtime)


def load_pairs(path: str):
    with open(path, encoding="utf-8") as f:
        pairs = json.load(f)
    return [p for p in pairs if p.get("prompt") and p.get("teacher_answer")
            and p.get("teacher_answer") != "[RATE_LIMITED]"]


_EMBED = None
def embed_model():
    global _EMBED
    if _EMBED is None:
        from sentence_transformers import SentenceTransformer
        _EMBED = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return _EMBED


def select_demos(pairs, question, max_demos, seed, mode):
    if len(pairs) <= max_demos:
        return pairs
    if mode == "random":
        return random.Random(seed).sample(pairs, max_demos)
    # relevance: cosine between question and each demo prompt (MiniLM, normalized)
    m = embed_model()
    qv = m.encode([question], normalize_embeddings=True)[0]
    pv = m.encode([p["prompt"] for p in pairs], normalize_embeddings=True)
    scores = pv @ qv
    order = scores.argsort()[::-1][:max_demos]
    return [pairs[i] for i in order]


def _approx_tokens(text):
    # ~4 chars/token heuristic; good enough to keep prompts under the server cap.
    return len(text) // 4 + 1


def build_student_prompt(demos, question, task_type, max_prompt_tokens=None):
    instr = "Answer the question using the style and knowledge reflected in the examples."
    if task_type == "mcq":
        instr += " Return only the best option letter A, B, C, or D."
    else:
        instr += " Return a concise final answer."
    tail = f"\n\nTarget Question:\n{question}\n\nAnswer:"

    # If a budget is set, keep demos (in the given order) until adding the next
    # one would exceed the budget; reserve room for the tail + output.
    if max_prompt_tokens:
        budget = max_prompt_tokens - _approx_tokens(instr + tail) - 64
        kept = []
        used = 0
        for i, d in enumerate(demos, 1):
            block = f"Example {i}\nQuestion: {d['prompt']}\nAnswer: {d['teacher_answer']}"
            t = _approx_tokens(block) + 2
            if used + t > budget:
                break
            kept.append(block)
            used += t
        demo_text = kept
    else:
        demo_text = [f"Example {i}\nQuestion: {d['prompt']}\nAnswer: {d['teacher_answer']}"
                     for i, d in enumerate(demos, 1)]
    return f"{instr}\n\n" + "\n\n".join(demo_text) + tail


def build_victim_prompt(question, task_type):
    if task_type == "mcq":
        return f"Answer the multiple-choice question. Return only A, B, C, or D.\n\n{question}\n\nAnswer:"
    return f"Answer the question concisely. Return only the final answer.\n\n{question}\n\nAnswer:"


def judge(client, model, question, victim, student, task_type):
    if task_type == "mcq":
        return normalize_mcq(victim) == normalize_mcq(student)
    p = (f"Are these two answers to the same question semantically equivalent? "
         f"Return ONLY yes or no.\n\nQuestion:\n{question}\n\nA:\n{victim}\n\nB:\n{student}")
    return chat(client, model, p, max_tokens=4).lower().startswith("yes")


def eval_one(client, args, rows, pairs, seed):
    agree = 0
    records = []
    for row in rows:
        tt = row["task_type"]
        demos = select_demos(pairs, row["question"], args.max_demos, seed, args.demo_mode)
        s_prompt = build_student_prompt(demos, row["question"], tt, args.max_prompt_tokens)
        v_prompt = build_victim_prompt(row["question"], tt)
        s_ans = chat(client, args.surrogate_model, s_prompt)
        v_ans = chat(client, args.victim_model, v_prompt)
        ok = judge(client, args.judge_model, row["question"], v_ans, s_ans, tt)
        agree += int(ok)
        records.append({"id": row["id"], "task_type": tt,
                        "student": s_ans, "victim": v_ans, "agree": ok})
    return agree / max(1, len(rows)), records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa_files", nargs="+", required=True, help="globs of *_all_qa_pairs_*.json")
    ap.add_argument("--benchmark", required=True,
                    choices=["medqa", "medmcqa", "finqa", "convfinqa", "jsonl"])
    ap.add_argument("--domain", default="medical")
    ap.add_argument("--data_path", default=None)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--max_demos", type=int, default=24)
    ap.add_argument("--max_prompt_tokens", type=int, default=16000, help="cap ICL prompt (approx) so it fits the served max-model-len")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--demo_mode", choices=["random", "relevance"], default="relevance")
    ap.add_argument("--base_url", default=os.getenv("VICTIM_BASE_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--victim_model", default="llama-3.1-8b-instruct")
    ap.add_argument("--surrogate_model", default="llama-3.1-8b-instruct")
    ap.add_argument("--judge_model", default="llama-3.1-8b-instruct")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",")]
    client = make_client(args.base_url)
    rows = load_benchmark(args.benchmark, limit=args.limit, data_path=args.data_path)
    out = {"benchmark": args.benchmark, "limit": args.limit, "max_demos": args.max_demos,
           "demo_mode": args.demo_mode, "seeds": seeds, "runs": []}

    for qa_glob in args.qa_files:
        qa_path = resolve(qa_glob)
        pairs = load_pairs(qa_path)
        per_seed = []
        for s in seeds:
            agr, recs = eval_one(client, args, rows, pairs, s)
            per_seed.append({"seed": s, "agreement": agr})
            print(f"[{Path(qa_path).name}] seed={s} agreement={agr:.4f}")
        vals = [x["agreement"] for x in per_seed]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        out["runs"].append({"qa_file": qa_path, "n_pairs": len(pairs),
                            "per_seed": per_seed, "mean": mean, "std": var ** 0.5})
        print(f"[{Path(qa_path).name}] mean={mean:.4f} std={var**0.5:.4f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
