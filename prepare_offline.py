import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from dynasteer_core import (
    compute_math_score,
    encode_text,
    generate_continuation,
    load_model_and_tokenizer,
    load_problem_dataset,
    model_input_device,
    render_math_prompt,
    seed_everything,
    split_into_sentence_chunks,
)


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Build high-entropy Truth/Fallacy sentence forks for DynaSteer.")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--datasets", default="gsm8k,math")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--limit_per_dataset", type=int, default=0)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--enable_thinking", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_baseline_tokens", type=int, default=1024)
    ap.add_argument("--max_sentence_tokens", type=int, default=256)
    ap.add_argument("--max_completion_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--fork_temperature", type=float, default=1.0)
    ap.add_argument("--n_forks", type=int, default=10)
    ap.add_argument("--label_rollouts", type=int, default=10)
    ap.add_argument("--entropy_quantile", type=float, default=80.0)
    ap.add_argument("--entropy_mode", choices=["full_vocab", "topk_renorm"], default="full_vocab")
    ap.add_argument("--entropy_topk", type=int, default=20)
    args = ap.parse_args()

    seed_everything(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(args.model, args.dtype)
    device = model_input_device(model)

    baseline_rows = []
    all_sentence_entropies = []
    problem_counter = 0

    for dataset_name in [x.strip() for x in args.datasets.split(",") if x.strip()]:
        data = load_problem_dataset(dataset_name, split="train")
        if args.limit_per_dataset > 0:
            data = data[: args.limit_per_dataset]

        for local_idx, item in enumerate(tqdm(data, desc=f"baseline:{dataset_name}")):
            prompt_text = render_math_prompt(tokenizer, item["problem"], args.enable_thinking)
            prompt_ids = encode_text(tokenizer, prompt_text, device)
            gen = generate_continuation(
                model,
                tokenizer,
                prompt_ids,
                max_new_tokens=args.max_baseline_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                entropy_mode=args.entropy_mode,
                entropy_topk=args.entropy_topk,
                stop_at_sentence=False,
            )
            chunks = split_into_sentence_chunks(tokenizer, gen["token_ids"], gen["entropies"])
            all_sentence_entropies.extend([c["avg_entropy"] for c in chunks])
            baseline_rows.append(
                {
                    "problem_id": problem_counter,
                    "dataset": dataset_name,
                    "dataset_index": local_idx,
                    "problem": item["problem"],
                    "ground_truth": item["answer"],
                    "prompt_text": prompt_text,
                    "baseline_text": gen["text"],
                    "baseline_score": compute_math_score(gen["text"][-1000:], item["answer"]),
                    "token_ids": gen["token_ids"],
                    "chunks": chunks,
                }
            )
            problem_counter += 1

    if not all_sentence_entropies:
        raise RuntimeError("No sentence chunks were produced; cannot compute the P80 threshold.")

    gamma = float(np.percentile(all_sentence_entropies, args.entropy_quantile))
    stats = {
        "model": args.model,
        "entropy_mode": args.entropy_mode,
        "entropy_topk": args.entropy_topk,
        "quantile": args.entropy_quantile,
        "gamma": gamma,
        "n_sentences": len(all_sentence_entropies),
        "mean": float(np.mean(all_sentence_entropies)),
        "std": float(np.std(all_sentence_entropies)),
        "p50": float(np.percentile(all_sentence_entropies, 50)),
        "p80": float(np.percentile(all_sentence_entropies, 80)),
        "p90": float(np.percentile(all_sentence_entropies, 90)),
    }
    (out_dir / "entropy_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    write_jsonl(out_dir / "baseline_chunks.jsonl", baseline_rows)
    print(f"Global entropy threshold gamma (P{args.entropy_quantile:g}) = {gamma:.6f}")

    fork_rows = []
    for row in tqdm(baseline_rows, desc="high-entropy forks"):
        prior_text = ""
        for chunk_idx, chunk in enumerate(row["chunks"]):
            if chunk["avg_entropy"] <= gamma:
                prior_text += chunk["text"]
                continue

            context_text = row["prompt_text"] + prior_text
            context_ids = encode_text(tokenizer, context_text, device)
            seen = set()
            positive, negative, ambiguous = [], [], []

            for _ in range(args.n_forks):
                fork = generate_continuation(
                    model,
                    tokenizer,
                    context_ids,
                    max_new_tokens=args.max_sentence_tokens,
                    temperature=args.fork_temperature,
                    top_p=1.0,
                    top_k=-1,
                    entropy_mode=args.entropy_mode,
                    entropy_topk=args.entropy_topk,
                    stop_at_sentence=True,
                )
                fork_text = fork["text"]
                if not fork_text or fork_text in seen:
                    continue
                seen.add(fork_text)

                fork_prefix_ids = encode_text(tokenizer, context_text + fork_text, device)
                rollout_scores = []
                for _ in range(args.label_rollouts):
                    completion = generate_continuation(
                        model,
                        tokenizer,
                        fork_prefix_ids,
                        max_new_tokens=args.max_completion_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        entropy_mode=args.entropy_mode,
                        entropy_topk=args.entropy_topk,
                        stop_at_sentence=False,
                    )
                    full_answer = fork_text + completion["text"]
                    rollout_scores.append(compute_math_score(full_answer[-1200:], row["ground_truth"]))

                fork_item = {
                    "text": fork_text,
                    "fork_avg_entropy": fork["avg_entropy"],
                    "rollout_scores": rollout_scores,
                }
                if rollout_scores and all(s == 1.0 for s in rollout_scores):
                    positive.append(fork_item)
                elif rollout_scores and all(s == 0.0 for s in rollout_scores):
                    negative.append(fork_item)
                else:
                    ambiguous.append(fork_item)

            if positive and negative:
                fork_rows.append(
                    {
                        "problem_id": row["problem_id"],
                        "dataset": row["dataset"],
                        "chunk_idx": chunk_idx,
                        "problem": row["problem"],
                        "ground_truth": row["ground_truth"],
                        "context": context_text,
                        "ori_chunk_text": chunk["text"],
                        "ori_chunk_entropy": chunk["avg_entropy"],
                        "positive_forks": positive,
                        "negative_forks": negative,
                        "ambiguous_count": len(ambiguous),
                    }
                )
            prior_text += chunk["text"]

    write_jsonl(out_dir / "forks.jsonl", fork_rows)
    print(f"Saved {len(fork_rows)} contrastive fork contexts to {out_dir / 'forks.jsonl'}")


if __name__ == "__main__":
    main()
