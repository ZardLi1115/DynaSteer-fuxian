import argparse
import json
from pathlib import Path

from tqdm import tqdm

from dynasteer_core import (
    SteeringInjector,
    compute_math_score,
    encode_text,
    generate_continuation,
    load_model_and_tokenizer,
    load_pickle,
    load_problem_dataset,
    model_input_device,
    render_math_prompt,
    seed_everything,
)


def dynasteer_generate(model, tokenizer, prompt_ids, injector, args, gamma):
    chosen_ids = []
    chosen_text_parts = []
    log = []
    extra_candidate_tokens = 0

    for sentence_idx in range(1, args.max_sentences + 1):
        if len(chosen_ids) >= args.max_new_tokens:
            break
        remaining = args.max_new_tokens - len(chosen_ids)
        sentence_budget = min(args.max_sentence_tokens, remaining)
        prefix = prompt_ids.new_tensor([prompt_ids[0].tolist() + chosen_ids])

        candidate = generate_continuation(
            model,
            tokenizer,
            prefix,
            max_new_tokens=sentence_budget,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            entropy_mode=args.entropy_mode,
            entropy_topk=args.entropy_topk,
            stop_at_sentence=True,
        )
        if not candidate["token_ids"]:
            break

        high_entropy = args.no_entropy_gating or candidate["avg_entropy"] > gamma
        inside_decay = args.tau_decay_sentences < 0 or sentence_idx <= args.tau_decay_sentences
        intervene = high_entropy and inside_decay

        if intervene:
            extra_candidate_tokens += len(candidate["token_ids"])
            prefix = prompt_ids.new_tensor([prompt_ids[0].tolist() + chosen_ids])
            steered = generate_continuation(
                model,
                tokenizer,
                prefix,
                max_new_tokens=sentence_budget,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                entropy_mode=args.entropy_mode,
                entropy_topk=args.entropy_topk,
                stop_at_sentence=True,
                hook_context=injector.hooks(model),
            )
            selected = steered if steered["token_ids"] else candidate
        else:
            selected = candidate

        chosen_ids.extend(selected["token_ids"])
        chosen_text_parts.append(selected["text"])
        log.append(
            {
                "sentence_idx": sentence_idx,
                "candidate_entropy": candidate["avg_entropy"],
                "triggered": bool(intervene),
                "candidate_text": candidate["text"],
                "selected_text": selected["text"],
            }
        )
        if selected["hit_eos"]:
            break

    return "".join(chosen_text_parts), log, extra_candidate_tokens


def main():
    ap = argparse.ArgumentParser(description="Run reconstructed DynaSteer online rollback + activation steering.")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--dataset", choices=["gsm8k", "math500", "jsonl"], default="math500")
    ap.add_argument("--jsonl_path", default=None)
    ap.add_argument("--steering", required=True)
    ap.add_argument("--entropy_stats", default=None)
    ap.add_argument("--entropy_threshold", type=float, default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--mode", choices=["plain", "dynasteer"], default="dynasteer")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--enable_thinking", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=3.0)
    ap.add_argument("--top_heads", type=int, default=32)
    ap.add_argument("--tau_decay_sentences", type=int, default=-1)
    ap.add_argument("--no_entropy_gating", action="store_true")
    ap.add_argument("--entropy_mode", choices=["full_vocab", "topk_renorm"], default="full_vocab")
    ap.add_argument("--entropy_topk", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--max_sentence_tokens", type=int, default=256)
    ap.add_argument("--max_sentences", type=int, default=64)
    args = ap.parse_args()

    seed_everything(args.seed)
    model, tokenizer = load_model_and_tokenizer(args.model, args.dtype)
    device = model_input_device(model)
    steering_cfg = load_pickle(args.steering)
    injector = SteeringInjector(steering_cfg, alpha=args.alpha, top_heads=args.top_heads)

    gamma = args.entropy_threshold
    if gamma is None and args.entropy_stats:
        stats = json.loads(Path(args.entropy_stats).read_text(encoding="utf-8"))
        if stats.get("entropy_mode") != args.entropy_mode:
            raise ValueError(
                f"Entropy mode mismatch: stats={stats.get('entropy_mode')} eval={args.entropy_mode}. "
                "Use matching modes or pass --entropy_threshold explicitly."
            )
        gamma = float(stats["gamma"])
    if gamma is None:
        if args.mode == "plain" or args.no_entropy_gating:
            gamma = float("inf")
        else:
            raise ValueError("Provide --entropy_stats or --entropy_threshold for DynaSteer entropy gating.")

    split = "test" if args.dataset in {"gsm8k", "math500"} else None
    data = load_problem_dataset(args.dataset, split=split, jsonl_path=args.jsonl_path)
    data = data[args.offset :]
    if args.limit > 0:
        data = data[: args.limit]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    correct = 0.0
    total_interventions = 0
    total_extra_tokens = 0

    with output_path.open("w", encoding="utf-8") as f:
        for idx, item in enumerate(tqdm(data, desc=args.mode)):
            prompt_text = render_math_prompt(tokenizer, item["problem"], args.enable_thinking)
            prompt_ids = encode_text(tokenizer, prompt_text, device)

            if args.mode == "plain":
                gen = generate_continuation(
                    model,
                    tokenizer,
                    prompt_ids,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    entropy_mode=args.entropy_mode,
                    entropy_topk=args.entropy_topk,
                    stop_at_sentence=False,
                )
                text, trajectory_log, extra_tokens = gen["text"], [], 0
            else:
                text, trajectory_log, extra_tokens = dynasteer_generate(
                    model, tokenizer, prompt_ids, injector, args, gamma
                )

            score = compute_math_score(text[-1200:], item["answer"])
            correct += score
            interventions = sum(1 for x in trajectory_log if x["triggered"])
            total_interventions += interventions
            total_extra_tokens += extra_tokens

            record = {
                "index": args.offset + idx,
                "source": item["source"],
                "problem": item["problem"],
                "ground_truth": item["answer"],
                "model_pred": text,
                "score": score,
                "gamma": gamma,
                "interventions": interventions,
                "rollback_extra_tokens": extra_tokens,
                "trajectory": trajectory_log,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    n = max(len(data), 1)
    print(f"Accuracy: {correct / n:.4f} ({correct:.0f}/{len(data)})")
    print(f"Interventions: {total_interventions}, mean/sample={total_interventions / n:.3f}")
    print(f"Rollback extra tokens: {total_extra_tokens}, mean/sample={total_extra_tokens / n:.2f}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
