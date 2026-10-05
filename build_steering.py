import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dynasteer_core import encode_text, load_model_and_tokenizer, model_input_device, save_pickle, seed_everything


def capture_attention_last(model, input_ids: torch.Tensor) -> np.ndarray:
    layers = model.model.layers
    n_layers = len(layers)
    n_heads = model.config.num_attention_heads
    hidden_size = model.config.hidden_size
    head_dim = hidden_size // n_heads
    captured = [None] * n_layers
    handles = []

    for layer_idx, layer in enumerate(layers):
        def make_hook(idx):
            def hook(_module, inputs):
                x = inputs[0]
                captured[idx] = x[0, -1].detach().float().cpu().numpy()
            return hook
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(make_hook(layer_idx)))

    try:
        with torch.inference_mode():
            _ = model(input_ids=input_ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    if any(x is None for x in captured):
        raise RuntimeError("Failed to capture attention outputs from every layer.")
    arr = np.stack(captured, axis=0).reshape(n_layers, n_heads, head_dim)
    return arr.astype(np.float16)


def lda_direction(pos: np.ndarray, neg: np.ndarray):
    pos = pos.astype(np.float64)
    neg = neg.astype(np.float64)
    mu_p, mu_n = pos.mean(0), neg.mean(0)
    dim = pos.shape[1]

    def cov(x):
        if len(x) >= 2:
            return LedoitWolf().fit(x).covariance_
        return np.eye(dim) * 1e-3

    sw = cov(pos) + cov(neg)
    reg = 1e-4 * (np.trace(sw) / max(dim, 1)) + 1e-8
    sw = sw + reg * np.eye(dim)
    delta = mu_p - mu_n
    w = np.linalg.solve(sw, delta)
    w = w / (np.linalg.norm(w) + 1e-12)
    fdr = float((w @ delta) ** 2 / (w @ sw @ w + 1e-12))
    return w.astype(np.float32), fdr


def probe_accuracy(pos: np.ndarray, neg: np.ndarray, seed: int) -> float:
    x = np.vstack([pos, neg]).astype(np.float32)
    y = np.array([1] * len(pos) + [0] * len(neg))
    if len(pos) < 2 or len(neg) < 2 or len(x) < 6:
        return 0.5
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, stratify=y, random_state=seed
        )
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
        clf.fit(x_train, y_train)
        return float(clf.score(x_test, y_test))
    except Exception:
        return 0.5


def main():
    ap = argparse.ArgumentParser(description="Collect attention-head activations and solve clustered Fisher-LDA steering vectors.")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--forks", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--n_clusters", type=int, default=5)
    ap.add_argument("--top_heads", type=int, default=32)
    ap.add_argument("--max_per_class", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    model, tokenizer = load_model_and_tokenizer(args.model, args.dtype)
    device = model_input_device(model)

    rows = []
    with open(args.forks, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    groups = []
    for row in tqdm(rows, desc="activation collection"):
        pos_items = row["positive_forks"]
        neg_items = row["negative_forks"]
        if args.max_per_class > 0:
            pos_items = pos_items[: args.max_per_class]
            neg_items = neg_items[: args.max_per_class]
        pos_acts, neg_acts = [], []
        for item in pos_items:
            ids = encode_text(tokenizer, row["context"] + item["text"], device)
            pos_acts.append(capture_attention_last(model, ids))
        for item in neg_items:
            ids = encode_text(tokenizer, row["context"] + item["text"], device)
            neg_acts.append(capture_attention_last(model, ids))
        if pos_acts and neg_acts:
            groups.append(
                {
                    "problem_id": row["problem_id"],
                    "chunk_idx": row["chunk_idx"],
                    "pos": np.stack(pos_acts),
                    "neg": np.stack(neg_acts),
                }
            )

    if not groups:
        raise RuntimeError("No contrastive activation groups found.")

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads
    units = []

    for layer_idx in tqdm(range(n_layers), desc="Fisher-LDA layers"):
        for head_idx in range(n_heads):
            diffs = []
            valid_groups = []
            for group in groups:
                pos = group["pos"][:, layer_idx, head_idx].astype(np.float32)
                neg = group["neg"][:, layer_idx, head_idx].astype(np.float32)
                if len(pos) and len(neg):
                    diffs.append(pos.mean(0) - neg.mean(0))
                    valid_groups.append((pos, neg))
            if not diffs:
                continue

            diff_mat = np.stack(diffs)
            diff_norm = diff_mat / (np.linalg.norm(diff_mat, axis=1, keepdims=True) + 1e-12)
            n_clusters = min(args.n_clusters, len(valid_groups))
            labels = KMeans(n_clusters=n_clusters, random_state=args.seed, n_init=10).fit_predict(diff_norm)

            clusters = []
            cluster_probe, cluster_weights, cluster_fdr = [], [], []
            for cluster_id in range(n_clusters):
                idx = np.where(labels == cluster_id)[0]
                pos = np.concatenate([valid_groups[i][0] for i in idx], axis=0)
                neg = np.concatenate([valid_groups[i][1] for i in idx], axis=0)
                w, fdr = lda_direction(pos, neg)
                pos_proj = pos.astype(np.float32) @ w
                acc = probe_accuracy(pos, neg, args.seed)
                clusters.append(
                    {
                        "cluster_id": int(cluster_id),
                        "cluster_size": int(len(idx)),
                        "head_dim": int(head_dim),
                        "w_lda": w,
                        "mu_p": float(np.mean(pos_proj)),
                        "std_p": float(max(np.std(pos_proj), 1e-6)),
                        "probe_acc": acc,
                        "lda_fdr": fdr,
                    }
                )
                cluster_probe.append(acc)
                cluster_fdr.append(fdr)
                cluster_weights.append(len(idx))

            units.append(
                {
                    "id": f"Attn_L{layer_idx}_H{head_idx}",
                    "layer": int(layer_idx),
                    "head": int(head_idx),
                    "type": "attention",
                    "probe_acc": float(np.average(cluster_probe, weights=cluster_weights)),
                    "lda_fdr": float(np.average(cluster_fdr, weights=cluster_weights)),
                    "clusters": clusters,
                }
            )

    units_by_probe = sorted(units, key=lambda x: x["probe_acc"], reverse=True)
    cfg = {
        "format": "dynasteer-reconstruction-v1",
        "model": args.model,
        "n_clusters_requested": args.n_clusters,
        "top_heads_default": args.top_heads,
        "attention": units,
        "selected_heads": [
            {"layer": u["layer"], "head": u["head"], "probe_acc": u["probe_acc"]}
            for u in units_by_probe[: args.top_heads]
        ],
        "notes": {
            "ranking": "top heads selected by probe accuracy, following the paper text",
            "cluster_scope": "per-head difference-vector KMeans, compatible with the released offline implementation",
        },
    }
    save_pickle(cfg, args.output)
    Path(args.output + ".summary.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "groups": len(groups),
                "units": len(units),
                "top_heads": cfg["selected_heads"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved steering config to {args.output}")


if __name__ == "__main__":
    main()
