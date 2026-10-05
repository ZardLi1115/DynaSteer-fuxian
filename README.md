# DynaSteer reproduction completion

This repository is an **independent reproduction / completion** of the missing parts of the public DynaSteer code for the paper **“Search for Truth from Reasoning: A Dynamic Representation Editing Framework for Steering LLM Trajectories” (ICML 2026)**.

The authors' public repository currently contains the offline pipeline through steering-vector solving, but not the online rollback + activation-steering evaluation code used for the reported benchmark table. This repository implements that missing path from the paper's equations and Algorithm 1, and also provides a self-contained offline pipeline that does not depend on the authors' custom Hugging Face model classes.

> Important: this is not the authors' official online evaluator. The exact reported numbers are not guaranteed because several details are still unpublished, especially the exact temporal-decay threshold `tau_decay`, the final online KV-cache/intervention implementation, and parts of the offline rollout protocol. The upstream reproduction issue is still open: https://github.com/tianlwang/DynaSteer/issues/1

## What is implemented

- sentence-level reasoning chunks
- full-vocabulary token entropy, plus an optional top-20-renormalized entropy mode
- global P80 training-entropy threshold
- high-entropy fork sampling
- consistency-based Truth / Fallacy labeling
- attention-head activation extraction from the input of each attention `o_proj`
- per-head difference-vector KMeans clustering
- Fisher-LDA steering direction with Ledoit-Wolf covariance shrinkage
- top-head selection by probe accuracy
- adaptive intervention strength from Eq. 9
- sentence-level online entropy monitoring
- rollback of the unsteered candidate sentence
- one-shot activation injection at the fork state, followed by regeneration
- optional temporal gating `t <= tau_decay`
- GSM8K and MATH-500 evaluation
- plain-generation baseline in the same evaluator

The implementation intentionally uses standard `transformers` hooks instead of copying the authors' custom model files. For Qwen3 and Llama-family causal LMs, the attention intervention is applied to the concatenated per-head attention output immediately before `o_proj`, which is the same representation location used by the released activation collector.

## 5060 Ti 16 GB recommendation

Use **Qwen3-1.7B in BF16** first. This is the practical target for an RTX 5060 Ti 16 GB. Start with 50 to 200 training problems per dataset to validate the pipeline, then increase the data size. The expensive part is repeated rollout generation, not Fisher-LDA.

Install a recent CUDA-enabled PyTorch build that supports your RTX 50-series GPU and driver, then:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

On WSL2/Linux, it can also help to set:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## 1. Build the contrastive sentence-fork dataset

A consumer-GPU smoke test:

```bash
python prepare_offline.py \
  --model Qwen/Qwen3-1.7B \
  --datasets gsm8k,math \
  --output_dir runs/qwen3-1.7b \
  --limit_per_dataset 100 \
  --n_forks 10 \
  --label_rollouts 3 \
  --max_baseline_tokens 1024 \
  --max_completion_tokens 1024
```

Outputs:

- `baseline_chunks.jsonl`: baseline trajectories and sentence entropy
- `entropy_stats.json`: global entropy distribution and P80 threshold `gamma`
- `forks.jsonl`: contexts that contain both unanimous Truth and unanimous Fallacy forks

### Paper protocol vs current public-code behavior

The paper describes consistency labeling using `N=10` stochastic rollouts. The current public repository instead samples 10 candidate sentence forks and then uses **3 completions per fork** for the unanimous label. Because the authors have not yet clarified which exact protocol produced Table 1, this reproduction exposes the count explicitly:

- paper-oriented run: `--label_rollouts 10`
- closer to the currently released code: `--label_rollouts 3`

For a first 5060 Ti run, use 3. For a paper-faithful attempt, use 10.

## 2. Collect activations and solve Fisher-LDA steering vectors

```bash
python build_steering.py \
  --model Qwen/Qwen3-1.7B \
  --forks runs/qwen3-1.7b/forks.jsonl \
  --output runs/qwen3-1.7b/steering.pkl \
  --n_clusters 5 \
  --top_heads 32
```

The paper's hyper-parameter study reports its best setting around:

- `M = 5` clusters
- `alpha = 3.0` already reaching the reported plateau
- 32 selected attention heads

This script ranks heads by **probe accuracy**, matching the paper text. The released `solve_steering.py` sorts its saved units by LDA FDR, which is one of the unresolved implementation differences noted in the upstream issue.

## 3. Run a plain MATH-500 baseline

```bash
python evaluate_dynasteer.py \
  --model Qwen/Qwen3-1.7B \
  --dataset math500 \
  --mode plain \
  --steering runs/qwen3-1.7b/steering.pkl \
  --output outputs/math500_plain.jsonl \
  --limit 50
```

The `--steering` file is still required by the CLI so that the exact same environment is used for both modes, though plain mode does not inject activations.

## 4. Run DynaSteer with entropy gating and rollback

```bash
python evaluate_dynasteer.py \
  --model Qwen/Qwen3-1.7B \
  --dataset math500 \
  --mode dynasteer \
  --steering runs/qwen3-1.7b/steering.pkl \
  --entropy_stats runs/qwen3-1.7b/entropy_stats.json \
  --alpha 3.0 \
  --top_heads 32 \
  --tau_decay_sentences -1 \
  --output outputs/math500_dynasteer.jsonl \
  --limit 50
```

`--tau_decay_sentences -1` means **no temporal decay**, which is useful because the paper reports that ablation separately and the exact full-method `tau_decay` value is not disclosed in the current paper/code release. The temporal gate itself is fully implemented. Once a verified value is known, use for example:

```bash
--tau_decay_sentences 6
```

The number above is only an example of the CLI shape, not a claimed paper hyper-parameter.

## Entropy modes

The paper defines Shannon entropy over the next-token vocabulary. Therefore the default is:

```bash
--entropy_mode full_vocab
```

The currently released sampling code requests top-20 vLLM logprobs and renormalizes those values. To test that implementation variant, use:

```bash
--entropy_mode topk_renorm --entropy_topk 20
```

Use the same entropy mode for offline threshold construction and online evaluation.

## Ablations

Disable entropy gating and steer every sentence:

```bash
python evaluate_dynasteer.py ... --no_entropy_gating
```

Disable temporal decay:

```bash
python evaluate_dynasteer.py ... --tau_decay_sentences -1
```

Reduce intervention strength:

```bash
python evaluate_dynasteer.py ... --alpha 1.0
```

Change the number of heads:

```bash
python evaluate_dynasteer.py ... --top_heads 16
```

## Files

- `dynasteer_core.py`: generation, entropy, scoring, dataset adapters, and online activation injector
- `prepare_offline.py`: baseline sentence entropy and contrastive fork construction
- `build_steering.py`: attention activation collection, KMeans, probes, and Fisher-LDA
- `evaluate_dynasteer.py`: Algorithm-1-style online candidate generation, entropy test, rollback, steering, and regeneration
- `requirements.txt`: Python dependencies other than the CUDA-specific PyTorch wheel

## Reproduction notes

This code follows the paper where the paper is explicit and exposes flags where the public material is ambiguous. In particular:

1. The entropy threshold is computed **globally** over the training sentence distribution by default, because the methodology states that `gamma` is the 80th percentile of the training entropy distribution.
2. The online candidate sentence is generated without intervention. If its mean entropy exceeds `gamma`, the candidate is discarded and the same prefix is regenerated with a one-shot activation edit.
3. The edit for each selected head sums the Fisher-LDA cluster directions using the paper's adaptive strength formula `alpha * tanh(ReLU((mu_p - h_proj) / sigma_p))`.
4. The exact upstream `tau_decay`, KV-cache semantics, and final benchmark evaluator were not public when this repository was written. Those points are deliberately not guessed in the defaults.

## Upstream references

Paper/code authors' repository: https://github.com/tianlwang/DynaSteer

Open reproduction question about the missing online code: https://github.com/tianlwang/DynaSteer/issues/1
