import contextlib
import json
import math
import random
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MATH_INSTRUCTION = r"""Solve the following math problem step by step. The last line of your response should be of the form Answer: \\boxed{{$Answer}} where $Answer is the answer to the problem.

{problem}

Remember to put your answer on its own line after "Answer:"."""


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_dtype(name: str):
    name = name.lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def load_model_and_tokenizer(model_name: str, dtype: str = "bf16"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=resolve_dtype(dtype),
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def model_input_device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


def render_math_prompt(tokenizer, problem: str, enable_thinking: bool = False) -> str:
    user_text = MATH_INSTRUCTION.format(problem=problem)
    messages = [{"role": "user", "content": user_text}]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if "qwen" in tokenizer.__class__.__name__.lower() or "qwen" in str(getattr(tokenizer, "name_or_path", "")).lower():
        kwargs["enable_thinking"] = enable_thinking
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def encode_text(tokenizer, text: str, device: torch.device) -> torch.Tensor:
    return tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)


def full_vocab_entropy(logits: torch.Tensor) -> torch.Tensor:
    logits = logits.to(torch.float32)
    log_p = torch.log_softmax(logits, dim=-1)
    p = log_p.exp()
    return -(p * log_p).sum(dim=-1)


def topk_renorm_entropy(logits: torch.Tensor, k: int = 20) -> torch.Tensor:
    logits = logits.to(torch.float32)
    k = min(k, logits.shape[-1])
    vals, _ = torch.topk(logits, k=k, dim=-1)
    log_p = torch.log_softmax(vals, dim=-1)
    p = log_p.exp()
    return -(p * log_p).sum(dim=-1)


def entropy_from_logits(logits: torch.Tensor, mode: str = "full_vocab", topk: int = 20) -> float:
    if mode == "full_vocab":
        return float(full_vocab_entropy(logits).item())
    if mode == "topk_renorm":
        return float(topk_renorm_entropy(logits, topk).item())
    raise ValueError(f"Unknown entropy mode: {mode}")


def _sample_token(logits: torch.Tensor, temperature: float, top_p: float, top_k: int) -> int:
    logits = logits.to(torch.float32).clone()
    if temperature <= 0:
        return int(torch.argmax(logits, dim=-1).item())

    logits = logits / temperature
    if top_k and top_k > 0 and top_k < logits.shape[-1]:
        threshold = torch.topk(logits, k=top_k, dim=-1).values[..., -1, None]
        logits = torch.where(logits < threshold, torch.full_like(logits, -float("inf")), logits)

    if 0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(sorted_probs, dim=-1)
        remove = cum > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
        filtered = torch.full_like(logits, -float("inf"))
        filtered.scatter_(-1, sorted_idx, sorted_logits)
        logits = filtered

    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def _has_sentence_boundary(text: str) -> bool:
    return ".\n" in text or ".\r\n" in text


@torch.inference_mode()
def generate_continuation(
    model,
    tokenizer,
    prefix_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    entropy_mode: str = "full_vocab",
    entropy_topk: int = 20,
    stop_at_sentence: bool = False,
    hook_context=None,
) -> Dict:
    if prefix_ids.ndim != 2 or prefix_ids.shape[0] != 1:
        raise ValueError("This reproduction implementation currently expects batch size 1.")

    ctx = hook_context if hook_context is not None else contextlib.nullcontext()
    with ctx:
        out = model(input_ids=prefix_ids, use_cache=True)
    logits = out.logits[:, -1, :]
    past = out.past_key_values

    new_ids: List[int] = []
    entropies: List[float] = []
    eos_id = tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        entropies.append(entropy_from_logits(logits[0], entropy_mode, entropy_topk))
        token_id = _sample_token(logits[0], temperature, top_p, top_k)
        new_ids.append(token_id)

        if eos_id is not None and token_id == eos_id:
            break
        if stop_at_sentence:
            text_so_far = tokenizer.decode(new_ids, skip_special_tokens=False)
            if _has_sentence_boundary(text_so_far):
                break

        token = torch.tensor([[token_id]], device=prefix_ids.device, dtype=prefix_ids.dtype)
        out = model(input_ids=token, past_key_values=past, use_cache=True)
        logits = out.logits[:, -1, :]
        past = out.past_key_values

    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return {
        "token_ids": new_ids,
        "text": text,
        "entropies": entropies,
        "avg_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "hit_eos": bool(new_ids and eos_id is not None and new_ids[-1] == eos_id),
    }


def split_into_sentence_chunks(tokenizer, token_ids: Sequence[int], entropies: Sequence[float]) -> List[Dict]:
    chunks: List[Dict] = []
    start = 0
    for i in range(len(token_ids)):
        text = tokenizer.decode(token_ids[start : i + 1], skip_special_tokens=True)
        if _has_sentence_boundary(text):
            vals = list(entropies[start : i + 1])
            chunks.append(
                {
                    "start": start,
                    "end": i + 1,
                    "text": text,
                    "avg_entropy": float(np.mean(vals)) if vals else 0.0,
                }
            )
            start = i + 1
    if start < len(token_ids):
        text = tokenizer.decode(token_ids[start:], skip_special_tokens=True)
        vals = list(entropies[start:])
        chunks.append(
            {
                "start": start,
                "end": len(token_ids),
                "text": text,
                "avg_entropy": float(np.mean(vals)) if vals else 0.0,
            }
        )
    return chunks


def _extract_boxed(text: str) -> Optional[str]:
    matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if matches:
        return matches[-1].strip()
    matches = re.findall(r"Answer:\s*([^\n]+)", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].strip().strip("$ ")
    return None


@lru_cache(maxsize=1)
def _math_verify_func():
    try:
        from math_verify.metric import math_metric
        from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

        return math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        )
    except Exception:
        return None


def compute_math_score(model_output: str, ground_truth: str) -> float:
    verify = _math_verify_func()
    if verify is not None:
        try:
            score, _ = verify([f"\\boxed{{{ground_truth}}}"], [model_output])
            return float(score)
        except Exception:
            pass
    pred = _extract_boxed(model_output)
    if pred is None:
        return 0.0
    normalize = lambda x: re.sub(r"\s+", "", str(x)).strip("$.")
    return float(normalize(pred) == normalize(ground_truth))


def _extract_math_answer(example: Dict) -> str:
    for key in ("answer", "final_answer", "target"):
        if key in example and example[key] is not None:
            return str(example[key]).strip()
    solution = str(example.get("solution", ""))
    boxed = _extract_boxed(solution)
    if boxed is not None:
        return boxed
    raise KeyError(f"Could not infer answer field from keys: {list(example.keys())}")


def load_problem_dataset(name: str, split: Optional[str] = None, jsonl_path: Optional[str] = None) -> List[Dict]:
    if name == "gsm8k":
        split = split or "train"
        ds = load_dataset("openai/gsm8k", "main", split=split)
        return [
            {
                "problem": ex["question"],
                "answer": str(ex["answer"]).split("####")[-1].strip(),
                "source": "gsm8k",
            }
            for ex in ds
        ]
    if name == "math":
        split = split or "train"
        ds = load_dataset("nlile/hendrycks-MATH-benchmark", split=split)
        return [
            {"problem": ex["problem"], "answer": _extract_math_answer(dict(ex)), "source": "math"}
            for ex in ds
        ]
    if name == "math500":
        split = split or "test"
        ds = load_dataset("HuggingFaceH4/MATH-500", split=split)
        return [
            {"problem": ex["problem"], "answer": _extract_math_answer(dict(ex)), "source": "math500"}
            for ex in ds
        ]
    if name == "jsonl":
        if not jsonl_path:
            raise ValueError("--jsonl_path is required for dataset=jsonl")
        rows = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    ex = json.loads(line)
                    rows.append(
                        {
                            "problem": ex["problem"],
                            "answer": str(ex["answer"]),
                            "source": ex.get("source", "jsonl"),
                        }
                    )
        return rows
    raise ValueError(f"Unsupported dataset: {name}")


class SteeringInjector:
    def __init__(self, steering_cfg: Dict, alpha: float = 3.0, top_heads: int = 32):
        units = list(steering_cfg.get("attention", []))
        units = sorted(units, key=lambda x: x.get("probe_acc", 0.0), reverse=True)
        if top_heads > 0:
            units = units[:top_heads]
        self.units = units
        self.alpha = float(alpha)
        self.by_layer: Dict[int, List[Dict]] = {}
        for unit in units:
            self.by_layer.setdefault(int(unit["layer"]), []).append(unit)

    @contextlib.contextmanager
    def hooks(self, model):
        handles = []
        for layer_idx, units in self.by_layer.items():
            layer = model.model.layers[layer_idx]
            module = layer.self_attn.o_proj

            def make_hook(layer_units):
                def hook(_module, inputs):
                    x = inputs[0]
                    y = x.clone()
                    for unit in layer_units:
                        head = int(unit["head"])
                        clusters = unit.get("clusters", [])
                        if not clusters:
                            continue
                        head_dim = int(clusters[0].get("head_dim", len(clusters[0]["w_lda"])))
                        start, end = head * head_dim, (head + 1) * head_dim
                        h = y[:, -1, start:end]
                        h32 = h.to(torch.float32)
                        delta = torch.zeros_like(h32)
                        for cluster in clusters:
                            w = torch.as_tensor(cluster["w_lda"], device=h.device, dtype=torch.float32)
                            mu_p = float(cluster["mu_p"])
                            std_p = max(float(cluster["std_p"]), 1e-6)
                            proj = h32 @ w
                            z = torch.relu((mu_p - proj) / std_p)
                            eta = self.alpha * torch.tanh(z)
                            delta = delta + eta.unsqueeze(-1) * w.unsqueeze(0)
                        y[:, -1, start:end] = (h32 + delta).to(y.dtype)
                    return (y,) + tuple(inputs[1:])

                return hook

            handles.append(module.register_forward_pre_hook(make_hook(units)))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()


def load_pickle(path: str):
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, path: str) -> None:
    import pickle
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
