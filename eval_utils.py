"""
eval_utils.py - Shared evaluation module for Cross-Encoder Acronym WSD.

Single source of truth: used by both train_acronym.py and evaluate_acronym.py
to ensure consistent evaluation logic.
"""

import torch
from typing import Any, Dict, List, Optional, Set, Tuple


@torch.no_grad()
def score_candidates(
    model, tokenizer, marked_text: str, candidates: List[str],
    device: torch.device, max_length: int = 128,
) -> List[float]:
    """Score all candidates for a single marked context. Returns list of logit scores."""
    if not candidates:
        return []
    if len(candidates) == 1:
        return [1.0]

    encodings = tokenizer(
        [marked_text] * len(candidates),
        candidates,
        max_length=max_length,
        padding=True,
        truncation="only_first",
        return_tensors="pt",
    ).to(device)

    logits = model(**encodings).logits.squeeze(-1)
    scores = logits.cpu().tolist()
    if isinstance(scores, float):
        scores = [scores]
    return scores


def evaluate_cross_encoder(
    model,
    tokenizer,
    acronym_dict: Dict[str, List[str]],
    samples: List[Dict[str, Any]],
    device: torch.device,
    train_acronyms: Optional[Set[str]] = None,
    max_length: int = 128,
    name: str = "",
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    Evaluate Cross-Encoder on a list of raw JSON samples.

    Args:
        model: HuggingFace model with .logits output.
        tokenizer: Tokenizer with <e>/<e> special tokens.
        acronym_dict: {acronym: [expansion1, expansion2, ...]}.
        samples: Raw samples [{text, start_char_idx, length_acronym, expansion}, ...].
        device: torch device.
        train_acronyms: Set of acronyms seen during training (for seen/unseen split).
        max_length: Max sequence length.
        name: Prefix for metric keys (e.g. "dev" → "dev_accuracy").

    Returns:
        (predictions_dict, metrics_dict)
    """
    model.eval()
    prefix = f"{name}_" if name else ""

    predictions: Dict[str, str] = {}
    total = correct = 0
    seen_total = seen_correct = 0
    unseen_total = unseen_correct = 0
    mrr_sum = 0.0

    for idx, sample in enumerate(samples):
        text = sample["text"]
        start = sample["start_char_idx"]
        length = sample["length_acronym"]
        acronym = text[start: start + length]
        gold = sample["expansion"]

        # Entity marking
        marked_text = text[:start] + "<e>" + acronym + "</e>" + text[start + length:]
        candidates = acronym_dict.get(acronym, [])

        # Predict
        if not candidates:
            pred = acronym
        elif len(candidates) == 1:
            pred = candidates[0]
            # MRR for single candidate
            if pred == gold:
                mrr_sum += 1.0
        else:
            scores = score_candidates(model, tokenizer, marked_text, candidates, device, max_length)
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            pred = candidates[ranked[0]]

            # MRR
            for rank, i in enumerate(ranked, 1):
                if candidates[i] == gold:
                    mrr_sum += 1.0 / rank
                    break

        predictions[str(idx)] = pred
        total += 1
        if pred == gold:
            correct += 1

        # Seen/unseen
        if train_acronyms is not None:
            if acronym in train_acronyms:
                seen_total += 1
                if pred == gold:
                    seen_correct += 1
            else:
                unseen_total += 1
                if pred == gold:
                    unseen_correct += 1

    metrics = {
        f"{prefix}accuracy": correct / max(total, 1) * 100,
        f"{prefix}mrr": mrr_sum / max(total, 1),
        f"{prefix}total": total,
        f"{prefix}seen_acc": seen_correct / max(seen_total, 1) * 100,
        f"{prefix}unseen_acc": unseen_correct / max(unseen_total, 1) * 100,
        f"{prefix}seen_total": seen_total,
        f"{prefix}unseen_total": unseen_total,
    }
    return predictions, metrics
