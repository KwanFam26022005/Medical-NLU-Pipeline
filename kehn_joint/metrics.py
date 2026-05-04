"""
metrics.py — Metrics cho KEHN: Topic, Intent, NER, và Semantic Accuracy.
"""

import numpy as np
from collections import Counter
from typing import Dict, List, Tuple


def compute_topic_metrics(preds: np.ndarray, labels: np.ndarray, label_names: list = None) -> dict:
    """
    Metrics cho Topic Classification (single-label, multi-class).
    
    Returns:
        dict with accuracy, macro_f1, weighted_f1, per_class_f1
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        classification_report,
    )
    
    accuracy = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    macro_precision = precision_score(labels, preds, average="macro", zero_division=0)
    macro_recall = recall_score(labels, preds, average="macro", zero_division=0)

    result = {
        "topic_accuracy": accuracy,
        "topic_macro_f1": macro_f1,
        "topic_weighted_f1": weighted_f1,
        "topic_precision": macro_precision,
        "topic_recall": macro_recall,
    }

    # Per-class F1 nếu có label names
    if label_names:
        per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)
        for i, name in enumerate(label_names):
            if i < len(per_class_f1):
                result[f"topic_f1_{name}"] = per_class_f1[i]

    return result


def compute_intent_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    """Metrics cho Intent Detection (single-label)."""
    from sklearn.metrics import accuracy_score, f1_score

    return {
        "intent_accuracy": accuracy_score(labels, preds),
        "intent_macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
    }


def compute_ner_metrics(
    pred_tags: List[List[str]],
    true_tags: List[List[str]],
) -> dict:
    """
    Metrics cho NER (entity-level F1 bằng seqeval).
    
    Args:
        pred_tags: List of predicted BIO tag sequences (strings)
        true_tags: List of ground truth BIO tag sequences (strings)
    """
    from seqeval.metrics import (
        f1_score as seq_f1,
        precision_score as seq_precision,
        recall_score as seq_recall,
    )

    return {
        "ner_f1": seq_f1(true_tags, pred_tags, zero_division=0),
        "ner_precision": seq_precision(true_tags, pred_tags, zero_division=0),
        "ner_recall": seq_recall(true_tags, pred_tags, zero_division=0),
    }


def compute_semantic_accuracy(
    topic_preds: np.ndarray,
    topic_labels: np.ndarray,
    intent_preds: np.ndarray,
    intent_labels: np.ndarray,
    ner_pred_tags: List[List[str]],
    ner_true_tags: List[List[str]],
) -> dict:
    """
    Semantic Accuracy (từ DCA-Net): % mẫu đúng CẢ 3 tasks đồng thời.
    
    Semantic_Acc = Σ[topic_ok(i) AND intent_ok(i) AND ner_ok(i)] / N
    """
    n = len(topic_preds)
    correct = 0

    for i in range(n):
        topic_ok = (topic_preds[i] == topic_labels[i])
        intent_ok = (intent_preds[i] == intent_labels[i])
        ner_ok = (ner_pred_tags[i] == ner_true_tags[i])
        if topic_ok and intent_ok and ner_ok:
            correct += 1

    return {
        "semantic_accuracy": correct / max(n, 1),
        "semantic_correct": correct,
        "semantic_total": n,
    }


def compute_all_metrics(
    topic_preds, topic_labels,
    intent_preds, intent_labels,
    ner_pred_tags, ner_true_tags,
    topic_label_names=None,
) -> dict:
    """Compute tất cả metrics trong 1 lần gọi."""
    result = {}
    result.update(compute_topic_metrics(topic_preds, topic_labels, topic_label_names))
    result.update(compute_intent_metrics(intent_preds, intent_labels))
    result.update(compute_ner_metrics(ner_pred_tags, ner_true_tags))
    result.update(compute_semantic_accuracy(
        topic_preds, topic_labels,
        intent_preds, intent_labels,
        ner_pred_tags, ner_true_tags,
    ))
    return result
