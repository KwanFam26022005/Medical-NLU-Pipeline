"""
Helper module: loss weight lookup cho curriculum phases.
Tách riêng để tránh circular import giữa kehn_model.py và config_joint.py.
"""

from .config_joint import TRAIN_CONFIG


def get_loss_weights(phase: str) -> dict:
    """Trả về dict {"topic": float, "intent": float, "ner": float} cho mỗi phase."""
    return TRAIN_CONFIG["loss_weights"].get(phase, TRAIN_CONFIG["loss_weights"]["full"])
