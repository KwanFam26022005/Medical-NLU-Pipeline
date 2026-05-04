"""
curriculum.py — Curriculum Learning Scheduler (từ OneNet).

4 giai đoạn huấn luyện:
  Phase 1 (epoch 1-3):   topic_only — Chỉ train Topic branch
  Phase 2 (epoch 4-6):   mining_only — Chỉ train Intent + NER  
  Phase 3 (epoch 7-10):  joint_no_prop — Train tất cả, chưa Stack-Propagation
  Phase 4 (epoch 11-30): full — Full Stack-Propagation + joint loss
"""

from .config_joint import TRAIN_CONFIG


class CurriculumScheduler:
    """
    Quản lý phase hiện tại dựa trên epoch number.
    
    Cung cấp:
    - Phase name cho model.forward()
    - Loss weights cho loss computation
    - Freeze/unfreeze instructions
    """

    def __init__(self):
        self.phases = TRAIN_CONFIG["loss_weights"]
        self.phase_ranges = {
            "topic_only": TRAIN_CONFIG["phase_topic_only"],
            "mining_only": TRAIN_CONFIG["phase_mining_only"],
            "joint_no_prop": TRAIN_CONFIG["phase_joint_no_prop"],
            "full": TRAIN_CONFIG["phase_full"],
        }

    def get_phase(self, epoch: int) -> str:
        """Trả về tên phase dựa trên epoch (1-indexed)."""
        for phase_name, (start, end) in self.phase_ranges.items():
            if start <= epoch <= end:
                return phase_name
        return "full"

    def get_loss_weights(self, epoch: int) -> dict:
        """Trả về loss weights cho epoch hiện tại."""
        phase = self.get_phase(epoch)
        return self.phases[phase]

    def apply_freeze(self, model, epoch: int):
        """
        Freeze/unfreeze modules theo curriculum phase.
        
        Phase 1 (topic_only): Freeze co_blocks, intent_fc, ner_fc, crf
        Phase 2 (mining_only): Freeze topic_classifier
        Phase 3+4: Unfreeze tất cả
        """
        phase = self.get_phase(epoch)

        # Unfreeze everything first
        for param in model.parameters():
            param.requires_grad = True

        if phase == "topic_only":
            # Freeze Feature Mining layer
            for module in [model.intent_fc, model.ner_fc, model.crf]:
                for param in module.parameters():
                    param.requires_grad = False
            for block in model.co_blocks:
                for param in block.parameters():
                    param.requires_grad = False
            # Label attention dùng weight của intent_fc/ner_fc → đã freeze

        elif phase == "mining_only":
            # Freeze Topic Decoder
            for param in model.topic_classifier.parameters():
                param.requires_grad = False

        # Phase 3 & 4: tất cả đều trainable

    def get_phase_info(self, epoch: int) -> str:
        """Pretty-print thông tin phase cho logging."""
        phase = self.get_phase(epoch)
        weights = self.get_loss_weights(epoch)
        return (
            f"Phase: {phase} | "
            f"L_topic={weights['topic']:.1f} "
            f"L_intent={weights['intent']:.1f} "
            f"L_ner={weights['ner']:.1f}"
        )
