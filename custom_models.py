import torch
import torch.nn as nn
from transformers import AutoModel
from torchcrf import CRF

class ViHealthBertCRF(nn.Module):
    """
    Kiến trúc SOTA cho Medical NER: ViHealthBERT kết hợp Conditional Random Fields (CRF)
    Giúp giải quyết triệt để lỗi Span-Noise và các ranh giới thực thể vô lý.
    """
    def __init__(self, model_name="demdecuong/vihealthbert-base-word", num_labels=7):
        super(ViHealthBertCRF, self).__init__()
        self.num_labels = num_labels
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_tags=num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        emissions = self.classifier(sequence_output) 
        mask = attention_mask.type(torch.uint8) if attention_mask is not None else None

        if labels is not None:
            labels_crf = labels.clone()
            # Bỏ qua các token padding (-100) của HuggingFace bằng cách set về 0 (đã có mask lo việc chặn loss)
            labels_crf[labels_crf == -100] = 0 
            loss = -self.crf(emissions, tags=labels_crf, mask=mask, reduction='mean')
            return (loss, emissions)
        else:
            return self.crf.decode(emissions, mask=mask)