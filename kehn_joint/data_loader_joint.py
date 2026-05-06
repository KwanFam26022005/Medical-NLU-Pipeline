"""
data_loader_joint.py — Dataset và DataLoader cho KEHN joint training.

Xử lý:
- Tokenize bằng PhoBERT/ViHealthBERT (word-level)
- Align NER BIO tags với sub-word tokens (first sub-word gets label)
- Trả về: input_ids, attention_mask, topic_label, intent_label, ner_labels
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from .config_joint import NER2ID, JOINT_DATA_DIR


class JointDataset(Dataset):
    """
    Dataset cho KEHN: mỗi sample chứa text + topic + intent + NER labels.
    
    Tokenization strategy (giống NERDataLoader Trạm 2A):
    - Tokenize từng word riêng → map NER label cho sub-token đầu tiên
    - Sub-tokens sau: ner_label = -100 (ignored by CRF and CE)
    """

    def __init__(
        self,
        data_path: str,
        tokenizer_name: str,
        max_seq_len: int = 128,
    ):
        self.max_seq_len = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Load data
        if data_path.endswith('.jsonl'):
            self.raw_data = []
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.raw_data.append(json.loads(line))
        else:
            with open(data_path, "r", encoding="utf-8") as f:
                self.raw_data = json.load(f)

        print(f"[JointDataset] Loaded {len(self.raw_data)} samples from {Path(data_path).name}")

    def __len__(self) -> int:
        return len(self.raw_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.raw_data[idx]
        words = item["words"]
        ner_tags = item["ner_tags"]
        token_intent_ids = item.get("token_intent_ids", [item.get("intent_label_id", 0)] * len(words))

        # Tokenize word-by-word (giống Trạm 2A bypass word_ids)
        input_ids = [self.tokenizer.cls_token_id]
        ner_label_ids = [-100]  # CLS token
        intent_label_ids = [-100]

        for word, ner_tag, intent_tag in zip(words, ner_tags, token_intent_ids):
            word_tokens = self.tokenizer.tokenize(word)
            if not word_tokens:
                continue
            w_ids = self.tokenizer.convert_tokens_to_ids(word_tokens)
            input_ids.extend(w_ids)

            # First sub-token gets the real label, rest = -100
            ner_label_ids.append(NER2ID.get(ner_tag, 0))
            ner_label_ids.extend([-100] * (len(w_ids) - 1))
            
            intent_label_ids.append(intent_tag)
            intent_label_ids.extend([-100] * (len(w_ids) - 1))

        # Add SEP token
        input_ids.append(self.tokenizer.sep_token_id)
        ner_label_ids.append(-100)
        intent_label_ids.append(-100)

        # Truncation
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[: self.max_seq_len - 1] + [self.tokenizer.sep_token_id]
            ner_label_ids = ner_label_ids[: self.max_seq_len - 1] + [-100]
            intent_label_ids = intent_label_ids[: self.max_seq_len - 1] + [-100]

        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "ner_labels": torch.tensor(ner_label_ids, dtype=torch.long),
            "topic_label": torch.tensor(item["topic_label_id"], dtype=torch.long),
            "intent_label": torch.tensor(item["intent_label_id"], dtype=torch.long),
            "token_intent_ids": torch.tensor(intent_label_ids, dtype=torch.long),
        }


class JointCollator:
    """
    Dynamic padding collator cho JointDataset.
    Pad tất cả sequences trong batch đến max length trong batch đó.
    """

    def __init__(self, pad_token_id: int = 1, ner_pad_id: int = -100):
        self.pad_token_id = pad_token_id
        self.ner_pad_id = ner_pad_id

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(item["input_ids"].size(0) for item in batch)

        input_ids_list = []
        attention_mask_list = []
        ner_labels_list = []
        topic_labels = []
        intent_labels = []
        token_intent_list = []

        for item in batch:
            seq_len = item["input_ids"].size(0)
            pad_len = max_len - seq_len

            input_ids_list.append(
                torch.cat([item["input_ids"], torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
            )
            attention_mask_list.append(
                torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
            )
            ner_labels_list.append(
                torch.cat([item["ner_labels"], torch.full((pad_len,), self.ner_pad_id, dtype=torch.long)])
            )
            topic_labels.append(item["topic_label"])
            intent_labels.append(item["intent_label"])
            if "token_intent_ids" in item:
                token_intent_list.append(
                    torch.cat([item["token_intent_ids"], torch.full((pad_len,), self.ner_pad_id, dtype=torch.long)])
                )

        result = {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_mask_list),
            "ner_labels": torch.stack(ner_labels_list),
            "topic_labels": torch.stack(topic_labels),
            "intent_labels": torch.stack(intent_labels),
        }
        if token_intent_list:
            result["token_intent_ids"] = torch.stack(token_intent_list)
        return result


def create_dataloaders(
    tokenizer_name: str,
    batch_size: int = 32,
    max_seq_len: int = 128,
    data_dir: str = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Tạo train/val/test DataLoaders."""
    data_dir = Path(data_dir) if data_dir else JOINT_DATA_DIR

    def get_path(split_name):
        # Ưu tiên .jsonl trong splits/, fallback về .json cũ
        if (data_dir / f"{split_name}.jsonl").exists():
            return str(data_dir / f"{split_name}.jsonl")
        elif (data_dir / f"joint_{split_name}.json").exists():
            return str(data_dir / f"joint_{split_name}.json")
        return str(data_dir / f"{split_name}.json")

    train_ds = JointDataset(get_path("train"), tokenizer_name, max_seq_len)
    val_ds = JointDataset(get_path("val"), tokenizer_name, max_seq_len)
    test_ds = JointDataset(get_path("test"), tokenizer_name, max_seq_len)

    # Get pad token id from tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    pad_id = tokenizer.pad_token_id or 1
    collator = JointCollator(pad_token_id=pad_id)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collator, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        collate_fn=collator, num_workers=0, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size * 2, shuffle=False,
        collate_fn=collator, num_workers=0, pin_memory=True,
    )

    return train_loader, val_loader, test_loader
