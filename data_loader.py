"""
data_loader.py - Module tải và tiền xử lý dữ liệu cho tất cả các bài toán.
Hỗ trợ: acrDrAid (WSD), ViMQ NER (BIO), ViMQ Intent, Topic CSV (pending).
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoTokenizer

from config import (
    ACRONYM_MODEL_NAME,
    DATA_DIR,
    INTENT_LABEL2ID,
    INTENT_NUM_LABELS,
    INTENT_MODEL_NAME,
    NER_LABEL2ID,
    NER_MODEL_NAME,
    TOPIC_CSV_FILES,
    TOPIC_DATASET_READY,
    TOPIC_MODEL_NAME,
)


# ============================================================
# 📍 TRẠM 1: ACRONYM DATA LOADER (acrDrAid — Cross-Encoder)
# Architecture: Binary scoring per (context, candidate) pair
# ============================================================

class AcronymDataset(TorchDataset):
    """
    Dataset cho Cross-Encoder Acronym Disambiguation.

    Mỗi sample gốc: {"text", "start_char_idx", "length_acronym", "expansion"}
    → Sinh N pairs: (marked_context [SEP] candidate_i) với label 1.0/0.0

    Entity Marking: Chèn <e>acronym</e> vào text để model biết vị trí cần disambiguate.
    """

    ENTITY_START = "<e>"
    ENTITY_END = "</e>"

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        acronym_dict: Dict[str, List[str]],
        tokenizer: AutoTokenizer,
        max_length: int = 128,
        mode: str = "train",
    ) -> None:
        """
        Args:
            samples: List of raw samples from data.json.
            acronym_dict: {acronym_str: [expansion1, expansion2, ...]}.
            tokenizer: HuggingFace tokenizer (must already have <e>, </e> tokens).
            max_length: Max sequence length for tokenizer.
            mode: "train" or "eval". Train returns flattened pairs; eval returns grouped.
        """
        self.samples = samples
        self.acronym_dict = acronym_dict
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode

        # Preprocess: extract acronym text and build marked context
        self._preprocess()

    def _preprocess(self) -> None:
        """Extract acronym string from each sample and validate against dictionary."""
        self.processed = []
        skipped = 0
        for sample in self.samples:
            text = sample["text"]
            start = sample["start_char_idx"]
            length = sample["length_acronym"]
            acronym = text[start: start + length]
            expansion = sample["expansion"]

            if acronym not in self.acronym_dict:
                skipped += 1
                continue

            # Entity marking: insert <e> and </e> around the acronym
            marked_text = (
                text[:start]
                + self.ENTITY_START + acronym + self.ENTITY_END
                + text[start + length:]
            )

            self.processed.append({
                "marked_text": marked_text,
                "acronym": acronym,
                "expansion": expansion,
            })

        if skipped > 0:
            print(f"  ⚠️ Skipped {skipped} samples (acronym not in dictionary)")

    def __len__(self) -> int:
        return len(self.processed)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.processed[idx]
        marked_text = item["marked_text"]
        acronym = item["acronym"]
        correct_expansion = item["expansion"]
        candidates = self.acronym_dict[acronym]

        if self.mode == "train":
            # Flatten: return one (context, candidate, label) pair per call
            # We sample: the positive + all negatives
            pairs = []
            for cand in candidates:
                label = 1.0 if cand == correct_expansion else 0.0
                enc = self.tokenizer(
                    marked_text,
                    cand,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation="only_first",
                    return_tensors="pt",
                )
                pairs.append({
                    "input_ids": enc["input_ids"].squeeze(0),
                    "attention_mask": enc["attention_mask"].squeeze(0),
                    "label": torch.tensor(label, dtype=torch.float),
                })
            return pairs  # List[Dict] — collate_fn will flatten

        else:
            # Eval mode: return all candidates grouped for ranking
            encodings = []
            for cand in candidates:
                enc = self.tokenizer(
                    marked_text,
                    cand,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation="only_first",
                    return_tensors="pt",
                )
                encodings.append({
                    "input_ids": enc["input_ids"].squeeze(0),
                    "attention_mask": enc["attention_mask"].squeeze(0),
                })
            return {
                "encodings": encodings,
                "candidates": candidates,
                "correct_expansion": correct_expansion,
                "acronym": acronym,
                "n_candidates": len(candidates),
            }


def acronym_train_collate_fn(
    batch: List[List[Dict[str, torch.Tensor]]],
) -> Dict[str, torch.Tensor]:
    """
    Collate function for train mode.
    Each item in batch is a List of (input_ids, attention_mask, label) dicts.
    Flatten all pairs into a single batch.
    """
    all_input_ids = []
    all_attention_mask = []
    all_labels = []

    for pairs in batch:
        for pair in pairs:
            all_input_ids.append(pair["input_ids"])
            all_attention_mask.append(pair["attention_mask"])
            all_labels.append(pair["label"])

    return {
        "input_ids": torch.stack(all_input_ids),
        "attention_mask": torch.stack(all_attention_mask),
        "labels": torch.stack(all_labels),
    }


def acronym_eval_collate_fn(
    batch: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Collate function for eval mode.
    Each item has variable number of candidates — keep as list, don't stack.
    """
    return batch  # Process one-by-one in eval loop


class AcronymDataLoader:
    """
    Tải và xử lý dataset acrDrAid cho Cross-Encoder Acronym Disambiguation.

    Data format: data/acrDrAid/{train,dev,test}/data.json
    Dictionary: data/acrDrAid/dictionary.json

    Cross-Encoder: Encode (marked_context, candidate) pairs → binary score.
    Tại inference, score tất cả candidates từ dictionary → argmax.
    """

    SPECIAL_TOKENS = ["<e>", "</e>"]

    def __init__(
        self,
        data_dir: str = "data/acrDrAid",
        tokenizer_name: str = ACRONYM_MODEL_NAME,
        max_length: int = 128,
    ) -> None:
        """
        Args:
            data_dir: Path to acrDrAid dataset root (contains train/, dev/, test/, dictionary.json).
            tokenizer_name: HuggingFace model name for tokenizer.
            max_length: Max sequence length.
        """
        self.data_dir = Path(data_dir)
        self.max_length = max_length

        # Load tokenizer and add special entity tokens
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        num_added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": self.SPECIAL_TOKENS}
        )
        print(f"[AcronymDataLoader] Added {num_added} special tokens: {self.SPECIAL_TOKENS}")

        # Load dictionary
        self.acronym_dict = self._load_dictionary()
        print(
            f"[AcronymDataLoader] Dictionary: {len(self.acronym_dict)} acronyms, "
            f"{sum(len(v) for v in self.acronym_dict.values())} expansions"
        )

    def _load_dictionary(self) -> Dict[str, List[str]]:
        """Load acronym dictionary from dictionary.json."""
        dict_path = self.data_dir / "dictionary.json"
        if not dict_path.exists():
            raise FileNotFoundError(f"Dictionary not found: {dict_path}")
        with open(dict_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_samples(self, split: str) -> List[Dict[str, Any]]:
        """Load raw samples from data.json for a given split."""
        data_path = self.data_dir / split / "data.json"
        if not data_path.exists():
            raise FileNotFoundError(f"Data not found: {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_datasets(
        self,
    ) -> Tuple[AcronymDataset, AcronymDataset, AcronymDataset]:
        """
        Build train, dev, test AcronymDataset objects.

        Returns:
            (train_dataset, dev_dataset, test_dataset)
        """
        train_samples = self._load_samples("train")
        dev_samples = self._load_samples("dev")
        test_samples = self._load_samples("test")

        print(f"\n📊 Dataset stats:")
        print(f"   Train: {len(train_samples)} samples")
        print(f"   Dev:   {len(dev_samples)} samples")
        print(f"   Test:  {len(test_samples)} samples")

        train_ds = AcronymDataset(
            train_samples, self.acronym_dict, self.tokenizer,
            max_length=self.max_length, mode="train",
        )
        dev_ds = AcronymDataset(
            dev_samples, self.acronym_dict, self.tokenizer,
            max_length=self.max_length, mode="eval",
        )
        test_ds = AcronymDataset(
            test_samples, self.acronym_dict, self.tokenizer,
            max_length=self.max_length, mode="eval",
        )

        # Compute seen/unseen stats
        train_acronyms = set(s["acronym"] for s in train_ds.processed)
        test_acronyms = set(s["acronym"] for s in test_ds.processed)
        unseen = test_acronyms - train_acronyms
        print(f"   Train acronyms: {len(train_acronyms)}")
        print(f"   Test acronyms:  {len(test_acronyms)}")
        print(f"   Unseen in test: {len(unseen)} ({len(unseen)/max(len(test_acronyms),1)*100:.1f}%)")

        return train_ds, dev_ds, test_ds

    def get_train_loader(
        self, batch_size: int = 8, shuffle: bool = True,
    ) -> "torch.utils.data.DataLoader":
        """Create DataLoader for training."""
        from torch.utils.data import DataLoader
        train_ds, _, _ = self.get_datasets()
        return DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=acronym_train_collate_fn,
            num_workers=0,
        )

    def save_dictionary(self, save_path: Path) -> None:
        """Save dictionary + tokenizer to model directory."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        with open(save_path / "acronym_dict.json", "w", encoding="utf-8") as f:
            json.dump(self.acronym_dict, f, ensure_ascii=False, indent=2)

        self.tokenizer.save_pretrained(save_path)
        print(f"  → Saved dictionary + tokenizer to {save_path}")


# ============================================================
# 📍 TRẠM 2A: NER DATA LOADER (ViMQ - CoNLL BIO Format)
# ============================================================

class NERDataLoader:
    """
    Tải dataset NER ở định dạng CoNLL (BIO tagging).
    Hỗ trợ nhãn: SYMPTOM_AND_DISEASE, MEDICAL_PROCEDURE, MEDICINE.
    
    ⚠️ QUAN TRỌNG: Dataset ViMQ NER đã được word-segmented sẵn.
    Khi tokenize bằng WordPiece, cần align lại nhãn cho sub-tokens.
    """

    def __init__(
        self,
        tokenizer_name: str = NER_MODEL_NAME,
        max_length: int = 256,
        label2id: Optional[Dict[str, int]] = None,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.label2id = label2id or NER_LABEL2ID

    def load_conll_file(self, file_path: Path) -> List[Dict[str, List[str]]]:
        """
        Đọc file CoNLL format.
        Mỗi câu là 1 block cách nhau bằng dòng trống.
        Mỗi dòng: <token>\t<label> hoặc <token> <label>
        """
        file_path = Path(file_path)
        sentences: List[Dict[str, List[str]]] = []
        current_tokens: List[str] = []
        current_labels: List[str] = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("-DOCSTART-"):
                    if current_tokens:
                        sentences.append({
                            "tokens": current_tokens,
                            "ner_tags": current_labels,
                        })
                        current_tokens = []
                        current_labels = []
                    continue

                # Tách token và label (hỗ trợ cả tab và space)
                parts = line.split("\t") if "\t" in line else line.split()
                if len(parts) >= 2:
                    current_tokens.append(parts[0])
                    current_labels.append(parts[-1])

        # Câu cuối cùng (nếu file không kết thúc bằng dòng trống)
        if current_tokens:
            sentences.append({
                "tokens": current_tokens,
                "ner_tags": current_labels,
            })

        print(f"[NERDataLoader] Đã load {len(sentences)} câu từ {file_path.name}")
        return sentences

    def _align_labels_with_tokens(
        self,
        labels: List[str],
        word_ids: List[Optional[int]],
    ) -> List[int]:
        """
        Align nhãn NER với sub-tokens sau WordPiece tokenization.
        
        Quy tắc:
        - Sub-token đầu tiên của word: giữ nguyên nhãn gốc
        - Các sub-token tiếp theo: 
          + Nếu nhãn gốc là B-xxx -> đổi thành I-xxx (tránh lặp B-tag)
          + Nếu nhãn gốc là I-xxx hoặc O -> giữ nguyên
        - Special tokens ([CLS], [SEP], [PAD]): gán -100 (ignore trong loss)
        """
        aligned_labels: List[int] = []
        previous_word_idx: Optional[int] = None

        for word_idx in word_ids:
            if word_idx is None:
                # Special token -> ignore
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                # Sub-token ĐẦU TIÊN của word -> giữ nhãn gốc
                aligned_labels.append(self.label2id.get(labels[word_idx], 0))
            else:
                # Sub-token TIẾP THEO -> chuyển B- thành I-
                label = labels[word_idx]
                if label.startswith("B-"):
                    i_label = "I-" + label[2:]
                    aligned_labels.append(self.label2id.get(i_label, 0))
                else:
                    aligned_labels.append(self.label2id.get(label, 0))
            previous_word_idx = word_idx

        return aligned_labels

    def tokenize_and_align(
        self, sentences: List[Dict[str, List[str]]]
    ) -> Dataset:
        """
        Tokenize tokens đã word-segmented và align nhãn NER.
        """
        all_input_ids: List[List[int]] = []
        all_attention_masks: List[List[int]] = []
        all_labels: List[List[int]] = []

        for sent in sentences:
            tokens = sent["tokens"]
            ner_tags = sent["ner_tags"]

            encoding = self.tokenizer(
                tokens,
                is_split_into_words=True,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
            )

            word_ids = encoding.word_ids()
            aligned_labels = self._align_labels_with_tokens(ner_tags, word_ids)

            all_input_ids.append(encoding["input_ids"])
            all_attention_masks.append(encoding["attention_mask"])
            all_labels.append(aligned_labels)

        dataset = Dataset.from_dict({
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks,
            "labels": all_labels,
        })
        dataset.set_format("torch")
        return dataset

    def prepare_datasets(
        self,
        train_path: Path,
        val_path: Optional[Path] = None,
        test_split: float = 0.15,
    ) -> DatasetDict:
        """Pipeline: load CoNLL -> tokenize & align -> split."""
        train_sents = self.load_conll_file(train_path)

        if val_path and Path(val_path).exists():
            val_sents = self.load_conll_file(val_path)
            return DatasetDict({
                "train": self.tokenize_and_align(train_sents),
                "validation": self.tokenize_and_align(val_sents),
            })
        else:
            full_ds = self.tokenize_and_align(train_sents)
            split = full_ds.train_test_split(test_size=test_split, seed=42)
            return DatasetDict({"train": split["train"], "validation": split["test"]})


# ============================================================
# 📍 TRẠM 2B: TOPIC DATA LOADER (CSV - PENDING)
# ============================================================

class TopicDataLoader:
    """
    Tải dataset Topic Classification từ nhiều file CSV.
    
    ⚠️ TRẠNG THÁI: ĐANG CHỜ HOÀN THIỆN DATASET.
    Các file CSV (train_ml.csv, alobacsi_processed.csv, tamanh.csv) chưa sẵn sàng.
    Class này chỉ là KHUNG CODE (template), sẽ hoàn thiện khi data ready.
    """

    def __init__(
        self,
        tokenizer_name: str = TOPIC_MODEL_NAME,
        max_length: int = 256,
        text_column: str = "text",
        label_column: str = "khoa",
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.text_column = text_column
        self.label_column = label_column
        self.label2id: Dict[str, int] = {}
        self.id2label: Dict[int, str] = {}

    def check_data_ready(self) -> bool:
        """Kiểm tra xem tất cả file CSV đã có chưa."""
        if not TOPIC_DATASET_READY:
            print("⚠️  [TopicDataLoader] Flag TOPIC_DATASET_READY = False.")
            print("    Dataset Topic chưa sẵn sàng. Đang chờ hoàn thiện...")
            return False

        missing = [f for f in TOPIC_CSV_FILES if not f.exists()]
        if missing:
            print(f"⚠️  [TopicDataLoader] Thiếu file: {[str(f) for f in missing]}")
            return False

        return True

    def load_and_merge_csv(self) -> pd.DataFrame:
        """Gộp nhiều file CSV thành 1 DataFrame thống nhất."""
        if not self.check_data_ready():
            raise RuntimeError("Dataset chưa sẵn sàng! Không thể load.")

        dfs: List[pd.DataFrame] = []
        for csv_path in TOPIC_CSV_FILES:
            df = pd.read_csv(csv_path)
            # Đảm bảo có đủ cột cần thiết
            if self.text_column in df.columns and self.label_column in df.columns:
                dfs.append(df[[self.text_column, self.label_column]])
            else:
                print(f"⚠️  Bỏ qua {csv_path.name}: thiếu cột '{self.text_column}' hoặc '{self.label_column}'")

        merged = pd.concat(dfs, ignore_index=True).dropna()
        print(f"[TopicDataLoader] Đã gộp {len(merged)} samples từ {len(dfs)} files.")
        return merged

    def build_label_mapping(self, df: pd.DataFrame) -> None:
        """Xây dựng label mapping từ DataFrame."""
        unique_labels = sorted(df[self.label_column].unique())
        self.label2id = {label: idx for idx, label in enumerate(unique_labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        print(f"[TopicDataLoader] {len(self.label2id)} topic labels: {list(self.label2id.keys())[:10]}...")

    def prepare_datasets(self, test_split: float = 0.15) -> DatasetDict:
        """Pipeline hoàn chỉnh: merge CSV -> build mapping -> tokenize -> split."""
        df = self.load_and_merge_csv()
        self.build_label_mapping(df)

        texts = df[self.text_column].tolist()
        labels = [self.label2id[l] for l in df[self.label_column].tolist()]

        encodings = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )

        dataset = Dataset.from_dict({
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels,
        })
        dataset.set_format("torch")

        split = dataset.train_test_split(test_size=test_split, seed=42)
        return DatasetDict({"train": split["train"], "validation": split["test"]})


# ============================================================
# 📍 TRẠM 2C: INTENT DATA LOADER (ViMQ Intent - Multi-label)
# ============================================================

class IntentDataLoader:
    """
    Tải dataset ViMQ Intent cho bài toán Multi-label Classification.
    4 nhãn Intent: Diagnosis, Treatment, Severity, Cause.
    
    ⚠️ LƯU Ý: Đây là bài toán MULTI-LABEL (1 câu có thể có nhiều intent).
    Labels được mã hóa dạng binary vector [0,1,1,0] thay vì single integer.
    """

    def __init__(
        self,
        tokenizer_name: str = INTENT_MODEL_NAME,
        max_length: int = 256,
    ) -> None:
        from config import INTENT_MODEL_NAME
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.label2id = INTENT_LABEL2ID
        self.id2label = {v: k for k, v in INTENT_LABEL2ID.items()}

    def load_raw_data(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Đọc file ViMQ Intent.
        Hỗ trợ format: JSON list hoặc JSONL.
        Mỗi sample: {"text": "...", "intents": ["Diagnosis", "Treatment"]}
        hoặc:        {"text": "...", "intent": "Diagnosis"}
        """
        file_path = Path(file_path)
        samples: List[Dict[str, Any]] = []

        if file_path.suffix == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
                samples = raw if isinstance(raw, list) else raw.get("data", [])
        elif file_path.suffix == ".jsonl":
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        samples.append(json.loads(line))
        elif file_path.suffix == ".csv":
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                sample: Dict[str, Any] = {"text": row["text"]}
                # Hỗ trợ cả 2 format: cột "intents" (multi) hoặc cột "intent" (single)
                if "intents" in df.columns:
                    intents = row["intents"]
                    if isinstance(intents, str):
                        sample["intents"] = [i.strip() for i in intents.split(",")]
                    else:
                        sample["intents"] = [intents]
                elif "intent" in df.columns:
                    sample["intents"] = [row["intent"]]
                samples.append(sample)
        else:
            raise ValueError(f"Unsupported: {file_path.suffix}")

        print(f"[IntentDataLoader] Đã load {len(samples)} samples từ {file_path.name}")
        return samples

    def _encode_multi_label(self, intents: List[str]) -> List[float]:
        """
        Chuyển danh sách intent thành binary vector.
        VD: ["Diagnosis", "Cause"] -> [1.0, 0.0, 0.0, 1.0]
        Dùng float vì BCEWithLogitsLoss yêu cầu target dạng float.
        """
        label_vector = [0.0] * INTENT_NUM_LABELS
        for intent in intents:
            if intent in self.label2id:
                label_vector[self.label2id[intent]] = 1.0
        return label_vector

    def compute_class_weights(
        self, samples: List[Dict[str, Any]]
    ) -> torch.Tensor:
        """
        Tính pos_weight cho BCEWithLogitsLoss để xử lý Imbalanced Data.
        
        Công thức: pos_weight[i] = num_negative[i] / num_positive[i]
        
        Giải thích:
        - Với mỗi class i, đếm số samples DƯƠNG (có nhãn i) và ÂM (không có nhãn i).
        - Class nào có ít samples dương (thiểu số) -> pos_weight cao -> loss phạt nặng hơn
          khi model dự đoán sai class đó -> buộc model học tốt hơn các class hiếm.
        - VD: nếu class "Severity" chỉ có 50 positive / 950 negative
          -> pos_weight = 950/50 = 19.0 (phạt gấp 19 lần khi miss class này).
        """
        # Đếm số positive samples cho mỗi class
        positive_counts = [0] * INTENT_NUM_LABELS
        total = len(samples)

        for sample in samples:
            intents = sample.get("intents", [sample.get("intent", "")])
            if isinstance(intents, str):
                intents = [intents]
            for intent in intents:
                if intent in self.label2id:
                    positive_counts[self.label2id[intent]] += 1

        # Tính pos_weight = negative_count / positive_count
        pos_weights: List[float] = []
        for i, count in enumerate(positive_counts):
            if count > 0:
                neg_count = total - count
                weight = neg_count / count
            else:
                # Nếu class không có sample nào -> weight = 1.0 (mặc định)
                weight = 1.0
            pos_weights.append(weight)

        print(f"[IntentDataLoader] Phân bố Intent labels:")
        for i, label in enumerate(INTENT_LABELS):
            print(f"  - {label}: {positive_counts[i]}/{total} samples "
                  f"(pos_weight = {pos_weights[i]:.2f})")

        return torch.tensor(pos_weights, dtype=torch.float32)

    def tokenize_and_encode(
        self, samples: List[Dict[str, Any]]
    ) -> Dataset:
        """Tokenize và encode multi-label."""
        texts: List[str] = []
        labels: List[List[float]] = []

        for sample in samples:
            texts.append(sample["text"])
            intents = sample.get("intents", [sample.get("intent", "")])
            if isinstance(intents, str):
                intents = [intents]
            labels.append(self._encode_multi_label(intents))

        encodings = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )

        dataset = Dataset.from_dict({
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels,
        })
        dataset.set_format("torch")
        return dataset

    def prepare_datasets(
        self,
        train_path: Path,
        val_path: Optional[Path] = None,
        test_split: float = 0.15,
    ) -> Tuple[DatasetDict, torch.Tensor]:
        """
        Pipeline hoàn chỉnh.
        Returns: (DatasetDict, pos_weight tensor cho loss function)
        """
        train_samples = self.load_raw_data(train_path)
        pos_weight = self.compute_class_weights(train_samples)

        if val_path and Path(val_path).exists():
            val_samples = self.load_raw_data(val_path)
            datasets = DatasetDict({
                "train": self.tokenize_and_encode(train_samples),
                "validation": self.tokenize_and_encode(val_samples),
            })
        else:
            full_ds = self.tokenize_and_encode(train_samples)
            split = full_ds.train_test_split(test_size=test_split, seed=42)
            datasets = DatasetDict({"train": split["train"], "validation": split["test"]})

        return datasets, pos_weight