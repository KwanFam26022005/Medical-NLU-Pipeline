"""
test_acronym.py - Unit tests cho Cross-Encoder Acronym Pipeline.

Chạy:
    python test_acronym.py
    # hoặc trên Colab:
    !python test_acronym.py
"""

import json
import sys
import traceback
from pathlib import Path

import torch

# ============================================================
# TEST HELPERS
# ============================================================

PASS_COUNT = 0
FAIL_COUNT = 0


def run_test(name, func):
    global PASS_COUNT, FAIL_COUNT
    try:
        func()
        PASS_COUNT += 1
        print(f"  ✅ {name}")
    except Exception as e:
        FAIL_COUNT += 1
        print(f"  ❌ {name}: {e}")
        traceback.print_exc()


# ============================================================
# TESTS
# ============================================================

def test_data_loader_init():
    """Test AcronymDataLoader khởi tạo đúng, load dictionary, add tokens."""
    from data_loader import AcronymDataLoader

    loader = AcronymDataLoader(
        data_dir="data/acrDrAid",
        tokenizer_name="demdecuong/vihealthbert-base-syllable",
        max_length=128,
    )

    assert len(loader.acronym_dict) > 0, "Dictionary should not be empty"
    assert "<e>" in loader.tokenizer.get_added_vocab(), "<e> not in special tokens"
    assert "</e>" in loader.tokenizer.get_added_vocab(), "</e> not in special tokens"

    print(f"    → Dict: {len(loader.acronym_dict)} acronyms")
    print(f"    → Vocab size: {len(loader.tokenizer)}")


def test_dataset_train_output():
    """Test AcronymDataset (train mode) trả về List[Dict] với đúng shape."""
    from data_loader import AcronymDataLoader

    loader = AcronymDataLoader(
        data_dir="data/acrDrAid",
        tokenizer_name="demdecuong/vihealthbert-base-syllable",
        max_length=64,  # short for fast test
    )
    train_ds, _, _ = loader.get_datasets()

    # Get first item
    item = train_ds[0]
    assert isinstance(item, list), f"Train item should be list, got {type(item)}"
    assert len(item) > 0, "Train item should have at least 1 pair"

    # Check each pair
    for pair in item:
        assert "input_ids" in pair, "Missing input_ids"
        assert "attention_mask" in pair, "Missing attention_mask"
        assert "label" in pair, "Missing label"
        assert pair["input_ids"].shape == (64,), f"input_ids shape wrong: {pair['input_ids'].shape}"
        assert pair["label"].dtype == torch.float, f"label dtype wrong: {pair['label'].dtype}"

    # Check that exactly one pair has label=1.0
    positive_count = sum(1 for p in item if p["label"].item() == 1.0)
    assert positive_count == 1, f"Expected 1 positive label, got {positive_count}"

    print(f"    → Sample 0: {len(item)} candidate pairs")
    print(f"    → input_ids shape: {item[0]['input_ids'].shape}")


def test_dataset_eval_output():
    """Test AcronymDataset (eval mode) trả về Dict với grouped candidates."""
    from data_loader import AcronymDataLoader

    loader = AcronymDataLoader(
        data_dir="data/acrDrAid",
        tokenizer_name="demdecuong/vihealthbert-base-syllable",
        max_length=64,
    )
    _, dev_ds, _ = loader.get_datasets()

    item = dev_ds[0]
    assert isinstance(item, dict), f"Eval item should be dict, got {type(item)}"
    assert "encodings" in item, "Missing encodings"
    assert "candidates" in item, "Missing candidates"
    assert "correct_expansion" in item, "Missing correct_expansion"
    assert "acronym" in item, "Missing acronym"

    # Check encodings count matches candidates
    assert len(item["encodings"]) == len(item["candidates"]), \
        f"Encodings ({len(item['encodings'])}) != candidates ({len(item['candidates'])})"

    # Check correct_expansion is in candidates
    assert item["correct_expansion"] in item["candidates"], \
        f"correct_expansion '{item['correct_expansion']}' not in candidates"

    print(f"    → Acronym: '{item['acronym']}'")
    print(f"    → Candidates: {len(item['candidates'])}")
    print(f"    → Correct: '{item['correct_expansion']}'")


def test_collate_fn_train():
    """Test acronym_train_collate_fn flattens variable-length batches correctly."""
    from data_loader import AcronymDataLoader, acronym_train_collate_fn

    loader = AcronymDataLoader(
        data_dir="data/acrDrAid",
        tokenizer_name="demdecuong/vihealthbert-base-syllable",
        max_length=64,
    )
    train_ds, _, _ = loader.get_datasets()

    # Simulate a batch of 4 samples
    batch = [train_ds[i] for i in range(min(4, len(train_ds)))]
    collated = acronym_train_collate_fn(batch)

    assert "input_ids" in collated
    assert "attention_mask" in collated
    assert "labels" in collated

    # Total pairs = sum of candidates per sample
    total_pairs = sum(len(b) for b in batch)
    assert collated["input_ids"].shape[0] == total_pairs, \
        f"Expected {total_pairs} pairs, got {collated['input_ids'].shape[0]}"
    assert collated["input_ids"].shape[1] == 64, \
        f"Seq len should be 64, got {collated['input_ids'].shape[1]}"
    assert collated["labels"].shape == (total_pairs,), \
        f"Labels shape wrong: {collated['labels'].shape}"

    print(f"    → Batch of 4 samples → {total_pairs} pairs")
    print(f"    → input_ids: {collated['input_ids'].shape}")
    print(f"    → labels: {collated['labels'].shape}")


def test_model_forward_pass():
    """Test AcronymCrossEncoder (untrained) forward pass produces correct output shapes."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("demdecuong/vihealthbert-base-syllable")
    tokenizer.add_special_tokens({"additional_special_tokens": ["<e>", "</e>"]})

    model = AutoModelForSequenceClassification.from_pretrained(
        "demdecuong/vihealthbert-base-syllable", num_labels=1
    )
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    # Encode a sample pair
    context = "Bệnh nhân có <e>kt</e> khối u 3cm"
    candidate = "kích thước"

    enc = tokenizer(context, candidate, max_length=64, padding="max_length",
                    truncation="only_first", return_tensors="pt")

    with torch.no_grad():
        outputs = model(**enc)

    logits = outputs.logits
    assert logits.shape == (1, 1), f"Expected logits (1,1), got {logits.shape}"

    score = logits.squeeze(-1).item()
    print(f"    → Logit: {score:.4f}")
    print(f"    → Sigmoid: {torch.sigmoid(torch.tensor(score)).item():.4f}")


def test_entity_marking():
    """Test entity marking đúng vị trí."""
    from data_loader import AcronymDataset

    text = "Kết quả XQ cho thấy tổn thương phổi"
    start = 8
    length = 2  # "XQ"

    marked = (
        text[:start]
        + AcronymDataset.ENTITY_START + text[start:start+length] + AcronymDataset.ENTITY_END
        + text[start+length:]
    )

    expected = "Kết quả <e>XQ</e> cho thấy tổn thương phổi"
    assert marked == expected, f"Expected '{expected}', got '{marked}'"
    print(f"    → '{text}' → '{marked}'")


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 50)
    print("🧪 ACRONYM CROSS-ENCODER — UNIT TESTS")
    print("=" * 50 + "\n")

    run_test("1. DataLoader Init", test_data_loader_init)
    run_test("2. Dataset Train Output Shape", test_dataset_train_output)
    run_test("3. Dataset Eval Output Shape", test_dataset_eval_output)
    run_test("4. Collate Function (Train)", test_collate_fn_train)
    run_test("5. Model Forward Pass", test_model_forward_pass)
    run_test("6. Entity Marking", test_entity_marking)

    print(f"\n{'='*50}")
    print(f"📊 Results: {PASS_COUNT} passed, {FAIL_COUNT} failed")
    print(f"{'='*50}")

    if FAIL_COUNT > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
