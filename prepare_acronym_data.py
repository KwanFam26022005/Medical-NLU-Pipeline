"""
prepare_acronym_data.py - Chuyển đổi dữ liệu thô acrDrAid sang format
tương thích với data_loader.py.

🔧 v3 - DICTIONARY-CONSTRAINED WSD:
  ✅ Labels build từ TRAINING DATA (280 classes) — không phải dictionary (424)
  ✅ Save dictionary_constraint.json (acronym → valid label IDs)
  ✅ Validate test set với gold.json

Cách chạy:
    python prepare_acronym_data.py
"""

import json
from collections import Counter
from pathlib import Path


def extract_acronym_from_sample(sample: dict) -> dict:
    """Trích xuất acronym từ text gốc dựa trên start_char_idx và length_acronym."""
    text = sample["text"]
    start = sample["start_char_idx"]
    length = sample["length_acronym"]
    acronym = text[start : start + length].strip()
    return {
        "acronym": acronym,
        "sentence": text.strip(),
        "expansion": sample["expansion"].strip(),
    }


def process_split(input_path: Path) -> tuple:
    """Xử lý 1 split (train/dev/test)."""
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    converted = []
    errors = 0
    for sample in raw_data:
        try:
            result = extract_acronym_from_sample(sample)
            if result["acronym"] and result["sentence"] and result["expansion"]:
                converted.append(result)
            else:
                errors += 1
        except (IndexError, KeyError):
            errors += 1

    return converted, errors


def build_dictionary_constraint(dictionary: dict, expansion2id: dict) -> dict:
    """
    Build dictionary_constraint: acronym → list of valid label IDs.
    
    Đây là KEY của Dictionary-Constrained WSD:
    - Mỗi acronym chỉ có 2-7 nghĩa hợp lệ
    - Tại inference: mask logits để chỉ chọn từ các nghĩa này
    - Tại training: focus model vào phân biệt các nghĩa hợp lệ
    
    Ví dụ:
        "bt" → [id("bất thường"), id("buồng trứng"), id("bao tuyến"), id("bên trái")]
        "kt" → [id("kích thước"), id("khu trú"), id("khuếch tán"), id("kỹ thuật")]
    """
    constraint = {}
    unmapped = {}

    for acronym, expansions in dictionary.items():
        valid_ids = []
        missing = []
        for exp in expansions:
            exp_lower = exp.strip().lower()
            exp_original = exp.strip()
            # Try both lowercase and original case
            if exp_original in expansion2id:
                valid_ids.append(expansion2id[exp_original])
            elif exp_lower in expansion2id:
                valid_ids.append(expansion2id[exp_lower])
            else:
                missing.append(exp_original)

        if valid_ids:
            constraint[acronym] = valid_ids

        if missing:
            unmapped[acronym] = missing

    return constraint, unmapped


def main():
    base_dir = Path("data/acrDrAid")
    output_dir = Path("data")

    print("=" * 60)
    print("🔄 CHUẨN BỊ DỮ LIỆU acrDrAid (v3 - DICTIONARY CONSTRAINED)")
    print("=" * 60)

    # ──────────────────────────────────────────────
    # 1. Load dictionary.json
    # ──────────────────────────────────────────────
    dict_path = base_dir / "dictionary.json"
    if not dict_path.exists():
        print(f"❌ Không tìm thấy {dict_path}")
        return

    with open(dict_path, "r", encoding="utf-8") as f:
        dictionary = json.load(f)
    print(f"\n📚 Dictionary: {len(dictionary)} acronyms")

    # ──────────────────────────────────────────────
    # 2. Process each split
    # ──────────────────────────────────────────────
    splits = {
        "train": base_dir / "train" / "data.json",
        "val":   base_dir / "dev" / "data.json",
        "test":  base_dir / "test" / "data.json",
    }

    all_data = {}
    for split_name, input_path in splits.items():
        if not input_path.exists():
            print(f"\n⚠️  Bỏ qua {split_name}: {input_path} không tồn tại")
            continue

        converted, errors = process_split(input_path)
        output_path = output_dir / f"acronym_{split_name}.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(converted, f, ensure_ascii=False, indent=2)

        all_data[split_name] = converted

        acronym_counter = Counter(s["acronym"] for s in converted)
        exp_counter = Counter(s["expansion"] for s in converted)

        print(f"\n📊 {split_name.upper()}: {len(converted)} samples (lỗi: {errors})")
        print(f"   Unique acronyms: {len(acronym_counter)}")
        print(f"   Unique expansions: {len(exp_counter)}")
        print(f"   Top 5 acronyms: {', '.join(f'{a}({c})' for a, c in acronym_counter.most_common(5))}")
        print(f"   → Saved: {output_path}")

    # ──────────────────────────────────────────────
    # 3. Build label mapping từ TRAINING DATA (không phải dictionary!)
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("🏷️  BUILD LABEL MAPPING (từ TRAINING + VAL data)")
    print("=" * 60)

    all_samples = all_data.get("train", []) + all_data.get("val", [])
    unique_expansions = sorted(set(s["expansion"] for s in all_samples))
    expansion2id = {exp: idx for idx, exp in enumerate(unique_expansions)}
    id2expansion = {idx: exp for idx, exp in enumerate(unique_expansions)}

    num_labels = len(expansion2id)
    print(f"   Số labels (từ data): {num_labels}")
    print(f"   ⚠️  vs dictionary (all): {sum(len(v) for v in dictionary.values())} total entries")

    # ──────────────────────────────────────────────
    # 4. Build Dictionary Constraint
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("🔒 BUILD DICTIONARY CONSTRAINT")
    print("=" * 60)

    constraint, unmapped = build_dictionary_constraint(dictionary, expansion2id)

    total_constrained = sum(len(v) for v in constraint.values())
    avg_per_acr = total_constrained / max(len(constraint), 1)

    print(f"   Acronyms with constraint: {len(constraint)}/{len(dictionary)}")
    print(f"   Avg valid expansions per acronym: {avg_per_acr:.1f}")
    print(f"   → Softmax space: ~{avg_per_acr:.0f} classes (thay vì {num_labels}!)")

    # Show examples
    print(f"\n   Ví dụ constraint:")
    for acr in ["bt", "kt", "tn", "nm", "th"][:5]:
        if acr in constraint:
            ids = constraint[acr]
            exps = [id2expansion[i] for i in ids]
            print(f"      '{acr}' → {exps} ({len(ids)} choices)")

    if unmapped:
        total_unmapped = sum(len(v) for v in unmapped.values())
        print(f"\n   ⚠️  {total_unmapped} expansions trong dictionary nhưng KHÔNG có trong training data")
        for acr, exps in list(unmapped.items())[:5]:
            print(f"      '{acr}': {exps}")

    # ──────────────────────────────────────────────
    # 5. Save all files
    # ──────────────────────────────────────────────
    # Save dictionary constraint
    constraint_path = output_dir / "dictionary_constraint.json"
    with open(constraint_path, "w", encoding="utf-8") as f:
        json.dump(constraint, f, ensure_ascii=False, indent=2)
    print(f"\n   → Saved: {constraint_path}")

    # Save expansion mapping (id → expansion)
    mapping_path = output_dir / "expansion_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(id2expansion, f, ensure_ascii=False, indent=2)
    print(f"   → Saved: {mapping_path}")

    # ──────────────────────────────────────────────
    # 6. Validate test với gold.json
    # ──────────────────────────────────────────────
    gold_path = base_dir / "gold.json"
    if "test" in all_data and gold_path.exists():
        with open(gold_path, "r", encoding="utf-8") as f:
            gold = json.load(f)
        test_data = all_data["test"]
        matches = 0
        total = 0
        for idx_str, gold_exp in gold.items():
            idx = int(idx_str)
            if idx < len(test_data):
                if test_data[idx]["expansion"].lower() == gold_exp.lower():
                    matches += 1
                total += 1
        print(f"\n🏆 Gold validation: {matches}/{total} match ({matches/max(total,1)*100:.1f}%)")

    # ──────────────────────────────────────────────
    # 7. Class imbalance analysis
    # ──────────────────────────────────────────────
    if "train" in all_data:
        train = all_data["train"]
        exp_counts = Counter(s["expansion"] for s in train)
        max_count = exp_counts.most_common(1)[0][1]
        min_count = exp_counts.most_common()[-1][1]
        rare = sum(1 for c in exp_counts.values() if c <= 5)

        print(f"\n⚖️  Class imbalance (train):")
        print(f"   Classes: {len(exp_counts)}, Max: {max_count}, Min: {min_count}")
        print(f"   Ratio: {max_count / max(min_count, 1):.1f}x")
        print(f"   Rare classes (≤5): {rare} ({rare/len(exp_counts)*100:.1f}%)")

    # ──────────────────────────────────────────────
    # 8. Summary
    # ──────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"✅ HOÀN TẤT!")
    print(f"   data/acronym_train.json")
    print(f"   data/acronym_val.json")
    print(f"   data/acronym_test.json")
    print(f"   data/expansion_mapping.json  ({num_labels} labels)")
    print(f"   data/dictionary_constraint.json ({len(constraint)} acronyms)")
    print(f"\n💡 Tiếp: python train_acronym.py \\")
    print(f"     --train_data data/acronym_train.json \\")
    print(f"     --val_data data/acronym_val.json")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
