import json
import os
from collections import Counter

# Đường dẫn tới thư mục chứa data
data_dir = "kehn_joint/data/"
splits = ["train.jsonl", "val.jsonl", "test.jsonl"]

# Ánh xạ ID sang tên nhãn dựa trên config_joint.py
ID2TOPIC = {
    0: "cardiology", 1: "dentistry", 2: "dermatology", 3: "endocrinology",
    4: "ent", 5: "gastroenterology", 6: "internal_medicine", 7: "neurology",
    8: "nutrition", 9: "obstetrics_gynecology", 10: "ophthalmology",
    11: "orthopedics", 12: "pediatrics", 13: "reproductive_endocrinology",
    14: "rheumatology", 15: "urology", 16: "oncology"
}

print(f"{'='*70}")
print(f"📊 BÁO CÁO PHÂN BỐ NHÃN TOPIC (TOPIC_LABEL_ID)")
print(f"{'='*70}\n")

total_invalid_samples = 0

for split in splits:
    file_path = os.path.join(data_dir, split)
    if not os.path.exists(file_path):
        print(f"⚠️ Không tìm thấy file: {split}\n")
        continue

    label_counts = Counter()
    total_samples = 0

    # Đọc và đếm nhãn
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                topic_id = item.get("topic_label_id", -1)
                label_counts[topic_id] += 1
                total_samples += 1

    print(f"📁 Tệp: {split} (Tổng số mẫu: {total_samples})")
    print(f"{'ID':<5} | {'Tên Nhãn (Theo Config)':<30} | {'Số lượng':<10} | {'Tỷ lệ':<10}")
    print("-" * 70)

    # In ra theo thứ tự ID tăng dần
    for topic_id in sorted(label_counts.keys()):
        count = label_counts[topic_id]
        percentage = (count / total_samples) * 100 if total_samples > 0 else 0
        
        # Đánh dấu cảnh báo nếu ID nằm ngoài phạm vi 0-16
        if topic_id >= 17 or topic_id < 0:
            warning = " ❌ [LỖI OUT-OF-BOUNDS]"
            total_invalid_samples += count
            topic_name = "UNKNOWN_OR_DROPPED"
        else:
            warning = ""
            topic_name = ID2TOPIC.get(topic_id, "UNKNOWN")
        
        print(f"{topic_id:<5} | {topic_name:<30} | {count:<10} | {percentage:.2f}%{warning}")
    
    print("\n")

print(f"{'='*70}")
print(f"🚨 TỔNG KẾT: Phát hiện {total_invalid_samples} mẫu có nhãn lỗi (ID >= 17 hoặc ID < 0) trên toàn bộ dữ liệu.")
print(f"{'='*70}")