import json
max_id = 0
with open("data/train.jsonl", "r",encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        max_id = max(max_id, data['topic_label_id'])
print(f"Max Topic ID in data: {max_id}")
# Nếu con số này >= 17, bạn phải re-map lại dữ liệu.