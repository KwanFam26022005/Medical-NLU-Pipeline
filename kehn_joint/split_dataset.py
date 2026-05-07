import json
import sys
import os
from collections import Counter
from sklearn.model_selection import train_test_split

def load_data(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def save_data(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def print_stats(split_name, data):
    print(f"\n--- {split_name.upper()} SPLIT STATS ---")
    total = len(data)
    print(f"Total samples: {total}")
    
    topic_counts = Counter(item['topic_label_id'] for item in data)
    intent_counts = Counter(item.get('intent_label_id', item.get('token_intent_ids', [0])[0]) for item in data)
    
    print("\nTopic Label Distribution:")
    for label, count in sorted(topic_counts.items()):
        print(f"  Topic {label}: {count} ({count/total*100:.2f}%)")
        
    print("\nIntent Label Distribution:")
    for label, count in sorted(intent_counts.items()):
        print(f"  Intent {label}: {count} ({count/total*100:.2f}%)")

def main():
    input_file = sys.argv[1] if len(sys.argv) > 1 else r'D:\Chatbot-Y tế\kehn_joint\data\medical_kehn_merged.jsonl'
    
    print(f"Loading data from {input_file}...")
    data = load_data(input_file)
    
    topics = [item['topic_label_id'] for item in data]
    
    # First split: 70% train, 30% temp (val + test)
    train_data, temp_data, _, temp_topics = train_test_split(
        data, topics, test_size=0.30, random_state=42, stratify=topics
    )
    
    # Second split: 50% val, 50% test from temp (which means 15% / 15% of total)
    val_data, test_data = train_test_split(
        temp_data, test_size=0.50, random_state=42, stratify=temp_topics
    )
    
    # Save splits
    save_data(train_data, 'data/train.jsonl')
    save_data(val_data, 'data/val.jsonl')
    save_data(test_data, 'data/test.jsonl')
    
    # Print stats
    print_stats("Train", train_data)
    print_stats("Val", val_data)
    print_stats("Test", test_data)

if __name__ == '__main__':
    main()
