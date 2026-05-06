import json
import collections
import statistics

file_path = r'd:\Chatbot-Y tế\kehn_joint\data\pseduo_kehn\merged_kehn.jsonl'
data = []

with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

print(f'Total size: {len(data)}')
if data:
    print(f'Keys: {list(data[0].keys())}')
    
intents = collections.Counter([d.get('intent_label') for d in data])
topics = collections.Counter([d.get('topic_label') for d in data])
ner_tags = collections.Counter([tag for d in data for tag in d.get('ner_tags', [])])

word_lengths = [len(d.get('words', [])) for d in data]

print(f'\\nIntents (Top 10): {intents.most_common(10)}')
print(f'Total Intents: {len(intents)}')
print(f'Topics (Top 10): {topics.most_common(10)}')
print(f'Total Topics: {len(topics)}')
print(f'NER tags (Top 15): {ner_tags.most_common(15)}')

if word_lengths:
    print(f'\\nSentence Lengths (tokens): Min={min(word_lengths)}, Max={max(word_lengths)}, Mean={statistics.mean(word_lengths):.2f}, Median={statistics.median(word_lengths)}')

print('\\nAnomalies:')
print('Missing intents:', sum(1 for d in data if 'intent_label' not in d or d['intent_label'] is None))
print('Empty intents:', sum(1 for d in data if d.get('intent_label') == ''))
print('Missing topics:', sum(1 for d in data if 'topic_label' not in d or d['topic_label'] is None))
print('Empty topics:', sum(1 for d in data if d.get('topic_label') == ''))
print('Missing words:', sum(1 for d in data if 'words' not in d or not d['words']))
print('Missing ner_tags:', sum(1 for d in data if 'ner_tags' not in d or not d['ner_tags']))

mismatch_len = sum(1 for d in data if len(d.get('words', [])) != len(d.get('ner_tags', [])))
print('Mismatched words/ner_tags length:', mismatch_len)

# Analyze confidence scores if present
has_conf = any('intent_confidence' in d for d in data)
if has_conf:
    intent_conf = [d.get('intent_confidence', 0) for d in data if d.get('intent_confidence') is not None]
    if intent_conf:
        print(f'Intent Confidence: Min={min(intent_conf):.2f}, Max={max(intent_conf):.2f}, Mean={statistics.mean(intent_conf):.2f}')
