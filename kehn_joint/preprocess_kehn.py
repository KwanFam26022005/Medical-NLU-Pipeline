"""
preprocess_kehn.py

This script preprocesses, normalizes, and merges Vietnamese medical datasets
(ViMQ and Hospital) into a strict JSONL format for the KEHN architecture.

Core Steps:
1. Punctuation Separation & Word Segmentation using PyVi.
2. Character-Level Index Tracking for NER Alignment.
3. Generate Token-level Intent IDs.
4. Local Deduplication (Hospital Dataset Only).
5. Fill Missing Confidences.
"""

import json
import re
import random
import difflib
import logging
from pathlib import Path

try:
    from pyvi import ViTokenizer
except ImportError:
    raise ImportError("Please install pyvi: pip install pyvi")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Immutable output schema requirements
# {
#   "source": "vimq|hospital",
#   "text": "...",
#   "words": ["..."],
#   "topic_label": "...",
#   "topic_label_id": 0,
#   "topic_confidence": 1.0,
#   "intent_label": "...",
#   "intent_label_id": 0,
#   "intent_confidence": 1.0,
#   "token_intent_ids": [0, 0, 0],
#   "ner_tags": ["B-SYM", "I-SYM", "O"],
#   "ner_tag_ids": [1, 2, 0],
#   "ner_confidence": 1.0
# }

NER2ID = {
    "O": 0,
    "B-SYM": 1,
    "I-SYM": 2,
    "B-PRO": 3,
    "I-PRO": 4,
    "B-DRU": 5,
    "I-DRU": 6
}

def clean_and_segment(text: str) -> list:
    """Detaches punctuation and applies ViTokenizer."""
    # Add space around standard punctuation
    cleaned = re.sub(r'([.,?!;:()[\]{}"\'])', r' \1 ', text)
    # Remove multiple consecutive spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    # Segment words using pyvi
    segmented = ViTokenizer.tokenize(cleaned)
    return segmented.split()

def deduplicate_tokens(words: list, tags: list, intent_ids: list):
    """
    Detects and removes repetitive sentences (like "A B C A B C").
    Splits by end-of-sentence punctuation and compares string similarity.
    """
    sentences = []
    cur_words, cur_tags, cur_intents = [], [], []
    
    for w, t, i in zip(words, tags, intent_ids):
        cur_words.append(w)
        cur_tags.append(t)
        cur_intents.append(i)
        # End of sentence markers
        if w in [".", "?", "!"]:
            sentences.append((cur_words, cur_tags, cur_intents))
            cur_words, cur_tags, cur_intents = [], [], []
            
    if cur_words:
        sentences.append((cur_words, cur_tags, cur_intents))
        
    deduped_words, deduped_tags, deduped_intents = [], [], []
    seen_texts = []
    
    for w_list, t_list, i_list in sentences:
        sent_text = " ".join(w_list).lower()
        # Do not deduplicate very short natural communication phrases
        if len(w_list) <= 5:
            deduped_words.extend(w_list)
            deduped_tags.extend(t_list)
            deduped_intents.extend(i_list)
            seen_texts.append(sent_text)
            continue
            
        is_dup = False
        for seen in seen_texts:
            seq = difflib.SequenceMatcher(None, sent_text, seen)
            # Threshold set to 90% as requested
            if seq.ratio() > 0.90:
                is_dup = True
                break
                
        if not is_dup:
            deduped_words.extend(w_list)
            deduped_tags.extend(t_list)
            deduped_intents.extend(i_list)
            seen_texts.append(sent_text)
            
    return deduped_words, deduped_tags, deduped_intents


def process_sample(sample: dict, source_name: str) -> dict:
    orig_words = sample.get("words", [])
    orig_tags = sample.get("ner_tags", [])
    
    # 1. Punctuation Separation & Word Segmentation
    raw_text = " ".join(orig_words)
    # Remove spacing right before punctuation that might have been caused by raw joining
    raw_text = re.sub(r'\s+([.,?!;:()[\]{}"\'])', r'\1', raw_text)
    new_words = clean_and_segment(raw_text)
    
    # 2. Character-Level Index Tracking for NER Alignment
    # Build original character array mapped to NER tags (removing any spaces or underscores)
    char_to_tag = []
    for w, t in zip(orig_words, orig_tags):
        pure_w = w.replace("_", "").replace(" ", "")
        char_to_tag.extend([t] * len(pure_w))
        
    new_tags = []
    char_idx = 0
    
    for nw in new_words:
        pure_nw = nw.replace("_", "").replace(" ", "")
        
        # Rule: Newly split punctuation gets tag O
        if re.match(r'^[.,?!;:()[\]{}"\']+$', pure_nw):
            new_tags.append("O")
            char_idx += len(pure_nw) # Advance pointer to skip the punctuation characters
            continue
            
        # Collect tags for the characters forming this new word
        tags_for_word = []
        for _ in range(len(pure_nw)):
            if char_idx < len(char_to_tag):
                tags_for_word.append(char_to_tag[char_idx])
            else:
                tags_for_word.append("O")
            char_idx += 1
            
        # Rule: If B- and I- are merged, the new word gets B-
        has_b = [t for t in tags_for_word if t.startswith("B-")]
        has_i = [t for t in tags_for_word if t.startswith("I-")]
        
        if has_b:
            new_tags.append(has_b[0])
        elif has_i:
            new_tags.append(has_i[0])
        else:
            new_tags.append("O")
            
    # 3. Generate Token-level Intent IDs
    intent_label = sample.get("intent_label", "unknown")
    intent_label_id = sample.get("intent_label_id", 0)
    token_intent_ids = [intent_label_id] * len(new_words)
    
    # 4. Local Deduplication (Hospital Dataset Only)
    if source_name == "hospital":
        new_words, new_tags, token_intent_ids = deduplicate_tokens(new_words, new_tags, token_intent_ids)
        
    # Generate NER tag IDs mapping
    new_tag_ids = [NER2ID.get(t, 0) for t in new_tags]
    
    # Strict Guardrails & Stop Conditions
    if not (len(new_words) == len(new_tags) == len(new_tag_ids) == len(token_intent_ids)):
        logging.warning(f"Length mismatch in sample. Words: {len(new_words)}, Tags: {len(new_tags)}. Skipping.")
        return None
        
    # 5. Fill Missing Confidences
    topic_conf = sample.get("topic_confidence", 1.0)
    intent_conf = sample.get("intent_confidence", 1.0)
    ner_conf = sample.get("ner_confidence", 1.0)
    
    # Reconstruct text from normalized words
    final_text = " ".join(new_words)
    
    # Construct strictly compliant output JSON
    output = {
        "source": source_name,
        "text": final_text,
        "words": new_words,
        "topic_label": sample.get("topic_label", "unknown"),
        "topic_label_id": sample.get("topic_label_id", 0),
        "topic_confidence": topic_conf,
        "intent_label": intent_label,
        "intent_label_id": intent_label_id,
        "intent_confidence": intent_conf,
        "token_intent_ids": token_intent_ids,
        "ner_tags": new_tags,
        "ner_tag_ids": new_tag_ids,
        "ner_confidence": ner_conf
    }
    
    return output

def process_file(input_path: str, source_name: str, output_path: str) -> list:
    results = []
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logging.error(f"Could not read {input_path}: {e}")
        return results
        
    valid_count = 0
    with open(output_path, 'w', encoding='utf-8') as out_f:
        for sample in data:
            processed = process_sample(sample, source_name)
            if processed:
                results.append(processed)
                out_f.write(json.dumps(processed, ensure_ascii=False) + '\n')
                valid_count += 1
                
    logging.info(f"Processed {source_name}: {valid_count}/{len(data)} valid samples.")
    return results

def main():
    # Update these paths to match your actual local file structure if needed
    vimq_input = "data/joint_train.json"
    hospital_input = "data/pseudo_new/pseudo_new_joint.json"
    
    vimq_output = "vimq_kehn.jsonl"
    hospital_output = "hospital_kehn.jsonl"
    merged_output = "merged_kehn.jsonl"
    
    logging.info("Starting processing for ViMQ...")
    vimq_data = process_file(vimq_input, "vimq", vimq_output)
    
    logging.info("Starting processing for Hospital...")
    hospital_data = process_file(hospital_input, "hospital", hospital_output)
    
    # Merge and Shuffle
    merged_data = vimq_data + hospital_data
    random.seed(42)
    random.shuffle(merged_data)
    
    with open(merged_output, 'w', encoding='utf-8') as out_f:
        for item in merged_data:
            out_f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    logging.info(f"Successfully created {merged_output} with {len(merged_data)} samples.")

if __name__ == "__main__":
    main()
