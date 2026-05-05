import json
import os
import sys
import torch
import numpy as np
from tqdm import tqdm

# Trỏ vào thư mục source của ViMQ để lấy các hàm tiện ích. Dùng insert(0) để ưu tiên thư viện ViMQ thay vì thư mục model/ hiện tại
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../ViMQ-main/ViMQ-main/src")))

from utils import MODEL_CLASSES, load_tokenizer, get_entity_label, spacy_to_iob

# Đường dẫn config
HOSPITAL_DATA_PATH = "data/pseduo_kehn/hospital_kehn.jsonl"
OUTPUT_DATA_PATH = "data/pseduo_kehn/hospital_kehn_vimq.jsonl"

# ⚠️ QUAN TRỌNG: Sửa đường dẫn này trỏ tới nơi lưu model ViMQ
# Nếu chạy trên Google Colab (theo hướng dẫn train): 
MODEL_DIR = "/content/drive/MyDrive/Medical-NLU-Pipeline/outputs/vimq_joint_model"
# Nếu tải về chạy Local:
# MODEL_DIR = "../outputs/vimq_joint_model"
VIMQ_DATA_DIR = "../ViMQ-main/ViMQ-main/data"

def load_vimq_model():
    print(f"Loading ViMQ model from {MODEL_DIR}...")
    
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"Thư mục {MODEL_DIR} không tồn tại. Hãy chạy train_vimq_colab.sh trước!")
        
    # Tải lại arguments lúc huấn luyện
    args = torch.load(os.path.join(MODEL_DIR, "training_args.bin"), map_location='cpu')
    
    # Ghi đè lại data_dir vì path lúc train có thể khác
    args.data_dir = VIMQ_DATA_DIR
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    
    tokenizer = load_tokenizer(args)
    label2index, index2label = get_entity_label(args)
    
    # Tải char_vocab
    char_vocab_path = os.path.join(args.data_dir, args.file_name_char2index)
    with open(char_vocab_path, 'r', encoding='utf-8') as f:
        char_vocab = json.load(f)
        
    # Tải model
    config_class, model_class, _ = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained("demdecuong/vihealthbert-base-word")
    model = model_class(config=config, args=args)
    checkpoint = torch.load(os.path.join(MODEL_DIR, "checkpoint.pth"), map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    
    return model, tokenizer, args, index2label, char_vocab, device

def span_decode(logits, index2label):
    arg_index = []
    for i in range(len(logits)):
        for j in range(i, len(logits[i])):
            if logits[i][j] > 0:
                arg_index.append([i, j, index2label.get(int(logits[i][j]), 'UNK')])
    return arg_index

def preprocess_text(tokenizer, words, char_vocab, args):
    """Giả lập lại tiền xử lý của ViMQ data_loader.py"""
    max_seq_len = args.max_seq_len
    
    # --- Tokenizer (Subword) ---
    input_ids = [tokenizer.cls_token_id]
    firstSWindices = [len(input_ids)]

    for word in words:
        word_token = tokenizer.encode(word)
        input_ids += word_token[1: (len(word_token) - 1)]
        firstSWindices.append(len(input_ids))

    firstSWindices = firstSWindices[: (len(firstSWindices) - 1)]
    input_ids.append(tokenizer.sep_token_id)
    attention_mask = [1] * len(input_ids)

    if len(input_ids) > max_seq_len:
        input_ids = input_ids[:max_seq_len]
        attention_mask = attention_mask[:max_seq_len]
        firstSWindices = firstSWindices[:max_seq_len]
    else:
        attention_mask = attention_mask + [0] * (max_seq_len - len(input_ids))
        input_ids = input_ids + [tokenizer.pad_token_id] * (max_seq_len - len(input_ids))
        firstSWindices = firstSWindices + [0]*(max_seq_len - len(firstSWindices))

    # --- Character Embedding ---
    char_seq = []
    for word in words:
        word_seq = []
        for i in range(args.max_char_len):
            try:
                char = word[i]
            except:
                char = args.pad_char
            word_seq.append(char)
        char_seq.append(word_seq)
        
    char_ids = []
    for word_chars in char_seq:
        word_char_ids = []
        for char in word_chars:
            if char not in char_vocab:
                word_char_ids.append(char_vocab.get("UNK"))
            else:
                word_char_ids.append(char_vocab.get(char))
        char_ids.append(word_char_ids)
        
    if len(char_ids) < max_seq_len:
        char_ids += [[char_vocab.get("PAD")] * args.max_char_len] * (max_seq_len - len(char_ids))
    else:
        char_ids = char_ids[:max_seq_len]

    return (
        torch.tensor([input_ids]), 
        torch.tensor([attention_mask]), 
        torch.tensor([firstSWindices]), 
        torch.tensor([char_ids])
    )

def main():
    try:
        model, tokenizer, args, index2label, char_vocab, device = load_vimq_model()
    except FileNotFoundError as e:
        print(e)
        return

    # Map nhãn BIO từ config của KEHN (có thể khác một chút với ViMQ)
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from config import NER_LABEL2ID

    print("Bắt đầu dán nhãn lại tập Hospital...")
    updated_data = []
    with open(HOSPITAL_DATA_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    for line in tqdm(lines, desc="Inference"):
        data = json.loads(line)
        words = data["words"]
        seq_len = len(words)
        
        # Tiền xử lý
        input_ids, attention_mask, first_subword, char_ids = preprocess_text(tokenizer, words, char_vocab, args)
        
        inputs = {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
            "first_subword": first_subword.to(device),
            "seq_len": torch.tensor([seq_len]).to(device),
            "char_ids": char_ids.to(device),
            "label": None
        }
        
        with torch.no_grad():
            score, _ = model(**inputs)
            
        # score shape: (1, max_seq_len, max_seq_len, C)
        score = score.detach().cpu().numpy()
        preds = np.argmax(score, axis=-1)[0] # Lấy mẫu đầu tiên trong batch
        
        # Decode Spans
        spacy_spans = span_decode(preds, index2label)
        
        # Chuyển đổi sang IOB tags
        iob_tags = spacy_to_iob(spacy_spans, seq_len)
        
        # Convert IOB to Tag IDs
        tag_ids = []
        for tag in iob_tags:
            # Map tag to ID (fallback to 0 if tag is not in NER_LABEL2ID)
            tag_ids.append(NER_LABEL2ID.get(tag, 0))
            
        data["ner_tags"] = iob_tags
        data["ner_tag_ids"] = tag_ids
        # Ta gán lại confidence là 1.0 (do dùng mô hình chất lượng cao)
        data["ner_confidence"] = 1.0 
        
        updated_data.append(data)
        
    # Lưu kết quả
    with open(OUTPUT_DATA_PATH, "w", encoding="utf-8") as f:
        for item in updated_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"✅ Đã dán nhãn lại thành công {len(updated_data)} dòng.")
    print(f"Lưu kết quả tại: {OUTPUT_DATA_PATH}")

if __name__ == "__main__":
    main()
