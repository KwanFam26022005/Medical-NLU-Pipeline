import os
import sys
import torch
import numpy as np

# Trỏ vào thư mục source của ViMQ để lấy các hàm tiện ích
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../ViMQ-main/ViMQ-main/src")))

from utils import MODEL_CLASSES, load_tokenizer, get_entity_label, spacy_to_iob

# Cấu hình đường dẫn model
# Nếu chạy trên Google Colab:
MODEL_DIR = "/content/drive/MyDrive/Medical-NLU-Pipeline/outputs/vimq_joint_model"
# Nếu tải về chạy Local:
# MODEL_DIR = "../outputs/vimq_joint_model"

VIMQ_DATA_DIR = "../ViMQ-main/ViMQ-main/data"

def load_vimq_model():
    print(f"Đang tải mô hình ViMQ từ {MODEL_DIR}...")
    
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"Thư mục {MODEL_DIR} không tồn tại. Hãy chắc chắn đường dẫn là chính xác!")
        
    # Tải lại arguments lúc huấn luyện
    args = torch.load(os.path.join(MODEL_DIR, "training_args.bin"), map_location='cpu', weights_only=False)
    args.data_dir = VIMQ_DATA_DIR
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    
    tokenizer = load_tokenizer(args)
    label2index, index2label = get_entity_label(args)
    
    import json
    char_vocab_path = os.path.join(args.data_dir, args.file_name_char2index)
    with open(char_vocab_path, 'r', encoding='utf-8') as f:
        char_vocab = json.load(f)
        
    # Tải model
    config_class, model_class, _ = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained("demdecuong/vihealthbert-base-word")
    model = model_class(config=config, args=args)
    
    checkpoint = torch.load(os.path.join(MODEL_DIR, "checkpoint.pth"), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    
    print("✅ Tải mô hình thành công!")
    return model, tokenizer, args, index2label, char_vocab, device

def span_decode(logits, index2label):
    arg_index = []
    for i in range(len(logits)):
        for j in range(i, len(logits[i])):
            if logits[i][j] > 0:
                arg_index.append([i, j, index2label.get(int(logits[i][j]), 'UNK')])
    return arg_index

def preprocess_text(tokenizer, words, char_vocab, args):
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
        
    firstSWindices = [min(idx, max_seq_len - 1) for idx in firstSWindices]

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

def predict(sentence, model, tokenizer, args, index2label, char_vocab, device):
    try:
        from pyvi import ViTokenizer
    except ImportError:
        print("Đang cài đặt thư viện pyvi...")
        os.system("pip install pyvi")
        from pyvi import ViTokenizer
        
    # Phân tách từ (word segmentation)
    segmented_sentence = ViTokenizer.tokenize(sentence)
    words = segmented_sentence.split(' ')
    seq_len = len(words)
    
    print(f"\n[+] Câu sau khi tách từ: {words}")
    
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
        
    score = score.detach().cpu().numpy()
    preds = np.argmax(score, axis=-1)[0]
    
    spacy_spans = span_decode(preds, index2label)
    iob_tags = spacy_to_iob(spacy_spans, seq_len)
    
    entities = []
    for span in spacy_spans:
        start_idx = span[0]
        end_idx = span[1]
        entity_label = span[2]
        entity_word = " ".join(words[start_idx:end_idx+1]).replace("_", " ")
        entities.append({"word": entity_word, "label": entity_label})
        
    return entities, iob_tags

if __name__ == "__main__":
    try:
        model, tokenizer, args, index2label, char_vocab, device = load_vimq_model()
    except Exception as e:
        print(f"Lỗi khởi tạo mô hình: {e}")
        sys.exit(1)
        
    print("\n" + "="*50)
    print("🔥 CÔNG CỤ TEST MÔ HÌNH ViMQ (Nhận diện thực thể Y Tế) 🔥")
    print("Nhập 'q' hoặc 'exit' để thoát.")
    print("="*50)
    
    while True:
        text = input("\nNhập câu hỏi bệnh lý: ")
        if text.strip().lower() in ['q', 'exit']:
            break
            
        if not text.strip():
            continue
            
        entities, iob_tags = predict(text, model, tokenizer, args, index2label, char_vocab, device)
        
        print("\n[KẾT QUẢ DỰ ĐOÁN]")
        print("-" * 30)
        print("1. Nhãn BIO (IOB Tags):")
        print("   " + " ".join(iob_tags))
        print("\n2. Thực thể tìm thấy:")
        if len(entities) == 0:
            print("   (Không tìm thấy thực thể nào)")
        else:
            for i, ent in enumerate(entities, 1):
                print(f"   {i}. [{ent['label']}] : {ent['word']}")
        print("-" * 30)
