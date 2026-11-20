import os
from datasets import load_dataset
from transformers import AutoTokenizer

# Xác định thư mục gốc của dự án D:\pubmedqa_eval
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models", "biobert")
OUTPUT_DIR = os.path.join(DATA_DIR, "processed_data")

# 1. Nạp dữ liệu PubMedQA thô từ thư mục ./data
print("[1/5] Đang nạp dữ liệu thô từ thư mục data...")
dataset = load_dataset(
    "qiaojin/PubMedQA", 
    "pqa_labeled", 
    cache_dir=DATA_DIR
)

# 2. Nạp Tokenizer của BioBERT đã tải về trong ./models/biobert
print(f"[2/5] Đang nạp Tokenizer BioBERT từ {MODEL_DIR}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# 3. Định nghĩa từ điển ánh xạ nhãn
label2id = {"yes": 0, "no": 1, "maybe": 2}

# 4. Hàm tiền xử lý: ghép văn bản, mã hóa token và chuyển đổi nhãn
def preprocess_fn(examples):
    # Ghép các câu rời trong tóm tắt y khoa (abstract) thành một đoạn văn hoàn chỉnh
    contexts = [" ".join(c["contexts"]) for c in examples["context"]]
    
    # Tokenize cặp (Question, Context), giới hạn chiều dài 512 tokens
    model_inputs = tokenizer(
        examples["question"],
        contexts,
        max_length=512,
        padding="max_length",
        truncation="only_second"  # Nếu vượt quá 512, chỉ cắt bớt context, giữ nguyên question
    )
    
    # Chuyển đổi nhãn chuỗi ("yes", "no", "maybe") sang số nguyên (0, 1, 2)
    model_inputs["labels"] = [label2id[ans] for ans in examples["final_decision"]]
    return model_inputs

print("[3/5] Đang tokenize và mã hóa toàn bộ 1.000 mẫu dữ liệu...")
tokenized_data = dataset.map(preprocess_fn, batched=True)

# 5. Phân chia tập dữ liệu: 90% Huấn luyện (Train) và 10% Kiểm thử (Test)
print("[4/5] Đang chia tập dữ liệu (900 train / 100 test, seed=42)...")
split_data = tokenized_data["train"].train_test_split(test_size=0.1, seed=42)

# 6. Lưu dữ liệu đã xử lý xuống ổ cứng để train.py nạp trực tiếp
print(f"[5/5] Đang lưu dữ liệu đã tiền xử lý vào {OUTPUT_DIR}...")
split_data.save_to_disk(OUTPUT_DIR)

print("\n" + "="*50)
print("[THÀNH CÔNG] ĐÃ XỬ LÝ VÀ ĐÓNG GÓI DỮ LIỆU!")
print(f"- Tập Train : {len(split_data['train'])} mẫu")
print(f"- Tập Test  : {len(split_data['test'])} mẫu")
print(f"- Vị trí lưu: {OUTPUT_DIR}")
print("="*50)
