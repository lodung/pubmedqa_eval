import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Tạo thư mục lưu mô hình cục bộ
model_dir = "./models/biobert"
os.makedirs(model_dir, exist_ok=True)

model_id = "dmis-lab/biobert-v1.1"
print(f"Đang tải mô hình {model_id} về thư mục: {model_dir}...")

# 1. Tải và lưu Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained(model_dir)

# 2. Tải và lưu Trọng số mô hình BioBERT
label2id = {"yes": 0, "no": 1, "maybe": 2}
id2label = {0: "yes", 1: "no", 2: "maybe"}
model = AutoModelForSequenceClassification.from_pretrained(
    model_id, 
    num_labels=3, 
    label2id=label2id, 
    id2label=id2label
)
model.save_pretrained(model_dir)

print("\n" + "="*45)
print("[THÀNH CÔNG] ĐÃ TẢI XONG MÔ HÌNH!")
print(f"Toàn bộ file mô hình đã nằm gọn trong: {model_dir}")
print("="*45)