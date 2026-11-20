import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments
)

# 1. Thiết lập đường dẫn động chuẩn
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed_data")
MODEL_PATH = os.path.join(BASE_DIR, "models", "biobert")
OUTPUT_MODEL_DIR = os.path.join(BASE_DIR, "models", "biobert_finetuned")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "results", "checkpoints")

print("="*60)
print("     BẮT ĐẦU QUY TRÌNH HUẤN LUYỆN (FINE-TUNE) BIOBERT")
print("="*60)

# 2. Nạp dữ liệu huấn luyện đã chuẩn bị
print(f"[1/4] Đang nạp tập dữ liệu huấn luyện từ: {DATA_PATH}...")
dataset = load_from_disk(DATA_PATH)
train_dataset = dataset["train"]
print(f"--> Số lượng mẫu huấn luyện: {len(train_dataset)} câu.")

# 3. Nạp Tokenizer và Mô hình BioBERT gốc
print(f"[2/4] Đang nạp mô hình và Tokenizer từ: {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Kiểm tra phần cứng tính toán
use_cuda = torch.cuda.is_available()
device_name = torch.cuda.get_device_name(0) if use_cuda else "CPU"
print(f"--> Thiết bị tính toán sử dụng: {device_name}")

# 4. Cấu hình tham số huấn luyện (TrainingArguments)
print("[3/4] Cấu hình tham số huấn luyện...")
training_args = TrainingArguments(
    output_dir=CHECKPOINTS_DIR,
    num_train_epochs=3,                     # 3 epochs chuẩn theo bài báo gốc Jin et al. (2019)
    per_device_train_batch_size=8,          # Batch size 8 tối ưu cho bộ nhớ
    gradient_accumulation_steps=2,         # Tích lũy gradient (tương đương effective batch size = 16)
    learning_rate=2e-5,                     # Learning rate chuẩn cho fine-tuning BERT
    weight_decay=0.01,                      # Tránh overfitting
    logging_steps=10,                       # In tiến độ sau mỗi 10 bước
    save_strategy="no",                     # Không lưu checkpoint trung gian để tiết kiệm dung lượng
    fp16=use_cuda,                          # Bật tính toán số thực 16-bit nếu có GPU
    report_to="none"                        # Tắt logging bên thứ ba (wandb/mlflow)
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer
)

# 5. Tiến hành huấn luyện
print("\n--> [4/4] Bắt đầu quá trình tối ưu hóa trọng số mô hình...")
trainer.train()

# 6. Đóng gói và lưu mô hình đã fine-tune hoàn thiện
print(f"\nĐang lưu mô hình hoàn thiện vào: {OUTPUT_MODEL_DIR}...")
trainer.save_model(OUTPUT_MODEL_DIR)
tokenizer.save_pretrained(OUTPUT_MODEL_DIR)

print("\n" + "="*60)
print("[HOÀN TẤT] HUẤN LUYỆN BIOBERT THÀNH CÔNG!")
print(f"- Vị trí mô hình đã fine-tune: {OUTPUT_MODEL_DIR}")
print("- Bước tiếp theo: Chạy file test.py để đánh giá trên 100 câu Test.")
print("="*60)
