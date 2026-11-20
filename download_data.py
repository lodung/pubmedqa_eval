from datasets import load_dataset
import os

os.environ["HF_DATASETS_OFFLINE"] = "1"
print("Đang tải tập dữ liệu PubMedQA về máy...")

# Tải và lưu thẳng vào thư mục ./data
dataset = load_dataset(
    "qiaojin/PubMedQA", 
    "pqa_labeled", 
    cache_dir="./data"
)

print("\n--> [THÀNH CÔNG] Dữ liệu đã được tải về!")
print(f"Tổng số câu hỏi tải được: {len(dataset['train'])} câu.")

# In thử câu hỏi đầu tiên để kiểm tra
print("\n--- XEM THỬ CÂU ĐẦU TIÊN ---")
print("Câu hỏi:", dataset["train"][0]["question"])
print("Đáp án chuẩn:", dataset["train"][0]["final_decision"])