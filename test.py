import os
import csv
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Thiết lập đường dẫn động
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed_data")
MODEL_PATH = os.path.join(BASE_DIR, "models", "biobert_finetuned")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CSV_OUTPUT = os.path.join(RESULTS_DIR, "predictions.csv")
SUMMARY_OUTPUT = os.path.join(RESULTS_DIR, "evaluation_summary.txt")

print("="*65)
print("     ĐÁNH GIÁ MÔ HÌNH BIOBERT TRÊN TẬP KIỂM THỬ (TEST SET)")
print("="*65)

# 2. Nạp tập Test (100 câu)
print(f"[1/4] Đang nạp tập dữ liệu kiểm thử từ: {DATA_PATH}...")
dataset = load_from_disk(DATA_PATH)
test_dataset = dataset["test"]
print(f"--> Số lượng câu hỏi kiểm thử: {len(test_dataset)} câu.")

# 3. Nạp mô hình BioBERT đã qua huấn luyện
print(f"[2/4] Đang nạp mô hình đã fine-tune từ: {MODEL_PATH}...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print(f"--> Đang chạy suy luận trên thiết bị: {device.type.upper()}")

# 4. Chạy suy luận (Inference) không tính gradient
print("\n--> [3/4] Đang thực hiện dự đoán trên 100 câu hỏi y khoa...")

id2label = {0: "yes", 1: "no", 2: "maybe"}
predictions = []
probabilities = []

with torch.no_grad():
    for item in test_dataset:
        input_ids = torch.tensor([item["input_ids"]]).to(device)
        attention_mask = torch.tensor([item["attention_mask"]]).to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).squeeze().tolist()
        pred_id = int(torch.argmax(logits, dim=-1).item())
        
        predictions.append(pred_id)
        probabilities.append(probs)

true_labels = test_dataset["labels"]

# 5. Tính toán các độ đo chuẩn: Accuracy, Precision, Recall, Macro-F1
print("--> [4/4] Đang tính toán độ đo và ma trận nhầm lẫn...")

total = len(true_labels)
correct = sum(1 for t, p in zip(true_labels, predictions) if t == p)
accuracy = (correct / total) * 100.0

classes = [0, 1, 2]
class_names = ["yes", "no", "maybe"]
class_metrics = {}
f1_list = []

# Ma trận nhầm lẫn 3x3: conf_matrix[true][pred]
conf_matrix = {c: {p: 0 for p in classes} for c in classes}
for t, p in zip(true_labels, predictions):
    conf_matrix[t][p] += 1

for c in classes:
    tp = conf_matrix[c][c]
    fp = sum(conf_matrix[other][c] for other in classes if other != c)
    fn = sum(conf_matrix[c][other] for other in classes if other != c)
    support = sum(conf_matrix[c].values())
    
    prec = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
    rec = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    
    f1_list.append(f1)
    class_metrics[class_names[c]] = {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "support": support
    }

macro_f1 = sum(f1_list) / len(f1_list)

# 6. Xuất bảng dự đoán chi tiết ra CSV
with open(CSV_OUTPUT, mode="w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow([
        "STT", "PubMed_ID", "Question", "Ground_Truth", 
        "BioBERT_Prediction", "Is_Correct", "Prob_Yes", "Prob_No", "Prob_Maybe"
    ])
    for idx, (item, p_id, probs) in enumerate(zip(test_dataset, predictions, probabilities), start=1):
        t_label = id2label[item["labels"]]
        p_label = id2label[p_id]
        is_corr = "DUNG" if t_label == p_label else "SAI"
        writer.writerow([
            idx,
            item["pubid"],
            item["question"],
            t_label,
            p_label,
            is_corr,
            f"{probs[0]:.4f}",
            f"{probs[1]:.4f}",
            f"{probs[2]:.4f}"
        ])

# 7. Xuất bản tóm tắt kết quả (Summary)
summary_text = f"""=================================================================
          BÁO CÁO KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH BIOBERT (PUBMEDQA)
=================================================================
1. TỔNG QUAN HIỆU NĂNG:
   - Tổng số mẫu kiểm thử : {total} câu
   - Số câu trả lời đúng  : {correct} / {total}
   - Accuracy (Độ chính xác) : {accuracy:.2f}%
   - Macro-F1               : {macro_f1:.2f}%

2. ĐỐI SÁNH VỚI BASELINE BÀI BÁO GỐC (Jin et al., 2019):
   +-----------------------------------+--------------+--------------+
   | Phương pháp                       | Accuracy (%) | Macro-F1 (%) |
   +-----------------------------------+--------------+--------------+
   | Majority Baseline                 | 55.20%       |     --       |
   | BioBERT Baseline (Bài báo gốc)    | 68.10%       | 52.70%       |
   | BioBERT Nhóm 11 (Tự thực nghiệm)  | {accuracy:5.2f}%       | {macro_f1:5.2f}%       |
   | Human Expert (Chuyên gia y tế)    | 78.00%       |     --       |
   +-----------------------------------+--------------+--------------+

3. CHI TIẾT TỪNG LỚP DỰ ĐOÁN:
   +-------+-----------+--------+--------+---------+
   | Lớp   | Precision | Recall | F1     | Support |
   +-------+-----------+--------+--------+---------+
   | yes   | {class_metrics['yes']['precision']:6.2f}%   | {class_metrics['yes']['recall']:5.2f}% | {class_metrics['yes']['f1']:5.2f}% |   {class_metrics['yes']['support']:3d}   |
   | no    | {class_metrics['no']['precision']:6.2f}%   | {class_metrics['no']['recall']:5.2f}% | {class_metrics['no']['f1']:5.2f}% |   {class_metrics['no']['support']:3d}   |
   | maybe | {class_metrics['maybe']['precision']:6.2f}%   | {class_metrics['maybe']['recall']:5.2f}% | {class_metrics['maybe']['f1']:5.2f}% |   {class_metrics['maybe']['support']:3d}   |
   +-------+-----------+--------+--------+---------+

4. MA TRẬN NHẦM LẪN (CONFUSION MATRIX):
   (Hàng: Nhãn thực tế, Cột: Nhãn BioBERT dự đoán)
             Pred: yes   Pred: no   Pred: maybe
   True: yes     {conf_matrix[0][0]:3d}        {conf_matrix[0][1]:3d}         {conf_matrix[0][2]:3d}
   True: no      {conf_matrix[1][0]:3d}        {conf_matrix[1][1]:3d}         {conf_matrix[1][2]:3d}
   True: maybe   {conf_matrix[2][0]:3d}        {conf_matrix[2][1]:3d}         {conf_matrix[2][2]:3d}
=================================================================
"""

with open(SUMMARY_OUTPUT, mode="w", encoding="utf-8") as f:
    f.write(summary_text)

print(summary_text)
print(f"[THÀNH CÔNG] Đã xuất file dự đoán chi tiết: {CSV_OUTPUT}")
print(f"[THÀNH CÔNG] Đã xuất file tóm tắt báo cáo : {SUMMARY_OUTPUT}")
