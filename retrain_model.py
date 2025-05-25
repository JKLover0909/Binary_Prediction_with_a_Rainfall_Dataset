import sys
import os
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from datetime import datetime

MODEL_PATH = 'catboost_rainfall_model.cbm'
LOG_PATH = 'retrain_log.txt'

# Lấy danh sách file csv từ argument
csv_files = sys.argv[1:]
if not csv_files:
    print("Không có file CSV nào được truyền vào.")
    exit(1)

# Đọc và gộp dữ liệu từ các file csv được truyền vào
df_list = [pd.read_csv(f) for f in csv_files]
df = pd.concat(df_list, ignore_index=True)

X = df.drop(columns=['rainfall'])
y = df['rainfall']

# Chia train/val để đánh giá
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Nếu đã có mô hình thì load, không thì khởi tạo mới
if os.path.exists(MODEL_PATH):
    print("Đang load mô hình cũ để huấn luyện tiếp...")
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    model.fit(X_train, y_train, init_model=MODEL_PATH, verbose=0)
else:
    print("Không tìm thấy mô hình cũ, khởi tạo mô hình mới...")
    model = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, verbose=0)
    model.fit(X_train, y_train)

# Đánh giá mô hình
y_pred = model.predict(X_val)
y_proba = model.predict_proba(X_val)[:, 1]

auc = roc_auc_score(y_val, y_proba)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)

# Lưu lại mô hình đã huấn luyện
model.save_model(MODEL_PATH)

# Ghi log
with open(LOG_PATH, 'a') as f:
    f.write(f"=== Retrain at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    f.write(f"Files used: {', '.join(csv_files)}\n")
    f.write(f"AUC: {auc:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 score: {f1:.4f}\n")
    f.write("========================================\n\n")

print("Đã lưu mô hình vào", MODEL_PATH)
print("AUC:", auc)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)
print("Đã ghi log vào", LOG_PATH)