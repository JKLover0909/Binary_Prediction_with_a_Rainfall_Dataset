import sys
import os
import pandas as pd
from catboost import CatBoostClassifier

MODEL_PATH = 'catboost_rainfall_model.cbm'

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

# Nếu đã có mô hình thì load, không thì khởi tạo mới
if os.path.exists(MODEL_PATH):
    print("Đang load mô hình cũ để huấn luyện tiếp...")
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    model.fit(X, y, init_model=MODEL_PATH, verbose=100)
else:
    print("Không tìm thấy mô hình cũ, khởi tạo mô hình mới...")
    model = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, verbose=100)
    model.fit(X, y)

# Lưu lại mô hình đã huấn luyện
model.save_model(MODEL_PATH)
print("Đã lưu mô hình vào", MODEL_PATH)