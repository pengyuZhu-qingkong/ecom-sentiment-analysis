"""拆分方面级标注数据: 4000 train / 500 val / 500 test"""
import pandas as pd
import os

IN_PATH = r"D:\dev\ec_review_analysis\deepseek_api\aspect_data\aspects_labeled.csv"
OUT_DIR = r"D:\dev\ec_review_analysis\bert\Cleaned_data_for_Bert\aspect"
RANDOM_SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(IN_PATH)
df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

n_train, n_val = 4000, 500
train_df = df.iloc[:n_train].reset_index(drop=True)
val_df = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
test_df = df.iloc[n_train + n_val:].reset_index(drop=True)

print(f"train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")

for name, data in [("train", train_df), ("val", val_df), ("test", test_df)]:
    data.to_csv(os.path.join(OUT_DIR, f"{name}.csv"), index=False, encoding="utf-8")
    print(f"  {name}.csv saved")

print(f"\n已保存到: {OUT_DIR}")
