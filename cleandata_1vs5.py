# -*- coding: utf-8 -*-
"""
极简二分类: 只保留 1星(差评) vs 5星(好评)
=====================================
去除评分2/3/4，消除标签模糊地带。
"""

import pandas as pd
import re
import os

RAW_DIR = r"D:\dev\ec_review_analysis\Initial_data"
OUT_DIR = r"D:\dev\ec_review_analysis\bert\Cleaned_data_for_Bert\1vs5"
TARGET_SIZE = 200_000
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
RANDOM_SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)


def is_valid_review(text):
    if not isinstance(text, str) or text.strip() == "":
        return False
    chinese = re.findall(r"[一-鿿]", text)
    if len(chinese) < 5:
        return False
    if len(chinese) / max(len(text), 1) < 0.1:
        return False
    return True


print("=" * 60)
print("极简二分类: 1星 vs 5星")
print("=" * 60)

# 1. 加载
print(f"\n[1] 加载数据...")
df = pd.read_csv(
    os.path.join(RAW_DIR, "full_train.csv"),
    header=None,
    names=["label", "product", "review"],
    dtype={"label": int, "product": str, "review": str},
    encoding="utf-8",
    quoting=1,
    on_bad_lines="skip",
)
df = df[["label", "review"]]
print(f"  原始: {len(df):,}")

# 2. 只保留 1星 和 5星
print(f"\n[2] 只保留 1星 + 5星...")
df = df[df["label"].isin([1, 5])].reset_index(drop=True)
print(f"  1星: {sum(df['label']==1):,}  5星: {sum(df['label']==5):,}  合计: {len(df):,}")

# 3. 过滤脏数据
print(f"\n[3] 过滤脏数据...")
before = len(df)
df = df[df["review"].apply(is_valid_review)].reset_index(drop=True)
print(f"  {before:,} → {len(df):,}")

# 4. 标签映射 1→0, 5→1
print(f"\n[4] 标签映射 1→0(差评)  5→1(好评)...")
df["label"] = df["label"].map({1: 0, 5: 1})
print(f"  差评: {sum(df['label']==0):,}  好评: {sum(df['label']==1):,}")

# 5. 去重
print(f"\n[5] 文本去重...")
before = len(df)
df = df.drop_duplicates(subset="review", keep="first").reset_index(drop=True)
print(f"  {before:,} → {len(df):,}")

# 6. 均衡 + 抽样
print(f"\n[6] 均衡 + 抽样到 {TARGET_SIZE:,}...")
counts = df["label"].value_counts()
per_class = min(TARGET_SIZE // 2, counts.min())
print(f"  每类取 {per_class:,}")

parts = []
for lid in [0, 1]:
    label_df = df[df["label"] == lid]
    sampled = label_df.sample(n=per_class, random_state=RANDOM_SEED)
    parts.append(sampled)
    name = "差评" if lid == 0 else "好评"
    print(f"  {name}: {len(sampled):,}")

df_balanced = pd.concat(parts, ignore_index=True)
df_balanced = df_balanced.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
print(f"  合计: {len(df_balanced):,}")

# 7. 8:1:1 划分
print(f"\n[7] 划分 train/val/test...")
train_parts, val_parts, test_parts = [], [], []
for lid in [0, 1]:
    label_df = df_balanced[df_balanced["label"] == lid].copy()
    n = len(label_df)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    label_df = label_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    train_parts.append(label_df.iloc[:n_train])
    val_parts.append(label_df.iloc[n_train:n_train + n_val])
    test_parts.append(label_df.iloc[n_train + n_val:])

train_df = pd.concat(train_parts, ignore_index=True).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
val_df = pd.concat(val_parts, ignore_index=True).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
test_df = pd.concat(test_parts, ignore_index=True).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

print(f"  train: {len(train_df):,}  val: {len(val_df):,}  test: {len(test_df):,}")

# 8. 保存
print(f"\n[8] 保存 → {OUT_DIR}")
train_df.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False, encoding="utf-8")
val_df.to_csv(os.path.join(OUT_DIR, "val.csv"), index=False, encoding="utf-8")
test_df.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False, encoding="utf-8")

print(f"  ✓ train.csv  {len(train_df):,}")
print(f"  ✓ val.csv    {len(val_df):,}")
print(f"  ✓ test.csv   {len(test_df):,}")
print("完成!")
