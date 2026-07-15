# -*- coding: utf-8 -*-
"""
BERT 五分类数据清洗 (1星→0, 2星→1, 3星→2, 4星→3, 5星→4)
=====================================
标签无映射歧义，保留原始评分信息。
"""

import pandas as pd
import re
import os

RAW_DIR = r"D:\dev\ec_review_analysis\Initial_data"
OUT_DIR = r"D:\dev\ec_review_analysis\bert\Cleaned_data_for_Bert\5class"
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
print("BERT 五分类数据清洗")
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
print(f"  原始: {len(df):,} 条")

# 2. 过滤脏数据
print(f"\n[2] 过滤脏数据...")
before = len(df)
df = df[df["review"].apply(is_valid_review)].reset_index(drop=True)
print(f"  {before:,} → {len(df):,}")

# 3. 标签映射 1→0, 2→1, 3→2, 4→3, 5→4
print(f"\n[3] 标签映射 1-5 → 0-4...")
df["label"] = df["label"] - 1
for lid in [0, 1, 2, 3, 4]:
    cnt = (df["label"] == lid).sum()
    print(f"  {lid+1}星→{lid}: {cnt:,} ({cnt/len(df)*100:.1f}%)")

# 4. 去重
print(f"\n[4] 文本去重...")
before = len(df)
df = df.drop_duplicates(subset="review", keep="first").reset_index(drop=True)
print(f"  {before:,} → {len(df):,}")

# 5. 均衡 + 抽样
print(f"\n[5] 均衡 + 抽样到 {TARGET_SIZE:,}...")
n_per_class = TARGET_SIZE // 5
parts = []
for lid in range(5):
    label_df = df[df["label"] == lid]
    n = min(n_per_class, len(label_df))
    sampled = label_df.sample(n=n, random_state=RANDOM_SEED)
    parts.append(sampled)
    print(f"  {lid+1}星: {n:,}")

df_balanced = pd.concat(parts, ignore_index=True)
df_balanced = df_balanced.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
print(f"  合计: {len(df_balanced):,}")

# 6. 8:1:1 划分
print(f"\n[6] 划分...")
train_parts, val_parts, test_parts = [], [], []
for lid in range(5):
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

# 7. 保存
print(f"\n[7] 保存 → {OUT_DIR}")
train_df.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False, encoding="utf-8")
val_df.to_csv(os.path.join(OUT_DIR, "val.csv"), index=False, encoding="utf-8")
test_df.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False, encoding="utf-8")

print(f"  ✓ train.csv  {len(train_df):,}")
print(f"  ✓ val.csv    {len(val_df):,}")
print(f"  ✓ test.csv   {len(test_df):,}")
print("完成!")
