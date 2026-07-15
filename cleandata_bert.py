# -*- coding: utf-8 -*-
"""
BERT 数据清洗脚本
=====================================
处理流程:
    1. 加载 full_train.csv (300万)
    2. 过滤脏数据
    3. 标签映射: 三分类(1,2→0 / 3→1 / 4,5→2) 或 二分类(1,2,3→0 / 4,5→1)
    4. 文本去重
    5. 标签均衡 + 抽样到 20万
    6. 8:1:1 划分 train/val/test
    7. 保存到 bert/Cleaned_data_for_Bert/  (CSV格式: label + review原始文本)

BERT 用 AutoTokenizer，不需要 jieba 分词，保留原始 review 即可。
"""

import pandas as pd
import re
import os

# ============================================================
# 0. 全局配置
# ============================================================

RAW_DIR = r"D:\dev\ec_review_analysis\Initial_data"
OUT_BASE = r"D:\dev\ec_review_analysis\bert\Cleaned_data_for_Bert"
TARGET_SIZE = 200_000
TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
RANDOM_SEED = 42


def is_valid_review(text):
    if not isinstance(text, str) or text.strip() == "":
        return False
    chinese_chars = re.findall(r"[一-鿿]", text)
    if len(chinese_chars) < 5:
        return False
    if len(chinese_chars) / max(len(text), 1) < 0.1:
        return False
    return True


def clean(mode="3class"):
    """
    mode: "3class" → 标签 0=负 1=中 2=正
          "binary" → 标签 0=差评 1=好评 (中评归差评)
    """
    out_dir = os.path.join(OUT_BASE, mode)
    os.makedirs(out_dir, exist_ok=True)

    # ---- 标签映射 ----
    if mode == "3class":
        label_map = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}
        label_names = {0: "负面", 1: "中性", 2: "正面"}
        n_classes = 3
    else:
        label_map = {1: 0, 2: 0, 3: 0, 4: 1, 5: 1}
        label_names = {0: "差评", 1: "好评"}
        n_classes = 2

    # ---- 加载 ----
    print("=" * 60)
    print(f"BERT 数据清洗 [{mode}]")
    print("=" * 60)
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

    # ---- 过滤脏数据 ----
    print(f"\n[2] 过滤脏数据...")
    before = len(df)
    df = df[df["review"].apply(is_valid_review)].reset_index(drop=True)
    print(f"  {before:,} → {len(df):,} (丢掉 {before - len(df):,})")

    # ---- 标签映射 ----
    print(f"\n[3] 标签映射 5→{n_classes}...")
    df["label"] = df["label"].map(label_map)
    for lid, lname in label_names.items():
        cnt = (df["label"] == lid).sum()
        print(f"  {lname}: {cnt:,} ({cnt/len(df)*100:.1f}%)")

    # ---- 去重 ----
    print(f"\n[4] 文本去重...")
    before = len(df)
    df = df.drop_duplicates(subset="review", keep="first").reset_index(drop=True)
    print(f"  {before:,} → {len(df):,} (丢掉 {before - len(df):,})")

    # ---- 标签均衡 + 抽样到 20万 ----
    print(f"\n[5] 标签均衡 + 抽样到 {TARGET_SIZE:,}...")
    counts = df["label"].value_counts()
    min_count = counts.min()
    per_class = min(TARGET_SIZE // n_classes, min_count)
    print(f"  各类最少: {min_count:,}  每类抽取: {per_class:,}")

    parts = []
    for lid in sorted(label_names.keys()):
        label_df = df[df["label"] == lid]
        sampled = label_df.sample(n=per_class, random_state=RANDOM_SEED)
        parts.append(sampled)
        print(f"  {label_names[lid]}: {len(sampled):,}")

    df_balanced = pd.concat(parts, ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    total = len(df_balanced)
    print(f"  均衡后: {total:,}")

    # ---- 8:1:1 划分 ----
    print(f"\n[6] 划分 train/val/test (8:1:1)...")
    train_parts, val_parts, test_parts = [], [], []

    for lid in sorted(label_names.keys()):
        label_df = df_balanced[df_balanced["label"] == lid].copy()
        n = len(label_df)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        label_df = label_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        train_parts.append(label_df.iloc[:n_train])
        val_parts.append(label_df.iloc[n_train:n_train + n_val])
        test_parts.append(label_df.iloc[n_train + n_val:])

    train_df = pd.concat(train_parts, ignore_index=True).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    val_df   = pd.concat(val_parts, ignore_index=True).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    test_df  = pd.concat(test_parts, ignore_index=True).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    for name, data in [("train", train_df), ("val", val_df), ("test", test_df)]:
        dist = ", ".join([f"{label_names[lid]}={sum(data['label']==lid):,}" for lid in sorted(label_names.keys())])
        print(f"  {name}: {len(data):,}  ({dist})")

    # ---- 保存 ----
    print(f"\n[7] 保存 → {out_dir}/")
    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False, encoding="utf-8")
    val_df.to_csv(os.path.join(out_dir, "val.csv"), index=False, encoding="utf-8")
    test_df.to_csv(os.path.join(out_dir, "test.csv"), index=False, encoding="utf-8")

    print(f"  ✓ train.csv  {len(train_df):,} 条")
    print(f"  ✓ val.csv    {len(val_df):,} 条")
    print(f"  ✓ test.csv   {len(test_df):,} 条")
    print(f"  合计:         {len(train_df)+len(val_df)+len(test_df):,} 条")
    print()


if __name__ == "__main__":
    clean("3class")
    clean("binary")
