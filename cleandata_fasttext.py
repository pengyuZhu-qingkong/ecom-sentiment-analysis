# -*- coding: utf-8 -*-
"""
FastText 数据清洗脚本
=====================================
处理流程:
    1. 加载 full_train.csv (300万) + full_test.csv (25万)
    2. 过滤脏数据 (中文字符 < 5 或占比 < 10%)
    3. 标签映射: 5分类(1-5分) → 3分类(负面/中性/正面)
    4. 文本去重
    5. 标签均衡: 3个类别各取相同数量
    6. 每个类别内部 8:1:1 划分 train/val/test
    7. 保存到 Cleaned_data_for_fasttext/
"""

import pandas as pd
import numpy as np
import re
import os

# ============================================================
# 0. 全局配置
# ============================================================

RAW_DIR = "Initial_data"
OUT_DIR = "Cleaned_data_for_fasttext"
TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
RANDOM_SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)


def is_valid_review(text):
    """判断是否为有效中文评论"""
    if not isinstance(text, str) or text.strip() == "":
        return False
    chinese_chars = re.findall(r"[一-鿿]", text)
    if len(chinese_chars) < 5:
        return False
    if len(chinese_chars) / max(len(text), 1) < 0.1:
        return False
    return True


def load_and_clean(csv_path, name):
    """加载并清洗单个CSV文件"""
    print(f"\n{'=' * 60}")
    print(f"[加载] {name}: {csv_path}")

    df = pd.read_csv(
        csv_path,
        header=None,
        names=["label", "product", "review"],
        dtype={"label": int, "product": str, "review": str},
        encoding="utf-8",
        quoting=1,
        on_bad_lines="skip",
    )

    # 丢掉商品名列
    df = df[["label", "review"]]
    print(f"  原始行数: {len(df):,}")

    # ---- 过滤脏数据 ----
    before = len(df)
    mask = df["review"].apply(is_valid_review)
    df = df[mask].reset_index(drop=True)
    print(f"  过滤脏数据: {before:,} → {len(df):,} (丢掉 {before - len(df):,})")

    # ---- 标签映射 5→3 ----
    label_map = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}
    df["label"] = df["label"].map(label_map)

    label_names = {0: "负面(neg)", 1: "中性(neu)", 2: "正面(pos)"}
    for lid in [0, 1, 2]:
        cnt = (df["label"] == lid).sum()
        print(f"  {label_names[lid]}: {cnt:,} ({cnt/len(df)*100:.1f}%)")

    # ---- 文本去重 ----
    before = len(df)
    df = df.drop_duplicates(subset="review", keep="first").reset_index(drop=True)
    print(f"  去重: {before:,} → {len(df):,} (丢掉 {before - len(df):,})")

    return df


def split_stratified(df, name_prefix):
    """按标签分层划分 train/val/test (8:1:1)，每类取等量"""
    print(f"\n{'=' * 60}")
    print(f"[划分] {name_prefix}: 标签均衡 + 8:1:1 划分")

    # 1. 先算每个类别最少有多少条
    counts = df["label"].value_counts()
    min_count = counts.min()
    print(f"  各类数量: {counts.to_dict()}")
    print(f"  以最少类别为准: {min_count:,} / 类")

    # 2. 每个类别取等量
    balanced_parts = []
    for lid in [0, 1, 2]:
        label_df = df[df["label"] == lid]
        sampled = label_df.sample(n=min_count, random_state=RANDOM_SEED)
        balanced_parts.append(sampled)

    df_balanced = pd.concat(balanced_parts, ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    total = len(df_balanced)
    print(f"  均衡后总量: {total:,} (3 × {min_count:,})")

    # 3. 每个类别内部分别 8:1:1
    train_parts, val_parts, test_parts = [], [], []

    for lid in [0, 1, 2]:
        label_df = df_balanced[df_balanced["label"] == lid].copy()
        n = len(label_df)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)

        label_df = label_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

        train_parts.append(label_df.iloc[:n_train])
        val_parts.append(label_df.iloc[n_train:n_train + n_val])
        test_parts.append(label_df.iloc[n_train + n_val:])

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    # 最终打乱
    train_df = train_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"  train: {len(train_df):,}  (pos={sum(train_df['label']==2):,} neg={sum(train_df['label']==0):,} neu={sum(train_df['label']==1):,})")
    print(f"  val:   {len(val_df):,}  (pos={sum(val_df['label']==2):,} neg={sum(val_df['label']==0):,} neu={sum(val_df['label']==1):,})")
    print(f"  test:  {len(test_df):,}  (pos={sum(test_df['label']==2):,} neg={sum(test_df['label']==0):,} neu={sum(test_df['label']==1):,})")

    return train_df, val_df, test_df


# ============================================================
# 主流程
# ============================================================

if __name__ == "__main__":
    # --- 处理训练集 ---
    df_train_raw = load_and_clean(
        os.path.join(RAW_DIR, "full_train.csv"), "训练集"
    )
    train_df, val_df, test_df = split_stratified(df_train_raw, "训练集")

    # --- 处理测试集 (full_test.csv) ---
    df_test_extra = load_and_clean(
        os.path.join(RAW_DIR, "full_test.csv"), "额外测试集"
    )

    # --- 保存 ---
    print(f"\n{'=' * 60}")
    print(f"[保存] → {OUT_DIR}/")

    train_df.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False, encoding="utf-8")
    val_df.to_csv(os.path.join(OUT_DIR, "val.csv"), index=False, encoding="utf-8")
    test_df.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False, encoding="utf-8")
    df_test_extra.to_csv(os.path.join(OUT_DIR, "test_extra.csv"), index=False, encoding="utf-8")

    # --- 摘要 ---
    print(f"\n{'=' * 60}")
    print("完成! 文件清单:")
    print(f"  train.csv       {len(train_df):>12,} 条")
    print(f"  val.csv         {len(val_df):>12,} 条")
    print(f"  test.csv        {len(test_df):>12,} 条")
    print(f"  test_extra.csv  {len(df_test_extra):>12,} 条 (原始测试集, 供最终评估)")
    total = len(train_df) + len(val_df) + len(test_df)
    print(f"  {'合计':<16} {total:>12,} 条")
