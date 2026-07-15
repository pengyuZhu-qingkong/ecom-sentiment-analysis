# -*- coding: utf-8 -*-
"""
FastText 二分类数据清洗脚本
=====================================
处理流程:
    1. 加载 full_train.csv (300万) + full_test.csv (25万)
    2. 过滤脏数据 (中文字符 < 5 或占比 < 10%)
    3. 标签映射: 1,2,3→0(差评)  4,5→1(好评)
    4. 文本去重
    5. 二分类标签均衡: 差评/好评各取等量
    6. 每个类别内部 8:1:1 划分 train/val/test
    7. 保存到 Cleaned_data_for_fasttext_binary/
"""

import pandas as pd
import numpy as np
import re
import os

# ============================================================
# 0. 全局配置
# ============================================================

BASE_DIR = r"D:\dev\ec_review_analysis"
RAW_DIR = os.path.join(BASE_DIR, "Initial_data")
OUT_DIR = os.path.join(BASE_DIR, "Cleaned_data_for_fasttext_binary")
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

    df = df[["label", "review"]]
    print(f"  原始行数: {len(df):,}")

    # ---- 过滤脏数据 ----
    before = len(df)
    mask = df["review"].apply(is_valid_review)
    df = df[mask].reset_index(drop=True)
    print(f"  过滤脏数据: {before:,} → {len(df):,} (丢掉 {before - len(df):,})")

    # ---- 标签映射 5→2 (二分类: 中评归入差评) ----
    label_map = {1: 0, 2: 0, 3: 0, 4: 1, 5: 1}
    df["label"] = df["label"].map(label_map)

    for lid, name in [(0, "差评"), (1, "好评")]:
        cnt = (df["label"] == lid).sum()
        print(f"  {name}: {cnt:,} ({cnt/len(df)*100:.1f}%)")

    # ---- 文本去重 ----
    before = len(df)
    df = df.drop_duplicates(subset="review", keep="first").reset_index(drop=True)
    print(f"  去重: {before:,} → {len(df):,} (丢掉 {before - len(df):,})")

    return df


def split_stratified(df, name_prefix):
    """二分类标签均衡 + 8:1:1 划分"""
    print(f"\n{'=' * 60}")
    print(f"[划分] {name_prefix}: 二分类均衡 + 8:1:1 划分")

    counts = df["label"].value_counts()
    min_count = counts.min()
    print(f"  各类数量: 差评={counts.get(0,0):,}  好评={counts.get(1,0):,}")
    print(f"  以最少类别为准: {min_count:,} / 类")

    # 每个类别取等量
    balanced_parts = []
    for lid in [0, 1]:
        label_df = df[df["label"] == lid]
        sampled = label_df.sample(n=min_count, random_state=RANDOM_SEED)
        balanced_parts.append(sampled)

    df_balanced = pd.concat(balanced_parts, ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    total = len(df_balanced)
    print(f"  均衡后总量: {total:,} (2 × {min_count:,})")

    # 每个类别内部 8:1:1
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

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    train_df = train_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"  train: {len(train_df):,}  (好评={sum(train_df['label']==1):,} 差评={sum(train_df['label']==0):,})")
    print(f"  val:   {len(val_df):,}  (好评={sum(val_df['label']==1):,} 差评={sum(val_df['label']==0):,})")
    print(f"  test:  {len(test_df):,}  (好评={sum(test_df['label']==1):,} 差评={sum(test_df['label']==0):,})")

    return train_df, val_df, test_df


def make_fasttext_txt(df, out_path):
    """将 DataFrame 导出为 FastText 训练格式 (jieba分词 + __label__)"""
    import jieba
    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            words = " ".join(jieba.lcut(row["review"])[:50])
            f.write(f"__label__{row['label']} {words}\n")


# ============================================================
# 主流程
# ============================================================

if __name__ == "__main__":
    import jieba

    # --- 处理训练集 ---
    df_train_raw = load_and_clean(
        os.path.join(RAW_DIR, "full_train.csv"), "训练集"
    )
    train_df, val_df, test_df = split_stratified(df_train_raw, "训练集")

    # --- 保存 CSV ---
    print(f"\n{'=' * 60}")
    print(f"[保存CSV] → {OUT_DIR}/")
    train_df.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False, encoding="utf-8")
    val_df.to_csv(os.path.join(OUT_DIR, "val.csv"), index=False, encoding="utf-8")
    test_df.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False, encoding="utf-8")

    # --- 生成 FastText txt (jieba分词 + __label__) ---
    print(f"\n{'=' * 60}")
    print(f"[生成FastText训练文件] → {OUT_DIR}/")
    print("  正在分词并生成 train.txt ...")
    make_fasttext_txt(train_df, os.path.join(OUT_DIR, "train.txt"))
    print("  正在分词并生成 val.txt ...")
    make_fasttext_txt(val_df, os.path.join(OUT_DIR, "val.txt"))
    print("  正在分词并生成 test.txt ...")
    make_fasttext_txt(test_df, os.path.join(OUT_DIR, "test.txt"))

    # --- 摘要 ---
    print(f"\n{'=' * 60}")
    print("完成! 文件清单:")
    print(f"  train.csv / train.txt    {len(train_df):>12,} 条")
    print(f"  val.csv   / val.txt      {len(val_df):>12,} 条")
    print(f"  test.csv  / test.txt     {len(test_df):>12,} 条")
    print(f"  合计:                    {len(train_df)+len(val_df)+len(test_df):>12,} 条")
