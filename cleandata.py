# -*- coding: utf-8 -*-
"""
电商用户评论智能分析系统 — 数据清洗脚本
===========================================
处理流程:
    1. 加载原始CSV, 只保留 label 和 review_text (丢掉商品名列)
    2. 过滤脏数据 (无中文字符 / 纯符号行)
    3. 标签映射: 5分类(1-5分) → 3分类(负面/中性/正面)
    4. 文本去重
    5. 分层抽样到 20万 条
    6. 按 8:1:1 划分 train / val / test
    7. 保存到 Cleaned_data/ 目录
"""

import pandas as pd
import numpy as np
import re
import os

# ============================================================
# 0. 全局配置
# ============================================================

# --- 文件路径 ---
RAW_DIR = "Initial_data"          # 原始数据目录
OUT_DIR = "Cleaned_data"          # 清洗后输出目录
INPUT_FILE = os.path.join(RAW_DIR, "full_train.csv")

# --- 参数 ---
TARGET_SIZE = 200_000             # 最终抽样数量
TRAIN_RATIO = 0.8                 # 训练集比例
VAL_RATIO   = 0.1                 # 验证集比例 (test = 1 - train - val)
RANDOM_SEED = 42                  # 随机种子, 保证每次运行结果一致

# 确保输出目录存在
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 1. 加载原始数据
# ============================================================
# CSV格式: "label","product_name","review_text"
# 第1列=评分(1-5), 第2列=商品名(不需要), 第3列=评论文本
print("=" * 60)
print("[步骤1] 加载原始数据...")

df = pd.read_csv(
    INPUT_FILE,
    header=None,                    # 没有表头
    names=["label", "product", "review"],  # 给三列起名
    dtype={"label": int, "product": str, "review": str},  # 指定数据类型
    encoding="utf-8",
    quoting=1,                      # QUOTE_ALL, 正确解析引号包裹的字段
    on_bad_lines="skip"             # 遇到解析异常的行直接跳过
)

# 丢弃第2列(商品名), 只要label和review
df = df[["label", "review"]]
print(f"  加载完成, 原始行数: {len(df):,}")

# ============================================================
# 2. 过滤脏数据
# ============================================================
# 脏数据特征:
#   - review为空或空白
#   - review全是符号(反引号/波浪号/句号等), 没有实际中文内容
#   - review中文字符太少 (< 5个), 不构成有效评论
print("\n[步骤2] 过滤脏数据...")

def is_valid_review(text):
    """判断一条评论是否为有效中文评论

    参数:
        text: 评论字符串

    返回:
        True  → 保留
        False → 丢弃
    """
    # 2a. 空值检查
    if not isinstance(text, str) or text.strip() == "":
        return False

    # 2b. 统计中文字符数量 (Unicode范围: 一-鿿, 覆盖常用汉字)
    chinese_chars = re.findall(r"[一-鿿]", text)
    chinese_count = len(chinese_chars)

    # 2c. 至少需要5个中文字符, 才算有效评论
    if chinese_count < 5:
        return False

    # 2d. 中文字符占比至少10%, 排除"~~~~~~~~"这类纯符号行
    total_len = max(len(text), 1)  # 避免除以0
    if chinese_count / total_len < 0.1:
        return False

    return True

# 应用过滤
before_filter = len(df)
df = df[df["review"].apply(is_valid_review)].reset_index(drop=True)
dropped = before_filter - len(df)
print(f"  过滤前: {before_filter:,} 条")
print(f"  过滤后: {len(df):,} 条")
print(f"  丢弃:   {dropped:,} 条 ({dropped/before_filter*100:.1f}%)")

# ============================================================
# 3. 标签映射: 5分类 → 3分类
# ============================================================
# 原始标签说明:
#   1分 = 非常不满意
#   2分 = 不满意
#   3分 = 一般
#   4分 = 满意
#   5分 = 非常满意
#
# 映射规则 (符合电商情感分析的通用做法):
#   1, 2 → 0 (负面 negative)
#   3    → 1 (中性 neutral)
#   4, 5 → 2 (正面 positive)
print("\n[步骤3] 标签映射: 5分类 → 3分类...")

# 定义映射字典
label_map = {
    1: 0,   # 负面
    2: 0,   # 负面
    3: 1,   # 中性
    4: 2,   # 正面
    5: 2,   # 正面
}

# 应用映射
df["label"] = df["label"].map(label_map)

# 检查是否有映射失败的值 (理论上不应该有)
if df["label"].isna().any():
    print("  ⚠ 警告: 存在无法映射的标签值!")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

# 打印各标签数量
label_names = {0: "负面(neg)", 1: "中性(neu)", 2: "正面(pos)"}
for label_id in [0, 1, 2]:
    count = (df["label"] == label_id).sum()
    print(f"  {label_names[label_id]}: {count:,} 条 ({count/len(df)*100:.1f}%)")

# ============================================================
# 4. 文本去重
# ============================================================
# 完全相同的评论文本只保留第一条
# 注意: 不同标签但相同文本也视为重复 (可能是数据错误)
print("\n[步骤4] 文本去重...")

before_dedup = len(df)
# subset="review": 按评论文本去重
# keep="first": 保留第一次出现的
df = df.drop_duplicates(subset="review", keep="first").reset_index(drop=True)
dedup_dropped = before_dedup - len(df)
print(f"  去重前: {before_dedup:,} 条")
print(f"  去重后: {len(df):,} 条")
print(f"  去除:   {dedup_dropped:,} 条 ({dedup_dropped/before_dedup*100:.1f}%)")

# ============================================================
# 5. 分层抽样到 20万
# ============================================================
# 按标签比例均匀抽样, 保证三个类别平衡
print(f"\n[步骤5] 分层抽样到 {TARGET_SIZE:,} 条...")

# 计算每个标签应该抽多少条 (平均分配)
unique_labels = sorted(df["label"].unique())
n_per_label = TARGET_SIZE // len(unique_labels)  # 20万 ÷ 3 ≈ 66,667

sampled_parts = []
for label_id in unique_labels:
    label_df = df[df["label"] == label_id]
    available = len(label_df)
    n_sample = min(n_per_label, available)  # 不够就全部取

    if available >= n_sample:
        sampled = label_df.sample(n=n_sample, random_state=RANDOM_SEED)
    else:
        # 如果某个类别不够, 全部保留
        print(f"  ⚠ 标签 {label_id} 只有 {available:,} 条, 不足 {n_per_label:,}, 全部保留")
        sampled = label_df

    sampled_parts.append(sampled)
    label_name = label_names.get(label_id, str(label_id))
    print(f"  标签{label_id} ({label_name}): 抽取 {len(sampled):,} / {available:,} 条")

df = pd.concat(sampled_parts, ignore_index=True)

# 随机打乱 (抽样后各标签是拼在一起的)
df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
print(f"  最终抽样数量: {len(df):,} 条")

# ============================================================
# 6. 划分数据集: train / val / test (8:1:1)
# ============================================================
print("\n[步骤6] 划分数据集 (train:val:test = 8:1:1)...")

# 先按标签分组, 每个标签内部分别划分, 保证各集合标签分布一致
train_parts, val_parts, test_parts = [], [], []

for label_id in unique_labels:
    label_df = df[df["label"] == label_id].copy()
    n_total = len(label_df)

    # 计算各部分的条数
    n_train = int(n_total * TRAIN_RATIO)
    n_val   = int(n_total * VAL_RATIO)
    # n_test = n_total - n_train - n_val  (剩下的全给test)

    # 随机打乱后切片
    label_df = label_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    train_parts.append(label_df.iloc[:n_train])
    val_parts.append(label_df.iloc[n_train:n_train + n_val])
    test_parts.append(label_df.iloc[n_train + n_val:])

# 合并各标签的数据
train_df = pd.concat(train_parts, ignore_index=True)
val_df   = pd.concat(val_parts, ignore_index=True)
test_df  = pd.concat(test_parts, ignore_index=True)

# 最后再打乱一次 (现在是按标签顺序的)
train_df = train_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
val_df   = val_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
test_df  = test_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

print(f"  训练集 (train): {len(train_df):,} 条")
print(f"  验证集 (val):   {len(val_df):,} 条")
print(f"  测试集 (test):  {len(test_df):,} 条")

# ============================================================
# 7. 保存结果
# ============================================================
print("\n[步骤7] 保存清洗后的数据...")

train_path = os.path.join(OUT_DIR, "train.csv")
val_path   = os.path.join(OUT_DIR, "val.csv")
test_path  = os.path.join(OUT_DIR, "test.csv")

# 保存为无表头、无索引的CSV (方便后续notebook直接用)
# header=False: 和原始数据格式保持一致
# index=False: 不保存pandas行号
train_df.to_csv(train_path, index=False, encoding="utf-8")
val_df.to_csv(val_path, index=False, encoding="utf-8")
test_df.to_csv(test_path, index=False, encoding="utf-8")

print(f"  ✓ {train_path}")
print(f"  ✓ {val_path}")
print(f"  ✓ {test_path}")

# ============================================================
# 8. 输出统计摘要
# ============================================================
print("\n" + "=" * 60)
print("清洗完成! 统计摘要:")
print("=" * 60)
print(f"{'':<12} {'总数':>8} {'负面(neg)':>10} {'中性(neu)':>10} {'正面(pos)':>10}")
print("-" * 60)

for name, data in [("train", train_df), ("val", val_df), ("test", test_df)]:
    neg = (data["label"] == 0).sum()
    neu = (data["label"] == 1).sum()
    pos = (data["label"] == 2).sum()
    print(f"{name:<12} {len(data):>8,} {neg:>10,} {neu:>10,} {pos:>10,}")

total = len(train_df) + len(val_df) + len(test_df)
print("-" * 60)
print(f"{'合计':<12} {total:>8,}")
print(f"\n文件已保存到: {os.path.abspath(OUT_DIR)}/")
