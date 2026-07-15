"""
验证: 方面级反推三分类
规则: 6个方面中非-1的值求和
  sum > 0  → 好评(2)
  sum < 0  → 差评(0)
  sum = 0  → 中评(1)
"""
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 加载方面级测试集
df = pd.read_csv(r"D:\dev\ec_review_analysis\bert\Cleaned_data_for_Bert\aspect\test.csv")
ASPECTS = ["商品质量", "外观设计", "使用体验", "物流配送", "价格", "客服服务"]

# 反推三分类标签
def aspect_to_label(row):
    s = 0
    count = 0
    for a in ASPECTS:
        v = row[a]
        if v != -1:  # 只计算被提及的方面
            s += v
            count += 1
    if count == 0:   # 全部未提及
        return 1      # 中性
    if s > 0:
        return 2      # 好评
    elif s < 0:
        return 0      # 差评
    else:
        return 1      # 中性 (正负打平)

df["pred_3class"] = df.apply(aspect_to_label, axis=1)
df["true_3class"] = df["original_label"]  # 真实的二分类标签，需要映射到三分类

# 统计
print("预测分布:")
for lid, name in [(0, "差评"), (1, "中评"), (2, "好评")]:
    n = (df["pred_3class"] == lid).sum()
    print(f"  {name}: {n} ({n/len(df)*100:.1f}%)")

print(f"\n真实标签分布 (原始):")
for lid in [0, 1]:
    n = (df["true_3class"] == lid).sum()
    name = "差评" if lid == 0 else "好评"
    print(f"  {name}: {n} ({n/len(df)*100:.1f}%)")

# 由于 original_label 是二分类(0/1)，需要将其映射到三分类来评估
# 0(差评) → 0, 1(好评) → 2
df["true_mapped"] = df["true_3class"].map({0: 0, 1: 2})

print(f"\n=== 方面级反推 vs 真实标签 ===")
acc = accuracy_score(df["true_mapped"], df["pred_3class"])
f1 = f1_score(df["true_mapped"], df["pred_3class"], average="macro")
print(f"Accuracy: {acc:.4f}")
print(f"F1: {f1:.4f}")
print(f"\n{classification_report(df['true_mapped'], df['pred_3class'], target_names=['差评', '中评', '好评'])}")

# 看看每个方面的贡献
print("\n各维度正负分布:")
for a in ASPECTS:
    pos = (df[a] == 1).sum()
    neg = (df[a] == 0).sum()
    noneed = (df[a] == -1).sum()
    print(f"  {a}: +{pos}  -{neg}  未提及{noneed}")
