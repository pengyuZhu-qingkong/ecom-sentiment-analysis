"""拆分联合标注数据: 三分类均衡 + 方面级"""
import pandas as pd
import os

IN_PATH = r"D:\dev\ec_review_analysis\deepseek_api\aspect_3class_data\labeled_5000.csv"
OUT_DIR = r"D:\dev\ec_review_analysis\bert\Cleaned_data_for_Bert\aspect_3class"
RANDOM_SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(IN_PATH)
ASPECTS = ["商品质量", "外观设计", "使用体验", "物流配送", "价格", "客服服务"]
print(f"原始: {len(df):,}")

# 均衡三分类: 取 min(差评,中评,好评)
counts = df["sentiment"].value_counts()
per_class = min(counts[0], counts[1], counts[2])  # ~1324
print(f"每类取 {per_class} (差评={counts.get(0,0)}, 中评={counts.get(1,0)}, 好评={counts.get(2,0)})")

parts = []
for lid, name in [(0, "差评"), (1, "中评"), (2, "好评")]:
    pool = df[df["sentiment"] == lid]
    sampled = pool.sample(n=per_class, random_state=RANDOM_SEED)
    parts.append(sampled)
    print(f"  {name}: {len(sampled)}")

df_balanced = pd.concat(parts, ignore_index=True)
df_balanced = df_balanced.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
total = len(df_balanced)
print(f"均衡后: {total:,} (3 × {per_class:,})")

# 8:1:1 划分
n_train = int(total * 0.8)
n_val = int(total * 0.1)

train_df = df_balanced.iloc[:n_train].reset_index(drop=True)
val_df = df_balanced.iloc[n_train:n_train+n_val].reset_index(drop=True)
test_df = df_balanced.iloc[n_train+n_val:].reset_index(drop=True)

for name, data in [("train", train_df), ("val", val_df), ("test", test_df)]:
    data.to_csv(os.path.join(OUT_DIR, f"{name}.csv"), index=False, encoding="utf-8")
    # 统计三分类
    sc = data["sentiment"].value_counts().sort_index()
    a_stats = ", ".join([f"{k}:+{(data[k]==1).sum()}/-{(data[k]==0).sum()}" for k in ASPECTS[:3]])
    print(f"  {name}: {len(data):,}  (差评={sc.get(0,0)} 中评={sc.get(1,0)} 好评={sc.get(2,0)})")

print(f"\n已保存到: {OUT_DIR}")
