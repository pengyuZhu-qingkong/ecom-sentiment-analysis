"""
DeepSeek 联合标注: 方面级 + 三分类
从原始数据重新抽取5000条，一次性标注两个方面
"""
import pandas as pd
import numpy as np
import requests
import time
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from config import Config

config = Config()

OUT_DIR = os.path.join(config.deepseek_root, "aspect_3class_data")
os.makedirs(OUT_DIR, exist_ok=True)

ASPECTS = {
    "商品质量": "质量、做工、材质、耐用、正品、假货、瑕疵",
    "外观设计": "颜值、外观、颜色、款式、大小、尺寸、设计、手感",
    "使用体验": "功能、性能、速度、续航、电池、操作、流畅度、稳定性",
    "物流配送": "发货速度、快递、物流、配送、包装、送达时间",
    "价格": "价格、性价比、便宜、贵、实惠、划算、值不值",
    "客服服务": "客服、售后、态度、回复速度、退换货、服务",
}
ASPECT_KEYS = list(ASPECTS.keys())

SYSTEM_PROMPT = """你是电商评论分析专家。对每条评论做两件事:

1. 六方面情感标注 (1=正面 0=负面 -1=未提及)
2. 整体三分类 (2=好评 1=中评 0=差评)

只返回JSON: {"aspects":{"商品质量":1,"外观设计":-1,...}, "sentiment":2}"""


def build_user_prompt(text):
    aspects_desc = "\n".join([f"  {k}: {v}" for k, v in ASPECTS.items()])
    return f"""分析以下京东评论:

方面(1=正面 0=负面 -1=未提及):
{aspects_desc}

整体情感: 2=好评 1=中评 0=差评

评论: "{text}"

返回JSON:"""


def label_one(text):
    text = str(text).strip()
    if not text:
        return None
    for _ in range(3):
        try:
            resp = requests.post(
                config.api_url,
                headers={"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"},
                json={
                    "model": config.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": build_user_prompt(text)},
                    ],
                    "temperature": 0.0, "max_tokens": 200,
                    "response_format": {"type": "json_object"},
                },
                timeout=30, proxies={"http": None, "https": None},
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            data = json.loads(content)
            aspects = data["aspects"]
            sentiment = int(data["sentiment"])
            # 验证
            for k in ASPECT_KEYS:
                aspects[k] = int(aspects.get(k, -1))
            return {"aspects": aspects, "sentiment": sentiment}
        except:
            if _ < 2: time.sleep(1)
    return None


def label_batch(texts, workers=20):
    results = [None] * len(texts)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(label_one, t): i for i, t in enumerate(texts)}
        for f in tqdm(as_completed(futures), total=len(futures), desc="DeepSeek 联合标注"):
            results[futures[f]] = f.result()
    return results


if __name__ == "__main__":
    # 从原始数据重新抽取 5000 条（不同于之前的5000条）
    print("加载原始数据...")
    df = pd.read_csv(
        r"D:\dev\ec_review_analysis\Initial_data\full_train.csv",
        header=None, names=["label", "product", "review"],
        dtype={"label": int, "product": str, "review": str},
        encoding="utf-8", quoting=1, on_bad_lines="skip",
    )
    # 过滤脏数据
    import re
    def valid(text):
        if not isinstance(text, str) or text.strip() == "": return False
        cn = len(re.findall(r"[一-鿿]", text))
        return cn >= 5 and cn / max(len(text), 1) >= 0.1
    df = df[df["review"].apply(valid)].reset_index(drop=True)
    print(f"  清洗后: {len(df):,}")

    # 分层抽样 5000 条(按原始1-5评分均匀)
    sample_size = 5000
    per_label = sample_size // 5
    parts = []
    for rating in [1, 2, 3, 4, 5]:
        pool = df[df["label"] == rating]
        n = min(per_label, len(pool))
        parts.append(pool.sample(n=n, random_state=123))
    df_sample = pd.concat(parts, ignore_index=True)
    df_sample = df_sample.sample(frac=1, random_state=123).reset_index(drop=True)
    print(f"  抽取: {len(df_sample):,} (每评分{per_label}条)")

    # 标注
    print("\n开始联合标注 (预计3-5分钟)...")
    results = label_batch(df_sample["review"])

    # 解析
    rows = []
    for i, r in enumerate(results):
        row = {"review": df_sample.iloc[i]["review"],
               "original_rating": df_sample.iloc[i]["label"]}
        if r:
            row["sentiment"] = r["sentiment"]
            for k in ASPECT_KEYS:
                row[k] = r["aspects"][k]
        else:
            row["sentiment"] = -1
            for k in ASPECT_KEYS:
                row[k] = -1
        rows.append(row)

    df_result = pd.DataFrame(rows)
    fail = (df_result["sentiment"] == -1).sum()

    # 统计
    print(f"\n成功: {len(df_result)-fail}  失败: {fail}")
    print(f"\n三分类分布:")
    for lid, name in [(0, "差评"), (1, "中评"), (2, "好评")]:
        n = (df_result["sentiment"] == lid).sum()
        print(f"  {name}: {n} ({n/len(df_result)*100:.1f}%)")

    print(f"\n各方面分布:")
    for k in ASPECT_KEYS:
        pos = (df_result[k] == 1).sum()
        neg = (df_result[k] == 0).sum()
        noneed = (df_result[k] == -1).sum()
        print(f"  {k}: +{pos}  -{neg}  未提及{noneed}")

    # 保存
    out_path = os.path.join(OUT_DIR, "labeled_5000.csv")
    df_result.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n已保存: {out_path}")
