"""
DeepSeek 方面级情感标注脚本
从训练集中抽取 5000 条评论，用 DeepSeek API 批量打标
输出: aspects_labeled.csv (review + 6个方面标签)
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

# 输出目录
OUT_DIR = os.path.join(config.deepseek_root, "aspect_data")
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 方面定义
# ============================================================
ASPECTS = {
    "商品质量": "质量、做工、材质、耐用、正品、假货、瑕疵、包装完整性",
    "外观设计": "颜值、外观、颜色、款式、大小、尺寸、设计、手感",
    "使用体验": "功能、性能、速度、续航、电池、操作、流畅度、稳定性",
    "物流配送": "发货速度、快递、物流、配送、包装、送达时间",
    "价格": "价格、性价比、便宜、贵、实惠、划算、值不值",
    "客服服务": "客服、售后、态度、回复速度、退换货、服务",
}

ASPECT_KEYS = list(ASPECTS.keys())

# ============================================================
# Prompt 模板
# ============================================================
SYSTEM_PROMPT = """你是一个电商评论分析专家。对于每条评论，你需要判断6个方面是否被提及以及情感倾向。
规则:
- 1 = 正面 (好评/满意)
- 0 = 负面 (差评/不满意)
- -1 = 未提及 (评论没提到这个方面)

只返回JSON格式，不要输出其他内容。
JSON格式示例: {"商品质量":1,"外观设计":-1,"使用体验":0,"物流配送":1,"价格":-1,"客服服务":-1}"""


def build_user_prompt(review_text):
    aspects_desc = "\n".join([f"  - {k}: {v}" for k, v in ASPECTS.items()])
    return f"""请分析以下京东商品评论的6个方面情感:

方面说明:
{aspects_desc}

评论: "{review_text}"

请返回JSON (1=正面 0=负面 -1=未提及):"""


# ============================================================
# 单条调用
# ============================================================
def label_one(text):
    text = str(text).strip()
    if not text:
        return None

    for retry in range(3):
        try:
            resp = requests.post(
                config.api_url,
                headers={
                    "Authorization": f"Bearer {config.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": config.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": build_user_prompt(text)},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 200,
                    "response_format": {"type": "json_object"},  # 强制JSON输出
                },
                timeout=30,
                proxies={"http": None, "https": None},
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()

            # 解析 JSON
            result = json.loads(content)
            # 验证格式
            parsed = {}
            for k in ASPECT_KEYS:
                parsed[k] = int(result.get(k, -1))
            return parsed

        except (json.JSONDecodeError, KeyError):
            if retry < 2:
                time.sleep(1)
        except Exception:
            if retry < 2:
                time.sleep(1)

    return None


# ============================================================
# 批量标注
# ============================================================
def label_batch(texts, workers=20):
    results = [None] * len(texts)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(label_one, t): i for i, t in enumerate(texts)}
        for future in tqdm(as_completed(futures), total=len(futures), desc="DeepSeek 方面标注"):
            idx = futures[future]
            results[idx] = future.result()

    return results


# ============================================================
# 主流程
# ============================================================
if __name__ == "__main__":
    # 从原始训练集抽取 5000 条
    print("加载数据...")
    df = pd.read_csv(r"D:\dev\ec_review_analysis\Cleaned_data_for_fasttext_binary\train.csv")
    sample_size = min(5000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    print(f"抽取: {len(df_sample):,} 条")
    print(f"标签分布: {df_sample['label'].value_counts().to_dict()}")

    # 批量标注
    print("\n开始标注 (预计 3-5 分钟)...")
    results = label_batch(df_sample["review"])

    # 解析结果
    print("\n解析结果...")
    rows = []
    success = 0
    for i, r in enumerate(results):
        row = {"review": df_sample.iloc[i]["review"],
               "original_label": df_sample.iloc[i]["label"]}
        if r is not None:
            for k in ASPECT_KEYS:
                row[k] = r[k]
            success += 1
        else:
            for k in ASPECT_KEYS:
                row[k] = -1  # 失败的默认未提及
        rows.append(row)

    df_result = pd.DataFrame(rows)
    fail = len(df_result) - success

    # 统计
    print(f"\n成功: {success}  失败: {fail}")
    print("\n各维度的正面/负面/未提及分布:")
    for k in ASPECT_KEYS:
        pos = (df_result[k] == 1).sum()
        neg = (df_result[k] == 0).sum()
        noneed = (df_result[k] == -1).sum()
        print(f"  {k}: 正面={pos:>4,}  负面={neg:>4,}  未提及={noneed:>4,}")

    # 保存
    out_path = os.path.join(OUT_DIR, "aspects_labeled.csv")
    df_result.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n已保存: {out_path}")
    print(f"  rows: {len(df_result)}, cols: {list(df_result.columns)}")
