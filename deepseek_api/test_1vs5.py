"""快速验证: 1星 vs 5星 二分类"""
import pandas as pd
import numpy as np
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
from config import Config

config = Config()

df = pd.read_csv(r"D:\dev\ec_review_analysis\bert\Cleaned_data_for_Bert\5class\test_15_only.csv")
print(f"测试集: {len(df):,}  (差评=4000, 好评=4000)")


def predict_one(text):
    text = str(text).strip()
    if not text:
        return -1
    for _ in range(3):
        try:
            resp = requests.post(
                config.api_url,
                headers={"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"},
                json={
                    "model": config.model,
                    "messages": [
                        {"role": "system", "content": "只输出0或1。"},
                        {"role": "user", "content": f"判断评论情感，0=差评 1=好评。只输出一个数字。\n评论: {text}"},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 5,
                },
                timeout=30,
                proxies={"http": None, "https": None},
            )
            resp.raise_for_status()
            r = resp.json()["choices"][0]["message"]["content"].strip()
            if "1" in r:
                return 1
            if "0" in r:
                return 0
            return -1
        except Exception:
            time.sleep(1)
    return -1


results = [-1] * len(df)
with ThreadPoolExecutor(max_workers=20) as ex:
    futures = {ex.submit(predict_one, t): i for i, t in enumerate(df["review"])}
    for f in tqdm(as_completed(futures), total=len(futures), desc="DeepSeek 1vs5"):
        results[futures[f]] = f.result()

mask = np.array(results) != -1
y_true = df["label"].values[mask]
y_pred = np.array(results)[mask]

print(f"\n=== DeepSeek 1星 vs 5星 ===")
print(f"有效: {len(y_true)}  (失败: {len(df)-len(y_true)})")
print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
print(f"F1: {f1_score(y_true, y_pred, average='macro'):.4f}")
print(f"\n{classification_report(y_true, y_pred, target_names=['差评(1星)', '好评(5星)'])}")
