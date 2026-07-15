import pandas as pd
import numpy as np
import requests
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

from config import Config

config = Config()


def predict_one(text, mode):
    """单条调用 DeepSeek，返回预测标签"""
    if mode == "binary":
        labels_desc = "0=差评(包含一般/中评), 1=好评"
        choices = ["0", "1"]
    elif mode == "3class":
        labels_desc = "0=差评, 1=中评, 2=好评"
        choices = ["0", "1", "2"]
    else:  # 5class
        labels_desc = "0=1星(非常差), 1=2星(较差), 2=3星(一般), 3=4星(较好), 4=5星(非常好)"
        choices = ["0", "1", "2", "3", "4"]

    text = str(text).strip()
    if not text:
        return -1

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
                        {"role": "system", "content": "你是一个电商评论情感分析助手，只输出数字标签。"},
                        {"role": "user", "content": f"判断以下评论的情感倾向，只输出一个数字。{labels_desc}。\n评论: {text}"},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 5,
                },
                timeout=30,
                proxies={"http": None, "https": None},
            )
            resp.raise_for_status()
            result = resp.json()["choices"][0]["message"]["content"].strip()
            for c in choices:
                if c in result:
                    return int(c)
            return -1
        except Exception:
            if retry < 2:
                time.sleep(1)

    return -1


def call_deepseek(texts, mode="binary", workers=20):
    """并发调用 DeepSeek API"""
    results = [-1] * len(texts)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(predict_one, t, mode): i for i, t in enumerate(texts)}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"DeepSeek {mode}"):
            idx = futures[future]
            results[idx] = future.result()

    return results


def evaluate(label_true, label_pred, name):
    mask = np.array(label_pred) != -1
    y_true = label_true[mask]
    y_pred = np.array(label_pred)[mask]
    skip = len(label_pred) - len(y_true)

    if name == "binary":
        target_names = ["差评", "好评"]
    elif name == "3class":
        target_names = ["负面", "中性", "正面"]
    else:
        target_names = ["1星", "2星", "3星", "4星", "5星"]

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    print(f"\n{'=' * 50}")
    print(f"DeepSeek {name}")
    print(f"  有效: {len(y_true)}  (失败: {skip})")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1:       {f1:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=target_names)}")

    df_result = pd.DataFrame({"true": y_true, "pred": y_pred})
    csv_path = os.path.join(config.result_root, f"deepseek_{name}.csv")
    df_result.to_csv(csv_path, index=False)
    print(f"  已保存: {csv_path}")

    return acc, f1


def run(mode="binary", sample_size=None):
    print(f"=== DeepSeek API [{mode}] ===")

    if mode == "binary":
        test_path = config.test_path_binary
    elif mode == "3class":
        test_path = config.test_path_3class
    else:
        test_path = config.test_path_5class
    df = pd.read_csv(test_path)
    if sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    print(f"测试集: {len(df):,}  标签: {df['label'].value_counts().sort_index().to_dict()}")

    preds = call_deepseek(df["review"], mode)
    evaluate(df["label"].values, preds, mode)


if __name__ == "__main__":
    # run(mode="binary", sample_size=None)
    # run(mode="3class", sample_size=None)
    run(mode="5class", sample_size=None)
