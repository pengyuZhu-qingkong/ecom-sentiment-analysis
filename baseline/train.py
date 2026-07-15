import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from baseline.config import Config

config = Config()
os.makedirs(config.model_root, exist_ok=True)
os.makedirs(config.result_root, exist_ok=True)


# 模块级函数, joblib 保存时需要
def split_tokenizer(text):
    """按空格拆分已分词的文本"""
    return text.split()


def train():
    # ================================================================
    # 1. 加载数据
    # ================================================================
    print("=" * 60)
    print("[1] 加载数据...")
    df_train = pd.read_csv(config.final_data_path_train, sep='\t', header=0)
    words = df_train["words"]
    labels = df_train["label"]
    # 中性合并到负面, 做二分类: 0=差评, 1=好评
    labels = labels.map({0: 0, 1: 0, 2: 1})
    print(f"  训练集: {len(words):,} 条")
    print(f"  标签分布: {labels.value_counts().to_dict()}")

    # ================================================================
    # 2. TF-IDF 向量化
    # ================================================================
    print("\n[2] TF-IDF 向量化...")
    tfidf = TfidfVectorizer(
        tokenizer=split_tokenizer,
        token_pattern=None,
        stop_words=None,
        max_features=5000,
        ngram_range=(1, 2),
        min_df=3,
    )
    X_all = tfidf.fit_transform(words)
    y_all = labels
    print(f"  特征维度: {X_all.shape[1]}")

    # ================================================================
    # 3. GridSearchCV 调参 (采样 1万 条)
    # ================================================================
    print("\n[3] GridSearchCV 调参 (采样 10,000 条)...")

    sample_size = min(10000, len(y_all))
    idx = np.random.RandomState(42).choice(len(y_all), size=sample_size, replace=False)
    X_sample = X_all[idx]
    y_sample = y_all.iloc[idx]
    print(f"  采样: {sample_size:,} 条")

    param_grid = {
        'n_estimators':      [100, 200, 300],
        'max_depth':         [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
    }

    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
    grid = GridSearchCV(
        rf_base,
        param_grid,
        cv=3,
        scoring='f1_macro',
        verbose=1,
        n_jobs=-1,
    )
    grid.fit(X_sample, y_sample)

    print(f"\n  最佳参数: {grid.best_params_}")
    print(f"  最佳CV F1: {grid.best_score_:.4f}")

    best_params = grid.best_params_

    # ================================================================
    # 4. 全量训练
    # ================================================================
    print("\n[4] 全量训练...")
    rf_final = RandomForestClassifier(
        n_estimators      = best_params['n_estimators'],
        max_depth         = best_params['max_depth'],
        min_samples_split = best_params['min_samples_split'],
        class_weight      = 'balanced',
        random_state      = 42,
        n_jobs            = -1,
    )
    rf_final.fit(X_all, y_all)
    print("  训练完成!")

    # ================================================================
    # 5. 测试集评估
    # ================================================================
    print("\n[5] 测试集评估...")
    df_test = pd.read_csv(config.final_data_path_test, sep='\t', header=0)
    X_test = tfidf.transform(df_test["words"])
    y_test = df_test["label"].map({0: 0, 1: 0, 2: 1})  # 同训练集: 中性→差评

    y_pred = rf_final.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1  = f1_score(y_test, y_pred, average='macro')

    print(f"  准确率 (Accuracy):  {acc:.4f}")
    print(f"  精确率 (Precision): {pre:.4f}")
    print(f"  召回率 (Recall):    {rec:.4f}")
    print(f"  F1值  (F1-score):   {f1:.4f}")
    print("\n  分类报告:")
    print(classification_report(y_test, y_pred, target_names=['差评', '好评']))

    # ================================================================
    # 6. 保存模型和向量器
    # ================================================================
    print("\n[6] 保存模型和向量器...")
    joblib.dump(rf_final, config.model_save_path)
    joblib.dump(tfidf, config.tfidf_save_path)
    print(f"  ✓ 模型: {config.model_save_path}")
    print(f"  ✓ TF-IDF: {config.tfidf_save_path}")

    print("\n" + "=" * 60)
    print("训练完成! 最佳参数:", best_params)


if __name__ == '__main__':
    train()
