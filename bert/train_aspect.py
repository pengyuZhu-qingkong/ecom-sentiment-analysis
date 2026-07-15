"""
BERT 方面级情感分析 —— 多标签多分类训练
6个方面，每个3分类: 0=负面 1=正面 2=未提及
"""
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score, classification_report

from config import Config

config = Config()
os.makedirs(config.model_root, exist_ok=True)
os.makedirs(config.result_root, exist_ok=True)

ASPECTS = ["商品质量", "外观设计", "使用体验", "物流配送", "价格", "客服服务"]
NUM_ASPECTS = len(ASPECTS)
NUM_CLASSES = 3  # 0=负面 1=正面 2=未提及


# ============================================================
# 1. 多标签 Dataset
# ============================================================
class AspectDataset(Dataset):
    def __init__(self, texts, labels_matrix, tokenizer, max_length=128):
        """
        labels_matrix: (n, 6) numpy array, 值域 {-1, 0, 1}
        训练时 -1 映射为 2 (未提及)
        """
        self.texts = [str(t) for t in texts]
        self.labels = labels_matrix.copy()
        self.labels[self.labels == -1] = 2  # 未提及 → class 2
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),  # shape (6,)
        }


# ============================================================
# 2. 多分类头模型
# ============================================================
class AspectBertModel(nn.Module):
    def __init__(self, model_name, num_aspects=NUM_ASPECTS, num_classes=NUM_CLASSES):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size  # 768
        # 6个独立分类头
        self.classifiers = nn.ModuleList([
            nn.Linear(hidden_size, num_classes) for _ in range(num_aspects)
        ])

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # [CLS] token, shape (batch, 768)

        # 6个头各自输出 logits
        logits_list = [head(pooled) for head in self.classifiers]  # 6 × (batch, 3)
        logits = torch.stack(logits_list, dim=1)  # (batch, 6, 3)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            losses = []
            for i in range(NUM_ASPECTS):
                losses.append(loss_fn(logits[:, i, :], labels[:, i]))
            loss = torch.stack(losses).mean()

        return {"loss": loss, "logits": logits}


# ============================================================
# 3. 自定义 Trainer (处理多标签 loss)
# ============================================================
class AspectTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=labels)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss


# ============================================================
# 4. 评估函数
# ============================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred  # logits: (n, 6, 3)  labels: (n, 6)
    preds = np.argmax(logits, axis=2)  # (n, 6)

    metrics = {}
    per_aspect_f1 = []

    for i, aspect in enumerate(ASPECTS):
        y_true = labels[:, i]
        y_pred = preds[:, i]
        f1 = f1_score(y_true, y_pred, average="macro")
        metrics[f"{aspect}_f1"] = f1
        per_aspect_f1.append(f1)

    metrics["f1_macro"] = np.mean(per_aspect_f1)
    return metrics


# ============================================================
# 5. 训练函数
# ============================================================
def train():
    print("=== BERT 方面级情感分析 ===")

    # 加载数据
    print(f"\n[1] 加载数据...")
    df_train = pd.read_csv(config.train_path_aspect)
    df_val   = pd.read_csv(config.val_path_aspect)
    df_test  = pd.read_csv(config.test_path_aspect)

    print(f"  train: {len(df_train):,}  val: {len(df_val):,}  test: {len(df_test):,}")

    # 提取 labels 矩阵
    train_labels = df_train[ASPECTS].values.astype(int)  # (n, 6)
    val_labels   = df_val[ASPECTS].values.astype(int)
    test_labels  = df_test[ASPECTS].values.astype(int)

    # 分布统计
    for i, aspect in enumerate(ASPECTS):
        pos = (train_labels[:, i] == 1).sum()
        neg = (train_labels[:, i] == 0).sum()
        noneed = (train_labels[:, i] == -1).sum()
        print(f"    {aspect}: 正面={pos} 负面={neg} 未提及={noneed}")

    # 加载 tokenizer & 模型
    print(f"\n[2] 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model)
    model = AspectBertModel(config.pretrained_model)

    # Dataset
    train_ds = AspectDataset(df_train["review"], train_labels, tokenizer, config.max_length)
    val_ds   = AspectDataset(df_val["review"],   val_labels,   tokenizer, config.max_length)
    test_ds  = AspectDataset(df_test["review"],  test_labels,  tokenizer, config.max_length)

    # 训练参数
    args = TrainingArguments(
        output_dir=os.path.join(config.result_root, "aspect_checkpoints"),
        num_train_epochs=10,              # 数据少(4000条)，多跑几轮
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        report_to="none",
        disable_tqdm=False,
    )

    trainer = AspectTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    # 训练
    print(f"\n[3] 训练 (epochs=10)...")
    trainer.train()

    # 测试集评估
    print(f"\n[4] 测试集评估...")
    test_result = trainer.evaluate(test_ds)

    print("\n" + "=" * 60)
    print("各方面 F1 (macro avg):")
    print("=" * 60)
    for aspect in ASPECTS:
        key = f"eval_{aspect}_f1"
        print(f"  {aspect:<8} F1={test_result[key]:.4f}")
    print(f"  {'平均':<8} F1={test_result['eval_f1_macro']:.4f}")

    # 保存
    save_dir = os.path.join(config.model_root, "bert_aspect")
    tokenizer.save_pretrained(save_dir)
    torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
    print(f"\n模型已保存: {save_dir}")


if __name__ == "__main__":
    train()
