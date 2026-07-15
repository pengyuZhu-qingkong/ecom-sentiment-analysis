"""
双头BERT: 三分类 + 方面级 — 公平对比
头1: 三分类 (0=差评 1=中评 2=好评)
头2: 6方面 × 3分类 (0=负面 1=正面 2=未提及)
"""
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, classification_report

from config import Config

config = Config()
os.makedirs(config.model_root, exist_ok=True)
os.makedirs(config.result_root, exist_ok=True)

ASPECTS = ["商品质量", "外观设计", "使用体验", "物流配送", "价格", "客服服务"]
NUM_ASPECTS = len(ASPECTS)
DATA_DIR = os.path.join(config.bert_root, "Cleaned_data_for_Bert", "aspect_3class")


# ============================================================
# Dataset
# ============================================================
class CombinedDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.texts = [str(t) for t in df["review"]]
        self.sentiments = df["sentiment"].values.astype(int)  # (n,)
        # 方面: -1→2, 0→0, 1→1
        self.aspects = df[ASPECTS].values.astype(int).copy()  # (n, 6)
        self.aspects[self.aspects == -1] = 2
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "sentiment_label": torch.tensor(self.sentiments[idx], dtype=torch.long),
            "aspect_labels": torch.tensor(self.aspects[idx], dtype=torch.long),  # (6,)
        }


# ============================================================
# 双头模型
# ============================================================
class DualHeadBert(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.sentiment_head = nn.Linear(hidden, 3)                    # 三分类
        self.aspect_heads = nn.ModuleList([nn.Linear(hidden, 3) for _ in range(NUM_ASPECTS)])  # 6方面

    def forward(self, input_ids, attention_mask, sentiment_label=None, aspect_labels=None):
        pooled = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        sent_logits = self.sentiment_head(pooled)  # (batch, 3)
        aspect_logits = torch.stack([h(pooled) for h in self.aspect_heads], dim=1)  # (batch, 6, 3)

        loss = None
        if sentiment_label is not None and aspect_labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            sent_loss = loss_fn(sent_logits, sentiment_label)
            aspect_losses = [loss_fn(aspect_logits[:, i, :], aspect_labels[:, i]) for i in range(NUM_ASPECTS)]
            aspect_loss = torch.stack(aspect_losses).mean()
            loss = sent_loss + aspect_loss  # 联合训练

        return {"loss": loss, "sent_logits": sent_logits, "aspect_logits": aspect_logits}


# ============================================================
# 自定义 Trainer
# ============================================================
class DualHeadTrainer:
    def __init__(self, model, train_ds, val_ds, test_ds, batch_size=32, lr=2e-5, epochs=10):
        self.model = model
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.epochs = epochs

        self.train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)
        self.test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(self.device)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch in self.train_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            self.optimizer.zero_grad()
            out = self.model(batch["input_ids"], batch["attention_mask"],
                             batch["sentiment_label"], batch["aspect_labels"])
            out["loss"].backward()
            self.optimizer.step()
            total_loss += out["loss"].item()
        return total_loss / len(self.train_loader)

    def evaluate(self, loader):
        self.model.eval()
        sent_preds, sent_true = [], []
        aspect_preds, aspect_true = [], []

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                out = self.model(batch["input_ids"], batch["attention_mask"])
                sent_preds.append(out["sent_logits"].argmax(-1).cpu())
                sent_true.append(batch["sentiment_label"].cpu())
                aspect_preds.append(out["aspect_logits"].argmax(-1).cpu())  # (batch, 6)
                aspect_true.append(batch["aspect_labels"].cpu())

        sent_preds = torch.cat(sent_preds).numpy()
        sent_true = torch.cat(sent_true).numpy()
        aspect_preds = torch.cat(aspect_preds).numpy()  # (n, 6)
        aspect_true = torch.cat(aspect_true).numpy()

        return sent_preds, sent_true, aspect_preds, aspect_true

    def run(self):
        best_f1 = 0
        for epoch in range(1, self.epochs + 1):
            loss = self.train_epoch()
            sent_preds, sent_true, _, _ = self.evaluate(self.val_loader)
            f1 = f1_score(sent_true, sent_preds, average="macro")
            print(f"  Epoch {epoch:2d}  loss={loss:.4f}  val_f1={f1:.4f}")
            if f1 > best_f1:
                best_f1 = f1
                torch.save(self.model.state_dict(), os.path.join(config.model_root, "dual_head_best.pt"))

        # 加载最佳模型
        self.model.load_state_dict(torch.load(os.path.join(config.model_root, "dual_head_best.pt")))
        self.model.to(self.device)

        # 测试集
        sent_preds, sent_true, aspect_preds, aspect_true = self.evaluate(self.test_loader)
        return sent_preds, sent_true, aspect_preds, aspect_true


# ============================================================
# 方面反推三分类
# ============================================================
def aspects_to_sentiment(aspect_labels):
    """Batched: aspect_labels shape (n, 6), values 0=neg 1=pos 2=未提及"""
    # 0→-1, 1→1, 2→0
    mapped = np.where(aspect_labels == 1, 1, np.where(aspect_labels == 0, -1, 0))
    sums = mapped.sum(axis=1)  # (n,)
    result = np.where(sums > 0, 2, np.where(sums < 0, 0, 1))
    return result


# ============================================================
# 主流程
# ============================================================
def train():
    print("=== 双头BERT: 三分类 + 方面级 ===")

    # 加载数据
    df_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_val = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    df_test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    print(f"train={len(df_train)} val={len(df_val)} test={len(df_test)}")

    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model)
    model = DualHeadBert(config.pretrained_model)

    train_ds = CombinedDataset(df_train, tokenizer, config.max_length)
    val_ds = CombinedDataset(df_val, tokenizer, config.max_length)
    test_ds = CombinedDataset(df_test, tokenizer, config.max_length)

    trainer = DualHeadTrainer(model, train_ds, val_ds, test_ds,
                              batch_size=config.batch_size,
                              lr=config.learning_rate,
                              epochs=10)
    sent_preds, sent_true, aspect_preds, aspect_true = trainer.run()

    # ======== 对比评估 ========
    # 1) 直接三分类
    sent_f1 = f1_score(sent_true, sent_preds, average="macro")
    print(f"\n{'=' * 60}")
    print(f"【直接三分类】F1={sent_f1:.4f}")
    print(classification_report(sent_true, sent_preds, target_names=["差评", "中评", "好评"]))

    # 2) 方面反推三分类 (用BERT预测的方面标签)
    reverse_preds = aspects_to_sentiment(aspect_preds)
    reverse_f1 = f1_score(sent_true, reverse_preds, average="macro")
    print(f"【方面反推三分类】F1={reverse_f1:.4f}")
    print(classification_report(sent_true, reverse_preds, target_names=["差评", "中评", "好评"]))

    # 3) 方面反推 (用DeepSeek真实方面标签，上界)
    gt_aspects = df_test[ASPECTS].values.astype(int)
    gt_aspects[gt_aspects == -1] = 2
    gt_reverse = aspects_to_sentiment(gt_aspects)
    gt_f1 = f1_score(sent_true, gt_reverse, average="macro")
    print(f"【方面反推-理想上界】F1={gt_f1:.4f}  (使用DeepSeek真值方面)")
    print(classification_report(sent_true, gt_reverse, target_names=["差评", "中评", "好评"]))

    # 汇总
    print(f"\n{'=' * 60}")
    print("对比汇总:")
    print(f"  直接三分类:          F1={sent_f1:.4f}")
    print(f"  方面反推 (BERT预测):  F1={reverse_f1:.4f}")
    print(f"  方面反推 (理想上界):  F1={gt_f1:.4f}")
    print(f"  反推差距 (BERT vs 上界): {gt_f1 - reverse_f1:.4f}")

    # 保存
    tokenizer.save_pretrained(os.path.join(config.model_root, "bert_combined"))
    print(f"\n模型已保存")


if __name__ == "__main__":
    train()
