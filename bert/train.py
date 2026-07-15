import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from config import Config

config = Config()
os.makedirs(config.model_root, exist_ok=True)
os.makedirs(config.result_root, exist_ok=True)


class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = [str(t) for t in texts]
        self.labels = labels
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
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro"),
    }


def train(mode="binary", sample_size=10000):
    print(f"=== BERT {mode} ({'采样 ' + f'{sample_size:,}' if sample_size else '全量'}) ===")

    paths = {
        "binary": (config.train_path_binary, config.val_path_binary, config.test_path_binary),
        "3class": (config.train_path_3class, config.val_path_3class, config.test_path_3class),
        "1vs5":   (config.train_path_1vs5,   config.val_path_1vs5,   config.test_path_1vs5),
    }
    train_path, val_path, test_path = paths[mode]
    num_labels = 3 if mode == "3class" else 2

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)
    if sample_size and sample_size < len(df_train):
        df_train = df_train.sample(n=sample_size, random_state=42).reset_index(drop=True)

    print(f"train={len(df_train):,}  val={len(df_val):,}  test={len(df_test):,}")
    print(f"labels: {df_train['label'].value_counts().sort_index().to_dict()}")

    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.pretrained_model, num_labels=num_labels
    )

    train_ds = ReviewDataset(df_train["review"], df_train["label"], tokenizer, config.max_length)
    val_ds   = ReviewDataset(df_val["review"],   df_val["label"],   tokenizer, config.max_length)
    test_ds  = ReviewDataset(df_test["review"],  df_test["label"],  tokenizer, config.max_length)

    args = TrainingArguments(
        output_dir=os.path.join(config.result_root, "checkpoints"),
        num_train_epochs=config.epochs_binary if mode == "binary" else config.epochs_3class,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    result = trainer.evaluate(test_ds)
    print(f"\ntest accuracy={result['eval_accuracy']:.4f}  f1={result['eval_f1']:.4f}")

    save_dir = os.path.join(config.model_root, f"bert_{mode}")
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"model saved: {save_dir}")


if __name__ == "__main__":
    # mode: "binary"=二分类(1+2+3→差评)  "3class"=三分类  "1vs5"=1星vs5星(干净!)
    # sample_size: None=全量(16万)  数字=采样条数
    train(mode="1vs5", sample_size=None)
    # train(mode="binary", sample_size=None)
    # train(mode="3class", sample_size=None)
