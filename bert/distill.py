"""
模型蒸馏: BERT(老师) → BiLSTM(学生)
数据: aspect_3class (DeepSeek干净标签)
学生: 三分类 + 6方面 — 双头输出
"""
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
from transformers import AutoTokenizer

from train_combined import DualHeadBert, ASPECTS
from config import Config

config = Config()
os.makedirs(config.model_root, exist_ok=True)

NUM_ASPECTS = len(ASPECTS)
DATA_DIR = os.path.join(config.bert_root, "Cleaned_data_for_Bert", "aspect_3class")
# 蒸馏用大数据: 从二分类训练集取2万条（不需要标注，老师模型生成软标签）
DISTILL_DATA_PATH = os.path.join(config.bert_root, "Cleaned_data_for_Bert", "binary", "train.csv")
DISTILL_SAMPLES = 20000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ============================================================
# 1. 学生模型
# ============================================================
class StudentBiLSTM(nn.Module):
    def __init__(self, vocab_size=21128, embed_dim=512, hidden_dim=512, num_layers=2,
                 num_aspects=NUM_ASPECTS, num_sent_classes=3, num_aspect_classes=3, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.sent_head = nn.Linear(hidden_dim * 2, num_sent_classes)     # 三分类
        self.aspect_heads = nn.ModuleList(
            [nn.Linear(hidden_dim * 2, num_aspect_classes) for _ in range(num_aspects)]
        )

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)                           # (batch, seq, 256)
        lstm_out, _ = self.lstm(embedded)                              # (batch, seq, 512)
        lstm_out = self.dropout(lstm_out)

        # 池化: 只取有效token的均值
        mask = attention_mask.unsqueeze(-1).float()                    # (batch, seq, 1)
        pooled = (lstm_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        sent_logits = self.sent_head(pooled)                           # (batch, 3)
        aspect_logits = torch.stack([h(pooled) for h in self.aspect_heads], dim=1)
        return sent_logits, aspect_logits


# ============================================================
# 2. 蒸馏 Dataset (数据+老师软标签)
# ============================================================
class DistillDataset(Dataset):
    def __init__(self, texts, hard_sent_labels, hard_aspect_labels,
                 soft_sent_logits, soft_aspect_logits, tokenizer, max_length=128):
        """soft_logits: 预计算的老师输出，不用每次推理"""
        self.texts = [str(t) for t in texts]
        self.hard_sent = hard_sent_labels  # DeepSeek三分类标签
        self.hard_aspects = hard_aspect_labels.copy()
        self.hard_aspects[self.hard_aspects == -1] = 2  # 映射未提及→2
        self.soft_sent = soft_sent_logits     # 老师三分类logits
        self.soft_aspects = soft_aspect_logits  # 老师方面logits
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
            "hard_sent": torch.tensor(self.hard_sent[idx], dtype=torch.long),
            "hard_aspects": torch.tensor(self.hard_aspects[idx], dtype=torch.long),
            "soft_sent": torch.tensor(self.soft_sent[idx], dtype=torch.float),
            "soft_aspects": torch.tensor(self.soft_aspects[idx], dtype=torch.float),
        }


# ============================================================
# 3. 生成老师软标签
# ============================================================
def generate_soft_labels(teacher, tokenizer, texts, batch_size=32):
    """用老师模型对全部数据生成软标签"""
    teacher.eval()
    all_sent_logits = []
    all_aspect_logits = []

    loader = DataLoader(list(texts), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch_texts in tqdm(loader, desc="生成老师软标签"):
            enc = tokenizer(
                batch_texts, max_length=128, padding="max_length",
                truncation=True, return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = teacher(enc["input_ids"], enc["attention_mask"])
            all_sent_logits.append(out["sent_logits"].cpu())
            all_aspect_logits.append(out["aspect_logits"].cpu())

    return (torch.cat(all_sent_logits).numpy(),
            torch.cat(all_aspect_logits).numpy())


# ============================================================
# 4. 蒸馏损失
# ============================================================
def distill_loss(student_sent_logits, student_aspect_logits,
                 soft_sent, soft_aspects, hard_sent, hard_aspects,
                 alpha=0.7, T=4.0):
    """alpha: 软标签权重  T: 温度"""
    kl = nn.KLDivLoss(reduction="batchmean")
    log_softmax = nn.LogSoftmax(dim=-1)
    softmax = nn.Softmax(dim=-1)
    ce = nn.CrossEntropyLoss()

    # 软损失: KL(学生||老师)，温度T
    s_sent_soft = log_softmax(student_sent_logits / T)
    t_sent_soft = softmax(soft_sent / T)
    loss_sent_soft = kl(s_sent_soft, t_sent_soft) * (T * T)

    loss_aspect_soft = 0
    for i in range(NUM_ASPECTS):
        s_a = log_softmax(student_aspect_logits[:, i, :] / T)
        t_a = softmax(soft_aspects[:, i, :] / T)
        loss_aspect_soft += kl(s_a, t_a) * (T * T)

    # 硬损失: 交叉熵（直接学真实标签）
    loss_sent_hard = ce(student_sent_logits, hard_sent)
    loss_aspect_hard = 0
    for i in range(NUM_ASPECTS):
        loss_aspect_hard += ce(student_aspect_logits[:, i, :], hard_aspects[:, i])

    total = alpha * (loss_sent_soft + loss_aspect_soft) + \
            (1 - alpha) * (loss_sent_hard + loss_aspect_hard)
    return total


# ============================================================
# 5. 评估
# ============================================================
def evaluate_student(model, loader):
    model.eval()
    sent_preds, sent_true = [], []
    aspect_preds, aspect_true = [], []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if k != "hard_sent" and k != "hard_aspects" and k != "soft_sent" and k != "soft_aspects"
                     else v for k, v in batch.items()}

            input_ids = batch["input_ids"].to(device) if isinstance(batch["input_ids"], torch.Tensor) else batch["input_ids"]
            attention_mask = batch["attention_mask"].to(device)

            s_logits, a_logits = model(input_ids, attention_mask)
            sent_preds.append(s_logits.argmax(-1).cpu())
            sent_true.append(batch["hard_sent"].cpu())
            aspect_preds.append(a_logits.argmax(-1).cpu())
            aspect_true.append(batch["hard_aspects"].cpu())

    sent_preds = torch.cat(sent_preds).numpy()
    sent_true = torch.cat(sent_true).numpy()
    aspect_preds = torch.cat(aspect_preds).numpy()
    aspect_true = torch.cat(aspect_true).numpy()

    sent_f1 = f1_score(sent_true, sent_preds, average="macro")
    aspect_f1s = []
    for i in range(NUM_ASPECTS):
        aspect_f1s.append(f1_score(aspect_true[:, i], aspect_preds[:, i], average="macro"))
    aspect_f1 = np.mean(aspect_f1s)

    return sent_f1, aspect_f1, sent_preds, sent_true, aspect_preds, aspect_true


# ============================================================
# 6. 主流程
# ============================================================
def run():
    print("=== 模型蒸馏: BERT → BiLSTM ===")

    # ---- 加载老师模型 ----
    print("\n[1] 加载老师模型...")
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(config.model_root, "bert_combined"))
    teacher = DualHeadBert(config.pretrained_model)
    teacher.load_state_dict(torch.load(os.path.join(config.model_root, "dual_head_best.pt"), map_location=device))
    teacher.to(device)
    teacher.eval()
    print("  老师模型加载完成")

    # ---- 蒸馏数据: 2万条 (无需标注，老师生成软标签) ----
    print(f"\n[2] 加载蒸馏数据 ({DISTILL_SAMPLES:,} 条)...")
    df_raw = pd.read_csv(DISTILL_DATA_PATH)
    df_raw = df_raw.sample(n=DISTILL_SAMPLES, random_state=42).reset_index(drop=True)
    # 8:2 拆分
    split = int(len(df_raw) * 0.8)
    df_train = df_raw.iloc[:split]
    df_val = df_raw.iloc[split:]
    print(f"  训练集: {len(df_train):,}  验证集: {len(df_val):,}")

    # 老师生成软标签
    print("\n[3] 老师生成软标签...")
    train_soft_sent, train_soft_aspects = generate_soft_labels(
        teacher, tokenizer, df_train["review"].tolist(),
    )
    val_soft_sent, val_soft_aspects = generate_soft_labels(
        teacher, tokenizer, df_val["review"].tolist(),
    )
    print(f"  完成: {train_soft_sent.shape}, {val_soft_sent.shape}")

    # 硬标签: 二分类标签→三分类 (0→0差评, 1→2好评, 中评缺失)
    train_hard_sent = df_train["label"].map({0: 0, 1: 2}).values
    val_hard_sent = df_val["label"].map({0: 0, 1: 2}).values
    # 方面硬标签: 无, 全填2(未提及)，用 alpha=0.9 靠软标签学
    train_hard_aspects = np.full((len(df_train), NUM_ASPECTS), 2)
    val_hard_aspects = np.full((len(df_val), NUM_ASPECTS), 2)

    train_ds = DistillDataset(
        df_train["review"], train_hard_sent, train_hard_aspects,
        train_soft_sent, train_soft_aspects, tokenizer,
    )
    val_ds = DistillDataset(
        df_val["review"], val_hard_sent, val_hard_aspects,
        val_soft_sent, val_soft_aspects, tokenizer,
    )

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)

    # ---- 初始化学生模型 ----
    print("\n[4] 初始化学生模型...")
    student = StudentBiLSTM()
    student.to(device)
    params = sum(p.numel() for p in student.parameters())
    print(f"  学生参数量: {params:,}  (老师: 1.1亿)")
    print(f"  压缩比: {1.1e8/params:.0f}x")

    # ---- 训练 ----
    print("\n[5] 蒸馏训练 (alpha=0.9 主要模仿老师)...")
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    best_val_f1 = 0

    for epoch in range(1, 21):
        student.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            s_logits, a_logits = student(batch["input_ids"], batch["attention_mask"])
            loss = distill_loss(
                s_logits, a_logits,
                batch["soft_sent"], batch["soft_aspects"],
                batch["hard_sent"], batch["hard_aspects"],
                alpha=0.9, T=3.0,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        sent_f1, aspect_f1, _, _, _, _ = evaluate_student(student, val_loader)
        avg_f1 = (sent_f1 + aspect_f1) / 2
        print(f"  Epoch {epoch:2d}  loss={total_loss/len(train_loader):.4f}  "
              f"val_sent={sent_f1:.4f}  val_aspect={aspect_f1:.4f}  avg={avg_f1:.4f}")

        if avg_f1 > best_val_f1:
            best_val_f1 = avg_f1
            torch.save(student.state_dict(), os.path.join(config.model_root, "student_best.pt"))

    # ---- 用 aspect_3class 测试集最终评估 ----
    print(f"\n[6] 最终评估 (aspect_3class 测试集)...")
    student.load_state_dict(torch.load(os.path.join(config.model_root, "student_best.pt")))
    student.to(device)

    df_test_final = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    test_hard_aspects = df_test_final[ASPECTS].values.astype(int)
    test_soft_sent, test_soft_aspects = generate_soft_labels(
        teacher, tokenizer, df_test_final["review"].tolist(),
    )
    test_ds = DistillDataset(
        df_test_final["review"], df_test_final["sentiment"].values, test_hard_aspects,
        test_soft_sent, test_soft_aspects, tokenizer,
    )
    test_loader = DataLoader(test_ds, batch_size=config.batch_size)
    sent_f1, aspect_f1, _, _, _, _ = evaluate_student(student, test_loader)

    print(f"\n{'=' * 60}")
    print(f"  学生三分类 F1: {sent_f1:.4f}  (老师: 0.8919)")
    print(f"  学生方面级 F1: {aspect_f1:.4f}  (老师: 0.865)")
    print(f"  蒸馏数据:      {DISTILL_SAMPLES:,} 条")
    print(f"  模型大小:      {params*4/1024/1024:.0f} MB  (老师: ~400 MB)")
    print(f"  压缩比:        {1.1e8/params:.0f}x")

    save_path = os.path.join(config.model_root, "student_distilled.pt")
    torch.save({"state_dict": student.state_dict(), "params": params}, save_path)
    tokenizer.save_pretrained(os.path.join(config.model_root, "student_tokenizer"))
    print(f"\n  学生模型已保存: {save_path}")


if __name__ == "__main__":
    run()
