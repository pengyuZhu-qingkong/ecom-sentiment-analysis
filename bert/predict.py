"""
三分类 + 方面级 预测脚本（Flask 可调用）
加载 dual_head_best.pt + bert_combined tokenizer
"""
import torch
from transformers import AutoTokenizer

from train_combined import DualHeadBert, ASPECTS
from config import Config

config = Config()

# ============================================================
# 加载模型（全局单例，只加载一次）
# ============================================================
MODEL_PATH = r"D:\dev\ec_review_analysis\bert\save_model\dual_head_best.pt"
TOKENIZER_PATH = r"D:\dev\ec_review_analysis\bert\save_model\bert_combined"

# 本地用 bert-base-chinese 路径，服务器改用 "bert-base-chinese"
PRETRAINED = getattr(config, "pretrained_model", "bert-base-chinese")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
model = DualHeadBert(PRETRAINED)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

SENT_NAMES = {0: "差评", 1: "中评", 2: "好评"}
ASPECT_NAMES = {0: "负面", 1: "正面", 2: "未提及"}


# ============================================================
# 核心预测函数（供 Flask 调用）
# ============================================================
def predict(text: str) -> dict:
    """
    输入: 评论文本
    输出: {
        "sentiment": "好评" | "中评" | "差评",
        "sentiment_id": 0|1|2,
        "aspects": {
            "商品质量": "正面" | "负面" | "未提及",
            "外观设计": ...,
            "使用体验": ...,
            "物流配送": ...,
            "价格": ...,
            "客服服务": ...
        }
    }
    """
    text = str(text).strip()
    if not text:
        return {"error": "输入为空"}

    enc = tokenizer(text, max_length=128, padding="max_length",
                    truncation=True, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(enc["input_ids"], enc["attention_mask"])

    # 三分类
    sent_id = out["sent_logits"].argmax(-1).item()

    # 方面级
    aspect_ids = out["aspect_logits"].argmax(-1).squeeze(0).tolist()  # [2, 1, 0, 2, 2, 2] 等
    aspects = {}
    for i, name in enumerate(ASPECTS):
        aspects[name] = ASPECT_NAMES[aspect_ids[i]]

    return {
        "sentiment": SENT_NAMES[sent_id],
        "sentiment_id": sent_id,
        "aspects": aspects,
    }


# ============================================================
# 批量预测
# ============================================================
def predict_batch(texts: list) -> list:
    return [predict(t) for t in texts]


# ============================================================
# 命令行测试
# ============================================================
if __name__ == "__main__":
    tests = [
        "手机屏幕很大很清晰，电池也耐用，推荐购买",
        "物流太慢了，包装都压坏了，但是东西质量还不错",
        "一般般吧，一分钱一分货，凑合用",
        "客服态度极其恶劣，退个货拖了一星期，再也不买了",
    ]
    for t in tests:
        result = predict(t)
        print(f"\n评论: {t}")
        print(f"  整体: {result.get('sentiment', 'error')}")
        if result.get("aspects"):
            print(f"  方面: {result['aspects']}")
