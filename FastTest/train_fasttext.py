import fasttext
import os

from FastTest.Config import Config

config = Config()
os.makedirs(config.model_root, exist_ok=True)


def train_3class():
    """三分类训练: lr=0.5 epoch=25 dim=100 wordNgrams=2"""
    print("=" * 60)
    print("FastText 三分类训练")
    print("=" * 60)

    train_path = config.final_root + "\\train.txt"
    val_path   = config.final_root + "\\val.txt"
    test_path  = config.final_root + "\\test.txt"

    model = fasttext.train_supervised(
        input=train_path,
        lr=0.5,
        epoch=25,
        dim=100,
        wordNgrams=2,
        loss='softmax',
        thread=4,
        verbose=2,
    )

    # 验证集
    _, val_p, val_r = model.test(val_path)
    val_f1 = 2 * val_p * val_r / (val_p + val_r) if (val_p + val_r) > 0 else 0
    print(f"验证集: P={val_p:.4f}  R={val_r:.4f}  F1={val_f1:.4f}")

    # 测试集
    _, test_p, test_r = model.test(test_path)
    test_f1 = 2 * test_p * test_r / (test_p + test_r) if (test_p + test_r) > 0 else 0
    print(f"测试集: P={test_p:.4f}  R={test_r:.4f}  F1={test_f1:.4f}")

    model.save_model(config.model_save_path)
    print(f"模型已保存: {config.model_save_path}")


def train_binary():
    """二分类训练 (中评归入差评): lr=0.5 epoch=25 dim=100 wordNgrams=2"""
    print("=" * 60)
    print("FastText 二分类训练")
    print("=" * 60)

    model = fasttext.train_supervised(
        input=config.binary_train_txt,
        lr=0.5,
        epoch=25,
        dim=100,
        wordNgrams=2,
        loss='softmax',
        thread=4,
        verbose=2,
    )

    # 验证集
    _, val_p, val_r = model.test(config.binary_val_txt)
    val_f1 = 2 * val_p * val_r / (val_p + val_r) if (val_p + val_r) > 0 else 0
    print(f"验证集: P={val_p:.4f}  R={val_r:.4f}  F1={val_f1:.4f}")

    # 测试集
    _, test_p, test_r = model.test(config.binary_test_txt)
    test_f1 = 2 * test_p * test_r / (test_p + test_r) if (test_p + test_r) > 0 else 0
    print(f"测试集: P={test_p:.4f}  R={test_r:.4f}  F1={test_f1:.4f}")

    model.save_model(config.binary_model_path)
    print(f"模型已保存: {config.binary_model_path}")


if __name__ == '__main__':
    train_3class()
    # train_binary()
