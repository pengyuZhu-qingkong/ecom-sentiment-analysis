import os


class Config():
    def __init__(self):
        # 项目根目录 (本地 D 盘 / 服务器 均自动识别)
        self.bert_root = os.path.dirname(os.path.abspath(__file__))

        # 清洗好的数据
        self.data_root_3class = os.path.join(self.bert_root, "Cleaned_data_for_Bert", "3class")
        self.data_root_binary = os.path.join(self.bert_root, "Cleaned_data_for_Bert", "binary")

        # 三分类数据
        self.train_path_3class = os.path.join(self.data_root_3class, "train.csv")
        self.val_path_3class   = os.path.join(self.data_root_3class, "val.csv")
        self.test_path_3class  = os.path.join(self.data_root_3class, "test.csv")

        # 二分类数据
        self.train_path_binary = os.path.join(self.data_root_binary, "train.csv")
        self.val_path_binary   = os.path.join(self.data_root_binary, "val.csv")
        self.test_path_binary  = os.path.join(self.data_root_binary, "test.csv")

        # 1星 vs 5星 (极简二分类，无标签噪声)
        self.data_root_1vs5 = os.path.join(self.bert_root, "Cleaned_data_for_Bert", "1vs5")
        self.train_path_1vs5 = os.path.join(self.data_root_1vs5, "train.csv")
        self.val_path_1vs5   = os.path.join(self.data_root_1vs5, "val.csv")
        self.test_path_1vs5  = os.path.join(self.data_root_1vs5, "test.csv")

        # 方面级分析数据
        self.data_root_aspect = os.path.join(self.bert_root, "Cleaned_data_for_Bert", "aspect")
        self.train_path_aspect = os.path.join(self.data_root_aspect, "train.csv")
        self.val_path_aspect   = os.path.join(self.data_root_aspect, "val.csv")
        self.test_path_aspect  = os.path.join(self.data_root_aspect, "test.csv")

        # 模型保存目录
        self.model_root = os.path.join(self.bert_root, "save_model")
        # 结果目录
        self.result_root = os.path.join(self.bert_root, "result")

        # 预训练模型 (本地路径 或 HuggingFace 自动下载)
        self.pretrained_model = r"D:\dev\NLP\day04\bert-base-chinese"
        # 服务器上改为:  self.pretrained_model = "bert-base-chinese"

        # ========== 训练参数 ==========
        self.max_length = 128        # 最大 token 数
        self.batch_size = 32         # GPU(4090)用32, CPU用8
        self.epochs_3class = 3       # 三分类轮次
        self.epochs_binary = 3       # 二分类轮次
        self.learning_rate = 2e-5    # 学习率 (BERT微调标准值)


if __name__ == '__main__':
    config = Config()
    print("bert_root:", config.bert_root)
    print("data_root_3class:", config.data_root_3class)
    print("data_root_binary:", config.data_root_binary)
    print("model_root:", config.model_root)
