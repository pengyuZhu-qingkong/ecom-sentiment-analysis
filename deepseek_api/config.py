import os


class Config:
    def __init__(self):
        self.deepseek_root = os.path.dirname(os.path.abspath(__file__))

        # BERT 数据 (复用)
        bert_data = r"D:\dev\ec_review_analysis\bert\Cleaned_data_for_Bert"
        self.test_path_binary = os.path.join(bert_data, "binary", "test.csv")
        self.test_path_3class = os.path.join(bert_data, "3class", "test.csv")
        self.test_path_5class = os.path.join(bert_data, "5class", "test.csv")

        # 结果保存
        self.result_root = os.path.join(self.deepseek_root, "result")
        os.makedirs(self.result_root, exist_ok=True)

        # DeepSeek API
        self.api_key = "sk-c970018e9a174100b978987d5d242544"          # 替换成你的 key
        self.api_url = "https://api.deepseek.com/chat/completions"
        self.model = "deepseek-chat"           # 或 deepseek-reasoner

        # 测试参数
        self.max_samples = None                # None=全部 20000 条, 数字=采样
