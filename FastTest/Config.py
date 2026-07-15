import os


class Config():
    def __init__(self):
        self.data_root = r"D:\dev\ec_review_analysis\Cleaned_data_for_fasttext"
        self.model_root = r"D:\dev\ec_review_analysis\FastTest\save_model"
        self.result_root = r"D:\dev\ec_review_analysis\FastTest\result"
        self.final_root=r"D:\dev\ec_review_analysis\FastTest\final_data"
        self.train_path = os.path.join(self.data_root, "train.csv")
        self.val_path = os.path.join(self.data_root, "val.csv")
        self.test_path = os.path.join(self.data_root, "test.csv")
        self.stop_word_path=os.path.join(self.final_root, "stopwords.csv")
        self.final_data_path_train=os.path.join(self.final_root, "train_process.csv")
        self.final_data_path_test=os.path.join(self.final_root, "test_process.csv")
        self.final_data_path_val=os.path.join(self.final_root, "val_process.csv")
        self.model_save_path=os.path.join(self.model_root, "fasttext_model.bin")
        self.result_save_path=os.path.join(self.result_root, "result1.csv")
        # 二分类数据路径
        self.binary_root = r"D:\dev\ec_review_analysis\Cleaned_data_for_fasttext_binary"
        self.binary_train_txt = os.path.join(self.binary_root, "train.txt")
        self.binary_val_txt   = os.path.join(self.binary_root, "val.txt")
        self.binary_test_txt  = os.path.join(self.binary_root, "test.txt")
        self.binary_model_path = os.path.join(self.model_root, "fasttext_binary_model.bin")
if __name__ == '__main__':
    config = Config()
    print(config.train_path)
    print(config.val_path)
    print(config.test_path)
    print(config.model_save_path)
    print(config.tfidf_save_path)
    print(config.result_save_path)
