import os


class Config():
    def __init__(self):
        self.data_root = r"D:\dev\ec_review_analysis\Cleaned_data"
        self.model_root = r"D:\dev\ec_review_analysis\baseline\save_model"
        self.result_root = r"D:\dev\ec_review_analysis\baseline\result"
        self.final_root=r"D:\dev\ec_review_analysis\baseline\final_data"
        self.train_path = os.path.join(self.data_root, "train.csv")
        self.val_path = os.path.join(self.data_root, "val.csv")
        self.test_path = os.path.join(self.data_root, "test.csv")
        self.stop_word_path=os.path.join(self.final_root, "stopwords.csv")
        self.final_data_path_train=os.path.join(self.final_root, "train_process.csv")
        self.final_data_path_test=os.path.join(self.final_root, "test_process.csv")
        self.final_data_path_val=os.path.join(self.final_root, "val_process.csv")
        self.model_save_path=os.path.join(self.model_root, "rf_model1.pkl")
        self.tfidf_save_path=os.path.join(self.model_root, "tfidf_model1.pkl")
        self.result_save_path=os.path.join(self.result_root, "result1.csv")
if __name__ == '__main__':
    config = Config()
    print(config.train_path)
    print(config.val_path)
    print(config.test_path)
    print(config.model_save_path)
    print(config.tfidf_save_path)
    print(config.result_save_path)
