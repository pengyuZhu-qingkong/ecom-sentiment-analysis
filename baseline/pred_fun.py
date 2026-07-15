import pickle

import jieba
import joblib

from baseline.config import Config
def split_tokenizer(text):
    """按空格拆分已分词的文本"""
    return text.split()
#加载配置/加载模型/加载向量化器
config=Config()
rf_model = joblib.load(config.model_save_path)
tfidf = joblib.load(config.tfidf_save_path)

def pred_fun(data):
    #加载数据并分词
    text=data["text"]
    words=" ".join(jieba.lcut(text)[:50])
    #向量化
    featrues=tfidf.transform([words])
    #预测
    pred_id=rf_model.predict(featrues)[0]
    #加载类别
    label_list=["差评","好评"]
    pred=label_list[pred_id]
    data["label"]= pred
    print(data)
    return data

if __name__ == '__main__':
    pred_fun({"text":"这个手机真好"})