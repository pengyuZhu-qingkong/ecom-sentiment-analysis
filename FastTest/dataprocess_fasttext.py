import jieba
import pandas as pd

from FastTest.Config import Config

config=Config()
def process_data(data_path,data_process_path):
    df_data=pd.read_csv(data_path,sep=',',header=0)
    df_data["words"]=df_data["review"].apply(lambda x: " ".join(jieba.lcut(x)[:50]))
    df_data.to_csv(data_process_path,sep='\t',header=True,index=False)
def gene_txt(path):
    df_data = pd.read_csv(path, sep='\t', header=0)
    words = df_data["words"]
    labels = df_data["label"]
    with open(f"{config.final_root}/train.txt", "w", encoding="utf-8") as f:
        for _, row in df_data.iterrows():
            f.write(f"__label__{row['label']} {row['words']}\n")
if __name__ == '__main__':
    # process_data(config.train_path,config.final_data_path_train)
    # process_data(config.val_path,config.final_data_path_val)
    # process_data(config.test_path,config.final_data_path_test)
    gene_txt(config.final_data_path_train)
    # gene_txt(config.final_data_path_val)
    # gene_txt(config.final_data_path_test)
