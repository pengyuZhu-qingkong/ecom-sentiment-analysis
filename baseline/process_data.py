import jieba
import pandas as pd

from config import Config
config=Config()
def process_data(data_path,data_process_path):
    df_data=pd.read_csv(data_path,sep=',',header=0)
    print(f"df_data->",df_data.head(10))
    df_data['words'] = df_data['review'].apply(lambda x: " ".join(jieba.lcut(x)[:50]))
    print(df_data.head())
    df_data.to_csv(data_process_path,sep='\t',header=True,index=False)

if __name__ == '__main__':
    process_data(config.train_path,config.final_data_path_train)
    process_data(config.val_path,config.final_data_path_val)
    process_data(config.test_path,config.final_data_path_test)