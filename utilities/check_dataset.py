import pandas as pd
import os
from utilities.parse_config import parameters
import numpy as np

data_dir = parameters['data_dir']
window_size = parameters['win_len']
stide = parameters['win_overlap']

def check_dataset(rootpath, filename):
    filepath = os.path.join(rootpath,  'temp', filename)
    df = pd.read_csv(filepath).to_numpy()
    df = np.array(df)
    df = pd.DataFrame(df)
    df_cleaned = df.dropna()

    df_cleaned.to_csv(filepath, index=False,header=False,encoding='utf-8')
    print(f"已删除包含NaN值的行并覆盖保存到 {filepath}")
