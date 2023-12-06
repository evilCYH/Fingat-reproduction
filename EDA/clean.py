import pandas as pd
import os

SP500_name = pd.read_csv("./datasets/sp500_companies.csv", encoding='ISO-8859-1')

for target in SP500_name["Symbol"].unique():
    try:
        df = pd.read_csv(f"./SP500_dataset/{target}.csv", encoding='ISO-8859-1')
    except FileNotFoundError:
        print(f"File for {target} not found, skipping.")
        continue

    df['Date'] = pd.to_datetime(df['Date'])

    # 检查第一个日期是否是 2001-01-02 and 最后一个
    if df['Date'].iloc[0] == pd.Timestamp('2015-01-02') and df['Date'].iloc[-1] == pd.Timestamp('2022-12-30'):
        continue
    else:
        # 删除不符合条件的行
        SP500_name = SP500_name[SP500_name["Symbol"] != target]

        # 删除相应的股票数据文件
        os.remove(f"./SP500_dataset/{target}.csv")

# 保存
SP500_name.to_csv("./datasets/sp500_companies.csv", index=False, encoding='ISO-8859-1')
