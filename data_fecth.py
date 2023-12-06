import os
import pandas as pd
import random
import time
import numpy as np
import yfinance as yf


def _load_symbols(path):
    df_sp500 = pd.read_csv(path, encoding='ISO-8859-1')
    stock_symbols = df_sp500['Symbol'].unique().tolist()
    print("Loaded %d stock symbols" % len(stock_symbols))
    return df_sp500, stock_symbols


def fetch_prices(symbol, out_name):
    print("Fetching {} ...".format(symbol))

    # 需要提前打开代理，端口7890
    df = yf.download(symbol, start="2015-01-01", end="2023-01-01", proxy="http://127.0.0.1:7890")
    df.to_csv(out_name, encoding='ISO-8859-1')
    data = pd.read_csv(out_name)
    if data.empty:
        print("Remove {} because the data set is empty.".format(out_name))
        os.remove(out_name)
    else:
        dates = data.iloc[:, 0].tolist()
        print("# Fetched rows: %d [%s to %s]" % (data.shape[0], dates[-1], dates[0]))

    sleep_time = np.round(np.random.uniform(low=1, high=3), 2)
    print("Sleeping ... %.2fs" % sleep_time)
    time.sleep(sleep_time)
    return True


def check_date(symbol, out_name):
    try:
        df = pd.read_csv(out_name, encoding='ISO-8859-1')
    except FileNotFoundError:
        print(f"File for {symbol} not found, skipping.")
        return False
    df['Date'] = pd.to_datetime(df['Date'])

    # 检查第一个日期是否是 2015-01-02 and 最后一个 2022-12-30
    if df['Date'].iloc[0] == pd.Timestamp('2015-01-02') and df['Date'].iloc[-1] == pd.Timestamp('2022-12-30'):
        return True
    else:
        # 删除相应的股票数据文件
        os.remove(out_name)
        return False


def arrange_companies(path):
    df_sp500 = pd.read_csv(path, encoding='ISO-8859-1')
    sector_list = df_sp500["Sector"].unique()
    sorted_df = pd.concat([df_sp500[df_sp500["Sector"] == sector] for sector in sector_list])
    sorted_df.to_csv(path, index=False, encoding='ISO-8859-1')


if __name__ == '__main__':
    STOCK_DIR = ".\\datasets\\SP500_datasets"
    SP500_LIST_PATH = ".\\datasets\\SP500_Companies.csv"

    num_failure = 0
    data_failure = 0

    df_sp500, symbols = _load_symbols(SP500_LIST_PATH)

    print("==================== start fetch data ====================")
    for idx, sym in enumerate(symbols):
        out_name = os.path.join(STOCK_DIR, sym + ".csv")

        succeeded = fetch_prices(sym, out_name)
        num_failure += int(not succeeded)

        if idx % 10 == 0:
            print("# Failures so far [%d/%d]: %d" % (idx + 1, len(symbols), num_failure))

    print("\n==================== start check data ====================")
    for idx, sym in enumerate(symbols):
        out_name = os.path.join(STOCK_DIR, sym + ".csv")

        date_succeeded = check_date(sym, out_name)
        if not date_succeeded:
            # 删除不符合条件的行
            df_sp500 = df_sp500[df_sp500["Symbol"] != sym]

        data_failure += int(not date_succeeded)

    df_sp500.to_csv(SP500_LIST_PATH, index=False, encoding='ISO-8859-1')

    print("\n==================== start arrange data ====================")
    arrange_companies(SP500_LIST_PATH)

    print("\nall over")
    print(f"total : fetch failure {num_failure}, date not qualified {data_failure}")
