import click
import os
import pandas as pd
import random
import time
import numpy as np
import yfinance as yf

DATA_DIR = "SP500_dataset"
RANDOM_SLEEP_TIMES = (1, 5)

SP500_LIST_PATH = "./datasets/SP500_Companies.csv"


def _load_symbols():
    df_sp500 = pd.read_csv(SP500_LIST_PATH, encoding='ISO-8859-1')
    stock_symbols = df_sp500['Symbol'].unique().tolist()
    print("Loaded %d stock symbols" % len(stock_symbols))
    return stock_symbols


def fetch_prices(symbol, out_name):
    print("Fetching {} ...".format(symbol))

    df = yf.download(symbol, start="2015-01-01", end="2023-01-01", proxy="http://127.0.0.1:7890")
    df.to_csv(f'./SP500_dataset/{symbol}.csv', encoding='ISO-8859-1')
    data = pd.read_csv(out_name)
    if data.empty:
        print("Remove {} because the data set is empty.".format(out_name))
        os.remove(out_name)
    else:
        dates = data.iloc[:, 0].tolist()
        print("# Fetched rows: %d [%s to %s]" % (data.shape[0], dates[-1], dates[0]))

    sleep_time = np.round(np.random.uniform(low=1, high=3), 2)
    print("Sleeping ... %ds" % sleep_time)
    time.sleep(sleep_time)
    return True


@click.command(help="Fetch stock prices data")
@click.option('--continued', is_flag=True)
def main(continued):
    random.seed(time.time())
    num_failure = 0

    symbols = _load_symbols()
    for idx, sym in enumerate(symbols):
        out_name = os.path.join(DATA_DIR, sym + ".csv")
        if continued and os.path.exists(out_name):
            print("Fetched", sym)
            continue

        succeeded = fetch_prices(sym, out_name)
        num_failure += int(not succeeded)

        if idx % 10 == 0:
            print("# Failures so far [%d/%d]: %d" % (idx + 1, len(symbols), num_failure))


if __name__ == "__main__":
    main()
