import pandas as pd
import numpy as np
import os
import math
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

# the information of stocks(i.e. name & category of sector).
SP500_name = pd.read_csv("./datasets4/arrange_sp500_companies.csv", encoding='ISO-8859-1')


def stock_preprocess(csv_path, agg_week):
    # the information of stocks(i.e. name & category of sector).
    SP500_name = pd.read_csv(csv_path, encoding='ISO-8859-1')

    # original information of stock price.
    SP500_stock = {}
    for target in SP500_name["Symbol"].unique():
        da = {}
        da["category"] = SP500_name[SP500_name.Symbol == target]["Sector"].iloc[0]
        da["stock_price"] = pd.read_csv("./SP500_dataset/%s.csv" % (target), encoding='ISO-8859-1')
        SP500_stock[target] = da

    # Let all stocks has the same date.
    print(SP500_stock.keys())
    need_day = np.array(SP500_stock["AAPL"]["stock_price"]["Date"])
    for target in SP500_stock.keys():
        SP500_stock[target]["stock_price"] = SP500_stock[target]["stock_price"][
            SP500_stock[target]["stock_price"]["Date"].isin(need_day)].reset_index(
            drop=True
        )
        SP500_stock[target]["stock_price"].index = SP500_stock[target]["stock_price"]["Date"]
    print('same date over')

    ### feature ###

    # normalize stock price
    normalize_scalar = {}
    for target in SP500_stock.keys():
        scaler = StandardScaler()
        nor_data = scaler.fit_transform(np.array(SP500_stock[target]["stock_price"]["Close"]).reshape(-1, 1)).ravel()
        SP500_stock[target]["stock_price"]["nor_close"] = nor_data
        normalize_scalar[target] = scaler

    print('normalize over')

    # calculate return ratio
    for target in SP500_stock.keys():
        return_tratio = []
        data = np.array(SP500_stock[target]["stock_price"]["Close"])
        for i in range(len(data)):
            if i == 0:
                return_tratio.append(0)
            else:
                return_tratio.append((data[i] - data[i - 1]) / data[i - 1])
        SP500_stock[target]["stock_price"]["return ratio"] = return_tratio
    print('calculate return ratio over')

    # feature of c_open / c_close / c_low
    for target in SP500_stock.keys():
        function = lambda x, y: (x / y) - 1
        data = SP500_stock[target]["stock_price"]
        data["c_open"] = list(map(function, data["Open"], data["Close"]))
        data["c_high"] = list(map(function, data["High"], data["Close"]))
        data["c_low"] = list(map(function, data["Low"], data["Close"]))
    print('calculate c_feature over')

    # 5 / 10 / 15 / 20 / 25 / 30 days moving average
    for target in SP500_stock.keys():
        data = SP500_stock[target]["stock_price"]["Close"]
        for i in [5, 10, 15, 20, 25, 30]:
            q = []
            for day in range(len(data)):
                if day >= i - 1:
                    q.append((np.mean(data.iloc[day - i + 1: day + 1]) / data.iloc[day]) - 1)
                if day < i - 1:
                    q.append(0)
            SP500_stock[target]["stock_price"]["%s-days" % (i)] = q
    print('calculate moving average over')

    # category of sector (one hot encoding)
    label = LabelEncoder()
    label.fit(SP500_name["Sector"].unique())

    for target in SP500_stock.keys():
        for label in SP500_name["Sector"].unique():
            cate = SP500_stock[target]["category"]
            if label != cate:
                SP500_stock[target]["stock_price"]["label_%s" % (label)] = 0
            if label == cate:
                SP500_stock[target]["stock_price"]["label_%s" % (label)] = 1
    print('one hot sector over')

    # total feature
    features = {}
    for target in SP500_stock.keys():
        features[target] = SP500_stock[target]["stock_price"].iloc[30:,
                           [4] + list(range(7, len(SP500_stock[target]["stock_price"].columns)))].reset_index(drop=True)
    print('all features over')

    # movement of stock
    Y_buy_or_not = {}
    for target in SP500_stock.keys():
        Y_buy_or_not[target] = (features[target]["return ratio"] >= 0) * 1
        features[target]['buy_or_not'] = Y_buy_or_not[target]
    print('movement label over')

    ## Trianing & Testing ##
    train_size = 0.8
    test_size = 0.2
    days = len(features["AAPL"])

    train_day = int(days * train_size)

    # data of training set and testing set
    train_data = {}
    test_data = {}

    train_Y_buy_or_not = {}
    test_Y_buy_or_not = {}

    train_return_ratio = {}
    test_return_ratio = {}

    for i in SP500_stock.keys():
        train_data[i] = features[i].drop(['return ratio', 'buy_or_not'], axis=1).iloc[:train_day, :]
        train_Y_buy_or_not[i] = features[i]['buy_or_not'][:train_day]
        train_return_ratio[i] = features[i]['return ratio'][:train_day]

        test_data[i] = features[i].drop(['return ratio', 'buy_or_not'], axis=1).iloc[train_day:, :]
        test_Y_buy_or_not[i] = features[i]['buy_or_not'][train_day:]
        test_return_ratio[i] = features[i]['return ratio'][train_day:]
    print('train&test over')

    # week represents the number of our inputs
    # train
    train = {}
    for w in range(agg_week):
        train_x = []
        for tr_ind in range(len(train_data["AAPL"]) - 7 - (agg_week - 2) - 1):
            tr = []
            valid_train = True  # 假设数据是有效的
            for target in SP500_stock.keys():
                data = train_data[target]
                slice_train = data.iloc[tr_ind + w: tr_ind + w + 7, :].values

                # 检查数据形状是否为 (7, 30)
                if slice_train.shape != (7, 30):
                    valid_train = False
                    break  # 如果数据形状不正确，跳出循环

                tr.append(slice_train)

            if valid_train:
                train_x.append(tr)
        train["x%s" % (w + 1)] = np.array(train_x)
    print('train_x over')

    train_y1, train_y2 = [], []
    for tr_ind in range(len(train_data["AAPL"]) - 7 - (agg_week - 2) - 1):
        all_stock_name = list(SP500_stock.keys())
        tr_y1, tr_y2 = [], []
        for target in SP500_stock.keys():
            data = train_data[target]
            tr_y1.append(train_return_ratio[target].iloc[tr_ind + (agg_week - 1) + 7])
            tr_y2.append(train_Y_buy_or_not[target].iloc[tr_ind + (agg_week - 1) + 7])
        train_y1.append(tr_y1)
        train_y2.append(tr_y2)
    train["y_return ratio"] = np.array(train_y1)
    train["y_up_or_down"] = np.array(train_y2)
    print('train_y over')
    print(train["y_return ratio"].shape)

    # test
    test = {}
    for w in range(agg_week):
        test_x = []
        for te_ind in range(len(test_data["AAPL"]) - 7 - (agg_week - 2) - 1):
            te = []
            valid_test = True  # 假设数据是有效的
            for target in SP500_stock.keys():
                data = test_data[target]
                slice_test = data.iloc[te_ind + w: te_ind + w + 7, :].values
                # 检查数据形状是否为 (7, 30)
                if slice_test.shape != (7, 30):
                    valid_test = False
                    break  # 如果数据形状不正确，跳出循环
                te.append(slice_test)
            if valid_test:
                test_x.append(te)
        test["x%s" % (w + 1)] = np.array(test_x)
    print('test_x over')

    test_y1, test_y2 = [], []
    for te_ind in range(len(test_data["AAPL"]) - 7 - (agg_week - 2) - 1):
        te_y1, te_y2 = [], []
        for target in SP500_stock.keys():
            data = test_data[target]
            te_y1.append(test_return_ratio[target].iloc[te_ind + (agg_week - 1) + 7])
            te_y2.append(test_Y_buy_or_not[target].iloc[te_ind + (agg_week - 1) + 7])
        test_y1.append(te_y1)
        test_y2.append(te_y2)
    test["y_return ratio"] = np.array(test_y1)
    test["y_up_or_down"] = np.array(test_y2)
    print('test_y over')
    print(test["y_return ratio"].shape)
    data = {"train": train, "test": test}

    return data


def create_edge(data_path, csv_path):
    df = pd.read_csv(csv_path, encoding='ISO-8859-1')
    company_list = df["Symbol"].unique()
    sector_list = df["Sector"].unique()

    outer_edge = []
    for i in range(len(sector_list)):
        for j in range(i, len(sector_list)):
            outer_edge.append((i, j))

    outer_edge = np.array(outer_edge)
    outer_dir = os.path.join(data_path, "outer_edge.npy")
    np.save(outer_dir, outer_edge)

    name2index = {}
    for i, key in enumerate(company_list):
        name2index[key] = i

    inner_edge = []
    for sector in sector_list:
        intra_companies = df[df.Sector == sector]["Symbol"].unique()
        for i in range(len(intra_companies)):
            for j in range(i, len(intra_companies)):
                inner_edge.append((name2index[intra_companies[i]], name2index[intra_companies[j]]))

    inner_edge = np.array(inner_edge)
    inner_dir = os.path.join(data_path, "inner_edge.npy")
    np.save(inner_dir, inner_edge)


if __name__ == '__main__':
    DATA_DIR = ".\\datasets"
    SP500_LIST_PATH = ".\\datasets\\SP500_Companies.csv"

    print("\n==================== start preprocess data ====================")
    data = stock_preprocess(csv_path=SP500_LIST_PATH, agg_week=4)
    with open("./datasets4/sp500_data.pkl", "wb") as f:
        pickle.dump(data, f)
    f.close()

    print("\n==================== start create edge ====================")
    create_edge(data_path=DATA_DIR, csv_path=SP500_LIST_PATH)

    print('\nall over')
