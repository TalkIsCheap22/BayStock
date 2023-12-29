import json
import csv
import numpy as np
from scipy.stats import norm

def read_stocks(stocks_file):
    datas, stock = [], []
    with open(stocks_file) as file:
        raw = list(csv.reader(file))
        stocks_points = raw[1:]
        labels = raw[0]
    name, T = stocks_points[0][6], 1
    while stocks_points[T][6] == name:
        T += 1
    N = len(stocks_points)
    for i in range(N):
        for j in range(1,6):
            if stocks_points[i][j] == "":
                stocks_points[i][j] = stocks_points[i][1]
        stock.append(stocks_points[i])
        if i % T == 0:
            datas.append(stock)
            stock = []
    return labels, datas

def train_test_divide(data, train_ratio):
    tot = len(data)
    train_len = int(train_ratio * tot)
    train_dataset = data[0:train_len]
    test_dataset = data[train_len:]
    return train_dataset, test_dataset



######


def read_single_stock(single_stock_file):
    datas = []
    with open(single_stock_file, mode="r", encoding="utf-8-sig") as file:
        reader = csv.reader(file)
        header = next(reader)
        time_stamp = 0
        for row in reader:
            data = {}
            data["time_stamp"] = time_stamp
            time_stamp += 1
            data["open"] = float(row[1])
            data["high"] = float(row[2])
            data["low"] = float(row[3])
            data["close"] = float(row[4])
            data["volumn"] = float(row[5])
            datas.append(data)
        for i in range(1,len(datas)-1):
            prev_diff, next_diff = round(datas[i]["close"]-datas[i-1]["close"], 2), round(datas[i+1]["close"]-datas[i]["close"], 2)
            datas[i]["prev_diff"] = prev_diff
            datas[i]["next_diff"] = next_diff
        diffs = [data["next_diff"] for data in datas[1:-1]]
        avg_diff = float(sum(diffs)) / len(diffs)
        errors = list(map(lambda x: (x - avg_diff) ** 2, diffs))
        sigma = sum(errors) / len(errors)
        diffs = list(map(lambda x: (x - avg_diff) / sigma, diffs))
        def normal_classifier(x):
            pos = norm.ppf(0.85)
            if x > pos:
                return 0
            elif x > 0:
                return 1
            elif x > -pos:
                return 2
            return 3
        diffs = list(map(normal_classifier, diffs))
        for i in range(1, len(datas)-1):
            datas[i]["price_trend"] = diffs[i-1]
    return datas[1:-1]