import json
import csv
import numpy as np
from scipy.stats import norm


def read_stocks(stocks_file):
    dataset, stock = [], []
    with open(stocks_file) as file:
        raw = list(csv.reader(file))
        stocks_points = raw[1:]
        labels = raw[0]
    name, T = stocks_points[0][6], 1
    while stocks_points[T][6] == name:
        T += 1
    N = len(stocks_points)
    for i in range(N):
        if i != 0 and i % T == 0:
            dataset.append(stock)
            stock = []
        for j in range(1,6):
            if stocks_points[i][j] == "":
                stocks_points[i][j] = stocks_points[0][j]
        stocks_points[i][1:6] = list(map(lambda x:float(x), stocks_points[i][1:6]))
        stock.append(stocks_points[i])
    return labels, dataset

def train_test_divide(data, train_ratio):
    tot = len(data)
    train_len = int(train_ratio * tot)
    train_dataset = data[0:train_len]
    test_dataset = data[train_len:]
    return train_dataset, test_dataset

####add labels

def classify_3(val, one, two):
    if val > one:
        return 0
    elif val > two:
        return 1
    return 2

def avg_increase_5days(labels, dataset):
    labels.append("avg_increase_5days")
    T = len(dataset[0])
    for stock in dataset:
        for i in range(5,T):
            avg = (stock[i][4] / stock[i-5][4] - 1) / 5 
            stock[i].append(classify_3(avg, 0.008, -0.005))
        for i in range(5):
            stock[i].append(stock[5][-1])
    return labels, dataset

def price_trend_1day(labels, dataset):      #0.014 is the 75 percent point, while -0.011 is the 25 percent ones
    labels.append("price_trend_1day")
    T = len(dataset[0])
    for stock in dataset:
        for i in range(0,T-1):
            trend = stock[i+1][4] / stock[i][4] - 1
            stock[i].append(classify_3(trend, 0.014, -0.011))
        stock[T-1].append(stock[T-2][-1])
    return labels, dataset

def price_trend_5days(labels, dataset):     #0.008 is the 75 percent point, while -0.005 is the 25 percent one
    labels.append("price_trend_5days")
    T = len(dataset[0])
    for stock in dataset:
        for i in range(0,T-5):
            avg = (stock[i+5][4] / stock[i][4] - 1) / 5 
            stock[i].append(classify_3(avg, 0.008, -0.005))
        for i in range(T-5,T):
            stock[i].append(stock[T-6][-1])
    return labels, dataset

def read_and_process(stocks_file):
    labels, dataset = read_stocks(stocks_file)
    labels, dataset = avg_increase_5days(labels, dataset)
    labels, dataset = price_trend_1day(labels, dataset)
    labels, dataset = price_trend_5days(labels, dataset)
    return labels, dataset

######depricated func

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