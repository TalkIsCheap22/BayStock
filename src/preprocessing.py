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

def increase_1day(labels, dataset):
    labels.append("increase_1day")
    T = len(dataset[0])
    for stock in dataset:
        for i in range(1,T):
            avg = stock[i][4] / stock[i-1][4] - 1 
            stock[i].append(avg)
        stock[0].append(stock[1][-1])
    return labels, dataset

def avg_increase_5days(labels, dataset):
    labels.append("avg_increase_5days")
    T = len(dataset[0])
    for stock in dataset:
        for i in range(5,T):
            avg = (stock[i][4] / stock[i-5][4] - 1) / 5 
            stock[i].append(avg)
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
    labels, dataset = increase_1day(labels, dataset)
    labels, dataset = avg_increase_5days(labels, dataset)
    labels, dataset = price_trend_1day(labels, dataset)
    labels, dataset = price_trend_5days(labels, dataset)
    return labels, dataset