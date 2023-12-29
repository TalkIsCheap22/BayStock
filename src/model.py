import numpy as np
import math

class SingleModel:
    def __init__(self, signal, label, K):
        self.K = K
        self.pi = [float(1) / self.K] * self.K
        self.mu = [0] * self.K
        self.signal = signal
        self.label = label
        self.sigma = 1

    def print_parameters(self):
        print("pi:{a}".format(a=self.pi))
        print("mu:{a}".format(a=self.mu))
        print("sigma:{a}".format(a=self.sigma))
    
    def train(self, dataset):
        print("--- Starting the training ---")
        N = len(dataset)
        signals = [data[self.signal] for data in dataset]
        labels = [data[self.label] for data in dataset]
        label_signals = [[] for i in range(self.K)]
        for i in range(N):
            label_signals[labels[i]].append(signals[i])
        tot_square = 0
        for k in range(self.K):
            self.pi[k] = len(label_signals[k]) / N
            self.mu[k] = sum(label_signals[k]) / len(label_signals[k])
            square = list(map(lambda x: (x-self.mu[k]) ** 2, label_signals[k]))
            tot_square += sum(square)
        self.sigma = tot_square / (N - self.K)
        print("--- Training finished ---")


    def evaluate(self, dataset):
        def forward(signal):
            val, idx = float("-inf"), 0
            for k in range(self.K):
                res = self.mu[k] * (2 * signal - 1) / (self.sigma ** 2) + math.log(self.pi[k])
                if res > val:
                    val = res
                    idx = k
            return idx
        print("---- staring the evaluation ---")
        self.print_parameters()
        correct1 = 0
        for data in dataset:
            pred = forward(data[self.signal])
            if pred == data[self.label]:
                correct1 += 1
            print("real: {a}, pred: {b}".format(a=data[self.label],b=pred))
        accuracy1 = float(correct1) / len(dataset)
        print("Evaluation finished, exactly correct rate is {a}.".format(a=accuracy1))