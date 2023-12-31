import numpy as np
import math 

class Model:
    def __init__(self, labels, ind, deps):
        self.ind = ind
        self.deps = deps
        self.labels = labels
        self.mu = []
        for _ in range(len(self.deps)):
            self.mu.append([0, 0, 0])
        self.sigma_square = [1] * len(deps)  
        print("The model is built successfully")
    
    def train(self, dataset):
        print(len(self.mu))
        print("Training session starts")
        idx_ind = self.labels.index(self.ind)
        for i in range(len(self.deps)):
            avg0s, avg1s, avg2s, sigmas_square = [], [], [], []
            idx_dep = self.labels.index(self.deps[i])
            for j in range(len(dataset)):
                data0, data1, data2 = list(filter(lambda lst: lst[idx_dep] == 0, dataset[i])), list(filter(lambda lst: lst[idx_dep] == 1, dataset[i])), list(filter(lambda lst: lst[idx_dep] == 2, dataset[i]))
                data0, data1, data2 = [x[idx_ind] for x in data0], [x[idx_ind] for x in data1], [x[idx_ind] for x in data2]
                avg0, avg1, avg2 = float(sum(data0))/len(data0), float(sum(data1))/len(data1), float(sum(data2))/len(data2)
                square0, square1, square2 = list(map(lambda x:(x-avg0) ** 2, data0)), list(map(lambda x:(x-avg1) ** 2, data1)), list(map(lambda x:(x-avg2) ** 2, data2))
                sigma_square = (sum(square0) + sum(square1) + sum(square2)) / (len(dataset[i]) - 3) 
                avg0s.append(avg0)
                avg1s.append(avg1)
                avg2s.append(avg2)
                sigmas_square.append(sigma_square)
            avg0 = sum(avg0s)/len(avg0s)
            avg1 = sum(avg1s)/len(avg1s)
            avg2 = sum(avg2s)/len(avg2s)
            sigma_square = sum(sigmas_square)/len(sigmas_square)    ##needs update
            self.mu[i] = [avg0, avg1, avg2]
            self.sigma_square[i] = sigma_square
        print("Training session ends")

    def evaluate(self, dataset):
        print("Start evaluation")
        idx_ind = self.labels.index(self.ind)
        idx_deps = list(map(lambda x:self.labels.index(x), self.deps))
        for stock in dataset:
            print("Predicting stock named {a}".format(a=stock[0][6]))
            correct = [0] * len(self.deps)
            for point in stock:
                for i in range(len(self.deps)):
                    #have not considered the impact of prior distribution
                    vals = [point[idx_ind] * self.mu[i][j] / self.sigma_square[j] - (self.mu[i][j] ** 2) / (2 * self.sigma_square[j]) + math.log(1) for j in range(len(self.deps))]
                    #lst = [0,1,1,2]
                    #pred = np.random.choice(lst)
                    pred = vals.index(max(vals))
                    true_label = point[idx_deps[i]]
                    if pred == true_label:
                        correct[i] += 1
                    print("dep {a}: pred = {b}, true_label = {c}".format(a=i+1,b=pred,c=true_label))
            ratio = list(map(lambda x:x/len(stock), correct))
            print("ratio : {a}".format(a=ratio))
        print("Evaluation ends")
