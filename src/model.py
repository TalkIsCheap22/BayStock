import numpy as np
import math 

class Model:
    def __init__(self, labels, inds, deps):
        self.inds = inds
        self.deps = deps
        self.labels = labels
        #mu[i][j][k]表示第i个因变量为第k类时自变量j的均值 shape = len(deps) * len(inds) * 3
        self.mu = []    
        self.sigma_square = []  #sigma_square[i][j]表示第i个因变量的意义下自变量j的方差 shape = len(deps) * len(inds)
        self.cov = []   #cov[i][k][j_0][j_1]表示第i个因变量的第k类协方差矩阵的[j_0][j_1] entry
        for _ in range(len(self.deps)):
            mu, sigma_square, cov = [], [], []
            for _ in range(len(self.inds)):
                mu.append([0, 0, 0])
                sigma_square.append([1, 1, 1])
            for _ in range(3):
                matrix = []
                for _ in range(len(self.inds)):
                    matrix.append([0] * len(self.inds))
                cov.append(matrix)
            self.mu.append(mu)
            self.sigma_square.append(sigma_square)  
            self.cov.append(cov)
        print("The model is built successfully")
    
    def train(self, dataset): #setting is a dict
        print("Training session starts")
        idx_inds = [self.labels.index(ind) for ind in self.inds]
        for i in range(len(self.deps)):
            idx_dep = self.labels.index(self.deps[i])
            data0, data1, data2 = [], [], []    #data0 is all whole datas with dep == self.deps[i]
            for j in range(len(dataset)):
                temp0, temp1, temp2 = list(filter(lambda lst: lst[idx_dep] == 0, dataset[i])), list(filter(lambda lst: lst[idx_dep] == 1, dataset[i])), list(filter(lambda lst: lst[idx_dep] == 2, dataset[i]))
                data0 += temp0
                data1 += temp1
                data2 += temp2
            cols0, cols1, cols2 = [], [], []
            for j in range(len(self.inds)):
                col0, col1, col2 = [x[idx_inds[j]] for x in data0], [x[idx_inds[j]] for x in data1], [x[idx_inds[j]] for x in data2]
                avg0, avg1, avg2 = float(sum(col0))/len(col0), float(sum(col1))/len(col1), float(sum(col2))/len(col2)
                square0, square1, square2 = list(map(lambda x:(x-avg0) ** 2, col0)), list(map(lambda x:(x-avg1) ** 2, col1)), list(map(lambda x:(x-avg2) ** 2, col2))
                sigma_square0, sigma_square1, sigma_square2 = sum(square0)/len(square0), sum(square1)/len(square1), sum(square2)/len(square2)
                self.mu[i][j] = [avg0, avg1, avg2]
                self.sigma_square[i][j] = [sigma_square0, sigma_square1, sigma_square2]
                cols0.append(col0)
                cols1.append(col1)
                cols2.append(col2)
            def cal_cov(lst1, lst2):
                avg1 = float(sum(lst1))/len(lst1)
                avg2 = float(sum(lst2))/len(lst2)
                tot = 0
                for i in range(len(lst1)):
                    tot += (lst1[i] - avg1) * (lst2[i] - avg2)
                return tot / (len(lst1) - 1)
            for j in range(len(self.inds)):
                for k in range(j+1):
                    cov0 = cal_cov(cols0[j], cols0[k])
                    cov1 = cal_cov(cols1[j], cols1[k])
                    cov2 = cal_cov(cols2[j], cols2[k])
                    self.cov[i][0][j][k] = self.cov[i][0][k][j] = cov0
                    self.cov[i][1][j][k] = self.cov[i][1][k][j] = cov1
                    self.cov[i][2][j][k] = self.cov[i][2][k][j] = cov2
        print("Training session ends")

    def evaluate(self, dataset, settings):
        def inv_mat(mat):
            a = np.array(mat)
            r = np.linalg.inv(a)
            return r.tolist()
        
        def det(mat):
            a = np.array(mat)
            return np.linalg.det(a)
        
        def mult_vec(vec1, vec2):
            ans = 0
            for a, b in zip(vec1, vec2):
                ans += (a * b)
            return ans

        def mult_matvec(mat, vec):
            ans = []
            for row in mat:
                ans.append(mult_vec(row, vec))
            return ans

        print("Start evaluation")
        tot_len = sum([len(stock) for stock in dataset])
        idx_deps_label = [self.labels.index(dep) for dep in settings["deps"]]
        idx_deps_self = [self.deps.index(dep) for dep in settings["deps"]]
        idx_inds_label = [self.labels.index(ind) for ind in settings["inds"]]
        idx_inds_self = [self.inds.index(ind) for ind in settings["inds"]]
        cov_rev = [self.cov[i] for i in idx_deps_self]  #filter the deps
        for i in range(len(cov_rev)):     #filter the inds
            for j in range(len(cov_rev[i])):
                cov_rev[i][j] = [cov_rev[i][j][l] for l in idx_inds_self]
                for k in range(len(cov_rev[i][j])):
                    cov_rev[i][j][k] = [cov_rev[i][j][k][l] for l in idx_inds_self]
                cov_rev[i][j] = inv_mat(cov_rev[i][j])
        
        tot_correct = [0] * len(settings["deps"])
        for stock in dataset:
            print("Predicting stock named {a}".format(a=stock[0][6]))
            prior, correct = [[0, 0, 0] for i in range(len(settings["deps"]))], [0] * len(settings["deps"])
            cnt = 0
            for point in stock:
                ind_vals = list(map(lambda x:point[x], idx_inds_label))
                for i in range(len(settings["deps"])):
                    #have not considered the impact of prior distribution
                    if cnt < 10:
                        pi = [0.33, 0.33, 0.33]
                    else:
                        pi = [prior[i][k]+1 for k in range(3)]
                        print("pi = {a}".format(a=pi))
                    cnt += 1
                    vals = [-0.5*mult_vec(mult_matvec(cov_rev[i][k],ind_vals),ind_vals)+mult_vec(mult_matvec(cov_rev[i][k],ind_vals),[self.mu[i][j][k] for j in idx_inds_self])-0.5*mult_vec(mult_matvec(cov_rev[i][k],[self.mu[i][j][k] for j in idx_inds_self]),[self.mu[i][j][k] for j in idx_inds_self])-0.5*abs(math.log(det(cov_rev[i][k])))+math.log(pi[k]) for k in range(3)]
                    pred = vals.index(max(vals))
                    #pred = np.random.randint(0,3)
                    true_label = point[idx_deps_label[i]]
                    prior[i][true_label] += 1
                    if pred == true_label:
                        correct[i] += 1
                    print("dep {a}: pred = {b}, true_label = {c}".format(a=i+1,b=pred,c=true_label))
            ratio = list(map(lambda x:x/len(stock), correct))
            for i in range(len(settings["deps"])):
                tot_correct[i] += correct[i]
            print("ratio : {a}".format(a=ratio))
        total_ratio = list(map(lambda x:x/tot_len, tot_correct))
        print("Evaluation ends, total ratio : {a}".format(a=total_ratio))
