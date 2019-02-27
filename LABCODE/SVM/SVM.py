import time
import numpy as np
import random
import sys


class SVMPattern:
    def __init__(self, data, labels, C, tol):
        self.X = data
        self.labelMat = labels
        self.C = C
        self.tol = tol
        self.m = np.shape(data)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))


def load(file):
    data = []
    labels = []
    fr = open(file)
    for line in fr.readlines():
        line = line.strip().split()
        line = list(map(float, line))
        data.append(line[0:-1])
        labels.append(line[-1])
    return data, labels


def chooseJ(i, pattern, Ei):
    max_k = -1
    max_delta_e = 0
    Ej = 0
    pattern.eCache[i] = [1, Ei]
    eca_list = np.nonzero(pattern.eCache[:, 0].A)[0]
    if (len(eca_list)) > 1:
        for k in eca_list:
            if k == i: continue
            Ek = float(
                np.multiply(pattern.alphas, pattern.labelMat).T * (pattern.X * pattern.X[k, :].T) + pattern.b) - float(
                pattern.labelMat[k])
            delta_e = abs(Ei - Ek)
            if delta_e > max_delta_e:
                max_k = k
                max_delta_e = delta_e
                Ej = Ek
        return max_k, Ej
    else:
        j = int(random.uniform(0, pattern.m))
        Ej = float(
            np.multiply(pattern.alphas, pattern.labelMat).T * (pattern.X * pattern.X[j, :].T) + pattern.b) - float(
            pattern.labelMat[j])
    return j, Ej


def changeErr(pattern, k):
    Ek = float(np.multiply(pattern.alphas, pattern.labelMat).T * (pattern.X * pattern.X[k, :].T) + pattern.b) - float(
        pattern.labelMat[k])
    pattern.eCache[k] = [1, Ek]


def innerL(i, pattern):
    Ei = float(np.multiply(pattern.alphas, pattern.labelMat).T * (pattern.X * pattern.X[i, :].T) + pattern.b) - float(
        pattern.labelMat[i])
    if ((pattern.labelMat[i] * Ei < -pattern.tol) and (pattern.alphas[i] < pattern.C)) or (
            (pattern.labelMat[i] * Ei > pattern.tol) and (pattern.alphas[i] > 0)):
        j, Ej = chooseJ(i, pattern, Ei)
        alphaIold = pattern.alphas[i].copy()
        alphaJold = pattern.alphas[j].copy()
        if pattern.labelMat[i] != pattern.labelMat[j]:
            L = max(0, pattern.alphas[j] - pattern.alphas[i])
            H = min(pattern.C, pattern.C + pattern.alphas[j] - pattern.alphas[i])
        else:
            L = max(0, pattern.alphas[j] + pattern.alphas[i] - pattern.C)
            H = min(pattern.C, pattern.alphas[j] + pattern.alphas[i])
        if L == H:
            return 0
        eta = 2.0 * pattern.X[i, :] * pattern.X[j, :].T - pattern.X[i, :] * pattern.X[i, :].T - pattern.X[j,
                                                                                                :] * pattern.X[j, :].T
        if eta >= 0:
            return 0
        pattern.alphas[j] -= pattern.labelMat[j] * (Ei - Ej) / eta
        pattern.alphas[j] = H if pattern.alphas[j] > H else max(pattern.alphas[j], L)
        changeErr(pattern, j)
        if abs(pattern.alphas[j] - alphaJold) < 0.00001:
            return 0
        pattern.alphas[i] += pattern.labelMat[j] * pattern.labelMat[i] * (alphaJold - pattern.alphas[j])
        changeErr(pattern, i)
        b1 = pattern.b - Ei - pattern.labelMat[i] * (pattern.alphas[i] - alphaIold) * pattern.X[i, :] * pattern.X[i,
                                                                                                        :].T - \
             pattern.labelMat[j] * (
                     pattern.alphas[j] - alphaJold) * pattern.X[i, :] * pattern.X[j, :].T
        b2 = pattern.b - Ej - pattern.labelMat[i] * (pattern.alphas[i] - alphaIold) * pattern.X[i, :] * pattern.X[j,
                                                                                                        :].T - \
             pattern.labelMat[j] * (
                     pattern.alphas[j] - alphaJold) * pattern.X[j, :] * pattern.X[j, :].T
        if (0 < pattern.alphas[i]) and (pattern.C > pattern.alphas[i]):
            pattern.b = b1
        elif (0 < pattern.alphas[j]) and (pattern.C > pattern.alphas[j]):
            pattern.b = b2
        else:
            pattern.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def SMO(data, labels, C, tol, max, limit_time):
    start_time = time.time()
    pattern = SVMPattern(np.mat(data), np.mat(labels).transpose(), C, tol)
    times = 0
    es = True
    changes = 0
    current_time = time.time() - start_time
    while (times < max) and ((changes > 0) or es) and current_time < limit_time - 5:
        changes = 0
        if es:
            for i in range(pattern.m):
                changes += innerL(i, pattern)
            times += 1
        else:
            IS = np.nonzero((pattern.alphas.A > 0) * (pattern.alphas.A < C))[0]
            for i in IS:
                changes += innerL(i, pattern)
            times += 1
        if es:
            es = False
        elif changes == 0:
            es = True
        current_time = time.time() - start_time
    return pattern.b, pattern.alphas



def pegasos(data, labels, lam, T):
    data = np.mat(data)
    m, n = np.shape(data)
    w = np.zeros(n)
    t = 0
    for a in range(1, T):
        for i in range(m):
            t += 1
            eta = 1.0 / (lam * t)
            p = w * data[i, :].T
            if labels[i] * p < 1:
                w = (1.0 - 1 / t) * w + eta * labels[i] * data[i, :]
            else:
                w = (1.0 - 1 / t) * w
    return w


if __name__ == '__main__':
    start_time = time.time()
    # random.seed(10)
    limit_time = int(sys.argv[4])
    data, labels = load(sys.argv[1])
    # b, alphas = SMO(data, labels, 0.6, 0.001, 40, limit_time)
    # X = np.mat(data)
    # labelsT = np.mat(labels).transpose()
    # m, n = np.shape(X)
    # w = np.zeros((n, 1))
    # for i in range(m):
    #     w += np.multiply(alphas[i] * labelsT[i], X[i, :].T)
    w = pegasos(data, labels, 1.8, 100)
    data, labels = load(sys.argv[2])
    test_count = 0
    right_count = 0
    for i in range(len(labels)):
        test_count += 1
        if w[0] * np.mat(data[i]).T > 0 : #np.dot(data[i], w.T[0]) + b > 0:
            print(1)
            if labels[i] == 1:
                right_count += 1
        else:
            print(-1)
            if labels[i] == -1:
                right_count += 1
    print(right_count / test_count)
    print(time.time() - start_time)
