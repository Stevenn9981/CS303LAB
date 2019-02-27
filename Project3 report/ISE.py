import sys
import numpy as np
import copy
import random
import time

__author__ = "Wentao Ning"


graph_matrix = list()
graph_list = dict()
graph_parent_list = dict()
neighbor_prob = dict()


def preprocess(network):
    file = open(network)
    line1 = file.readline().split()
    num_nodes = int(line1[0])
    global graph_matrix
    global graph_list
    global neighbor_prob
    global graph_parent_list
    graph_matrix = np.zeros((num_nodes + 1, num_nodes + 1))
    data = file.readlines()
    for i in range(num_nodes + 1):
        graph_list[i] = list()
        graph_parent_list[i] = list()
        neighbor_prob[i] = 0
    for edge in data:
        tmp = edge.split()
        graph_list[int(tmp[0])].append(int(tmp[1]))
        graph_matrix[int(tmp[0]), int(tmp[1])] = 1
        neighbor_prob[int(tmp[1])] = float(tmp[2])
        graph_parent_list[int(tmp[1])].append(int(tmp[0]))


def calculate_influence(network, seed_set, model, time_limit):
    start = time.time()
    preprocess(network)
    file = open(seed_set)
    data = file.readlines()
    seeds = list()
    sum = 0
    times = 10000
    # print(neighbor_prob)
    # print(graph_list[10])
    # print(graph_parent_list[6])
    for seed in data:
        seeds.append(int(seed))
    if model.upper() == "IC":
        for i in range(times):
            sum += one_IC_sample(seeds)
        # while time.time() - start < time_limit - 0.05:
        #     sum += one_IC_sample(seeds)
        #     times += 1
        return  sum / times
    elif model.upper() == "LT":
        for i in range(times):
            sum += one_LT_sample(seeds)
        # while time.time() - start < time_limit - 0.05:
        #     sum += one_LT_sample(seeds)
        #     times += 1
        return sum / times


def one_IC_sample(seed: list):
    global graph_matrix
    global graph_list
    global neighbor_prob
    activated = set(seed)
    activitySet = set(seed)
    count = len(activitySet)
    while activitySet:
        newActivitySet = set()
        for se in activitySet:
            for neighbor in graph_list[se]:
                if neighbor not in activated:
                    if random.random() < neighbor_prob[neighbor]:
                        activated.add(neighbor)
                        newActivitySet.add(neighbor)
        count += len(newActivitySet)
        activitySet = newActivitySet
    return count


def one_LT_sample(seed: list):
    global graph_matrix
    global graph_list
    global neighbor_prob
    global graph_parent_list
    activated = set(seed)
    activitySet = set(seed)
    thresholds = dict()
    while activitySet:
        newActivitySet = set()
        for se in activitySet:
            tmp1 = graph_list[se]
            for neighbor in tmp1:
                if neighbor not in activated:
                    w_total = 0
                    # print(activated)
                    tmp2 = graph_parent_list[neighbor]
                    for node in tmp2:
                        # print(node)
                        if node in activated:
                            w_total += neighbor_prob[neighbor]
                    if neighbor not in thresholds:
                        thresholds[neighbor] = random.random()
                    if w_total >= thresholds[neighbor]:
                        activated.add(neighbor)
                        newActivitySet.add(neighbor)
        activitySet = newActivitySet
    return len(activated)


if __name__ == '__main__':
    # start = time.time()
    network = sys.argv[2]
    seed_set = sys.argv[4]
    model = sys.argv[6]
    time_limit = float(sys.argv[8])
    print(calculate_influence(network, seed_set, model, time_limit))
    # print(time.time() - start)




