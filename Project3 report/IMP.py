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


def calculate_influence(seeds, model):
    sum = 0
    times = 2000
    if model.upper() == "IC":
        for i in range(times):
            sum += one_IC_sample(seeds)
        return  sum / times
    elif model.upper() == "LT":
        for i in range(times):
            sum += one_LT_sample(seeds)
        return sum / times


def one_IC_sample(seed: set):
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


def one_LT_sample(seed: set):
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


def CELF(seed_num: int, model):
    global graph_matrix
    global graph_list
    global neighbor_prob
    global graph_parent_list
    influence_dict = dict()
    for i in range(1, len(neighbor_prob)):
        influence_dict[i] = calculate_influence({i}, model)
    seeds = set()
    # nodes = set(neighbor_prob.keys())
    tmp_max_node = max(influence_dict, key=influence_dict.get)
    seeds.add(tmp_max_node)
    influence_dict.pop(tmp_max_node)
    # nodes.remove(tmp_max_node)
    while len(seeds) < seed_num:
        tmp_max_node = max(influence_dict, key=influence_dict.get)
        influence_dict[tmp_max_node] = calculate_influence(seeds | {tmp_max_node}, model) - calculate_influence(seeds, model)
        tmp_max_node2 = max(influence_dict, key=influence_dict.get)
        if tmp_max_node == tmp_max_node2:
            seeds.add(tmp_max_node2)
            influence_dict.pop(tmp_max_node2)
        else:
            continue
    return seeds


def print_results(seeds: set):
    # print(calculate_influence(seeds, "IC"))
    while seeds:
        print(seeds.pop())


if __name__ == '__main__':
    start = time.time()
    preprocess(sys.argv[2])
    print_results(CELF(int(sys.argv[4]), sys.argv[6]))
    # print(time.time() - start)




