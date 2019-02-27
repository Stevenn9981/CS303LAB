import copy
import random
import sys
import time
import multiprocessing as mp
import numpy as np

# python 3 only!!!
import collections
import math

args = list()
IF_PREPROCESSED = False
co = list()
count_cal_pro = 0
requied_edges_num = 0
vertices_num = 0
depot = 1
capacity = 0


class Graph:
    ''' graph class inspired by https://gist.github.com/econchick/4666413
    '''

    def __init__(self, mygraph):
        self.vertices = set()

        # makes the default value for all vertices an empty list
        self.edges = collections.defaultdict(list)
        self.weights = {}
        for i in mygraph:
            self.add_vertex(i)
        for i in mygraph:
            for j in mygraph[i]:
                self.add_edge(i, j, mygraph[i][j][0])

    def add_vertex(self, value):
        self.vertices.add(value)

    def add_edge(self, from_vertex, to_vertex, distance):
        if from_vertex == to_vertex: pass  # no cycles allowed
        self.edges[from_vertex].append(to_vertex)
        self.weights[(from_vertex, to_vertex)] = distance

    def __str__(self):
        string = "Vertices: " + str(self.vertices) + "\n"
        string += "Edges: " + str(self.edges) + "\n"
        string += "Weights: " + str(self.weights)
        return string


def dijkstra(graph, start):
    # initializations
    S = set()

    # delta represents the length shortest distance paths from start -> v, for v in delta.
    # We initialize it so that every vertex has a path of infinity (this line will break if you run python 2)
    delta = dict.fromkeys(list(graph.vertices), math.inf)
    previous = dict.fromkeys(list(graph.vertices), None)

    # then we set the path length of the start vertex to 0
    delta[start] = 0

    # while there exists a vertex v not in S
    while S != graph.vertices:
        # let v be the closest vertex that has not been visited...it will begin at 'start'
        v = min((set(delta.keys()) - S), key=delta.get)

        # for each neighbor of v not in S
        for neighbor in set(graph.edges[v]) - S:
            new_path = delta[v] + graph.weights[v, neighbor]

            # is the new path from neighbor through
            if new_path < delta[neighbor]:
                # since it's optimal, update the shortest path for neighbor
                delta[neighbor] = new_path

                # set the previous vertex of neighbor to v
                previous[neighbor] = v
        S.add(v)
    return (delta, previous)


def shortest_path(cost_matrix, start, end):
    global requied_edges_num
    global vertices_num
    global count_cal_pro
    count_cal_pro += 1
    return floyd(cost_matrix, start, end, False if count_cal_pro == 1 else True)


def floyd(cost_matrix, start, end, IF_PREPROCESSED):
    global co
    if not IF_PREPROCESSED:
        for a in range(1, len(cost_matrix)):
            for b in range(1, len(cost_matrix)):
                for c in range(1, len(cost_matrix)):
                    if cost_matrix[b, a] + cost_matrix[a, c] < cost_matrix[b, c]:
                        cost_matrix[b, c] = cost_matrix[b, a] + cost_matrix[a, c]
        co = cost_matrix
        # print(co)
    return int(co[start, end])


def path_scanning(graph, cost_matrix, t_rule):
    global depot
    global capacity
    paths = []
    gra = copy.deepcopy(graph)
    count = -1
    loads = {}
    costs = {}
    while True:
        strat = depot
        count += 1  # k = k + 1
        loads[count] = 0
        costs[count] = 0
        path = []
        while True:
            tmp_cost = sys.maxsize
            tmp_edge = ()
            tmp_ratio = random.uniform(0.88, 1)
            for edge in gra:
                if loads[count] + edge[3] <= capacity * tmp_ratio:
                    d_se = shortest_path(cost_matrix, strat, edge[0])
                    if d_se < tmp_cost:
                        tmp_cost = d_se
                        tmp_edge = edge
                    elif d_se == tmp_cost and better(cost_matrix, edge, tmp_edge, depot, loads[count], strat, t_rule):
                        tmp_edge = edge
            if tmp_edge:
                [x, y, z, w] = tmp_edge
                path.append(tmp_edge)
                gra.remove(tmp_edge)
                gra.remove([y, x, z, w])
                loads[count] += tmp_edge[3]
                costs[count] += tmp_edge[2] + tmp_cost
                strat = tmp_edge[1]
            if not gra or tmp_cost == sys.maxsize:  # repeat...until free is empty pr d_bar is infinity
                break
        costs[count] += shortest_path(cost_matrix, strat, depot)
        paths.append(path)
        if not gra:  # repeat...until free is empty
            break
    return [paths, list(costs.values()), list(loads.values())]


def better(cost_matrix, tmp_edge, edge, src, load, strat, t_rule):
    rule = 1
    if t_rule == 1:
        rule = 1
    elif t_rule == 2:
        rule = 2
    elif t_rule == 3:
        rule = random.randint(1, 2)
    elif t_rule == 4:
        rule = random.randint(3, 5)
    elif t_rule == 5:
        rule = random.randint(1, 5)
    # rule = 2 #random.randint(1, 5)
    tp_cost = shortest_path(cost_matrix, strat, tmp_edge[0])
    cur_cost = shortest_path(cost_matrix, strat, edge[0])
    if not edge:
        return True
    if rule == 1:  # maximize c_ij/r_ij
        return (tp_cost + tmp_edge[2]) / tmp_edge[3] > (cur_cost + edge[2]) / edge[3]
    elif rule == 2:  # minimize c_ij/r_ij
        return (tp_cost + tmp_edge[2]) / tmp_edge[3] < (cur_cost + edge[2]) / edge[3]
    else:
        tmp_edge_cost = shortest_path(cost_matrix, tmp_edge[1], src)
        edge_cost = shortest_path(cost_matrix, edge[1], src)
        if rule == 3:  # maximize return cost
            return tmp_edge_cost > edge_cost
        if rule == 4:  # minimize return cost
            return tmp_edge_cost < edge_cost
        if rule == 5:
            if load < capacity / 2:  # less than half full capacity, apply rule 3
                return tmp_edge_cost > edge_cost
            else:  # else apply rule 4
                return tmp_edge_cost < edge_cost


def select(path_list,count):
    path_list = sorted(path_list, key=lambda x: sum(x[1]))
    return path_list[:count]


def print_list(path_list):
    str1 = "s "
    for i in path_list[0]:
        str1 += '0,'
        for j in i:
            str1 += '(' + str(j[0]) + ',' + str(j[1]) + ')' + ','
        str1 += '0,'
    str1 = str1[:-1]
    str1 += '\nq ' + str(sum(path_list[1]))
    print(str1)


def cal_path_cost(cost_matrix, path):
    global depot
    start = depot
    total_cost = 0

    for i in path:
        total_cost += shortest_path(cost_matrix, start, i[0]) + i[2]
        start = i[1]
    total_cost += shortest_path(cost_matrix, start, depot)
    return total_cost


def cal_total_cost(cost_matrix, paths):
    cc = 0
    for i in paths:
        cc += cal_path_cost(cost_matrix, i)
    return cc


def single_insertion(cost_matrix, p_list):
    # print(p_list)
    path_list = p_list.copy()
    path_index = random.randint(0, len(path_list[0]) - 1)
    tmp_path = path_list[0][path_index]
    edge_index = random.randint(0, len(tmp_path) - 1)
    [s, e, c, d] = tmp_path[edge_index]
    now_cost = path_list[1][path_index]
    del tmp_path[edge_index]  # tmp_path.remove(edge_index)
    expand_list = list()
    # opt_index = 0
    for i in range(0, len(tmp_path) - 1):
        if i == edge_index:
            continue
        tmp_path.insert(i, [s, e, c, d])
        t_cost = cal_path_cost(cost_matrix, tmp_path)
        path_list[1][path_index] = t_cost
        if t_cost < now_cost * 1.06:
            # opt_index = i
            expand_list.append(path_list)
        del tmp_path[i]  # tmp_path.remove(i)
        tmp_path.insert(i, [e, s, c, d])
        t_cost = cal_path_cost(cost_matrix, tmp_path)
        path_list[1][path_index] = t_cost
        if t_cost < now_cost * 1.06:
            # opt_index = i
            expand_list.append(path_list)
        del tmp_path[i]  # tmp_path.remove(i)
    tmp_path.insert(edge_index, [s, e, c, d])
    path_list[1][path_index] = now_cost
    # print(path_list == p_list)
    return expand_list


def double_insertion(cost_matrix, p):
    path = p[0]
    tmp1 = []
    tmp2 = []
    route_index = random.randint(0, len(path) - 1)
    for i in range(0, len(path)):
        t_edge = copy.deepcopy(path[i])
        new_path1 = new_path2 = t_edge
        if i == route_index:
            e_idx = random.randint(0, len(t_edge) - 1)

            [s, e, c, d] = t_edge[e_idx]
            inv_edge = [e, s, c, d]

            t_edge.remove([s, e, c, d])
            insert_index = random.randint(0, len(t_edge))
            dup_edge = copy.deepcopy(t_edge)
            new_path2 = dup_edge

            t_edge.insert(insert_index, [s, e, c, d])
            dup_edge.insert(insert_index, inv_edge)

        tmp1.append(new_path1)
        tmp2.append(new_path2)
        # print(muta1)
        [cs1, ls1] = cal_costs_and_loads(cost_matrix, tmp1)
        # muta1.append(cs1)
        # muta1.append(ls1)
        [cs2, ls2] = cal_costs_and_loads(cost_matrix, tmp2)
        # muta1.append(cs2)
        # muta1.append(ls2)
    # print(muta1)
    return [[tmp1, cs1, ls1], [tmp2, cs2, ls2]]


def cal_costs_and_loads(cost_matrix, paths):
    costs = []
    loads = []
    for path in paths:
        costs.append(cal_path_cost(cost_matrix, path))
        lt = 0
        for m in path:
            lt += m[3]
        loads.append(lt)
    return [costs, loads]


def _2opt(cost_matrix, paths):
    # print(paths)
    tmp_path = copy.deepcopy(paths[0])
    route_index = random.randint(0, len(tmp_path) - 1)
    if len(tmp_path[route_index]) == 1:
        return []
    [a, b] = random.sample(range(len(tmp_path[route_index])), 2)
    if a > b:
        a, b = b, a
    temp = copy.deepcopy(tmp_path[route_index])
    for i in range(b, a - 1, -1):
        [q, w, e, r] = temp[i]
        tmp_path[route_index][b + a - i] = [w, q, e, r]
    [cc, ss] = cal_costs_and_loads(cost_matrix, tmp_path)
    # print([tmp_path, cc, ss])
    return [[tmp_path, cc, ss]]


def reverse_edges(edges):
    tmp = copy.deepcopy(edges)
    tmp.reverse()
    for i in range(len(tmp)):
        [a, b, c, d] = tmp[i]
        tmp[i] = [b, a, c, d]
    return tmp


def check_path(path):
    global capacity
    loads = 0
    for edge in path:
        loads += edge[3]
    return loads <= capacity


def better_2opt(cost_matrix, paths):
    paths1_tmp = []
    paths2_tmp = []
    extend_routes = []
    # print(paths)
    [t_idx1, t_idx2] = random.sample(range(len(paths[0])), 2)
    if len(paths[0][t_idx1]) == 1 or len(paths[0][t_idx2]) == 1:
        return []
    a = random.randint(1, len(paths[0][t_idx1]) - 1)
    b = random.randint(1, len(paths[0][t_idx2]) - 1)

    for i in range(0, len(paths[0])):
        if i != t_idx1 and i != t_idx2:
            paths1_tmp.append(copy.deepcopy(paths[0][i]))
            paths2_tmp.append(copy.deepcopy(paths[0][i]))

    t3 = copy.deepcopy(paths[0][t_idx1][0:a]) + copy.deepcopy(paths[0][t_idx2][b:])
    if t3:
        paths1_tmp.append(t3)

    t3 = copy.deepcopy(paths[0][t_idx2][0:b]) + copy.deepcopy(paths[0][t_idx1][a:])
    if t3:
        paths1_tmp.append(t3)

    t3 = copy.deepcopy(paths[0][t_idx1][0:a]) + reverse_edges(paths[0][t_idx2][0:b])
    if t3:
        paths2_tmp.append(t3)

    t3 = reverse_edges(paths[0][t_idx1][a:]) + copy.deepcopy(paths[0][t_idx2][b:])
    if t3:
        paths2_tmp.append(t3)

    global capacity
    if paths1_tmp:
        [xx, yy] = cal_costs_and_loads(cost_matrix, paths1_tmp)
        zz = [_ for _ in yy if _ > capacity]
        if not zz:
            extend_routes.append([paths1_tmp, xx, yy])
    if paths2_tmp:
        [xx, yy] = cal_costs_and_loads(cost_matrix, paths2_tmp)
        zz = [_ for _ in yy if _ > capacity]
        if not zz:
            extend_routes.append([paths2_tmp, xx, yy])
    # print(extend_routes)
    # print("finish")
    return extend_routes


def check_route(route):
    global capacity
    zz = [_ for _ in route[2] if _ > capacity]
    return len(zz) == 0


def mutation(cost_matrix, paths):
    rule = random.randint(1, 5)
    if rule == 1:
        return single_insertion(cost_matrix, paths)
    if rule == 2:
        return double_insertion(cost_matrix, paths)
    if rule == 3:
        return swap(cost_matrix, paths)
    if rule == 4:
        return _2opt(cost_matrix, paths)
    if rule == 5:
        return better_2opt(cost_matrix, paths)


def swap(cost_matrix, paths):
    global capacity
    [ii, jj] = random.sample(range(0, len(paths[0])), 2)
    paths_list = list()
    m = random.randint(0, len(paths[0][ii]) - 1)
    [a1, b1, c1, d1] = paths[0][ii][m]
    for n in range(0, len(paths[0][jj]) - 1):
        [a2, b2, c2, d2] = paths[0][jj][n]
        # print(capacity)
        # print(paths[2][ii], d1, d2, paths[2][jj])
        if paths[2][ii] - d1 + d2 <= capacity and paths[2][jj] - d2 + d1 <= capacity:
            paths[0][ii][m] = [a2, b2, c2, d2]
            paths[0][jj][n] = [a1, b1, c1, d1]
            mmm = copy.deepcopy(paths[0])
            [cc1, s1] = cal_costs_and_loads(cost_matrix, mmm)
            paths_list.append([mmm, cc1, s1])
            paths[0][ii][m] = [b2, a2, c2, d2]
            xxx = copy.deepcopy(paths[0])
            [cc1, s1] = cal_costs_and_loads(cost_matrix, xxx)
            paths_list.append([xxx, cc1, s1])
            # print(paths[0][jj][n])
            paths[0][jj][n] = [b1, a1, c1, d1]
            # print(paths[0][jj][n])
            nnn = copy.deepcopy(paths[0])
            [cc1, s1] = cal_costs_and_loads(cost_matrix, nnn)
            paths_list.append([nnn, cc1, s1])
            # print(paths[0] == xxx)
            paths[0][ii][m] = [a1, b1, c1, d1]
            paths[0][jj][n] = [a2, b2, c2, d2]
        else:
            paths[0][ii][m] = [a1, b1, c1, d1]
            paths[0][jj][n] = [a2, b2, c2, d2]
            continue
    # print(paths_list)
    return paths_list


if __name__ == '__main__':
    start = time.time()
    file = open(sys.argv[1])
    termination = int(sys.argv[3])
    seed = int(sys.argv[5])
    tmp = file.readline()
    for i in range(7):
        tmp = file.readline()
        args.append(int(tmp[tmp.find(":") + 2:]))
    tmp = file.readline()
    requied_edges_num = args[2]
    vertices_num = args[0]
    depot = args[1]
    capacity = args[5]
    # print(args)
    graph_matrix = dict()  # adjacent list
    data = file.readlines()
    rt = list()
    for line in data:
        if line != "END":
            d = line.split()
            d = list(map(int, d))
            if d[3] != 0:
                rt.append([d[0], d[1], d[2], d[3]])
                rt.append([d[1], d[0], d[2], d[3]])
            if d[0] in graph_matrix:
                graph_matrix[d[0]][d[1]] = (d[2], d[3])
            else:
                graph_matrix[d[0]] = {d[1]: (d[2], d[3])}
            if d[1] in graph_matrix:
                graph_matrix[d[1]][d[0]] = (d[2], d[3])
            else:
                graph_matrix[d[1]] = {d[0]: (d[2], d[3])}

    # print(graph)
    cost_matrix = np.zeros((args[0] + 1, args[0] + 1))  # adjacent matrix of cost
    cost_matrix.fill(sys.maxsize)
    for i in range(len(cost_matrix)):
        cost_matrix[i, i] = 0
    count = 0
    for i in graph_matrix:
        for j in graph_matrix[i]:
            # print(graph[i][j][0])
            cost_matrix[i, j] = graph_matrix[i][j][0]
            count += 1

    # for i in graph:
    #     for j in graph[i]:
    #         # print((i, j))
    #         if graph[i][j][1] != 0 and i < j:
    #             rt.append([i, j, graph[i][j][0], graph[i][j][1]])
    # random.seed(seed)
    # print(rt)
    # print(time.time() - start)
    costList = list()
    for i in range(50):
        costList.append(path_scanning(rt, cost_matrix, 1))
        costList.append(path_scanning(rt, cost_matrix, 2))
    for i in range(350):
        costList.append(path_scanning(rt, cost_matrix, 3))
        costList.append(path_scanning(rt, cost_matrix, 4))
    for i in range(1600):
        costList.append(path_scanning(rt, cost_matrix, 5))
    costList = select(costList, 300)
    # print(costList)
    print_list(costList[0])
    # print(str(sum(costList[0][1])))
    cur_time = time.time() - start
    print(cur_time)
    # ccc = int(time.time() - start)
    while cur_time < termination - 1.2:
        # if int(time.time() - start) >= ccc:
        #     print(str(sum(costList[0][1])) + " " + str(sum(costList[len(costList) - 1][1])))
        #     ccc += 1
        if sum(costList[0][1]) == sum(costList[int((len(costList) - 1) * 0.9)][1]):
            costList = select(costList, 20)
            for i in range(0, len(costList) - 1):
                tm = better_2opt(cost_matrix, costList[i])
                if tm:
                    costList.extend(tm)
                tm = _2opt(cost_matrix, costList[i])
                if tm:
                    costList.extend(tm)
        for i in range(0, len(costList) - 1):
            tm = mutation(cost_matrix, costList[i])
            if tm:
                costList.extend(tm)
            # print(costList)
        costList = select(costList, 250)
        cur_time = time.time() - start
    print_list(costList[0])
    # print(time.time() - start)
