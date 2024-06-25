import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from deap import base, creator, tools, algorithms
import sys

# 初始化遗传算法环境
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# 功能函数：计算给定路径的总距离
def evalTSP(individual, distance_matrix, start_node):
    path = [start_node] + individual
    distance = 0
    for i in range(len(path) - 1):
        distance += distance_matrix[path[i], path[i + 1]]
    distance += distance_matrix[path[-1], start_node]  # 确保路径封闭
    return distance,

# 生成图
def generate_graph(edges, num_nodes):
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)
    max_weight = sys.maxsize
    for u, v in G.edges():
        if 'weight' not in G.edges[u, v]:
            G.edges[u, v]['weight'] = max_weight
    return G

# 自定义交叉函数
def custom_cxOrdered(ind1, ind2):
    size = min(len(ind1), len(ind2))
    cxpoint1, cxpoint2 = sorted(random.sample(range(size), 2))
    temp1 = ind1[cxpoint1:cxpoint2+1] + ind1[:cxpoint1] + ind1[cxpoint2+1:]
    temp2 = ind2[cxpoint1:cxpoint2+1] + ind2[:cxpoint1] + ind2[cxpoint2+1:]
    ind1[:] = temp1
    ind2[:] = temp2
    return ind1, ind2

def main():
    num_nodes = int(input("Enter the number of nodes: "))
    num_edges = int(input("Enter the number of edges: "))
    edges = []
    for i in range(num_edges):
        u, v, w = map(int, input(f"Enter edge {i + 1} in the format 'u v w': ").split())
        edges.append((u, v, w))

    G = generate_graph(edges, num_nodes)
    start_node = int(input("Enter the start node: "))
    distance_matrix = nx.to_numpy_array(G, weight='weight')

    toolbox = base.Toolbox()
    node_list = list(range(num_nodes))
    node_list.remove(start_node)
    toolbox.register("indices", random.sample, node_list, len(node_list))
    toolbox.register("individual", tools.initIterate, creator.Individual, lambda: random.sample(node_list, len(node_list)))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", custom_cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evalTSP, distance_matrix=distance_matrix, start_node=start_node)

    population = toolbox.population(n=300)
    ngen = 200
    cxpb = 0.7
    mutpb = 0.2

    for gen in range(ngen):
        algorithms.eaSimple(population, toolbox, cxpb, mutpb, 1, verbose=False)

    best_individual = tools.selBest(population, 1)[0]
    path = [start_node] + best_individual + [start_node]

    # Visualization
    # ... (Add your visualization code here)

    print('最佳路径: ', path)
    print('最短距离: ', evalTSP(best_individual, distance_matrix, start_node)[0])

if __name__ == "__main__":
    main()
