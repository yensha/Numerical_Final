import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from deap import base, creator, tools, algorithms
import sys

# 初始化遺傳算法環境
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# 功能函數：計算給定路徑的總距離
def evalTSP(individual, distance_matrix, start_node):
    path = [start_node] + [node for node in individual if node != start_node]
    distance = 0
    for i in range(len(path) - 1):
        if distance_matrix[path[i], path[i + 1]] == sys.maxsize:
            return sys.maxsize,  # 如果路徑包含無限大權重，則返回無限大
        distance += distance_matrix[path[i], path[i + 1]]
    distance += distance_matrix[path[-1], path[0]]  # 路徑封閉
    return distance,

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

def init_individual(node_list, start_node):
    node_list = list(node_list)  # 複製以避免修改原始列表
    node_list.remove(start_node)  # 移除起始節點
    random.shuffle(node_list)  # 打亂剩餘節點
    print("Initialized individual:", node_list)
    return creator.Individual(node_list)  # 創建個體
  # 創建個體

def main():
    num_nodes = int(input("輸入節點數量: "))
    num_edges = int(input("輸入邊數量: "))
    edges = []
    for i in range(num_edges):
        u, v, w = map(int, input(f"輸入邊 {i + 1} 的格式為 'u v w': ").split())
        edges.append((u, v, w))
    
    G = generate_graph(edges, num_nodes)
    start_node = int(input("輸入起始節點: "))
    distance_matrix = nx.to_numpy_array(G, weight='weight')
    
    toolbox = base.Toolbox()
    node_list = list(range(num_nodes))
    
    # 註冊遺傳算法所需的工具
    toolbox.register("individual", init_individual, node_list=node_list, start_node=start_node)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=300)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evalTSP, distance_matrix=distance_matrix, start_node=start_node)
    
    population = toolbox.population()
    ngen = 200
    cxpb = 0.7
    mutpb = 0.2

    for gen in range(ngen):
        algorithms.eaSimple(population, toolbox, cxpb, mutpb, 1, verbose=False)
    
    best_individual = tools.selBest(population, 1)[0]
    path = [start_node] + best_individual + [start_node]  # 闭合路径并确保起点和终点为 start_node

    fig, ax = plt.subplots()
    pos = nx.spring_layout(G, seed=42)  # 布局算法并设置种子确保布局一致

    def update(num):
        ax.clear()
        nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightblue', node_size=500)
        nx.draw_networkx_nodes(G, pos, nodelist=[start_node], node_color='yellow', node_size=700)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
        if num > 0:
            nx.draw_networkx_edges(G, pos, edgelist=list(zip(path[:num], path[1:num+1])), width=2, edge_color='blue')
        ax.set_title(f'路徑步驟: {num}/{len(path)-1}')

    ani = FuncAnimation(fig, update, frames=len(path), repeat=True)
    plt.show()

    print('最佳路徑: ', path)
    print('最短距離: ', evalTSP(best_individual, distance_matrix, start_node)[0])

if __name__ == "__main__":
    main()
