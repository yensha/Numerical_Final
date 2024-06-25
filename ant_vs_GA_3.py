import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.animation as animation

def genetic_algorithm_tsp(G, population_size=10, num_generations=100, mutation_rate=0.01):
    def create_initial_population():
        population = []
        for _ in range(population_size):
            path = list(G.nodes())
            random.shuffle(path)
            path.append(path[0])  # Make it a round trip
            population.append(path)
        return population

    def calculate_fitness(path):
        fitness = 0
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            if v in G[u]:
                fitness += G[u][v]['weight']
            else:
                raise KeyError(f"Edge ({u}, {v}) not found in graph.")
        return fitness

    def select_parents(population):
        sorted_population = sorted(population, key=lambda x: calculate_fitness(x))
        return sorted_population[:2]  # Return top 2 as parents

    def crossover(parent1, parent2):
        size = len(G.nodes())
        child = [None] * size
        start, end = sorted(random.sample(range(size), 2))
        child[start:end] = parent1[start:end]

        child_pos = end
        for node in parent2:
            if node not in child:
                if child_pos >= size:
                    child_pos = 0
                child[child_pos] = node
                child_pos += 1

        child.append(child[0])  # Make it a round trip
        return child

    def mutate(path):
        index1, index2 = random.sample(range(len(path) - 1), 2)  # Exclude last as it's a repeat of first
        path[index1], path[index2] = path[index2], path[index1]

    population = create_initial_population()
    best_path = min(population, key=calculate_fitness)

    for _ in range(num_generations):
        parents = select_parents(population)
        offspring = [crossover(parents[0], parents[1]) for _ in range(population_size)]
        for individual in offspring:
            if random.random() < mutation_rate:
                mutate(individual)
        population = offspring
        current_best = min(population, key=calculate_fitness)
        if calculate_fitness(current_best) < calculate_fitness(best_path):
            best_path = current_best

    return best_path, calculate_fitness(best_path)

def ant_colony_optimization(G, num_ants=20, num_iterations=100, decay=0.5, alpha=1.0, beta=2.0):
    def initialize_pheromones():
        return np.ones((len(G), len(G))) * 0.1

    def calculate_probabilities(from_node, pheromones, visited):
        probabilities = np.zeros(len(G))
        for to_node in range(len(G)):
            if to_node not in visited:
                pheromone = pheromones[from_node][to_node] ** alpha
                visibility = (1.0 / G[from_node][to_node]['weight']) ** beta
                probabilities[to_node] = pheromone * visibility
        probabilities /= np.sum(probabilities)
        return probabilities

    def update_pheromones(paths, pheromones):
        for path, cost in paths:
            for i in range(len(path) - 1):
                pheromones[path[i]][path[i+1]] += 1.0 / cost
                pheromones[path[i+1]][path[i]] += 1.0 / cost  # Symmetric TSP
        pheromones *= decay

    pheromones = initialize_pheromones()
    best_path = None
    best_cost = float('inf')

    for _ in range(num_iterations):
        paths = []
        for _ in range(num_ants):
            path = [random.randint(0, len(G)-1)]
            while len(path) < len(G):
                probabilities = calculate_probabilities(path[-1], pheromones, set(path))
                next_node = np.random.choice(len(G), p=probabilities)
                path.append(next_node)
            path.append(path[0])  # Complete the cycle
            cost = sum(G[path[i]][path[i+1]]['weight'] for i in range(len(path) - 1))
            paths.append((path, cost))
            if cost < best_cost:
                best_path, best_cost = path, cost
        update_pheromones(paths, pheromones)

    return best_path, best_cost

# Helper function for generating a complete graph with random weights
def generate_graph(num_nodes, max_weight=100):
    G = nx.complete_graph(num_nodes)
    for u, v in G.edges():
        G[u][v]['weight'] = random.randint(1, max_weight)
    return G

# Ensure paths start from the same node
def ensure_start_node(path, start_node):
    start_index = path.index(start_node)
    return path[start_index:] + path[1:start_index + 1]

def main():
    num_nodes = 10
    start_node = 0  # 指定一个统一的起点
    G = generate_graph(num_nodes)
    
    # 调用蚂蚁算法和遗传算法时传入起点参数
    best_path_aco, best_distance_aco = ant_colony_optimization(G)
    best_path_ga, best_distance_ga = genetic_algorithm_tsp(G)

    # Ensure paths start from the start_node
    best_path_aco = ensure_start_node(best_path_aco, start_node)
    best_path_ga = ensure_start_node(best_path_ga, start_node)

    # 初始化图形界面，分成两个子图
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    pos = nx.circular_layout(G)  # 节点在圆上的布局

    # Draw the complete graph in both subplots
    nx.draw_networkx(G, pos, ax=axes[0], with_labels=True, node_color='lightblue', edge_color='gray')
    nx.draw_networkx(G, pos, ax=axes[1], with_labels=True, node_color='lightblue', edge_color='gray')

    # Initialize lines for animation
    line_aco, = axes[0].plot([], [], 'ro-', lw=2, label='ACO Best Path')
    line_ga, = axes[1].plot([], [], 'go-', lw=2, label='GA Best Path')
    axes[0].legend()
    axes[1].legend()

    def init():
        line_aco.set_data([], [])
        line_ga.set_data([], [])
        return line_aco, line_ga

    def animate(i):
        # ACO path
        x_aco = [pos[best_path_aco[j % num_nodes]][0] for j in range(i+1)]
        y_aco = [pos[best_path_aco[j % num_nodes]][1] for j in range(i+1)]
        line_aco.set_data(x_aco, y_aco)

        # GA path
        x_ga = [pos[best_path_ga[j % num_nodes]][0] for j in range(i+1)]
        y_ga = [pos[best_path_ga[j % num_nodes]][1] for j in range(i+1)]
        line_ga.set_data(x_ga, y_ga)

        return line_aco, line_ga

    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=num_nodes+1, interval=1000, blit=True)
    plt.tight_layout()
    plt.show()

    print("ACO Best Path:", best_path_aco, "Distance:", best_distance_aco)
    print("GA Best Path:", best_path_ga, "Distance:", best_distance_ga)

if __name__ == "__main__":
    main()
