import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.animation as animation
import flet as ft

def genetic_algorithm_tsp(G, start_node=None, population_size=10, num_generations=100, mutation_rate=0.01):
    nodes = list(G.nodes())
    if start_node is None:
        start_node = random.choice(nodes)

    def create_initial_population():
        population = []
        for _ in range(population_size):
            path = [start_node] + random.sample([n for n in nodes if n != start_node], len(nodes) - 1)
            path.append(start_node)  # Make it a round trip
            population.append(path)
        return population

    def calculate_fitness(path):
        return sum(G[path[i]][path[i+1]]['weight'] for i in range(len(path) - 1))

    def select_parents(population):
        sorted_population = sorted(population, key=lambda x: calculate_fitness(x))
        return sorted_population[:2]  # Return top 2 as parents

    def crossover(parent1, parent2):
        size = len(G.nodes())
        child = [None] * size
        start, end = sorted(random.sample(range(1, size), 2))
        child[start:end] = parent1[start:end]
        child[0] = start_node

        child_pos = end
        for node in parent2:
            if node not in child:
                if child_pos >= size:
                    child_pos = 1
                child[child_pos] = node
                child_pos += 1

        child.append(start_node)  # Make it a round trip
        return child

    def mutate(path):
        index1, index2 = random.sample(range(1, len(path) - 1), 2)  # Exclude first and last
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


def ant_colony_optimization(G, start_node=None, num_ants=20, num_iterations=100, decay=0.5, alpha=1.0, beta=2.0):
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

    if start_node is None:
        start_node = random.choice(list(G.nodes()))
    pheromones = initialize_pheromones()
    best_path = None
    best_cost = float('inf')

    for _ in range(num_iterations):
        paths = []
        for _ in range(num_ants):
            path = [start_node]
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

def tabu_search(G, start_node, max_iterations=100, tabu_tenure=10):
    def calculate_cost(path):
        return sum(G[path[i]][path[i+1]]['weight'] for i in range(len(path) - 1))

    best_path = list(G.nodes())
    random.shuffle(best_path)
    best_path = [start_node] + [node for node in best_path if node != start_node]
    best_path.append(start_node)
    best_cost = calculate_cost(best_path)

    tabu_list = []

    for _ in range(max_iterations):
        neighborhood = []
        for i in range(1, len(best_path) - 1):
            for j in range(i + 1, len(best_path) - 1):
                neighbor = best_path[:]
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                neighborhood.append((neighbor, calculate_cost(neighbor)))

        neighborhood.sort(key=lambda x: x[1])
        for neighbor, cost in neighborhood:
            if neighbor not in tabu_list:
                current_path = neighbor
                current_cost = cost
                break

        if current_cost < best_cost:
            best_path = current_path
            best_cost = current_cost

        tabu_list.append(current_path)
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)

    return best_path, best_cost

def simulated_annealing(G, start_node=None, initial_temp=100, final_temp=1, alpha=0.9):
    def calculate_cost(path):
        return sum(G[path[i]][path[i+1]]['weight'] for i in range(len(path) - 1))

    def get_neighbor(path):
        neighbor = path[:]
        i, j = random.sample(range(1, len(path) - 1), 2)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        return neighbor

    if start_node is None:
        start_node = random.choice(list(G.nodes()))
    current_path = list(G.nodes())
    random.shuffle(current_path)
    current_path = [start_node] + [node for node in current_path if node != start_node]
    current_path.append(start_node)
    current_cost = calculate_cost(current_path)
    best_path = current_path
    best_cost = current_cost
    temp = initial_temp

    while temp > final_temp:
        neighbor = get_neighbor(current_path)
        neighbor_cost = calculate_cost(neighbor)
        delta = neighbor_cost - current_cost

        if delta < 0 or random.random() < np.exp(-delta / temp):
            current_path = neighbor
            current_cost = neighbor_cost

        if current_cost < best_cost:
            best_path = current_path
            best_cost = current_cost

        temp *= alpha

    return best_path, best_cost

def nearest_neighbor(G, start_node=None):
    def calculate_cost(path):
        return sum(G[path[i]][path[i+1]]['weight'] for i in range(len(path) - 1))

    if start_node is None:
        start_node = random.choice(list(G.nodes()))
    path = [start_node]
    while len(path) < len(G):
        last_node = path[-1]
        next_node = min((node for node in G if node not in path), key=lambda node: G[last_node][node]['weight'])
        path.append(next_node)
    path.append(start_node)
    return path, calculate_cost(path)

def minimum_spanning_tree(G, start_node=None):
    T = nx.minimum_spanning_tree(G)
    path = list(nx.dfs_preorder_nodes(T, source=0))
    path.append(path[0])
    cost = sum(G[path[i]][path[i+1]]['weight'] for i in range(len(path) - 1))
    return path, cost

def generate_graph(num_nodes, max_weight=100):
    G = nx.complete_graph(num_nodes)
    for u, v in G.edges():
        G[u][v]['weight'] = random.randint(1, max_weight)
    return G

def ensure_start_node(path, start_node):
    start_index = path.index(start_node)
    return path[start_index:] + path[1:start_index + 1]

def main():
    algorithms = {
        '1': ("Genetic Algorithm", genetic_algorithm_tsp),
        '2': ("Ant Colony Optimization", ant_colony_optimization),
        '3': ("Tabu Search", tabu_search),
        '4': ("Simulated Annealing", simulated_annealing),
        '5': ("Nearest Neighbor", nearest_neighbor),
        '6': ("Minimum Spanning Tree", minimum_spanning_tree)
    }

    num_nodes = int(input("Enter the number of nodes in the graph: "))
    start_node = int(input("Enter the starting node: "))
    graph = generate_graph(num_nodes)

    print("Available algorithms:")
    for key, value in algorithms.items():
        print(f"{key}: {value[0]}")

    choice1 = input("Enter the number for the first algorithm to compare: ")
    choice2 = input("Enter the number for the second algorithm to compare: ")
    if choice1 not in algorithms or choice2 not in algorithms:
        print("Invalid choice.")
        return

    algorithm1_name, algorithm1 = algorithms[choice1]
    algorithm2_name, algorithm2 = algorithms[choice2]

    # Call algorithms with the start_node parameter
    if choice1 == '3':
        best_path1, best_distance1 = algorithm1(graph, start_node=start_node)
    else:
        best_path1, best_distance1 = algorithm1(graph) if 'start_node' not in algorithm1.__code__.co_varnames else algorithm1(graph, start_node=start_node)
    
    if choice2 == '3':
        best_path2, best_distance2 = algorithm2(graph, start_node=start_node)
    else:
        best_path2, best_distance2 = algorithm2(graph) if 'start_node' not in algorithm2.__code__.co_varnames else algorithm2(graph, start_node=start_node)

    # Ensure paths start from the specified start_node
    best_path1 = ensure_start_node(best_path1, start_node)
    best_path2 = ensure_start_node(best_path2, start_node)

    # Initialize the figure and subplots for animation
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    pos = nx.circular_layout(graph)  # Layout nodes in a circle

    # Draw the complete graph in both subplots
    nx.draw_networkx(graph, pos, ax=axes[0], with_labels=True, node_color='lightblue', edge_color='gray')
    nx.draw_networkx(graph, pos, ax=axes[1], with_labels=True, node_color='lightblue', edge_color='gray')

    # Initialize lines for animation
    line1, = axes[0].plot([], [], 'ro-', lw=2, label=f'{algorithm1_name} best path')
    line2, = axes[1].plot([], [], 'go-', lw=2, label=f'{algorithm2_name} best path')
    axes[0].legend()
    axes[1].legend()

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        return line1, line2

    def animate(i):
        # Path for algorithm 1
        x1 = [pos[best_path1[j % num_nodes]][0] for j in range(i+1)]
        y1 = [pos[best_path1[j % num_nodes]][1] for j in range(i+1)]
        line1.set_data(x1, y1)

        # Path for algorithm 2
        x2 = [pos[best_path2[j % num_nodes]][0] for j in range(i+1)]
        y2 = [pos[best_path2[j % num_nodes]][1] for j in range(i+1)]
        line2.set_data(x2, y2)

        return line1, line2

    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=num_nodes+1, interval=1000, blit=True)
    plt.tight_layout()
    plt.show()

    print(f"{algorithm1_name} best path:", best_path1, "Distance:", best_distance1)
    print(f"{algorithm2_name} best path:", best_path2, "Distance:", best_distance2)

if __name__ == "__main__":
    main()
