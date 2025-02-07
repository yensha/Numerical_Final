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
            path.append(start_node)  # Ensure a round trip
            population.append(path)
        return population

    def calculate_fitness(path):
        return sum(G[path[i]][path[i+1]]['weight'] for i in range(len(path) - 1))

    def select_parents(population):
        sorted_population = sorted(population, key=lambda x: calculate_fitness(x))
        return sorted_population[:2]  # Select top 2 as parents

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

        child.append(start_node)  # Ensure a round trip
        return child

    def mutate(path):
        index1, index2 = random.sample(range(1, len(path) - 1), 2)  # Exclude first and last node
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
