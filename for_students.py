from itertools import compress
import random
import time
import matplotlib.pyplot as plt

from data import *


def initial_population(individual_size, population_size):
    return [[random.choice([True, False]) for _ in range(individual_size)] for _ in range(population_size)]


def fitness(items, knapsack_max_capacity, individual):
    total_weight = sum(compress(items['Weight'], individual))
    if total_weight > knapsack_max_capacity:
        return 0
    return sum(compress(items['Value'], individual))


def population_best(items, knapsack_max_capacity, population):
    best_individual = None
    best_individual_fitness = -1
    for individual in population:
        individual_fitness = fitness(items, knapsack_max_capacity, individual)
        if individual_fitness > best_individual_fitness:
            best_individual = individual
            best_individual_fitness = individual_fitness
    return best_individual, best_individual_fitness


def fitness_for_all_population(items, knapsack_max_capacity, population):
    sum = 0;
    for individual in population:
        individual_fitness = fitness(items, knapsack_max_capacity, individual)
        sum += individual_fitness
    return sum


def individual_probability(items, knapsack_max_capacity, individual, population_fitness):
    return fitness(items, knapsack_max_capacity, individual) / population_fitness


def probability_list(items, knapsack_max_capacity, population):
    population_fitness = fitness_for_all_population(items, knapsack_max_capacity, population)
    p_list = []
    for individual in population:
        p_list.append(individual_probability(items, knapsack_max_capacity, individual, population_fitness))
    return p_list


items, knapsack_max_capacity = get_big()
print(items)

population_size = 100
generations = 200
n_selection = 20
n_elite = 5

start_time = time.time()
best_solution = None
best_fitness = 0
population_history = []
best_history = []
population = initial_population(len(items), population_size)


def make_parents_pairs(parents):
    pairs = []
    while len(parents) >= 2:
        pair = random.sample(parents, 2)
        pairs.append(pair)
        parents.remove(pair[0])
        parents.remove(pair[1])
    return pairs


def crossover(vector1, vector2):
    combined_vector = list(zip(vector1, vector2))
    # Obliczenie środka indeksu, aby podzielić wektory na połowy
    mid_index = len(combined_vector) // 2
    # Pobranie pierwszej połowy z pierwszego wektora i drugiej połowy z drugiego wektora
    new_vector = [x for x, _ in combined_vector[:mid_index]] + [y for _, y in combined_vector[mid_index:]]
    return new_vector


def mutation(population):
    for individual in population:
        index_to_change = random.randint(0, len(individual) - 1)
        element_to_change = individual[index_to_change]
        individual[index_to_change] = not element_to_change
    return population


def elite(list_of_probability, new_population,population):
    elite_probability = max(list_of_probability)
    elite_index = list_of_probability.index(elite_probability)
    list_of_probability[elite_index] = 0
    elite_individual = population[elite_index]
    new_population_list_of_probability = probability_list(items, knapsack_max_capacity, new_population)
    worst_individual = min(new_population_list_of_probability)
    individual_index_to_change = new_population_list_of_probability.index(worst_individual)
    new_population_list_of_probability[individual_index_to_change] = 1
    new_population[individual_index_to_change] = elite_individual
    return new_population


for _ in range(generations):
    population_history.append(population)

    # TODO: implement genetic algorithm

    # wybor rodzicow metoda ruletki

    list_of_probability = probability_list(items, knapsack_max_capacity, population)

    # Wybierz losowo jeden indeks z listy przy użyciu wag
    parents = []
    for i in range(n_selection):
        random_index = random.choices(range(len(list_of_probability)), weights=list_of_probability)[0]
        parents.append(population[random_index])
    new_population = []
    # krzyrzówka rodziców i tworzenie nowej populacji
    for i in range(int(population_size / n_selection)):
        pairs = make_parents_pairs(parents)
        for pair in pairs:
            new_population.append(crossover(pair[0], pair[1]))
            new_population.append(crossover(pair[1], pair[0]))
    #mutacja
    new_population = mutation(new_population)

    # elityzm
    for i in range(n_elite):
        new_population = elite(list_of_probability, new_population,population)

    population = new_population


    best_individual, best_individual_fitness = population_best(items, knapsack_max_capacity, population)
    if best_individual_fitness > best_fitness:
        best_solution = best_individual
        best_fitness = best_individual_fitness
    best_history.append(best_fitness)

end_time = time.time()
total_time = end_time - start_time
print('Best solution:', list(compress(items['Name'], best_solution)))
print('Best solution value:', best_fitness)
print('Time: ', total_time)

# plot generations
x = []
y = []
top_best = 10
for i, population in enumerate(population_history):
    plotted_individuals = min(len(population), top_best)
    x.extend([i] * plotted_individuals)
    population_fitnesses = [fitness(items, knapsack_max_capacity, individual) for individual in population]
    population_fitnesses.sort(reverse=True)
    y.extend(population_fitnesses[:plotted_individuals])
plt.scatter(x, y, marker='.')
plt.plot(best_history, 'r')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.show()
