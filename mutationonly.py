import random
import math
import matplotlib.pyplot as plt

def create_individual():
    return random.uniform(-0.5, 1)

def fitness(x):
    return x * math.sin(10 * math.pi * x) + 1

def mutate(x, mutation_prob=0.1):
    if random.random() < mutation_prob:
        x += random.gauss(0, 0.05)  
        x = max(-0.5, min(x, 1))    
    return x

def tournament_selection(population, fitnesses, k=3):
    selected_indices = random.sample(range(len(population)), k)

    best_index = selected_indices[0] 
    best_fitness = fitnesses[best_index]

    for index in selected_indices:
        if fitnesses[index] > best_fitness:
            best_index = index
            best_fitness = fitnesses[index]

    return population[best_index]

pop_size = 50
generations = 50
mutation_prob = 0.1

population = [create_individual() for _ in range(pop_size)]
best_fitnesses = []
best_individuals = []

for gen in range(generations):
    fits = [fitness(x) for x in population]
    best_idx = fits.index(max(fits))
    best_ind = population[best_idx]

    best_fitnesses.append(max(fits))
    best_individuals.append((best_ind))
    
    parents = [tournament_selection(population, fits) for _ in range(pop_size)]

    offspring = [mutate(parent, mutation_prob) for parent in parents]
    
    
    population = offspring

plt.figure(figsize=(10,5))
plt.plot(best_fitnesses, label='Best')

plt.scatter(range(generations), best_fitnesses, color='red', marker='o', label='Best Points')

plt.legend()
plt.title('Fitness over Generations')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.show()

x = [i/1000 for i in range(-600, 1001)]
y = [fitness(xi) for xi in x]
plt.figure(figsize=(10,5))
plt.plot(x, y, label='f(x)')
plt.scatter(best_individuals, best_fitnesses, color='red', marker='o', label='Best Individuals')
plt.scatter([best_individuals[-1]], [best_fitnesses[-1]], color='black', marker='*', s=100, label='Final GA Solution')

plt.legend()
plt.title('Function and GA Solution')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()

print(f"Best solution: x={best_individuals[-1]:.4f}, f(x)={best_fitnesses[-1]:.4f}")