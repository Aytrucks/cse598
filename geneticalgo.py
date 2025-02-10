import math
import random
import matplotlib.pyplot as plt

#Helper functions
#Use a tuple represent the sign variable and the 4 base10 integers as genes
def create_individual():
    sign = random.choice([-1, 1])
    if sign == -1:
        gene_1 = random.randint(0, 4)
    else:
        gene_1 = random.randint(0, 9)
    #print(sign, gene_1, random.randint(0,9), random.randint(0,9), random.randint(0,9))
    return (sign, gene_1, random.randint(0,9), random.randint(0,9), random.randint(0,9))

def create_individual_MUTATE():
    return random.uniform(-0.5, 1)

#translate the tuple into actual value x for fitness function
def decode(individual):
    sign, gene_1, gene_2, gene_3, gene_4 = individual
    return sign * (gene_1 * 0.1 + gene_2 * 0.01 + gene_3 * 0.001 + gene_4 * 0.0001)


#fitness function
def fitness(x):
    return x * math.sin(10 * math.pi * x) + 1

#ensure that after mutation or crossover that individual's genotype still adheres to x parameters
def repair(individual):
    sign, gene_1, gene_2, gene_3, gene_4 = individual
    if sign == -1 and gene_1 > 5:
        gene_1 = random.randint(0, 4)
    return (sign, gene_1, gene_2, gene_3, gene_4)

#uses both parents as templates for gene crossover, probabilty determined manually
#randomly determinses the single point where cross over occurs (cp), c1 and c2 represents 2 children
def crossover(p1, p2, crossover_prob):
    if random.random() < crossover_prob:
        cp = random.randint(1, 4)
        c1 = p1[:cp] + p2[cp:]
        c2 = p2[:cp] + p1[cp:]
        return (repair(c1), repair(c2))
    return (p1, p2)

#simulates random change in all genes including sign var
def mutate(ind, mutation_prob):
    sign, gene_1, gene_2, gene_3, gene_4 = ind
    if random.random() < mutation_prob:
        sign = -sign
    if sign == -1 and gene_1 > 5:
        gene_1 = random.randint(0, 5)
    for i in range(1, 5):
        if random.random() < mutation_prob:
            val = random.randint(0, 9)
            if i == 1 and sign == -1:
                val = random.randint(0, 5)
            ind = list(ind)
            ind[i] = val
    return repair(ind)


#selects random sample of 3 individuals and takes the best fit one
def tournament_selection(population, fitnesses, k=3):
    selected_indices = random.sample(range(len(population)), k)

    best_index = selected_indices[0] 
    best_fitness = fitnesses[best_index]

    for index in selected_indices:
        if fitnesses[index] > best_fitness:
            best_index = index
            best_fitness = fitnesses[index]

    return population[best_index]

if __name__ == "__main__":
    #Parameters
    pop_size = 50
    generations = 10
    crossover_prob = 0.8
    mutation_prob = 0.1

    #Initialize population (list of tuples)
    population = [create_individual() for _ in range(pop_size)]

    #these are for recording the plots later
    best_fitnesses = []
    avg_fitnesses = []
    worst_fitnesses = []
    best_individuals = []

    # main genetic algo loop
    for gen in range(generations):
        decoded = [decode(ind) for ind in population]
        fits = [fitness(x) for x in decoded]
        best_idx = fits.index(max(fits))
        best_ind = population[best_idx]
        
        #record stats
        best_fitnesses.append(max(fits))
        avg_fitnesses.append(sum(fits)/pop_size)
        worst_fitnesses.append(min(fits))
        best_individuals.append(decode(best_ind))
        
        #select parents and rproduction
        offspring = [] 
        for _ in range(pop_size // 2):
            p1 = tournament_selection(population, fits)
            p2 = tournament_selection(population, fits)
            
            c1, c2 = crossover(p1, p2, crossover_prob) 
            c1 = mutate(c1, mutation_prob)  
            c2 = mutate(c2, mutation_prob)  
            
            offspring.append(c1)
            offspring.append(c2)
      
        population = offspring[:pop_size] 

    #Plotting
    plt.figure(figsize=(10,5))
    plt.plot(best_fitnesses, label='Best')
    plt.plot(avg_fitnesses, label='Average')
    plt.plot(worst_fitnesses, label='Worst')
    plt.scatter(range(generations), best_fitnesses, color='red', marker='o', label='Best Points')
    plt.scatter(range(generations), avg_fitnesses, color='blue', marker='o', label='Average Points')
    plt.scatter(range(generations), worst_fitnesses, color='green', marker='o', label='Worst Points')
    plt.legend()
    plt.title('Fitness over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.show()

    plt.figure(figsize=(10,5))
    plt.plot(best_individuals)
    plt.scatter(range(generations), best_individuals, color='red', marker='o', label='Best Points')
    plt.title('Best Individual per Generation')
    plt.xlabel('Generation')
    plt.ylabel('x Value')
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