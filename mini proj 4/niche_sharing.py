import math
import random
import matplotlib.pyplot as plt

#Helper functions
#Use a tuple represent the sign variable and the 4 base10 integers as genes
def create_individual():
    return random.uniform(0,1)



#fitness function
def fitness(x):
    #return math.pow(math.sin(5 * math.pi * x), 6)
    return math.exp(-2 * math.log(2) * ((x - 0.01) / 0.8) ** 2) * math.sin(5 * math.pi * ((x ** 0.75) - 0.05)) ** 6

def gen_fitness_shared(population, fits, share, alpha):
    #f_shared = f_original / sum of sharing functions from all neighbors j of i
    fitness_shared = []
    for i, di in enumerate(population):
        ni = 1
        for j,dj in enumerate(population):
            if(i == j):
                ni += 0
            else:
                ni += sharing_function(di, dj, share, alpha)
        fitness_shared.append(fits[i] / ni)
    #print(fitness_shared)
    return fitness_shared

def sharing_function(di, dj, share, alpha):
    distance = abs(dj - di)
    if(distance < share):
        return (1 - (distance/share)**alpha)
    else:
        return 0



#uses both parents as templates for gene crossover, probabilty determined manually
#randomly determinses the single point where cross over occurs (cp), c1 and c2 represents 2 children
def crossover(p1, p2, crossover_prob):
    #print("Parent1", p1, "Parent2", p2)
    local_prob = random.random()
    #print(local_prob)
    if local_prob < crossover_prob:
        alpha = random.random()
        c1 = (alpha) * p1 + (1 - alpha) * p2
        c2 = (1 - alpha) * p1 + (alpha) * p2
        return (c1,c2)
    return (p1, p2)

#simulates random change in all genes including sign var
def mutate(ind, mutation_prob):

    if(random.random() < mutation_prob):
        mutation = random.uniform(-0.025, 0.025)
        ind += mutation
        
        return max(0, min(1, ind))
    return ind


#selects random sample of 2 individuals and takes the best fit one
def tournament_selection(population, fitnesses):
    k = 3
    #print(k)
    selected_indices = random.sample(range(len(population)), k)
    
    best_index = selected_indices[0] 
    best_fitness = fitnesses[best_index]

    for index in selected_indices:
        if fitnesses[index] > best_fitness:
            best_index = index
            best_fitness = fitnesses[index]

    return population[best_index]

def plot_best_av_worst(best, average, worst, generations):
    plt.figure(figsize=(10,5))
    plt.plot(best_fitnesses, color='orange', label='Best')
    plt.plot(avg_fitnesses, color = 'yellow',label='Average')
    plt.plot(worst_fitnesses, label='Worst')
    plt.scatter(range(generations), best, color='red', marker='o', label='Best Points')
    plt.scatter(range(generations), average, color='blue', marker='o', label='Average Points')
    plt.scatter(range(generations), worst, color='green', marker='o', label='Worst Points')
    plt.legend()
    plt.title('Fitness over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.show()

def plot_best_individuals(best, generations):
    plt.figure(figsize=(10,5))
    plt.plot(best)
    plt.scatter(range(generations), best, color='red', marker='o', label='Best Points')
    plt.title('Best Individual per Generation')
    plt.xlabel('Generation')
    plt.ylabel('x Value')
    plt.show()

def plot_end_fitness_curve(fin_pop, final_fit, best_individuals, best_fitnesses):
    x = [i/1000 for i in range(0, 1001)]
    y = [fitness(xi) for xi in x]
    
    plt.figure(figsize=(10,5))
    plt.plot(x, y, label='f(x)')
    plt.scatter(fin_pop, final_fit, color='red', marker='o', label='Best Individuals')
    plt.scatter([fin_pop[-1]], [final_fit[-1]], color='black', marker='*', s=100, label='Final GA Solution')

    plt.legend()
    plt.title('Function and GA Solution')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()

    print(f"Best solution: x={best_individuals[-1]:.4f}, f(x)={best_fitnesses[-1]:.4f}")

    plt.figure(figsize=(10,5))
    plt.plot(x, y, label='f(x)', linewidth=2)

if __name__ == "__main__":
    #Parameters
    pop_size = 120
    generations = 200
    crossover_prob = 0.4
    mutation_prob = 0.15
    share = 0.1
    alpha = 1.0

    #Initialize population (list of tuples)
    population = [create_individual() for _ in range(pop_size)]
    #print(population)
    #these are for recording the plots later
    fits = []
    best_fitnesses = []
    avg_fitnesses = []
    worst_fitnesses = []
    best_individuals = []

    # main genetic algo loop
    for gen in range(generations):
        #print(gen)
        #decoded = [decode(ind) for ind in population]
        #print("ENTIRE DECODED GEN", decoded)
        fits_orig = [fitness(x) for x in population]
        fits = gen_fitness_shared(population, fits_orig, share, alpha)
        best_idx = fits_orig.index(max(fits_orig))
        best_ind = population[best_idx]
        #print(best_ind)
        
        #record stats
        best_fitnesses.append(max(fits_orig))
        avg_fitnesses.append(sum(fits_orig)/pop_size)
        worst_fitnesses.append(min(fits_orig))
        best_individuals.append((best_ind))
        
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
        mutation_prob = mutation_prob - 0.0008 * random.random()
        #print("mut", mutation_prob)

    #Plotting
    for i in range(2):
        plot_best_av_worst(best_fitnesses,avg_fitnesses,worst_fitnesses,generations)

        plot_best_individuals(best_individuals, generations)

        fin_pop = population
        final_fit = [fitness(x) for x in fin_pop]

        plot_end_fitness_curve(fin_pop,final_fit,best_individuals,best_fitnesses)

    # Calculate raw fitness values for the best individuals
    raw_fitness_values = [fitness(x) for x in best_individuals]
