import math
import random

def fitness(individual):
        #Need to get rid of round before submit
        #Fitness function from report
        return round(individual * math.sin(10 * math.pi * individual) + 1.0, 4)

def encode():
    
    char1 = random.choice([-1,1])
    char2 = random.randint(0,9)
    char3 = random.randint(0,9)
    char4 = random.randint(0,9)
    char5 = random.randint(0,9)
    chromosome = "0."+ str(char2) + str(char3) + str(char4) + str(char5)
    chromosome = float(chromosome) * char1
    return chromosome

#Generates random individual from 4 base10 integers and random sign
def random_gen():
    chromosome = round(random.choice([-1, 1]) * random.uniform(0, 1), 4)
    while(chromosome > 1 or chromosome < -0.5):
        chromosome = round(random.choice([-1, 1]) * random.uniform(0, 1), 4)
    return chromosome

def genetic_algorithm():
    for gen in range(generation):
        initial_pop = {}
        for person in range(population):
            chromosome = random_gen()
            initial_pop[chromosome] = fitness(chromosome)
        print("Population of Gen", gen,":",initial_pop)
        max_fit = max(initial_pop, key=initial_pop.get)
        print("Fittest individual is:", max_fit, "of", initial_pop[max_fit],"\n" )


if __name__ == "__main__":
    population = 10
    generation = 5

    genetic_algorithm()


