import random
import numpy as np
import copy
import matplotlib.pyplot as plt
import math

knap_sack_items = []

# Open the file in read mode
with open('data/f2_l-d_kp_20_878.txt', 'r') as file:
    # Iterate over each line in the file
    for line in file:
        # Split the line into weight and profit
        profit,weight = map(int, line.split())
        # Append the tuple (weight, profit) to the data list
        knap_sack_items.append((weight, profit))

print(knap_sack_items)

KNAPSACK_WEIGHT = knap_sack_items[0][0]
TOTAL_ITEMS = knap_sack_items[0][1] - 1

knap_sack_items = knap_sack_items[1:]

class chormosome():

    def __init__(self) -> None:
        self.weight = 0
        self.value = 0
        self.chromosome = []

        self.intialize()

    def intialize(self):

        # print(TOTAL_ITEMS)

        for _ in range(TOTAL_ITEMS):
            numb = random.randint(0,1)
            self.chromosome.append(numb)

        self.weight = self.calculate_weight()
        self.value = self.calculate_value()

      
    def calculate_weight(self):
        chromo_weight = 0

        for i in range(TOTAL_ITEMS):
            if self.chromosome[i] == 1:
                chromo_weight+= knap_sack_items[i][1]
        
        
        return chromo_weight
    
    def calculate_value(self):
        chromo_value = 0

        for i in range(TOTAL_ITEMS):
            if self.chromosome[i] == 1:
                chromo_value+= knap_sack_items[i][0]
        
        
        return chromo_value
    
    def get_fitness(self):
        if self.weight > KNAPSACK_WEIGHT:
            return 0
        else:
            return self.value
        
    def mutation(self, mutation_probability):
        for i in range(TOTAL_ITEMS):
            if np.random.random() < mutation_probability:
                self.chromosome[i] = 1 - self.chromosome[i]

        

class Island():
    
    def __init__(self, population_size_per_island, mutation_rate):
         
        self.population_size = population_size_per_island # population of the islands
        self.mutation_rate = mutation_rate 

        # islands = [island_1, island_2, .....]
        #island_1 = [choromosome1, chormosome2, ....]
        self.island = self.initialize_island()


    def initialize_island(self):
        island = []
        for _ in range(self.population_size):
            # city_lst = copy.deepcopy(city_list)
            # np.random.shuffle(city_list)
            choromo1 = chormosome()
            island.append(choromo1)

        return island

    # would done idividually to each island and the chromosomes in it
    # def crossover(self, chrom1, chrom2, mut_rate):
    # # Randomly select start and end points for crossover
    #     start = random.randint(0, TOTAL_ITEMS - 1)
    #     end = random.randint(start, TOTAL_ITEMS - 1)

    #     # Create empty offspring chromosomes
    #     offspring1 = chormosome()
    #     offspring2 = chormosome()

    #     # Perform crossover between parents
    #     offspring1.chromosome = chrom1.chromosome[:start] + chrom2.chromosome[start:end] + chrom1.chromosome[end:]
    #     offspring2.chromosome = chrom2.chromosome[:start] + chrom1.chromosome[start:end] + chrom2.chromosome[end:]

    #     # Apply mutation to offspring chromosomes
    #     offspring1.mutation(mut_rate)
    #     offspring2.mutation(mut_rate)

    #     return [offspring1, offspring2]

    # def crossover(self, chrom1, chrom2, mut_rate):
    #     # scheme to be implemented using crossover scheme where a randomly selected set of parent 1 would displace the randomly selected set of parent 2

    #     start = random.randint(2,int(TOTAL_ITEMS/2))
    #     end = random.randint(start, TOTAL_ITEMS)
    #     # start = random.randint(1, 4)
    #     # end = random.randint(5, 8)
    #     parent1 = chrom1.chromosome[:]
    #     parent2 = chrom2.chromosome[:]

    #     offspring1 = [None] * TOTAL_ITEMS
    #     offspring2 = [None] * TOTAL_ITEMS

    #     offspring1[start:end+1] = parent1[start:end+1]
    #     offspring2[start:end+1] = parent2[start:end+1]
    #     # print(offspring1)
    #     # print(offspring2)
    #     # print(end)
    #     pointer = end + 1
    #     parent1_pointer = end + 1
    #     parent2_pointer = end + 1
    #     # print(start,end, parent2_pointer)
    #     counter = 0

    #     while pointer != start:
    #         # print(pointer, parent2[parent2_pointer])   
    #         if parent2[parent2_pointer] not in offspring1:    
    #             offspring1[pointer] = parent2[parent2_pointer]
    #             pointer += 1
    #         parent2_pointer += 1
    #         if parent2_pointer == len(offspring1):
    #             parent2_pointer = 0
    #         if pointer == len(offspring1):
    #             pointer = 0
    #         # counter+=1

    #     pointer = end + 1

    #     while pointer != start:
    #         # print(pointer, parent2[parent2_pointer])
    #         if parent1[parent1_pointer] not in offspring2: 
    #             offspring2[pointer] = parent1[parent1_pointer]
    #             pointer += 1
    #         parent1_pointer += 1
    #         if parent1_pointer == len(offspring2):
    #             parent1_pointer = 0
    #         if pointer == len(offspring2):
    #             pointer = 0

    #     offspring1 = chormosome(offspring1)
    #     offspring2 = chormosome(offspring2)
    #     offspring1.mutation(mut_rate)
    #     offspring2.mutation(mut_rate)

    #     return [offspring1, offspring2] 

    def crossover(self, parent1, parent2, mutation_rate):

        start = random.randint(1, TOTAL_ITEMS-2)
        end = random.randint(start, TOTAL_ITEMS-1)

        offspring1_genome = [-1 for _ in range(TOTAL_ITEMS)]
        offspring2_genome = [-1 for _ in range(TOTAL_ITEMS)]

        for i in range(start, end+1):
            offspring1_genome[i] = parent1.chromosome[i]
            offspring2_genome[i] = parent2.chromosome[i]

        parent1_pointer = (end + 1) % TOTAL_ITEMS
        parent2_pointer = (end + 1) % TOTAL_ITEMS
        # There's a more efficient way to do this I'm sure
        pointer = (end + 1) % TOTAL_ITEMS
        count = end-start+1
        while count < TOTAL_ITEMS:		
            offspring1_genome[pointer % TOTAL_ITEMS] = parent2.chromosome[parent2_pointer]
            count += 1
            pointer += 1
            parent2_pointer = (parent2_pointer + 1) % TOTAL_ITEMS

        pointer = (end + 1) % TOTAL_ITEMS
        count = end-start+1
        while count < TOTAL_ITEMS:						
            offspring2_genome[pointer % TOTAL_ITEMS] = parent1.chromosome[parent1_pointer]
            count += 1
            pointer += 1
            parent1_pointer = (parent1_pointer + 1) % TOTAL_ITEMS 

        offspring1 = chormosome()
        offspring1.chromosome = offspring1_genome

        offspring2 = chormosome()
        offspring2.chromosome = offspring2_genome

        offspring1.mutation(mutation_rate)
        offspring2.mutation(mutation_rate)

        
        return [offspring1, offspring2]
    
    def total_fitness(self):
        total = 0
        for i in self.island:
            total += i.get_fitness()
        return total
    
    def best_fitness(self):
        best = self.island[0].get_fitness()
        for i in self.island[1:]:
            if i.get_fitness() < best:
                best = i.get_fitness()
        return best
    
    def avg_fitnness(self):
        total = self.total_fitness()
        return total / self.population_size
    
    def random_selection(self, size):
        result = []
        for i in range(size):
            rand_num = random.randint(0, self.population_size - 1)
            result.append(self.island[rand_num])
        # print(result)
        return result

    def truncation_selection(self, size):
        result = []
        result = copy.deepcopy(self.island)
        result.sort(key=lambda k : k.get_fitness(), reverse=True)
        # print(result)
        return result[:size]
    
    def binary_tournament_selection(self, size):
        result= []
        for i in range(size):
            ind1, ind2 = random.sample(self.island, 2)
            selected = ind1 if ind1.get_fitness() < ind2.get_fitness() else ind2
            # print(ind1.get_fitness(), ind2.get_fitness())
            result.append(selected)
            # print(selected.get_fitness())
        return result
    
    def fitness_proportional_selection(self,size):
        total_fitness = sum(chromo.get_fitness() for chromo in self.island)
        selection_probs = [chromo.get_fitness() / total_fitness for chromo in self.island]
        selected_indices = np.random.choice(range(self.population_size), size=self.population_size, replace=True, p=selection_probs)
        return [self.island[i] for i in selected_indices[:size]]
    

    #     # print(normalized_range)
    #     return result
    def rank_selection(self, size):
        self.island.sort(key=lambda chromo: chromo.get_fitness())
        ranks = np.arange(1, self.population_size + 1)
        total_rank = np.sum(ranks)
        selection_probs = ranks / total_rank
        selected_indices = np.random.choice(range(self.population_size), size=self.population_size, replace=True, p=selection_probs)
        return [self.island[i] for i in selected_indices[:size]]

    # settnig the number of generation to a deault of 1 rn 
    def run_generation(self,parent_sel, survivor_sel, popul, offsp, num_gen = 1 ):
        
        parents = []
        for i in range(num_gen):
            if parent_sel == 1:
                parents = self.random_selection(offsp)
            elif parent_sel == 2:
                parents = self.binary_tournament_selection(offsp)
            elif parent_sel == 3:
                parents = self.fitness_proportional_selection(offsp)
            elif parent_sel == 4:
                parents = self.rank_selection(offsp)
            elif parent_sel == 5:
                parents = self.truncation_selection(offsp)
            
            for j in range(0,offsp,2):
                selected_parents = random.sample(parents, 2)
                self.island += self.crossover(selected_parents[0], selected_parents[1], self.mutation_rate)
            print("the total populatin size is ", len(self.island))
            if survivor_sel == 1:
                self.island = self.random_selection(popul)
            elif survivor_sel == 2:
                self.island = self.binary_tournament_selection(popul)
            elif survivor_sel == 3:
                self.island = self.fitness_proportional_selection(popul)
            elif survivor_sel == 4:
                self.island = self.rank_selection(popul)
            elif survivor_sel == 5:
                self.island = self.truncation_selection(popul)

            # print("the best fit for the ", i, " generation is: ", self.best_fitness())

    # def run_model(self):
    #     pass  

class IslandModels:
    def __init__(self, num_islands, population_size_per_island, num_generations, migration_interval, migration_rate, mutation_rate, tournament_size):
        self.num_islands = num_islands  # telling the number of islands present 
        self.population_size_per_island = population_size_per_island # population of the islands
        self.num_generations = num_generations # for how many
        self.migration_interval = migration_interval  # informing the number of genererations after migrations occurs
        self.migration_rate = migration_rate # number of individuals transfered during each migration
        self.mutation_rate = mutation_rate 
        self.tournament_size = tournament_size

        self.islands = [Island(population_size_per_island, mutation_rate) for _ in range(num_islands)]

    def migrate(self):
        migrants_per_island = int(self.population_size_per_island * self.migration_rate)
        
        # Select migrants from each island
        migrants = [island.fitness_proportional_selection(migrants_per_island) for island in self.islands]
        
        # Migrate between islands in a circular fashion
        for i in range(self.num_islands):
            target_island = (i + 1) % self.num_islands
            self.islands[target_island].island.extend(migrants[i])
            # After migration, trim each island's population to maintain size
            self.islands[target_island].island = self.islands[target_island].truncation_selection(self.population_size_per_island)
    
    def evolve(self):
        best_fitnesses = []
        avg_fitnesses = []
        for generation in range(self.num_generations):
            # Evolve each island
            for island_index, island in enumerate(self.islands):
                # Assuming the arguments are in order: parent selection method, survivor selection method, population size, offspring size
                island.run_generation(5, 5, self.population_size_per_island, self.tournament_size)
                best_fitness = island.best_fitness()
                avg_fitness = island.avg_fitnness()
                best_fitnesses.append(best_fitness)
                avg_fitnesses.append(avg_fitness)
                print(f"Best fitness for Island {island_index + 1} in generation {generation + 1}: {best_fitness}")
        
        return best_fitnesses, avg_fitnesses

            # Perform migration at the specified interval
            # if (generation + 1) % self.migration_interval == 0:
            #     self.migrate()
            #     print(f"Migration occurred at generation {generation + 1}.")

        
    

# Set parameters
num_islands = 1
population_size_per_island = 100
num_generations = 1000
migration_interval = 10
migration_rate = 0.6
mutation_rate = 0.5
tournament_size = 50

# Initialize island model
model = IslandModels(num_islands, population_size_per_island, num_generations, migration_interval, migration_rate, mutation_rate, tournament_size)

# Run evolution
best_fitnesses, avg_fitnesses = model.evolve()

x = np.arange(1, num_generations+1, 1)


plt.plot(x, best_fitnesses, label="Best fitness")
plt.plot(x, avg_fitnesses, label="Average fitness")
plt.xlabel("Num generations")
plt.ylabel("Average/Best fitness")
plt.savefig("knap.png")	

