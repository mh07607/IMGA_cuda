import numpy as np
import random
import tsp

class IslandModel:
    def __init__(self, num_islands, population_size_per_island, num_generations, migration_interval, migration_rate, mutation_rate, tournament_size):
        
        self.num_islands = num_islands
        self.population_size_per_island = population_size_per_island
        self.num_generations = num_generations
        self.migration_interval = migration_interval
        self.migration_rate = migration_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.islands = self.initialize_islands()

    def initialize_islands(self):
        islands = []
        for _ in range(self.num_islands):
            indi_islands = []
            for i in population_size_per_island:
                #  population size of 100 passed in the tsp problem
                chormosome = tsp.EvolAlgo(100)
                indi_islands.append(chormosome)

            islands.append(chormosome)

        return islands

    def selection(self, population):
        # indiviudal parameters set for TSP problem over here 
        for island_numb in range(self.num_islands):
            chormo = self.islands[island_numb]
            for indi_numb in range(self.population_size_per_island):
                chormo[indi_numb]


    def crossover(self, parent1, parent2):
        pass

    def mutation(self, individual):
        pass

    def evaluate_population(self, population):
        pass

    def migrate_between_islands(self, island1, island2):
        pass

    def evolve(self):
        for generation in range(self.num_generations):
            # Evaluate populations on each island
            for island in self.islands:
                island_fitness = self.evaluate_population(island)
            
            # Perform selection, crossover, and mutation on each island
            for island in self.islands:
                selected_individuals = self.selection(island)
                new_population = []
                for i in range(self.population_size_per_island):
                    parent1 = random.choice(selected_individuals)
                    parent2 = random.choice(selected_individuals)
                    child = self.crossover(parent1, parent2)
                    child = self.mutation(child)
                    new_population.append(child)
                island[:] = new_population
            
            # Perform migration between islands
            if generation % self.migration_interval == 0:
                for i in range(self.num_islands):
                    for j in range(i+1, self.num_islands):
                        self.migrate_between_islands(self.islands[i], self.islands[j])
            
            # Optionally, implement stopping criteria if a satisfactory solution is found
            
        # Select the best individual from all islands as the final solution
        best_solution = None
        best_fitness = float('inf')
        for island in self.islands:
            for individual in island:
                if self.evaluate_individual(individual) < best_fitness:
                    best_solution = individual
                    best_fitness = self.evaluate_individual(individual)

        print("Best solution:", best_solution)
        print("Best fitness:", best_fitness)

    def evaluate_individual(self, individual):
        pass


# Example usage
num_islands = 5
population_size_per_island = 50
num_generations = 100
migration_interval = 10
migration_rate = 0.1
mutation_rate = 0.1
tournament_size = 5
num_cities = 10  # You need to define this variable somewhere

island_model = IslandModel(num_islands, population_size_per_island, num_generations, migration_interval, migration_rate, mutation_rate, tournament_size)
island_model.evolve()





import numpy as np
import copy
import random


city_list = []
city_dict = {}

def read_and_convert_to_dict(file_path):

    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into parts
            parts = line.strip().split()

            # Extract key and coordinates
            key = int(parts[0])

            # adding the city to the list as well to keep record
            city_list.append(key)

            coordinates = tuple(map(float, parts[1:]))

            # Create dictionary entry
            city_dict[key] = coordinates

    return city_dict

# Example usage:
file_path = 'data.txt'  # Replace with the path to your text file
result_dict = read_and_convert_to_dict(file_path)

# Print the result
# print(result_dict)
print(city_list)

# inverse distance to be set and then seen as what needs to be achieved

class chormosome():
    def __init__(self, comb: list):
        self.indi = comb[:]

        self.distance = 0
        self.chorm_distance(self.indi) # allowing a new distance value to be generated


    def city_distance(self, x: tuple, y: tuple):
    # x and y are two 2D points each not 2 coordinates of one 2D point.
        return ((x[0]-y[0])**2 + (x[1]-y[1])**2)**0.5
    
    
    def get_distance(self):
        return self.distance
    

    def chorm_distance(self, indi: list):
        for i in range(len(indi)-1):
            self.distance += self.city_distance(city_dict[indi[i]], city_dict[indi[i+1]])
            
    
    # mutation algorithm to be changed rn
    def mutation(self, mutation_rate:int):
        rand_numb = random.randint(0,1)
        if rand_numb <= mutation_rate:
            # mutation defined by reversing orders for now and to be changed
            i = random.randint(0,193)
            j = random.randint(0,193)
            self.indi[i],self.indi[j] = self.indi[j], self.indi[i]
         
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
            city_lst = copy.deepcopy(city_list)
            np.random.shuffle(city_list)
            choromo1 = chormosome(city_lst)
            island.append(choromo1)

        return island

    # would done idividually to each island and the chromosomes in it
    def crossover(self, chrom1, chrom2, mut_rate):
        # scheme to be implemented using crossover scheme where a randomly selected set of parent 1 would displace the randomly selected set of parent 2

        start = random.randint(50,75)
        end = random.randint(76, 120)
        # start = random.randint(1, 4)
        # end = random.randint(5, 8)
        parent1 = chrom1.indi
        parent2 = chrom2.indi

        offspring1 = [None] * 194
        offspring2 = [None] * 194

        offspring1[start:end+1] = parent1[start:end+1]
        offspring2[start:end+1] = parent2[start:end+1]
        # print(offspring1)
        # print(offspring2)
        # print(end)
        pointer = end + 1
        parent1_pointer = end + 1
        parent2_pointer = end + 1
        # print(start,end, parent2_pointer)
        counter = 0

        while pointer != start:
            # print(pointer, parent2[parent2_pointer])   
            if parent2[parent2_pointer] not in offspring1:    
                offspring1[pointer] = parent2[parent2_pointer]
                pointer += 1
            parent2_pointer += 1
            if parent2_pointer == len(offspring1):
                parent2_pointer = 0
            if pointer == len(offspring1):
                pointer = 0
            # counter+=1

        pointer = end + 1

        while pointer != start:
            # print(pointer, parent2[parent2_pointer])
            if parent1[parent1_pointer] not in offspring2: 
                offspring2[pointer] = parent1[parent1_pointer]
                pointer += 1
            parent1_pointer += 1
            if parent1_pointer == len(offspring2):
                parent1_pointer = 0
            if pointer == len(offspring2):
                pointer = 0

        offspring1 = chormosome(offspring1)
        offspring2 = chormosome(offspring2)
        offspring1.mutation(mut_rate)
        offspring2.mutation(mut_rate)

        return [offspring1, offspring2] 

    def total_fitness(self):
        total = 0
        for i in self.island:
            total += i.get_distance()
        return total
    
    def best_fitness(self):
        best = self.island[0].get_distance()
        for i in self.population[1:]:
            if i.get_distance() < best:
                best = i.get_distance()
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
        result.sort(key=lambda k : k.get_distance())
        # print(result)
        return result[:size]
    
    def binary_tournament_selection(self, size):
        result= []
        for i in range(size):
            ind1, ind2 = random.sample(self.island, 2)
            selected = ind1 if ind1.get_distance() < ind2.get_distance() else ind2
            # print(ind1.get_distance(), ind2.get_distance())
            result.append(selected)
            # print(selected.get_distance())
        return result
    
    def fitness_proportional_selection(self,size):
        total_fitness = sum(chromo.distance for chromo in self.island)
        selection_probs = [chromo.distance / total_fitness for chromo in self.island]
        selected_indices = np.random.choice(range(self.population_size), size=self.population_size, replace=True, p=selection_probs)
        return [self.island[i] for i in selected_indices[:size]]
    

    #     # print(normalized_range)
    #     return result
    def rank_selection(self, size):
        self.island.sort(key=lambda chromo: chromo.distance)
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

            print("the best fit for the ", i, " generation is: ", self.best_fitness())

    # def run_model(self):
    #     pass  

class IslandModels:
    def __init__(self, num_islands, population_size_per_island, num_generations, migration_interval, migration_rate, mutation_rate, tournament_size):
        self.num_islands = num_islands  # telling the number of islands present 
        self.population_size_ = population_size_per_island # population of the islands
        self.num_generations = num_generations # for how many
        self.migration_interval = migration_interval  # informing the number of genererations after migrations occurs
        self.migration_rate = migration_rate # number of individuals transfered during each migration
        self.mutation_rate = mutation_rate 
        self.tournament_size = tournament_size

        self.islands = [Island(population_size_per_island, mutation_rate) for _ in range(num_islands)]

    def migrate(self):
        migrants_per_island = int(self.population_size_per_island * self.migration_rate)
        
        # Select migrants from each island
        migrants = [island.truncation_selection(migrants_per_island) for island in self.islands]
        
        # Migrate between islands in a circular fashion
        for i in range(self.num_islands):
            target_island = (i + 1) % self.num_islands
            self.islands[target_island].island.extend(migrants[i])
            # After migration, trim each island's population to maintain size
            self.islands[target_island].island = self.islands[target_island].truncation_selection(self.population_size_per_island)
    
    def evolve(self):
        for generation in range(self.num_generations):
            # Evolve each island
            for island in self.islands:
                # Assuming the arguments are in order: parent selection method, survivor selection method, population size, offspring size
                island.run_generation(2, 5, self.population_size_per_island, self.tournament_size)
            
            # Perform migration at the specified interval
            if (generation + 1) % self.migration_interval == 0:
                self.migrate()
                print(f"Migration occurred at generation {generation + 1}.")
    
        
myalgo = IslandModels(300)
a = (myalgo.population[0])
# print("list after printing is", a.indi)
# result = myalgo.crossover((myalgo.population[0]), (myalgo.population[1]),0.5)

# result = myalgo.binary_tournament_selection()

myalgo.run_generation(4,2,300,150,0.8,10000)